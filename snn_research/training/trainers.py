# matsushibadenki/snn/snn_research/training/trainers.py
# SNNモデルの学習と評価ループを管理するTrainerクラス (モニタリング・評価機能完備)
# 
# 機能:
# - Metal (mps) デバイスに対応。
# - TensorBoardと連携し、学習・検証のメトリクスをリアルタイムで可視化。
# - 検証データセットでモデル性能を評価する `evaluate` メソッドを実装。
# - 検証結果に基づき、最も性能の良いモデルを `best_model.pth` として保存する機能を追加。
# - チェックポイントの保存・読み込みロジックを修正し、バッファを除外して再開時のサイズミスマッチエラーを解消。

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import collections
from tqdm import tqdm  # type: ignore
from typing import Tuple, Dict, Any, Optional
import shutil

from snn_research.training.losses import CombinedLoss, DistillationLoss
from torch.utils.tensorboard import SummaryWriter

class BreakthroughTrainer:
    """モニタリングと評価機能を完備した、SNNの統合トレーニングシステム。"""
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                 scheduler: Optional[torch.optim.lr_scheduler.LRScheduler], device: str,
                 grad_clip_norm: float, rank: int, use_amp: bool, log_dir: str):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.grad_clip_norm = grad_clip_norm
        self.rank = rank
        # mpsはGradScalerをサポートしないため、use_ampを無効化
        self.use_amp = use_amp and self.device != 'mps'
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.best_metric = float('inf')
        
        if self.rank in [-1, 0]:
            self.writer = SummaryWriter(log_dir)
            print(f"✅ TensorBoard logging enabled. Log directory: {log_dir}")

    def _run_step(self, batch: Tuple[torch.Tensor, ...], is_train: bool) -> Dict[str, Any]:
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        input_ids, target_ids = [t.to(self.device) for t in batch[:2]]
        
        # mpsでもautocastは利用可能
        with torch.amp.autocast(device_type=self.device, enabled=self.use_amp):
            with torch.set_grad_enabled(is_train):
                logits, spike_data = self.model(input_ids, return_spikes=True)
                loss_dict = self.criterion(logits, target_ids, spike_data)
        
        if is_train:
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss_dict['total']).backward()
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict['total'].backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            if isinstance(self.criterion, CombinedLoss):
                ignore_idx = self.criterion.ce_loss_fn.ignore_index
                mask = target_ids != ignore_idx
                accuracy = (preds[mask] == target_ids[mask]).sum().float() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0)
                loss_dict['accuracy'] = accuracy

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        total_metrics: Dict[str, float] = collections.defaultdict(float)
        num_batches = len(dataloader)
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}", disable=(self.rank not in [-1, 0]))
        
        for batch in progress_bar:
            metrics = self._run_step(batch, is_train=True)
            for key, value in metrics.items(): total_metrics[key] += value

        if self.scheduler: self.scheduler.step()
        
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
        
        if self.rank in [-1, 0]:
            for key, value in avg_metrics.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)
            self.writer.add_scalar('Train/learning_rate', self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]['lr'], epoch)
        
        return avg_metrics

    def evaluate(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """検証データセットでモデルを評価する。"""
        total_metrics: Dict[str, float] = collections.defaultdict(float)
        num_batches = len(dataloader)
        progress_bar = tqdm(dataloader, desc=f"Evaluating Epoch {epoch}", disable=(self.rank not in [-1, 0]))
        
        with torch.no_grad():
            for batch in progress_bar:
                metrics = self._run_step(batch, is_train=False)
                for key, value in metrics.items(): total_metrics[key] += value
        
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
        
        if self.rank in [-1, 0]:
            print(f"Epoch {epoch} Validation Results: " + ", ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()]))
            for key, value in avg_metrics.items():
                self.writer.add_scalar(f'Validation/{key}', value, epoch)
        
        return avg_metrics

    def save_checkpoint(self, path: str, epoch: int, metric_value: float, **kwargs: Any):
        if self.rank in [-1, 0]:
            model_to_save = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
            
            buffer_names = {name for name, _ in model_to_save.named_buffers()}
            model_state = {k: v for k, v in model_to_save.state_dict().items() if k not in buffer_names}

            state = {
                'epoch': epoch, 'model_state_dict': model_state, 
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_metric': self.best_metric
            }
            if self.use_amp:
                state['scaler_state_dict'] = self.scaler.state_dict()
            if self.scheduler: 
                state['scheduler_state_dict'] = self.scheduler.state_dict()
            state.update(kwargs)
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(state, path)
            print(f"✅ チェックポイントを '{path}' に保存しました (Epoch: {epoch})。")
            
            is_best = metric_value < self.best_metric
            if is_best:
                self.best_metric = metric_value
                best_path = os.path.join(os.path.dirname(path), 'best_model.pth')
                
                temp_state_for_best = {'model_state_dict': model_state}
                temp_state_for_best.update(kwargs)
                torch.save(temp_state_for_best, best_path)
                print(f"🏆 新しいベストモデルを '{best_path}' に保存しました (Metric: {metric_value:.4f})。")

    def load_checkpoint(self, path: str) -> int:
        if not os.path.exists(path):
            print(f"⚠️ チェックポイントファイルが見つかりません: {path}。最初から学習を開始します。")
            return 0
            
        map_location = self.device
        checkpoint = torch.load(path, map_location=map_location)
        
        model_to_load = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
        
        checkpoint_state_dict = checkpoint['model_state_dict']
        model_buffer_names = {name for name, _ in model_to_load.named_buffers()}
        keys_to_remove = [k for k in checkpoint_state_dict if k in model_buffer_names]
        if keys_to_remove:
            print(f"古いチェックポイントからバッファキーを削除します: {keys_to_remove}")
            for k in keys_to_remove:
                del checkpoint_state_dict[k]

        model_to_load.load_state_dict(checkpoint_state_dict, strict=False)
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.best_metric = checkpoint.get('best_metric', float('inf'))
        start_epoch = checkpoint.get('epoch', 0)
        print(f"✅ チェックポイント '{path}' を正常にロードしました。Epoch {start_epoch} から学習を再開します。")
        return start_epoch


class DistillationTrainer(BreakthroughTrainer):
    """知識蒸留に特化したトレーナー（モニタリング・評価機能完備）。"""
    def _run_step(self, batch: Tuple[torch.Tensor, ...], is_train: bool) -> Dict[str, Any]:
        if is_train: self.model.train()
        else: self.model.eval()
            
        student_input, student_target, teacher_logits = [t.to(self.device) for t in batch]

        with torch.amp.autocast(device_type=self.device, enabled=self.use_amp):
            with torch.set_grad_enabled(is_train):
                student_logits, spike_data = self.model(student_input, return_spikes=True)
                assert isinstance(self.criterion, DistillationLoss)
                loss_dict = self.criterion(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    targets=student_target,
                    spikes=spike_data
                )
        
        if is_train:
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss_dict['total']).backward()
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict['total'].backward()
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
        
        return {k: v.cpu().item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
