# matsushibadenki/snn2/train.py
# (旧 snn_research/training/main.py)
#
# 新しい統合学習実行スクリプト (完全版)
#
# 機能:
# - DIコンテナを使用し、学習に必要なコンポーネントを動的に組み立てる。
# - --override_config 引数で、コマンドラインから任意の設定を上書き可能。
# - 分散学習 (`--distributed`) に対応。
# - 勾配ベース学習と生物学的学習のパラダイムをconfigファイルで切り替え可能。
# - 既存の機能をすべて維持し、省略しない完全なコード。

import argparse
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, DistributedSampler
from dependency_injector.wiring import inject, Provide
from typing import Optional, Tuple, List, Dict, Any

from app.containers import TrainingContainer
from snn_research.data.datasets import get_dataset_class, DistillationDataset, DataFormat, SNNBaseDataset

from snn_research.training.trainers import BreakthroughTrainer, SelfSupervisedTrainer # ✨インポート追加
from snn_research.training.bio_trainer import BioTrainer

# DIコンテナのセットアップ
container = TrainingContainer()

@inject
def train(
    args,
    config=Provide[TrainingContainer.config],
    tokenizer=Provide[TrainingContainer.tokenizer],
):
    """学習プロセスを実行するメイン関数"""
    is_distributed = args.distributed
    rank = int(os.environ.get("LOCAL_RANK", -1))
    device = f'cuda:{rank}' if is_distributed and torch.cuda.is_available() else get_auto_device()
    
    paradigm = config.training.paradigm()
    is_distillation = paradigm == "gradient_based" and config.training.gradient_based.type() == "distillation"
    
    # データパスが指定されていなければconfigの値を使用
    data_path = args.data_path or config.data.path()
    
    DatasetClass = get_dataset_class(DataFormat(config.data.format()))
    dataset = DistillationDataset(
        file_path=os.path.join(data_path, "distillation_data.jsonl"), data_dir=data_path,
        tokenizer=tokenizer, max_seq_len=config.model.time_steps()
    ) if is_distillation else DatasetClass(
        file_path=data_path, tokenizer=tokenizer, max_seq_len=config.model.time_steps()
    )
        
    train_size = int((1.0 - config.data.split_ratio()) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size(), shuffle=(train_sampler is None),
        sampler=train_sampler, collate_fn=collate_fn(tokenizer, is_distillation)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.training.batch_size(), shuffle=False,
        collate_fn=collate_fn(tokenizer, is_distillation)
    )

    print(f"🚀 学習パラダイム '{paradigm}' で学習を開始します...")

    if paradigm in ["gradient_based", "self_supervised", "physics_informed"]:
        if is_distributed and paradigm != "gradient_based":
            raise NotImplementedError(f"{paradigm} learning does not support DDP yet.")
        
        snn_model = container.snn_model().to(device)
        if is_distributed:
            snn_model = DDP(snn_model, device_ids=[rank], find_unused_parameters=True)
        
        astrocyte = container.astrocyte_network(snn_model=snn_model) if args.use_astrocyte else None
        
        trainer: BreakthroughTrainer
        if paradigm == "gradient_based":
            optimizer = container.optimizer(params=snn_model.parameters())
            scheduler = container.scheduler(optimizer=optimizer) if config.training.gradient_based.use_scheduler() else None
            trainer_provider = container.distillation_trainer if is_distillation else container.standard_trainer
            trainer = trainer_provider(model=snn_model, optimizer=optimizer, scheduler=scheduler, device=device, rank=rank, astrocyte_network=astrocyte)
        elif paradigm == "self_supervised":
            optimizer = container.ssl_optimizer(params=snn_model.parameters())
            scheduler = container.ssl_scheduler(optimizer=optimizer) if config.training.self_supervised.use_scheduler() else None
            trainer = container.self_supervised_trainer(model=snn_model, optimizer=optimizer, scheduler=scheduler, device=device, rank=rank, astrocyte_network=astrocyte)
        else: # physics_informed
            optimizer = container.pi_optimizer(params=snn_model.parameters())
            scheduler = container.pi_scheduler(optimizer=optimizer) if config.training.physics_informed.use_scheduler() else None
            trainer = container.physics_informed_trainer(model=snn_model, optimizer=optimizer, scheduler=scheduler, device=device, rank=rank, astrocyte_network=astrocyte)
        
        start_epoch = trainer.load_checkpoint(args.resume_path) if args.resume_path and paradigm == "gradient_based" else 0
        for epoch in range(start_epoch, config.training.epochs()):
            if train_sampler: train_sampler.set_epoch(epoch)
            trainer.train_epoch(train_loader, epoch)
            if rank in [-1, 0] and (epoch % config.training.eval_interval() == 0 or epoch == config.training.epochs() - 1):
                val_metrics = trainer.evaluate(val_loader, epoch)
                if paradigm == "gradient_based" and epoch % config.training.log_interval() == 0:
                    checkpoint_path = os.path.join(config.training.log_dir(), f"checkpoint_epoch_{epoch}.pth")
                    trainer.save_checkpoint(
                        path=checkpoint_path, epoch=epoch, metric_value=val_metrics.get('total', float('inf')),
                        tokenizer_name=config.data.tokenizer_name(), config=config.model.to_dict()
                    )

    elif paradigm == "biologically_plausible":
        if is_distributed: raise NotImplementedError("Biologically plausible learning does not support DDP yet.")
        trainer = container.bio_trainer()
        for epoch in range(config.training.epochs()):
            trainer.train_epoch(train_loader, epoch, config.model.time_steps())
            if rank in [-1, 0] and (epoch % config.training.eval_interval() == 0 or epoch == config.training.epochs() - 1):
                trainer.evaluate(val_loader, epoch, config.model.time_steps())
    else:
        raise ValueError(f"Unknown training paradigm: '{paradigm}'")

    if rank in [-1, 0]: print("✅ 学習が完了しました。")


def collate_fn(tokenizer, is_distillation: bool):
    def collate(batch: List[Tuple[torch.Tensor, ...]]):
        inputs = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
        padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=tokenizer.pad_token_id)
        if is_distillation:
            logits = [item[2] for item in batch]
            padded_logits = torch.nn.utils.rnn.pad_sequence(logits, batch_first=True, padding_value=0.0)
            return padded_inputs, padded_targets, padded_logits
        return padded_inputs, padded_targets
    return collate
    
def get_auto_device() -> str:
    """実行環境に最適なデバイスを自動的に選択する。"""
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"
    
def main():
    parser = argparse.ArgumentParser(description="SNN 統合学習スクリプト")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="基本設定ファイル")
    parser.add_argument("--model_config", type=str, help="モデルアーキテクチャ設定ファイル")
    parser.add_argument("--data_path", type=str, help="データセットのパス（configを上書き）")
    parser.add_argument("--override_config", type=str, action='append', help="設定を上書き (例: 'training.epochs=5')")
    parser.add_argument("--distributed", action="store_true", help="分散学習を有効にする")
    parser.add_argument("--resume_path", type=str, help="チェックポイントから学習を再開する (gradient_basedのみ)")
    parser.add_argument("--use_astrocyte", action="store_true", help="アストロサイトネットワークを有効にする (gradient_based系のみ)")
    args = parser.parse_args()

    container.config.from_yaml(args.config)
    if args.model_config: container.config.from_yaml(args.model_config)
    if args.data_path: container.config.data.path.from_value(args.data_path)
    if args.override_config:
        for override in args.override_config:
            key, value = override.split('=', 1)
            container.config.from_dict({key: value})
    if args.distributed: dist.init_process_group(backend="nccl")
    
    # DIコンテナのwiring: main関数内で行うことで、設定読み込み後に依存関係を解決
    container.wire(modules=[__name__])
    
    train(args)
    if args.distributed: dist.destroy_process_group()

if __name__ == "__main__":
    main()

