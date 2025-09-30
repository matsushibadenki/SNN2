# matsushibadenki/snn2/snn_research/benchmark/tasks.py
# ベンチマークタスクの定義ファイル
#
# 変更点:
# - mypyエラーを解消するため、型ヒントの修正、ライブラリimportへの# type: ignore追加、
#   len()呼び出しのキャストなどを行った。
# - BreakthroughSNNの呼び出しを修正し、型推論エラーを解消。

import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Callable, Sized, cast
from datasets import load_dataset  # type: ignore
from tqdm import tqdm  # type: ignore
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from snn_research.core.snn_core import BreakthroughSNN, AdaptiveLIFNeuron
from snn_research.benchmark.ann_baseline import ANNBaselineModel
from snn_research.benchmark.metrics import calculate_accuracy

# --- 共通データセットクラス ---
class GenericDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
    def __len__(self) -> int: return len(self.data)
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

# --- ベンチマークタスクの基底クラス ---
class BenchmarkTask(ABC):
    """ベンチマークタスクの抽象基底クラス。"""
    def __init__(self, tokenizer: PreTrainedTokenizerBase, device: str):
        self.tokenizer = tokenizer
        self.device = device

    @abstractmethod
    def prepare_data(self, data_dir: str) -> Tuple[Dataset, Dataset]:
        """データセットを準備し、train/validationのDatasetオブジェクトを返す。"""
        pass

    @abstractmethod
    def get_collate_fn(self) -> Callable:
        """タスク固有のcollate_fnを返す。"""
        pass

    @abstractmethod
    def build_model(self, model_type: str, vocab_size: int) -> nn.Module:
        """タスクに適したSNNまたはANNモデルを構築する。"""
        pass
    
    @abstractmethod
    def evaluate(self, model: nn.Module, loader: DataLoader) -> Dict[str, Any]:
        """モデルを評価し、結果を辞書で返す。"""
        pass

# --- 感情分析タスク (SST-2) ---
class SST2Task(BenchmarkTask):
    """GLUEベンチマークのSST-2 (感情分析) タスク。"""
    
    def prepare_data(self, data_dir: str = "data") -> Tuple[Dataset, Dataset]:
        os.makedirs(data_dir, exist_ok=True)
        dataset = load_dataset("glue", "sst2")
        
        def _load_split(split):
            data = []
            for ex in dataset[split]:
                data.append({"text": ex['sentence'], "label": ex['label']})
            return GenericDataset(data)
            
        return _load_split("train"), _load_split("validation")

    def get_collate_fn(self) -> Callable:
        def collate_fn(batch: List[Dict[str, Any]]):
            texts = [item['text'] for item in batch]
            targets = [item['label'] for item in batch]
            tokenized = self.tokenizer(
                texts, padding=True, truncation=True, max_length=64, return_tensors="pt"
            )
            return {
                "input_ids": tokenized['input_ids'],
                "attention_mask": tokenized['attention_mask'],
                "labels": torch.tensor(targets, dtype=torch.long)
            }
        return collate_fn

    def build_model(self, model_type: str, vocab_size: int) -> nn.Module:
        # 分類タスク用にモデルをラップする
        class SNNClassifier(nn.Module):
            def __init__(self, snn_backbone):
                super().__init__()
                self.snn_backbone = snn_backbone
                self.classifier = nn.Linear(self.snn_backbone.d_model, 2)
            
            def forward(self, input_ids, **kwargs):
                # NOTE: SNNの出力から最後のタイムステップの特徴量を取得して分類
                logits, spikes, mem = self.snn_backbone(input_ids, return_spikes=True)
                pooled_output = logits[:, -1, :] # 最後のトークンの特徴量を使用
                return self.classifier(pooled_output), spikes

        if model_type == 'SNN':
            backbone = BreakthroughSNN(
                vocab_size=vocab_size,
                d_model=64,
                d_state=32,
                num_layers=2,
                time_steps=64,
                n_head=2,
                neuron_config={'type': 'lif'}
            )
            return SNNClassifier(backbone)
        else:
            ann_params = {'d_model': 64, 'd_hid': 128, 'nlayers': 2, 'nhead': 2}
            return ANNBaselineModel(vocab_size=vocab_size, **ann_params, num_classes=2)

    def evaluate(self, model: nn.Module, loader: DataLoader) -> Dict[str, Any]:
        model.eval()
        true_labels: List[int] = []
        pred_labels: List[int] = []
        total_spikes = 0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating SST-2"):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                targets = inputs.pop("labels")
                
                outputs, spikes = model(**inputs)
                if spikes is not None:
                    total_spikes += spikes.sum().item()
                
                preds = torch.argmax(outputs, dim=1)
                pred_labels.extend(preds.cpu().numpy())
                true_labels.extend(targets.cpu().numpy())
        
        avg_spikes = total_spikes / len(cast(Sized, loader.dataset)) if total_spikes > 0 else 0.0
        return {
            "accuracy": calculate_accuracy(true_labels, pred_labels),
            "avg_spikes": avg_spikes
        }

# --- 文章要約タスク (XSum) ---
class XSumTask(BenchmarkTask):
    """XSumデータセットを用いた文章要約タスク。"""
    
    def prepare_data(self, data_dir: str = "data") -> Tuple[Dataset, Dataset]:
        os.makedirs(data_dir, exist_ok=True)
        dataset = load_dataset("xsum", split="validation[:1%]") # 検証用に一部データを使用
        
        data = [{"document": ex['document'], "summary": ex['summary']} for ex in dataset]
        return GenericDataset(data), GenericDataset(data) # Train/Valに同じデータを使用

    def get_collate_fn(self) -> Callable:
        def collate_fn(batch: List[Dict[str, Any]]):
            inputs = [item['document'] for item in batch]
            targets = [item['summary'] for item in batch]
            
            tokenized_inputs = self.tokenizer(
                inputs, padding=True, truncation=True, max_length=256, return_tensors="pt"
            )
            # ターゲットは評価時にデコードするため、ここではテキストのまま保持
            return {
                "input_ids": tokenized_inputs['input_ids'],
                "attention_mask": tokenized_inputs['attention_mask'],
                "summaries": targets
            }
        return collate_fn

    def build_model(self, model_type: str, vocab_size: int) -> nn.Module:
        # 生成タスクなので、ベースモデルをそのまま使用
        if model_type == 'SNN':
            return BreakthroughSNN(
                vocab_size=vocab_size,
                d_model=64,
                d_state=32,
                num_layers=2,
                time_steps=256,
                n_head=2,
                neuron_config={'type': 'lif'}
            )
        else:
            # ANNのベースラインも生成モデルである必要がある (ここではダミーとして分類器を流用)
            ann_params = {'d_model': 64, 'd_hid': 128, 'nlayers': 2, 'nhead': 2}
            return ANNBaselineModel(vocab_size=vocab_size, **ann_params, num_classes=vocab_size)

    def evaluate(self, model: nn.Module, loader: DataLoader) -> Dict[str, Any]:
        model.eval()
        # ROUGEスコアの代わりに、生成されたテキストの平均長を簡易的なメトリクスとする
        total_gen_len = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating XSum"):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'summaries'}
                
                # ダミーの生成ロジック
                # 本来は generate メソッドを実装する必要がある
                outputs, _ , _ = model(**inputs)
                generated_ids = torch.argmax(outputs, dim=-1)
                
                total_gen_len += generated_ids.shape[1]
                
        return {"avg_summary_length": total_gen_len / len(cast(Sized, loader.dataset))}

