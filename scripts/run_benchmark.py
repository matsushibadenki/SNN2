# matsushibadenki/snn/scripts/run_benchmark.py
# GLUEベンチマーク (SST-2) を用いたSNN vs ANN 性能評価スクリプト (最新アーキテクチャ対応版)
#
# 目的:
# - ロードマップ フェーズ2「2.3. アーキテクチャの進化」を評価。
# - 最新の `BreakthroughSNN` モデルをバックボーンとして使用し、その性能を直接測定する。
# - 旧式の `SNNClassifier` を廃止し、常にコアアーキテクチャの最新の性能を評価できるようにする。
# - [改善] 特定の学習済みモデルを直接評価するための --model_path 引数を追加。

import os
import json
import time
import argparse
import pandas as pd  # type: ignore
from datasets import load_dataset  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from tqdm import tqdm  # type: ignore
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import Dict, Any, List, Tuple

# 最新のSNNアーキテクチャをインポート
from snn_research.core.snn_core import BreakthroughSNN
from snn_research.benchmark.ann_baseline import ANNBaselineModel

# --- 1. データ準備 (変更なし) ---
def prepare_sst2_data(output_dir: str = "data") -> Dict[str, str]:
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    dataset = load_dataset("glue", "sst2")
    data_paths: Dict[str, str] = {}
    for split in ["train", "validation"]:
        jsonl_path = os.path.join(output_dir, f"sst2_{split}.jsonl")
        data_paths[split] = jsonl_path
        if os.path.exists(jsonl_path): continue
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for ex in tqdm(dataset[split], desc=f"Processing {split}"):
                f.write(json.dumps({"text": ex['sentence'], "label": ex['label']}) + "\n")
    return data_paths


# --- 2. 共通データセット (変更なし) ---
class ClassificationDataset(Dataset):
    def __init__(self, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
    def __len__(self) -> int: return len(self.data)
    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.data[idx]['text'], self.data[idx]['label']

def create_collate_fn_for_classification(tokenizer: PreTrainedTokenizerBase):
    def collate_fn(batch: List[Tuple[str, int]]):
        texts, targets = zip(*batch)
        tokenized = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
        return tokenized['input_ids'], tokenized['attention_mask'], torch.tensor(targets, dtype=torch.long)
    return collate_fn

# --- 3. 最新アーキテクチャを使用した分類モデル ---
class SNNClassifier(nn.Module):
    """BreakthroughSNNをバックボーンとして使用する分類器。"""
    def __init__(self, vocab_size: int, d_model: int, d_state: int, num_layers: int, time_steps: int, n_head: int, num_classes: int):
        super().__init__()
        # SNNバックボーンを初期化
        self.snn_backbone = BreakthroughSNN(vocab_size, d_model, d_state, num_layers, time_steps, n_head)
        # 分類用のヘッドを追加
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # バックボーンからプーリングされた特徴量とスパイク情報を取得
        # pool_method='mean' を使用して、分類に適した固定長ベクトルを得る
        pooled_features, spikes = self.snn_backbone(
            input_ids, 
            attention_mask=attention_mask,
            return_spikes=True, 
            pool_method='mean'
        )
        # 分類ヘッドでロジットを計算
        logits = self.classifier(pooled_features)
        return logits, spikes

# --- 4. 実行関数 (モデル呼び出し部分を更新) ---
def run_benchmark_for_model(
    model_type: str,
    data_paths: dict,
    tokenizer: PreTrainedTokenizerBase,
    model_params: dict,
    model_path: str = None
) -> Dict[str, Any]:
    print("\n" + "="*20 + f" 🚀 Starting {model_type} Benchmark " + "="*20)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset_val = ClassificationDataset(data_paths['validation'])
    
    collate_fn = create_collate_fn_for_classification(tokenizer)
    loader_val = DataLoader(dataset_val, batch_size=32, shuffle=False, collate_fn=collate_fn)

    vocab_size = tokenizer.vocab_size
    model: nn.Module
    if model_type == 'SNN':
        # 新しいSNNClassifierを使用
        model = SNNClassifier(vocab_size=vocab_size, **model_params, num_classes=2).to(device)
    else: # ANN
        model = ANNBaselineModel(vocab_size=vocab_size, **model_params, num_classes=2).to(device)

    # --- モデルのロードまたは学習 ---
    if model_path and os.path.exists(model_path):
        print(f"💾 学習済みモデルをロードします: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        print("🏋️ モデルをスクラッチから学習します...")
        dataset_train = ClassificationDataset(data_paths['train'])
        loader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, collate_fn=collate_fn)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(3):
            model.train()
            for input_ids, attention_mask, targets in tqdm(loader_train, desc=f"{model_type} Epoch {epoch+1}"):
                input_ids, attention_mask, targets = input_ids.to(device), attention_mask.to(device), targets.to(device)
                optimizer.zero_grad()
                
                if model_type == 'SNN':
                    outputs, _ = model(input_ids, attention_mask=attention_mask)
                else:
                    outputs = model(input_ids, src_padding_mask=(attention_mask == 0))

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

    print(f"{model_type} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    model.eval()
    true_labels: List[int] = []
    pred_labels: List[int] = []
    latencies: List[float] = []
    total_spikes: float = 0.0
    with torch.no_grad():
        for input_ids, attention_mask, targets in tqdm(loader_val, desc=f"{model_type} Evaluating"):
            input_ids, attention_mask, targets = input_ids.to(device), attention_mask.to(device), targets.to(device)
            start_time = time.time()
            if model_type == 'SNN':
                outputs, spikes = model(input_ids, attention_mask=attention_mask)
                total_spikes += spikes.sum().item()
            else:
                outputs = model(input_ids, src_padding_mask=(attention_mask == 0))

            latencies.append((time.time() - start_time) * 1000)
            preds = torch.argmax(outputs, dim=1)
            pred_labels.extend(preds.cpu().numpy())
            true_labels.extend(targets.cpu().numpy())
            
    accuracy = accuracy_score(true_labels, pred_labels)
    avg_latency_ms = sum(latencies) / len(latencies)
    avg_spikes_per_sample = total_spikes / len(dataset_val) if model_type == 'SNN' else 'N/A'

    print(f"  {model_type} Validation Accuracy: {accuracy:.4f}")
    print(f"  {model_type} Average Inference Time (per batch): {avg_latency_ms:.2f} ms")
    if model_type == 'SNN':
        print(f"  {model_type} Average Spikes per Sample: {avg_spikes_per_sample:,.2f}")
        
    return {"model": model_type, "accuracy": accuracy, "avg_latency_ms": avg_latency_ms, "avg_spikes_per_sample": avg_spikes_per_sample}

# --- 5. メイン実行ブロック (引数処理を追加) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNN vs ANN Benchmark Script")
    parser.add_argument("--model_path", type=str, help="評価対象の学習済みSNNモデルのパス。指定しない場合はSNNとANNをスクラッチから学習します。")
    args = parser.parse_args()

    pd.set_option('display.precision', 4)
    data_paths = prepare_sst2_data()
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    snn_params = {'d_model': 64, 'd_state': 32, 'num_layers': 2, 'time_steps': 64, 'n_head': 2}
    
    if args.model_path:
        # 指定されたSNNモデルのみを評価
        print(f"--- 特定のSNNモデルの評価を開始: {args.model_path} ---")
        snn_results = run_benchmark_for_model('SNN', data_paths, tokenizer, snn_params, model_path=args.model_path)
        print("\n\n" + "="*25 + " 🏆 Final Benchmark Results " + "="*25)
        results_df = pd.DataFrame([snn_results])
        print(results_df.to_string(index=False))
        print("="*75)
    else:
        # SNNとANNの両方を学習して比較
        snn_results = run_benchmark_for_model('SNN', data_paths, tokenizer, snn_params)
        ann_params = {'d_model': 64, 'd_hid': 128, 'nlayers': 2, 'nhead': 2}
        ann_results = run_benchmark_for_model('ANN', data_paths, tokenizer, ann_params)
        
        print("\n\n" + "="*25 + " 🏆 Final Benchmark Results " + "="*25)
        results_df = pd.DataFrame([snn_results, ann_results])
        print(results_df.to_string(index=False))
        print("="*75)
