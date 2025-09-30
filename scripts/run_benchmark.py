# matsushibadenki/snn/scripts/run_benchmark.py
# 複数のタスクを実行可能な、新しいベンチマークスイート
#
# 変更点:
# - mypyエラーを解消するため、型ヒントの修正と、結果を格納する辞書の扱いを変更。

import argparse
import time
import pandas as pd  # type: ignore
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import Dict, Any

from snn_research.benchmark.tasks import SST2Task, XSumTask

# タスク名とクラスのマッピング
TASK_REGISTRY = {
    "sst2": SST2Task,
    "xsum": XSumTask,
}

def run_single_task(task_name: str, device: str):
    """単一のベンチマークタスクを実行する。"""
    print("\n" + "="*20 + f" 🚀 Starting Benchmark for: {task_name.upper()} " + "="*20)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    TaskClass = TASK_REGISTRY[task_name]
    task = TaskClass(tokenizer, device)

    _, val_dataset = task.prepare_data(data_dir="data")
    val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=task.get_collate_fn())
    
    results = []
    for model_type in ['SNN', 'ANN']:
        print(f"\n--- Evaluating {model_type} model ---")
        model = task.build_model(model_type, tokenizer.vocab_size).to(device)
        
        start_time = time.time()
        metrics = task.evaluate(model, val_loader)
        duration = time.time() - start_time
        
        result_record: Dict[str, Any] = {
            "model": model_type,
            "task": task_name,
            "eval_time_sec": duration,
        }
        result_record.update(metrics)
        
        results.append(result_record)
        print(f"  - Results: {result_record}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="SNN vs ANN Benchmark Suite")
    parser.add_argument(
        "--task", 
        type=str, 
        default="all", 
        choices=["all"] + list(TASK_REGISTRY.keys()),
        help="実行するベンチマークタスクを選択します。"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tasks_to_run = TASK_REGISTRY.keys() if args.task == "all" else [args.task]
    
    all_results = []
    for task_name in tasks_to_run:
        all_results.extend(run_single_task(task_name, device))

    print("\n\n" + "="*25 + " 🏆 Final Benchmark Summary " + "="*25)
    df = pd.DataFrame(all_results)
    print(df.to_string())
    print("="*75)

if __name__ == "__main__":
    main()

