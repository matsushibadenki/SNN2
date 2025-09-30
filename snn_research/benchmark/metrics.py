# matsushibadenki/snn2/snn_research/benchmark/metrics.py
# ベンチマーク評価用のメトリクス関数

from typing import List

def calculate_accuracy(true_labels: List[int], pred_labels: List[int]) -> float:
    """分類タスクの正解率を計算する。"""
    if len(true_labels) == 0:
        return 0.0
    correct = sum(1 for true, pred in zip(true_labels, pred_labels) if true == pred)
    return correct / len(true_labels)
