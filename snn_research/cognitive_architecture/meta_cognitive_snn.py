# /snn_research/cognitive_architecture/meta_cognitive_snn.py
# Phase 3: メタ認知SNN (SNAKE -改-)
#
# 機能:
# - ネットワーク全体の予測誤差（自由エネルギー）を監視し、その最小化を司る。
# - 予測誤差が大きい情報源（ニューロン層）に動的に注意を割り当て、学習を促進する。
# - AstrocyteNetworkの恒常性維持機能に加え、より能動的な学習制御を行う。

import torch
import torch.nn as nn
from typing import List, Dict

from snn_research.core.snn_core import AdaptiveLIFNeuron

class MetaCognitiveSNN:
    """
    主要SNNモデルの活動と性能をメタレベルで監視・制御するネットワーク。
    通称 SNAKE (Spiking Neural Attention and Knowledge Engine)。
    """
    def __init__(
        self, 
        snn_model: nn.Module, 
        error_threshold: float = 0.8,
        modulation_strength: float = 0.05
    ):
        self.snn_model = snn_model
        self.error_threshold = error_threshold
        self.modulation_strength = modulation_strength
        
        # 監視対象となる適応的ニューロン層を登録
        self.monitored_neurons: List[AdaptiveLIFNeuron] = [
            m for m in self.snn_model.modules() if isinstance(m, AdaptiveLIFNeuron)
        ]
        
        print(f"🐍 メタ認知SNN (SNAKE) が {len(self.monitored_neurons)} 個のニューロン層の監視を開始しました。")

    @torch.no_grad()
    def monitor_and_modulate(self, current_loss: float):
        """
        現在の予測誤差（損失）を評価し、必要であれば特定の層への注意を変調させる。

        Args:
            current_loss (float): 現在の学習ステップにおける損失関数の値。
        """
        # 予測誤差が大きい場合（学習が困難な状況）に介入
        if current_loss > self.error_threshold and self.monitored_neurons:
            print(f"  - 🐍 SNAKE: 高い予測誤差 ({current_loss:.4f}) を検知。アテンションを変調します。")

            # 最も発火率が低い（＝活動が停滞している）層を見つけ、そこに注意を向ける
            # これにより、困難なタスクに対して未貢献のニューロンを活性化させる
            target_layer = min(
                self.monitored_neurons, 
                key=lambda layer: layer.adaptive_threshold.mean().item()
            )

            # 注意の変調: 適応強度を少し弱めることで、閾値が下がりやすくなり、発火を促す
            original_strength = target_layer.adaptation_strength
            new_strength = original_strength * (1.0 - self.modulation_strength)
            
            target_layer.adaptation_strength = new_strength
            
            print(f"    - 層 {target_layer.__class__.__name__} の活動を促進 (適応強度: {original_strength:.4f} -> {new_strength:.4f})")