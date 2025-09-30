# /snn_research/cognitive_architecture/astrocyte_network.py
# Phase 4: ニューロン群の活動を長期的に調整するアストロサイト・ネットワーク
#
# 機能:
# - グローバルな発火活動を監視し、ネットワーク全体の恒常性を維持する。
# - 特定のニューロン群の過活動や非活動を検知し、パラメータ（例: 発火閾値）を調整する。
# - 学習の安定化と、エネルギー効率の最適化に貢献する。

import torch
import torch.nn as nn
from typing import List, Dict

from snn_research.core.snn_core import AdaptiveLIFNeuron

class AstrocyteNetwork:
    """
    SNN全体の活動を監視し、恒常性を維持するためのグリア細胞様ネットワーク。
    """
    def __init__(self, snn_model: nn.Module, monitoring_interval: int = 100):
        self.snn_model = snn_model
        self.monitoring_interval = monitoring_interval
        self.step_counter = 0
        
        # 監視対象となる適応的ニューロン層を登録
        self.monitored_neurons: List[AdaptiveLIFNeuron] = [
            m for m in self.snn_model.modules() if isinstance(m, AdaptiveLIFNeuron)
        ]
        
        # 各層の長期的な平均発火率を記録する
        self.long_term_spike_rates: Dict[str, torch.Tensor] = {}
        print(f"✨ アストロサイト・ネットワークが {len(self.monitored_neurons)} 個のニューロン層の監視を開始しました。")

    def step(self):
        """
        学習または推論の各ステップで呼び出され、内部カウンターをインクリメントする。
        一定間隔で監視・調整ロジックをトリガーする。
        """
        self.step_counter += 1
        if self.step_counter % self.monitoring_interval == 0:
            self.monitor_and_regulate()

    @torch.no_grad()
    def monitor_and_regulate(self):
        """
        ネットワーク全体の活動を監視し、必要に応じて調整を行う。
        """
        print(f"\n🔬 アストロサイトによるグローバル活動監視 (ステップ: {self.step_counter})")
        
        for i, layer in enumerate(self.monitored_neurons):
            layer_name = f"AdaptiveLIF_{i}"
            
            # 現在のバッチでの平均発火率を取得 (spikeはバッファではないため直接アクセス不可)
            # このため、実際の監視はTrainer内から行うのがより現実的。
            # ここではダミーとして、閾値から発火率を推定する。
            current_threshold = layer.adaptive_threshold.mean().item()
            estimated_rate = 1 / (1 + current_threshold) # 簡易的な推定
            
            # 長期的な発火率を更新 (指数移動平均)
            if layer_name in self.long_term_spike_rates:
                self.long_term_spike_rates[layer_name] = (
                    0.99 * self.long_term_spike_rates[layer_name] + 0.01 * estimated_rate
                )
            else:
                self.long_term_spike_rates[layer_name] = torch.tensor(estimated_rate)

            long_term_rate = self.long_term_spike_rates[layer_name].item()
            target_rate = layer.target_spike_rate
            
            print(f"  - 層 {layer_name}: 長期平均発火率(推定)={long_term_rate:.4f} (目標: {target_rate:.4f})")

            # 恒常性維持のための調整 (ホメオスタティック可塑性)
            # 発火率が目標から大きく外れている場合、適応強度を調整する
            if abs(long_term_rate - target_rate) > target_rate * 0.5:
                if long_term_rate > target_rate:
                    # 発火しすぎている -> 閾値適応をより強くする
                    new_strength = layer.adaptation_strength * 1.05
                    print(f"    - ⚠️ 過活動を検知。適応強度を調整します: {layer.adaptation_strength:.4f} -> {new_strength:.4f}")
                    layer.adaptation_strength = new_strength
                else:
                    # 発火が少なすぎる -> 閾値適応を弱くする
                    new_strength = layer.adaptation_strength * 0.95
                    print(f"    - ⚠️ 活動低下を検知。適応強度を調整します: {layer.adaptation_strength:.4f} -> {new_strength:.4f}")
                    layer.adaptation_strength = new_strength