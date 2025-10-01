# /snn_research/cognitive_architecture/planner_snn.py
# Phase 3: 学習可能な階層的思考プランナーSNN
#
# 機能:
# - 自然言語のタスク要求を入力として受け取る。
# - 利用可能な専門家スキル（サブタスク）の最適な実行順序を予測して出力する。
# - BreakthroughSNNをベースアーキテクチャとして使用する。

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any

from snn_research.core.snn_core import BreakthroughSNN

class PlannerSNN(BreakthroughSNN):
    """
    タスク要求からサブタスクのシーケンスを生成することに特化したSNNモデル。
    """
    def __init__(self, vocab_size: int, d_model: int, d_state: int, num_layers: int, time_steps: int, n_head: int, num_skills: int, neuron_config: Optional[Dict[str, Any]] = None):
        """
        Args:
            num_skills (int): 予測対象となるスキル（サブタスク）の総数。
        """
        super().__init__(vocab_size, d_model, d_state, num_layers, time_steps, n_head, neuron_config=neuron_config)
        
        # BreakthroughSNNの出力層を、スキルを予測するための分類層に置き換える
        self.output_projection = nn.Linear(d_model, num_skills)
        print(f"🧠 学習可能プランナーSNNが {num_skills} 個のスキルを認識して初期化されました。")

    def forward(
        self, 
        input_ids: torch.Tensor, 
        return_spikes: bool = False, 
        return_full_mems: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        フォワードパスを実行し、スキル予測ロジット、スパイク、膜電位を返す。
        """
        # super().forward()を呼び出すと、このクラスで上書きされたself.output_projectionが内部で使われる。
        # その結果、skill_logits_over_timeは [batch, seq_len, num_skills] の形状を持つ。
        skill_logits_over_time, spikes, mem = super().forward(
            input_ids, 
            return_spikes=return_spikes, 
            return_full_mems=return_full_mems
        )
        
        # 最終タイムステップのロジットをプーリングして、最終的な計画予測とする
        final_skill_logits = skill_logits_over_time[:, -1, :]
        
        return final_skill_logits, spikes, mem
