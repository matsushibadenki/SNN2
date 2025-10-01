# snn_research/learning_rules/base_rule.py
# Title: 学習ルールの抽象基底クラス
# Description: 全ての学習ルールクラスが継承すべき基本構造を定義します。

from abc import ABC, abstractmethod
import torch
from typing import Dict, Any, Optional

class BioLearningRule(ABC):
    """生物学的学習ルールのための抽象基底クラス。"""

    @abstractmethod
    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        optional_params: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        シナプス重みの変化量を計算する。
        このメソッドは各タイムステップで呼び出される。

        Args:
            pre_spikes (torch.Tensor): シナプス前ニューロンの発火
            post_spikes (torch.Tensor): シナプス後ニューロンの発火
            weights (torch.Tensor): 現在のシナプス重み
            optional_params (Optional[Dict[str, Any]]): オプションのパラメータ（例: 報酬信号）

        Returns:
            torch.Tensor: 計算された重み変化量 (dw)
        """
        pass