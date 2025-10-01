# snn_research/learning_rules/reward_modulated_stdp.py
# Title: 報酬変調型STDP
# Description: STDPを大域的な報酬信号で変調する強化学習ルールを実装します。

import torch
from typing import Dict, Any, Optional
from .stdp import STDP # STDPのトレース更新ロジックを継承

class RewardModulatedSTDP(STDP):
    """STDPと適格性トレース(Eligibility Trace)を用いた報酬ベースの学習ルール。"""
    def __init__(self, learning_rate: float, a_plus: float, a_minus: float, tau_trace: float, tau_eligibility: float, dt: float = 1.0):
        super().__init__(learning_rate, a_plus, a_minus, tau_trace, dt)
        self.tau_eligibility = tau_eligibility
        self.eligibility_trace: Optional[torch.Tensor] = None

    def _initialize_eligibility_trace(self, weight_shape: tuple, device: torch.device):
        self.eligibility_trace = torch.zeros(weight_shape, device=device)

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        optional_params: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """報酬信号に基づいて重み変化量を計算する。"""
        # 親クラスのトレース初期化・更新を呼び出す
        if self.pre_trace is None or self.post_trace is None or self.pre_trace.shape[0] != pre_spikes.shape[0] or self.post_trace.shape[0] != post_spikes.shape[0]:
            self._initialize_traces(pre_spikes.shape[0], post_spikes.shape[0], pre_spikes.device)
        self._update_traces(pre_spikes, post_spikes)
        
        # 適格性トレースの初期化
        if self.eligibility_trace is None or self.eligibility_trace.shape != weights.shape:
            self._initialize_eligibility_trace(weights.shape, weights.device)
        
        # mypyにNoneでないことを伝える
        assert self.pre_trace is not None and self.post_trace is not None and self.eligibility_trace is not None

        # 1. STDPライクなルールで適格性トレースを更新
        self.eligibility_trace += self.a_plus * torch.outer(post_spikes, self.pre_trace)
        self.eligibility_trace -= self.a_minus * torch.outer(self.post_trace, pre_spikes).T

        # 2. 適格性トレースを時間減衰させる
        self.eligibility_trace -= (self.eligibility_trace / self.tau_eligibility) * self.dt
        
        # 3. 報酬信号を受け取った時にのみ、重みを更新
        reward = optional_params.get("reward", 0.0) if optional_params else 0.0
        
        if reward != 0.0:
            dw = self.learning_rate * reward * self.eligibility_trace
            self.eligibility_trace *= 0.0 # 更新後にリセット
            return dw
        
        return torch.zeros_like(weights)