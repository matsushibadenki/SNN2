# matsushibadenki/snn/snn_research/core/snn_core.py
# SNNモデルの定義、次世代ニューロンなど、中核となるロジックを集約したライブラリ
#
# 変更点:
# - AttentionalSpikingSSMLayerのforwardメソッドを修正し、
#   バッチごとにシーケンス長が異なる場合に隠れ状態を正しく再初期化するようにした。
# - STDPSynapseクラスに、実際のSTDP学習ロジックを実装。
# - Izhikevichニューロンモデルを新たに追加。
# - BreakthroughSNN.forwardに `pool_method` 引数を追加。
#   これにより、言語モデリングだけでなく、分類タスク用の特徴量抽出器としても機能するようになった。
# - 返り値の型ヒントを更新。

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate, functional  # type: ignore
from typing import Tuple, Dict, Any, Optional, List, Union
from torch.utils.data import DataLoader
import os
import collections
import math
from collections import deque
from tqdm import tqdm  # type: ignore

class TTFSEncoder(nn.Module):
    """
    Time-to-First-Spike (TTFS) 符号化器
    連続値を最初のスパイクまでの時間（レイテンシ）に変換
    """
    def __init__(self, d_model: int, time_steps: int, max_latency: int = 10):
        super().__init__()
        self.d_model = d_model
        self.time_steps = time_steps
        self.max_latency = min(max_latency, time_steps)
        self.scaling = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_activated = torch.sigmoid(self.scaling * x)
        spike_times = self.max_latency * (1.0 - x_activated)
        spike_times = torch.round(spike_times).long()
        spikes = torch.zeros(x.shape[0], self.time_steps, x.shape[1], x.shape[2], device=x.device)
        spike_times = torch.clamp(spike_times, 0, self.time_steps - 1)
        spikes = spikes.scatter(1, spike_times.unsqueeze(1), 1.0)
        return spikes

class AdaptiveLIFNeuron(nn.Module):
    """
    適応的閾値を持つLIFニューロン
    """
    adaptive_threshold: torch.Tensor

    def __init__(self, features: int, tau: float = 2.0, base_threshold: float = 1.0, adaptation_strength: float = 0.1):
        super().__init__()
        self.features = features
        self.tau = tau
        self.base_threshold = base_threshold
        self.adaptation_strength = adaptation_strength
        self.register_buffer('adaptive_threshold', torch.ones(features) * base_threshold)
        self.surrogate_function = surrogate.ATan(alpha=2.0)
        self.mem_decay = math.exp(-1.0 / tau)

    def forward(self, x: torch.Tensor, v_mem: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v_potential = v_mem * self.mem_decay + x
        spike = self.surrogate_function(v_potential - self.adaptive_threshold)
        v_mem_new = v_potential * (1.0 - spike.detach())
        with torch.no_grad():
            self.adaptive_threshold += self.adaptation_strength * (spike.mean(dim=(0, 1)) - 0.1)
        return spike, v_mem_new

class MetaplasticLIFNeuron(nn.Module):
    """
    メタ可塑性を持つLIFニューロン
    """
    adaptive_threshold: torch.Tensor

    def __init__(self, features: int, tau: float = 2.0, threshold: float = 1.0, metaplastic_tau: float = 1000.0, metaplastic_strength: float = 0.1):
        super().__init__()
        self.features = features
        self.tau = tau
        self.base_threshold = threshold
        self.metaplastic_tau = metaplastic_tau
        self.metaplastic_strength = metaplastic_strength
        self.register_buffer('adaptive_threshold', torch.ones(features) * threshold)
        self.surrogate_function = surrogate.ATan(alpha=2.0)
        self.mem_decay = math.exp(-1.0 / tau)
        self.meta_decay = math.exp(-1.0 / metaplastic_tau)
    
    def forward(self, x: torch.Tensor, v_mem: torch.Tensor, activity_history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v_potential = v_mem * self.mem_decay + x
        current_threshold = self.adaptive_threshold * (1.0 + self.metaplastic_strength * activity_history.mean(dim=(0,1)))
        spike = self.surrogate_function(v_potential - current_threshold)
        v_mem_new = v_potential * (1.0 - spike.detach())
        activity_history_new = activity_history * self.meta_decay + spike.detach() * (1 - self.meta_decay)
        return spike, v_mem_new, activity_history_new

class IzhikevichNeuron(nn.Module):
    """
    Izhikevichニューロンモデル。
    多様な発火パターンを再現できる、計算効率の良いモデル。
    """
    v: torch.Tensor
    u: torch.Tensor

    def __init__(self, features: int, a: float = 0.02, b: float = 0.2, c: float = -65.0, d: float = 8.0, threshold: float = 30.0):
        super().__init__()
        self.features = features
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.threshold = threshold
        self.surrogate_function = surrogate.ATan(alpha=2.0)
        
        # 膜電位vと回復変数uをバッファとして登録
        self.register_buffer('v', torch.ones(features) * self.c)
        self.register_buffer('u', torch.ones(features) * self.c * self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 状態変数を現在のバッチサイズに合わせる
        if self.v.shape[0] != x.shape[0] or self.v.dim() != x.dim():
            self.v = torch.ones_like(x) * self.c
            self.u = torch.ones_like(x) * self.v * self.b

        # 膜電位と回復変数の更新
        self.v += 0.5 * (0.04 * self.v**2 + 5 * self.v + 140 - self.u + x)
        self.v += 0.5 * (0.04 * self.v**2 + 5 * self.v + 140 - self.u + x) # 2回適用して精度向上
        self.u += self.a * (self.b * self.v - self.u)

        # スパイク発火
        spike = self.surrogate_function(self.v - self.threshold)
        
        # スパイク後のリセット (detachして勾配計算から切り離す)
        spike_d = spike.detach()
        self.v = self.v * (1 - spike_d) + self.c * spike_d
        self.u = self.u * (1 - spike_d) + (self.u + self.d) * spike_d
        
        return spike

class STDPSynapse(nn.Module):
    """ 
    Spike-Timing-Dependent Plasticity シナプス 
    - forwardパスでは通常の線形層として振る舞う。
    - pre/postスパイクを受け取り、STDP則に基づいて重みを更新する。
    """
    pre_trace: torch.Tensor
    post_trace: torch.Tensor

    def __init__(self, in_features: int, out_features: int, 
                 tau_pre: float = 20.0, tau_post: float = 20.0,
                 A_pos: float = 0.01, A_neg: float = 0.005,
                 w_min: float = 0.0, w_max: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.A_pos = A_pos
        self.A_neg = A_neg
        self.w_min = w_min
        self.w_max = w_max
        
        self.weight = nn.Parameter(torch.rand(out_features, in_features) * (w_max - w_min) + w_min)
        
        # スパイクのトレース（痕跡）を記録するバッファ
        self.register_buffer('pre_trace', torch.zeros(1, in_features))
        self.register_buffer('post_trace', torch.zeros(1, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 通常の線形変換として動作 """
        return F.linear(x, self.weight)

    @torch.no_grad()
    def update_weights(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """ STDP則に基づいてシナプス重みを更新する """
        # トレースの減衰
        self.pre_trace *= math.exp(-1.0 / self.tau_pre)
        self.post_trace *= math.exp(-1.0 / self.tau_post)

        # スパイクがあったニューロンのトレースを増加
        self.pre_trace += pre_spikes.mean(dim=0, keepdim=True)
        self.post_trace += post_spikes.mean(dim=0, keepdim=True)

        # 重み更新量の計算
        # 1. ポストシナプスニューロンが発火した場合 (LTP: Long-Term Potentiation)
        #    -> pre_trace（直前のプリ側の活動）に応じて重みを増加
        delta_w_pos = self.A_pos * torch.outer(post_spikes.mean(dim=0), self.pre_trace.squeeze(0))
        
        # 2. プリシナプスニューロンが発火した場合 (LTD: Long-Term Depression)
        #    -> post_trace（直前のポスト側の活動）に応じて重みを減少
        delta_w_neg = self.A_neg * torch.outer(self.post_trace.squeeze(0), pre_spikes.mean(dim=0))
        
        # 重みの更新とクリッピング
        new_weight = self.weight + delta_w_pos - delta_w_neg
        self.weight.data = torch.clamp(new_weight, self.w_min, self.w_max)

class STPSynapse(nn.Module):
    """ 短期シナプス可塑性 """
    u: torch.Tensor
    x: torch.Tensor

    def __init__(self, in_features: int, out_features: int, tau_fac: float = 100.0, tau_dep: float = 200.0, U: float = 0.5, use_facilitation: bool = True, use_depression: bool = True):
        super().__init__()
        # ... (実装は省略)
        self.weight = nn.Parameter(torch.rand(out_features, in_features) * 0.5 + 0.25)
        if use_facilitation: self.register_buffer('u', torch.ones(1, in_features) * U)
        if use_depression: self.register_buffer('x', torch.ones(1, in_features))
        # ... (実装は省略)

    # ... (forward の実装は省略)

class EventDrivenSSMLayer(nn.Module):
    """
    Event-driven Spiking State Space Model
    """
    h_state: torch.Tensor
    state_v_mem: torch.Tensor
    output_v_mem: torch.Tensor

    def __init__(self, d_model: int, d_state: int = 64, dt: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt = dt
        self.A = nn.Parameter(torch.randn(d_state, d_state))
        self.B = nn.Parameter(torch.randn(d_state, d_model))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model, d_model))
        self.state_lif = AdaptiveLIFNeuron(d_state)
        self.output_lif = AdaptiveLIFNeuron(d_model)
        self.register_buffer('h_state', torch.zeros(1, 1, d_state))
        self.register_buffer('state_v_mem', torch.zeros(1, 1, d_state))
        self.register_buffer('output_v_mem', torch.zeros(1, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, seq_len, _ = x.shape
        h, state_v, output_v = self.h_state, self.state_v_mem, self.output_v_mem
        if h.shape[0] != batch_size or h.shape[1] != seq_len:
            h = torch.zeros(batch_size, seq_len, self.d_state, device=x.device)
            state_v = torch.zeros(batch_size, seq_len, self.d_state, device=x.device)
        if output_v.shape[0] != batch_size or output_v.shape[1] != seq_len:
            output_v = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)
        h, state_v, output_v = h.detach(), state_v.detach(), output_v.detach()
        outputs = []
        for t in range(time_steps):
            x_t = x[:, t, :, :]
            state_transition = F.linear(h, self.A)
            if torch.any(x_t > 0):
                input_projection = F.linear(x_t, self.B)
                state_update = state_transition + input_projection
                h, state_v = self.state_lif(state_update, state_v)
                output_projection = F.linear(h, self.C)
                output_update = output_projection + F.linear(x_t, self.D)
                out_spike, output_v = self.output_lif(output_update, output_v)
            else:
                h, state_v = self.state_lif(state_transition, state_v)
                out_spike = torch.zeros_like(x_t)
            outputs.append(out_spike)
        self.h_state, self.state_v_mem, self.output_v_mem = h.detach(), state_v.detach(), output_v.detach()
        return torch.stack(outputs, dim=1)

class AttentionalSpikingSSMLayer(nn.Module):
    """
    SSMの状態遷移にアテンション機構を統合したEvent-driven Spiking State Space Model。
    入力スパイクから、現在の状態にとって重要な情報を動的に選択して状態を更新する。
    """
    h_state: torch.Tensor
    state_v_mem: torch.Tensor
    output_v_mem: torch.Tensor

    def __init__(self, d_model: int, d_state: int = 64, n_head: int = 4, dt: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.n_head = n_head
        self.d_head = d_state // n_head
        assert self.d_head * n_head == self.d_state, "d_state must be divisible by n_head"

        # SSM パラメータ
        self.A = nn.Parameter(torch.randn(d_state, d_state))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model, d_model))

        # アテンション用線形層
        self.q_proj = nn.Linear(d_state, d_state) # Q from h_state
        self.k_proj = nn.Linear(d_model, d_state) # K from x_t
        self.v_proj = nn.Linear(d_model, d_state) # V from x_t (これが状態遷移行列Bの代わりになる)
        
        self.out_proj = nn.Linear(d_state, d_state) # アテンション結果を状態空間に戻す

        # ニューロン
        self.state_lif = AdaptiveLIFNeuron(d_state)
        self.output_lif = AdaptiveLIFNeuron(d_model)

        # 状態バッファ
        self.register_buffer('h_state', torch.zeros(1, 1, d_state))
        self.register_buffer('state_v_mem', torch.zeros(1, 1, d_state))
        self.register_buffer('output_v_mem', torch.zeros(1, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, seq_len, _ = x.shape
        
        # 状態変数の初期化とデバイス移動
        # バッチサイズかシーケンス長が変わっていたら、隠れ状態をリセット
        if self.h_state.shape[0] != batch_size or self.h_state.shape[1] != seq_len:
            h = torch.zeros(batch_size, seq_len, self.d_state, device=x.device)
            state_v = torch.zeros(batch_size, seq_len, self.d_state, device=x.device)
            output_v = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)
        else:
            h = self.h_state.clone().detach()
            state_v = self.state_v_mem.clone().detach()
            output_v = self.output_v_mem.clone().detach()

        outputs = []
        for t in range(time_steps):
            x_t = x[:, t, :, :]
            
            # 状態遷移 (A*h_t-1)
            state_transition = F.linear(h, self.A)
            
            # イベント駆動: スパイクがある場合のみアテンションと状態更新を計算
            if torch.any(x_t > 0):
                q = self.q_proj(h).view(batch_size * seq_len, self.n_head, self.d_head)
                k = self.k_proj(x_t).view(batch_size * seq_len, self.n_head, self.d_head)
                v = self.v_proj(x_t).view(batch_size * seq_len, self.n_head, self.d_head)

                attn_scores = torch.matmul(q.transpose(0,1), k.transpose(0,1).transpose(-2, -1)) / math.sqrt(self.d_head)
                attn_weights = F.softmax(attn_scores, dim=-1)
                
                attended_v = torch.matmul(attn_weights, v.transpose(0,1)).transpose(0,1)
                attended_v = attended_v.contiguous().view(batch_size, seq_len, self.d_state)
                
                gated_input = self.out_proj(attended_v)
                
                # 状態更新 (A*h_t-1 + Attention(h_t-1, x_t))
                state_update = state_transition + gated_input
                h, state_v = self.state_lif(state_update, state_v)
                
                # 出力計算 (C*h_t + D*x_t)
                output_projection = F.linear(h, self.C)
                output_update = output_projection + F.linear(x_t, self.D)
                out_spike, output_v = self.output_lif(output_update, output_v)
            else:
                # スパイクがない場合は状態遷移のみ
                h, state_v = self.state_lif(state_transition, state_v)
                out_spike = torch.zeros_like(x_t)
            
            outputs.append(out_spike)
            
        # 次のフォワードパスのために状態を保存
        self.h_state, self.state_v_mem, self.output_v_mem = h.detach(), state_v.detach(), output_v.detach()
        
        return torch.stack(outputs, dim=1)


class BreakthroughSNN(nn.Module):
    """
    AttentionalSpikingSSMLayerを統合した、次世代のSNNアーキテクチャ。
    言語モデリングと分類タスクの両方に対応可能。
    """
    def __init__(self, vocab_size: int, d_model: int, d_state: int, num_layers: int, time_steps: int, n_head: int):
        super().__init__()
        self.time_steps = time_steps
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.spike_encoder = TTFSEncoder(d_model=d_model, time_steps=time_steps)
        
        self.ssm_layers = nn.ModuleList([
            AttentionalSpikingSSMLayer(d_model=d_model, d_state=d_state, n_head=n_head) 
            for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                return_spikes: bool = False,
                pool_method: Optional[str] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids (torch.Tensor): 入力トークンID (batch_size, seq_len)
            attention_mask (Optional[torch.Tensor]): アテンションマスク (batch_size, seq_len)
            return_spikes (bool): スパイク列を返すかどうか
            pool_method (Optional[str]): 
                - None: 言語モデルとして動作。ロジットを返す (batch_size, seq_len, vocab_size)。
                - 'mean': 分類タスク用。シーケンス全体の特徴量を平均プーリングして返す (batch_size, d_model)。
                - 'last': 分類タスク用。最後の非パディングトークンの特徴量を返す (batch_size, d_model)。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - pooled_features または logits
                - hidden_states (スパイク列)
        """
        token_emb = self.token_embedding(input_ids)
        spike_sequence = self.spike_encoder(token_emb)
        
        hidden_states = spike_sequence
        for layer in self.ssm_layers:
            hidden_states = layer(hidden_states)
            
        # 時間方向にスパイクを積分 (平均化)
        time_integrated = hidden_states.mean(dim=1)

        # SpikingJellyのモジュール状態をリセット
        functional.reset_net(self)

        output: torch.Tensor
        if pool_method == 'mean':
            if attention_mask is not None:
                # パディング部分をマスクして平均を計算
                mask = attention_mask.unsqueeze(-1).expand_as(time_integrated).float()
                sum_features = (time_integrated * mask).sum(dim=1)
                sum_mask = mask.sum(dim=1)
                output = sum_features / sum_mask.clamp(min=1e-9)
            else:
                output = time_integrated.mean(dim=1)
        elif pool_method == 'last':
             if attention_mask is not None:
                # 各バッチ要素の最後の非パディングトークンのインデックスを取得
                last_token_indices = attention_mask.sum(dim=1) - 1
                output = time_integrated[torch.arange(time_integrated.shape[0]), last_token_indices]
             else:
                output = time_integrated[:, -1, :]
        else:
            # デフォルトは言語モデリング
            output = self.output_projection(time_integrated)

        if return_spikes:
            return output, hidden_states
        return output, torch.empty(0, device=output.device)
