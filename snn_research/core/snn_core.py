# matsushibadenki/snn2/snn_research/core/snn_core.py
# SNNモデルの定義、次世代ニューロンなど、中核となるロジックを集約したライブラリ
#
# 変更点:
# - 外部AIからのフィードバックに基づき、廃止されていた高度なコンポーネント(TTFSEncoder, IzhikevichNeuronなど)を復活。
# - BreakthroughSNNがTTFSEncoderをオプションで使えるようにし、表現力を回復。
# - AttentionalSpikingSSMLayerの出力にLIFニューロンを再追加し、SNNとしての特性を強化。
# - 全体的な可読性と拡張性を向上させるリファクタリングを実施。

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import surrogate, functional
from typing import Tuple, Dict, Any, Optional
import math

class TTFSEncoder(nn.Module):
    """
    Time-to-First-Spike (TTFS) 符号化器 (復活)
    連続値を最初のスパイクまでの時間（レイテンシ）に変換する、リッチな時間符号化方式。
    """
    def __init__(self, d_model: int, time_steps: int, max_latency: int = 10):
        super().__init__()
        self.time_steps = time_steps
        self.max_latency = min(max_latency, time_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq, embed) -> (batch, time, seq, embed)
        x_activated = torch.sigmoid(x)
        spike_times = torch.round(self.max_latency * (1.0 - x_activated)).long()
        spike_times = torch.clamp(spike_times, 0, self.time_steps - 1)
        
        spikes = torch.zeros(x.shape[0], self.time_steps, x.shape[1], x.shape[2], device=x.device)
        # 4D tensorにscatterするためにインデックスを整形
        batch_idx = torch.arange(x.shape[0]).view(-1, 1, 1).expand_as(spike_times)
        seq_idx = torch.arange(x.shape[1]).view(1, -1, 1).expand_as(spike_times)
        embed_idx = torch.arange(x.shape[2]).view(1, 1, -1).expand_as(spike_times)
        
        spikes[batch_idx, spike_times, seq_idx, embed_idx] = 1.0
        return spikes

class AdaptiveLIFNeuron(nn.Module):
    """
    適応的発火閾値を持つLIFニューロン
    """
    def __init__(self, features: int, tau: float = 2.0, base_threshold: float = 1.0, adaptation_strength: float = 0.1, target_spike_rate: float = 0.02):
        super().__init__()
        self.tau = tau
        self.base_threshold = base_threshold
        self.adaptation_strength = adaptation_strength
        self.target_spike_rate = target_spike_rate
        self.surrogate_function = surrogate.ATan()
        self.mem_decay = math.exp(-1.0 / tau)
        self.register_buffer('adaptive_threshold', torch.full((features,), base_threshold))

    def forward(self, x: torch.Tensor, v_mem: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v_potential = v_mem * self.mem_decay + x
        spike = self.surrogate_function(v_potential - self.adaptive_threshold)
        v_mem_new = v_potential * (1.0 - spike.detach())
        
        with torch.no_grad():
            spike_rate_error = spike.mean() - self.target_spike_rate
            self.adaptive_threshold += self.adaptation_strength * spike_rate_error
            self.adaptive_threshold.clamp_(min=0.5)

        return spike, v_mem_new

class IzhikevichNeuron(nn.Module):
    """ Izhikevichニューロンモデル (復活) """
    def __init__(self, features: int, a: float = 0.02, b: float = 0.2, c: float = -65.0, d: float = 8.0):
        super().__init__()
        self.a, self.b, self.c, self.d = a, b, c, d
        self.surrogate_function = surrogate.ATan()
        self.register_buffer('v', torch.full((features,), self.c))
        self.register_buffer('u', torch.full((features,), self.c * self.b))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 状態変数を現在のバッチサイズに合わせる
        if self.v.shape != x.shape:
            self.v = torch.full_like(x, self.c)
            self.u = self.v * self.b

        self.v += 0.5 * (0.04 * self.v**2 + 5 * self.v + 140 - self.u + x)
        self.v += 0.5 * (0.04 * self.v**2 + 5 * self.v + 140 - self.u + x)
        self.u += self.a * (self.b * self.v - self.u)
        
        spike = self.surrogate_function(self.v - 30.0)
        spike_d = spike.detach()
        self.v = self.v * (1 - spike_d) + self.c * spike_d
        self.u = self.u * (1 - spike_d) + (self.u + self.d) * spike_d
        return spike

class STDPSynapse(nn.Module):
    """ STDPシナプス """
    def __init__(self, in_features: int, out_features: int, tau_pre: float = 20.0, tau_post: float = 20.0, A_pos: float = 0.01, A_neg: float = 0.005, w_min: float = 0.0, w_max: float = 1.0):
        super().__init__()
        self.tau_pre_decay = math.exp(-1.0 / tau_pre)
        self.tau_post_decay = math.exp(-1.0 / tau_post)
        self.A_pos = A_pos
        self.A_neg = A_neg
        self.w_min = w_min
        self.w_max = w_max
        self.weight = nn.Parameter(torch.rand(out_features, in_features) * (w_max - w_min) + w_min)
        self.register_buffer('pre_trace', torch.zeros(1, in_features))
        self.register_buffer('post_trace', torch.zeros(1, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)

    @torch.no_grad()
    def update_weights(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        self.pre_trace = self.pre_trace * self.tau_pre_decay + pre_spikes.mean(dim=0, keepdim=True)
        self.post_trace = self.post_trace * self.tau_post_decay + post_spikes.mean(dim=0, keepdim=True)
        delta_w_pos = self.A_pos * torch.outer(post_spikes.mean(dim=0), self.pre_trace.squeeze(0))
        delta_w_neg = self.A_neg * torch.outer(self.post_trace.squeeze(0), pre_spikes.mean(dim=0))
        self.weight.add_(delta_w_pos - delta_w_neg).clamp_(self.w_min, self.w_max)

class AttentionalSpikingSSMLayer(nn.Module):
    """ アテンション機構を統合したSpiking State Space Model """
    def __init__(self, d_model: int, d_state: int = 64, n_head: int = 4):
        super().__init__()
        self.d_state = d_state
        self.n_head = n_head
        self.d_head = d_state // n_head
        assert self.d_head * n_head == self.d_state, "d_state must be divisible by n_head"

        self.A = nn.Parameter(torch.randn(d_state, d_state))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        
        self.q_proj = nn.Linear(d_state, d_state)
        self.kv_proj = nn.Linear(d_model, d_state * 2)
        self.out_proj = nn.Linear(d_state, d_state)

        self.state_lif = AdaptiveLIFNeuron(d_state)
        self.output_lif = AdaptiveLIFNeuron(d_model) # SNN特性を強化するため出力LIFを復活

        self.register_buffer('h_state', torch.zeros(1, 1, d_state))
        self.register_buffer('state_v_mem', torch.zeros(1, 1, d_state))
        self.register_buffer('output_v_mem', torch.zeros(1, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, seq_len, _ = x.shape
        
        if self.h_state.shape[0] != batch_size or self.h_state.shape[1] != seq_len:
            h = torch.zeros(batch_size, seq_len, self.d_state, device=x.device)
            state_v = torch.zeros(batch_size, seq_len, self.d_state, device=x.device)
            output_v = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)
        else:
            h, state_v, output_v = self.h_state.clone().detach(), self.state_v_mem.clone().detach(), self.output_v_mem.clone().detach()

        outputs = []
        for t in range(time_steps):
            x_t = x[:, t, :, :]
            state_transition = F.linear(h, self.A)
            
            q = self.q_proj(h).view(batch_size * seq_len, self.n_head, self.d_head)
            k, v = self.kv_proj(x_t).chunk(2, dim=-1)
            k = k.contiguous().view(batch_size * seq_len, self.n_head, self.d_head)
            v = v.contiguous().view(batch_size * seq_len, self.n_head, self.d_head)

            attn_scores = torch.matmul(q.transpose(0,1), k.transpose(0,1).transpose(-2, -1)) / math.sqrt(self.d_head)
            attn_weights = F.softmax(attn_scores, dim=-1)
            attended_v = torch.matmul(attn_weights, v.transpose(0,1)).transpose(0,1).contiguous().view(batch_size, seq_len, self.d_state)
            
            state_update = state_transition + self.out_proj(attended_v)
            h, state_v = self.state_lif(state_update, state_v)
            
            output_potential = F.linear(h, self.C)
            out_spike, output_v = self.output_lif(output_potential, output_v)
            outputs.append(out_spike)
            
        self.h_state, self.state_v_mem, self.output_v_mem = h.detach(), state_v.detach(), output_v.detach()
        return torch.stack(outputs, dim=1)

class BreakthroughSNN(nn.Module):
    """
    全てのコンポーネントを統合したSNNアーキテクチャ
    """
    def __init__(self, vocab_size: int, d_model: int, d_state: int, num_layers: int, time_steps: int, n_head: int, encoder_type: str = 'expand'):
        super().__init__()
        self.time_steps = time_steps
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        if encoder_type == 'ttfs':
            self.spike_encoder = TTFSEncoder(d_model=d_model, time_steps=time_steps)
        else:
            self.spike_encoder = None
        
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

        token_emb = self.token_embedding(input_ids)

        if self.spike_encoder:
            hidden_states = self.spike_encoder(token_emb)
        else:
            hidden_states = token_emb.unsqueeze(1).expand(-1, self.time_steps, -1, -1)
        
        all_spikes = []
        for layer in self.ssm_layers:
            hidden_states = layer(hidden_states)
            if return_spikes:
                all_spikes.append(hidden_states)
            
        time_integrated = hidden_states.mean(dim=1)
        functional.reset_net(self)

        output: torch.Tensor
        if pool_method == 'mean':
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).expand_as(time_integrated).float()
                sum_features = (time_integrated * mask).sum(dim=1)
                sum_mask = mask.sum(dim=1)
                output = sum_features / sum_mask.clamp(min=1e-9)
            else:
                output = time_integrated.mean(dim=1)
        elif pool_method == 'last':
             if attention_mask is not None:
                last_token_indices = attention_mask.sum(dim=1) - 1
                output = time_integrated[torch.arange(time_integrated.shape[0]), last_token_indices]
             else:
                output = time_integrated[:, -1, :]
        else:
            output = self.output_projection(time_integrated)

        spikes_tensor = torch.stack(all_spikes).mean() if return_spikes and all_spikes else torch.empty(0, device=output.device)
        return output, spikes_tensor