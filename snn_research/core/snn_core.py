# matsushibadenki/snn2/snn_research/core/snn_core.py
# SNNモデルの定義、次世代ニューロンなど、中核となるロジックを集約したライブラリ
#
# 変更点:
# - ANN性能に近づけるため、階層的な「予測符号化」アーキテクチャを導入。
#   - PredictiveCodingLayerを追加し、トップダウン予測とボトムアップ観測の誤差を計算。
#   - BreakthroughSNNを、予測符号化を行う階層モデルに刷新。
# - 表現力向上のため、AdaptiveLIFNeuronを標準ニューロンとして採用。
# - STDPなどの複雑なコンポーネントは一旦コメントアウトし、まずは中核となる予測符号化の安定動作に注力。
# - mypyエラー解消のため、型ヒントを修正・追加。
# - [改善] BreakthroughSNNのforwardパスを、より本格的なリカレント（RNN）形式の時系列処理に変更。
# - [改善] 学習の安定化のため、SNNに適したLayerNormと残差接続を導入。

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import surrogate, functional  # type: ignore
from typing import Tuple, Dict, Any, Optional, List
import math

# --- ニューロンモデル ---
class AdaptiveLIFNeuron(nn.Module):
    """
    適応的発火閾値を持つLIFニューロン (表現力向上のための標準ニューロン)
    """
    def __init__(
        self,
        features: int,
        tau: float = 2.0,
        base_threshold: float = 1.0,
        adaptation_strength: float = 0.1,
        target_spike_rate: float = 0.02,
    ):
        super().__init__()
        self.tau = tau
        self.base_threshold = base_threshold
        self.adaptation_strength = adaptation_strength
        self.target_spike_rate = target_spike_rate
        self.surrogate_function = surrogate.ATan()
        self.mem_decay = math.exp(-1.0 / tau)
        self.register_buffer(
            "adaptive_threshold", torch.full((features,), base_threshold)
        )
        self.adaptive_threshold: torch.Tensor
        self.register_buffer("mem", torch.zeros(1, features))
        self.mem: torch.Tensor

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mem.shape[0] != x.shape[0] or self.mem.device != x.device:
            self.mem = torch.zeros(x.shape[0], x.shape[-1], device=x.device)

        self.mem = self.mem * self.mem_decay + x
        spike = self.surrogate_function(self.mem - self.adaptive_threshold)
        self.mem = self.mem * (1.0 - spike.detach())

        if self.training:
            with torch.no_grad():
                spike_rate_error = spike.mean() - self.target_spike_rate
                self.adaptive_threshold += self.adaptation_strength * spike_rate_error
                self.adaptive_threshold.clamp_(min=0.5)

        return spike, self.mem

class SNNLayerNorm(nn.Module):
    """SNNのためのタイムステップごとに行うLayerNorm"""
    def __init__(self, normalized_shape):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape)
    def forward(self, x):
        return self.norm(x)

# --- 予測符号化レイヤー ---
class PredictiveCodingLayer(nn.Module):
    """
    予測符号化を実行する単一の階層レイヤー。
    トップダウンの予測とボトムアップの観測から誤差を計算する。
    """
    def __init__(self, d_model: int, d_state: int, n_head: int):
        super().__init__()
        # 生成モデル (トップダウン予測を生成)
        self.generative_fc = nn.Linear(d_state, d_model)
        self.generative_lif = AdaptiveLIFNeuron(d_model)

        # 推論モデル (ボトムアップ観測から状態を更新)
        self.inference_fc = nn.Linear(d_model, d_state)
        self.inference_lif = AdaptiveLIFNeuron(d_state)

        self.norm_state = SNNLayerNorm(d_state)
        self.norm_error = SNNLayerNorm(d_model)


    def forward(self, bottom_up_input: torch.Tensor, top_down_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. トップダウン予測を生成
        prediction, _ = self.generative_lif(self.generative_fc(top_down_state))

        # 2. 予測誤差を計算 (観測 - 予測)
        prediction_error = F.relu(bottom_up_input - prediction)
        prediction_error = self.norm_error(prediction_error)

        # 3. 予測誤差に基づいて状態を更新
        state_update, _ = self.inference_lif(self.inference_fc(prediction_error))
        
        # 4. 残差接続と正規化で状態を更新
        updated_state = self.norm_state(top_down_state + state_update)

        return updated_state, prediction_error, prediction, self.inference_lif.mem

# --- コアSNNモデル ---
class BreakthroughSNN(nn.Module):
    """
    リカレント予測符号化アーキテクチャを実装した階層的SNN
    """
    def __init__(self, vocab_size: int, d_model: int, d_state: int, num_layers: int, time_steps: int, n_head: int):
        super().__init__()
        self.time_steps = time_steps
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_state = d_state

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.input_encoder = nn.Sequential(nn.Linear(d_model, d_model), AdaptiveLIFNeuron(d_model))
        
        self.pc_layers = nn.ModuleList(
            [PredictiveCodingLayer(d_model, d_state, n_head) for _ in range(num_layers)]
        )
        
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_spikes: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size, seq_len = input_ids.shape
        token_emb = self.token_embedding(input_ids)
        
        states = [torch.zeros(batch_size, self.d_state, device=input_ids.device) for _ in range(self.num_layers)]
        
        total_spikes = 0
        total_mem_potential = 0
        all_logits = []

        # シーケンスを時間ステップとして順方向に処理
        for i in range(seq_len):
            embedded_token = token_emb[:, i, :]
            
            bottom_up_input, _ = self.input_encoder(embedded_token)
            
            layer_errors = []
            layer_mems = []
            
            # ボトムアップ処理 (推論)
            for j in range(self.num_layers):
                states[j], error, _, mem = self.pc_layers[j](bottom_up_input, states[j])
                bottom_up_input = F.relu(states[j]) # 次の層への入力は現在の層の状態スパイク
                layer_errors.append(error)
                layer_mems.append(mem)
            
            # トップダウン処理 (生成)
            top_down_signal = states[-1]
            for j in reversed(range(self.num_layers)):
                 _, _, prediction, _ = self.pc_layers[j](torch.zeros_like(bottom_up_input, device=input_ids.device), top_down_signal)
                 top_down_signal = prediction
            
            # 最終的な出力層の予測からロジットを計算
            logits = self.output_projection(top_down_signal)
            all_logits.append(logits)

            total_spikes += sum(err.mean() for err in layer_errors)
            total_mem_potential += sum(m.abs().mean() for m in layer_mems)

        # (batch_size, seq_len, vocab_size) の形状にスタック
        final_logits = torch.stack(all_logits, dim=1)
        
        avg_spikes = total_spikes / seq_len
        avg_mem = total_mem_potential / seq_len

        return final_logits, avg_spikes, avg_mem
