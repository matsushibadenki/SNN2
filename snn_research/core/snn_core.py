# matsushibadenki/snn2/snn_research/core/snn_core.py
# SNNモデルの定義、次世代ニューロンなど、中核となるロジックを集約したライブラリ
#
# 変更点:
# - mypyエラー解消のため、neuron_classに明示的な型ヒントを追加。

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import surrogate, functional  # type: ignore
from typing import Tuple, Dict, Any, Optional, List, Type
import math

# --- ニューロンモデル ---
class AdaptiveLIFNeuron(nn.Module):
    """
    適応的発火閾値を持つLIFニューロン (表現力向上のための標準ニューロン)
    """
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
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

class DendriticNeuron(nn.Module):
    """
    Phase 4: 樹状突起演算を模倣したニューロン。
    複数の分岐（dendritic branches）を持ち、それぞれが異なる時空間パターンを学習する。
    """
    def __init__(self, input_features: int, num_branches: int, branch_features: int):
        super().__init__()
        self.num_branches = num_branches
        # 各分岐は独立した線形変換とLIFニューロンを持つ
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_features, branch_features),
                AdaptiveLIFNeuron(branch_features)
            ) for _ in range(num_branches)
        ])
        # Soma (細胞体): 各分岐からの出力を統合する
        self.soma_lif = AdaptiveLIFNeuron(branch_features * num_branches)
        self.output_projection = nn.Linear(branch_features * num_branches, input_features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        branch_outputs = []
        for branch in self.branches:
            # 各ブランチは同じ入力を受け取る
            branch_spike, _ = branch(x)
            branch_outputs.append(branch_spike)
        
        # 全ての分岐からの出力を結合
        concatenated_spikes = torch.cat(branch_outputs, dim=-1)
        
        # 細胞体で統合し、最終的な出力を生成
        soma_spike, soma_mem = self.soma_lif(concatenated_spikes)
        
        # 出力スパイクを元の次元に射影
        output = self.output_projection(soma_spike)
        
        return output, soma_mem


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
    def __init__(self, d_model: int, d_state: int, n_head: int, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]):
        super().__init__()
        # 生成モデル (トップダウン予測を生成)
        self.generative_fc = nn.Linear(d_state, d_model)
        if neuron_class == DendriticNeuron:
            self.generative_neuron = neuron_class(input_features=d_model, **neuron_params)
        else:  # AdaptiveLIFNeuronを想定
            self.generative_neuron = neuron_class(features=d_model)

        # 推論モデル (ボトムアップ観測から状態を更新)
        self.inference_fc = nn.Linear(d_model, d_state)
        if neuron_class == DendriticNeuron:
            self.inference_neuron = neuron_class(input_features=d_state, **neuron_params)
        else:  # AdaptiveLIFNeuronを想定
            self.inference_neuron = neuron_class(features=d_state)

        self.norm_state = SNNLayerNorm(d_state)
        self.norm_error = SNNLayerNorm(d_model)


    def forward(self, bottom_up_input: torch.Tensor, top_down_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. トップダウン予測を生成
        prediction, _ = self.generative_neuron(self.generative_fc(top_down_state))

        # 2. 予測誤差を計算 (観測 - 予測)
        prediction_error = F.relu(bottom_up_input - prediction)
        prediction_error = self.norm_error(prediction_error)

        # 3. 予測誤差に基づいて状態を更新
        state_update, inference_mem = self.inference_neuron(self.inference_fc(prediction_error))
        
        # 4. 残差接続と正規化で状態を更新
        updated_state = self.norm_state(top_down_state + state_update)

        return updated_state, prediction_error, prediction, inference_mem

# --- コアSNNモデル ---
class BreakthroughSNN(nn.Module):
    """
    リカレント予測符号化アーキテクチャを実装した階層的SNN
    """
    def __init__(self, vocab_size: int, d_model: int, d_state: int, num_layers: int, time_steps: int, n_head: int, neuron_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.time_steps = time_steps
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_state = d_state

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        if neuron_config is None:
            neuron_config = {"type": "lif"}
        
        neuron_type = neuron_config.get("type", "lif")
        
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        neuron_class: Type[nn.Module]
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

        if neuron_type == "dendritic":
            neuron_class = DendriticNeuron
            neuron_params = {
                "num_branches": neuron_config.get("num_branches", 4),
                "branch_features": neuron_config.get("branch_features", d_model // 4)
            }
            self.input_encoder = nn.Sequential(
                nn.Linear(d_model, d_model),
                DendriticNeuron(input_features=d_model, **neuron_params)
            )
            print("💡 BreakthroughSNN initialized with Dendritic Neurons.")
        else:  # デフォルトは "lif"
            neuron_class = AdaptiveLIFNeuron
            neuron_params = {}  # LIFに追加パラメータは不要
            self.input_encoder = nn.Sequential(
                nn.Linear(d_model, d_model),
                AdaptiveLIFNeuron(features=d_model)
            )
        
        self.pc_layers = nn.ModuleList(
            [PredictiveCodingLayer(d_model, d_state, n_head, neuron_class, neuron_params) for _ in range(num_layers)]
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
        
        total_spikes = torch.tensor(0.0, device=input_ids.device)
        total_mem_potential = torch.tensor(0.0, device=input_ids.device)
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

            if return_spikes:
                total_spikes += sum(err.mean() for err in layer_errors)
                total_mem_potential += sum(m.abs().mean() for m in layer_mems)

        # (batch_size, seq_len, vocab_size) の形状にスタック
        final_logits = torch.stack(all_logits, dim=1)
        
        avg_spikes = total_spikes / seq_len if return_spikes else torch.tensor(0.0)
        avg_mem = total_mem_potential / seq_len if return_spikes else torch.tensor(0.0)

        return final_logits, avg_spikes, avg_mem
# ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

