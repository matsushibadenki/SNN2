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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mem.shape[0] != x.shape[0]:
            self.mem = torch.zeros_like(x)

        self.mem = self.mem * self.mem_decay + x
        spike = self.surrogate_function(self.mem - self.adaptive_threshold)
        self.mem = self.mem * (1.0 - spike.detach())

        if self.training:
            with torch.no_grad():
                spike_rate_error = spike.mean() - self.target_spike_rate
                self.adaptive_threshold += self.adaptation_strength * spike_rate_error
                self.adaptive_threshold.clamp_(min=0.5)

        return spike

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

    def forward(self, bottom_up_input: torch.Tensor, top_down_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            bottom_up_input (torch.Tensor): 下位層からの観測スパイク (batch, d_model)
            top_down_state (torch.Tensor): 上位層からの状態 (予測の元) (batch, d_state)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - updated_state (torch.Tensor): 更新された状態 (batch, d_state)
                - prediction_error (torch.Tensor): 予測誤差スパイク (batch, d_model)
                - prediction (torch.Tensor): 予測スパイク (batch, d_model)
        """
        # 1. トップダウン予測を生成
        prediction = self.generative_lif(self.generative_fc(top_down_state))

        # 2. 予測誤差を計算 (観測 - 予測)
        #    ここでは単純な引き算の代わりに、観測があり予測がなかった場合にスパイクが立つようにする
        prediction_error = F.relu(bottom_up_input - prediction)

        # 3. 予測誤差に基づいて状態を更新
        state_update = self.inference_lif(self.inference_fc(prediction_error))
        updated_state = top_down_state + state_update # 残差接続的に状態を更新

        return updated_state, prediction_error, prediction


# --- コアSNNモデル ---
class BreakthroughSNN(nn.Module):
    """
    予測符号化アーキテクチャを実装した階層的SNN
    """
    def __init__(self, vocab_size: int, d_model: int, d_state: int, num_layers: int, time_steps: int, n_head: int):
        super().__init__()
        self.time_steps = time_steps
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_state = d_state

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # 入力を最初の観測スパイクに変換するエンコーダ
        self.input_encoder = nn.Sequential(nn.Linear(d_model, d_model), AdaptiveLIFNeuron(d_model))
        
        self.pc_layers = nn.ModuleList(
            [PredictiveCodingLayer(d_model, d_state, n_head) for _ in range(num_layers)]
        )
        
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_spikes: bool = False,
        # pool_method と attention_mask は予測符号化モデルでは一旦不要
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, seq_len = input_ids.shape
        token_emb = self.token_embedding(input_ids)
        
        # 状態変数の初期化
        states = [torch.zeros(batch_size, seq_len, self.d_state, device=input_ids.device) for _ in range(self.num_layers)]
        
        total_prediction_error = 0
        final_predictions = []

        # 時間ステップごとの処理
        for t in range(self.time_steps):
            # 各トークン位置に対して処理
            # (簡単のため、シーケンスをバッチのように扱う)
            # 本来はRNNのようにシーケンシャルに処理すべきだが、まずは基本構造を実装
            for i in range(seq_len):
                embedded_token = token_emb[:, i, :]
                
                # 入力トークンをスパイクに変換 (最下層の観測)
                bottom_up_input = self.input_encoder(embedded_token)
                
                layer_errors = []
                # ボトムアップ処理 (推論)
                for j in range(self.num_layers):
                    states[j][:, i, :], error, _ = self.pc_layers[j](bottom_up_input, states[j][:, i, :])
                    bottom_up_input = states[j][:, i, :] # 次の層への入力は現在の層の状態
                    layer_errors.append(error)
                
                # トップダウン処理 (生成)
                top_down_signal = states[-1][:, i, :]
                for j in reversed(range(self.num_layers)):
                    _, _, prediction = self.pc_layers[j](torch.zeros_like(bottom_up_input), top_down_signal)
                    top_down_signal = prediction # 次の層への予測は現在の層の予測
                
                # 最終的な出力層の予測を保存
                if i == seq_len - 1: # 最後のトークンの予測のみを使用
                    final_predictions.append(prediction)

                total_prediction_error += sum(err.sum() for err in layer_errors)

        # 最後の時間ステップの最後のトークンの予測を集計してロジットを計算
        if not final_predictions: # max_lenが0などでループが回らなかった場合
            final_output_activity = torch.zeros(batch_size, self.d_model, device=input_ids.device)
        else:
            final_output_activity = torch.stack(final_predictions).mean(dim=0)

        logits = self.output_projection(final_output_activity)
        
        # 損失計算用に予測誤差の合計（自由エネルギーの代理指標）を返す
        spikes_tensor = total_prediction_error / (self.time_steps * seq_len * batch_size)

        return logits, spikes_tensor
