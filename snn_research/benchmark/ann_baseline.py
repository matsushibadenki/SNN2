# matsushibadenki/snn/benchmark/ann_baseline.py
#
# SNNモデルとの性能比較を行うためのANNベースラインモデル
#
# 目的:
# - ロードマップ フェーズ1「1.2. ANNベースラインとの比較」に対応。
# - SNNとほぼ同等のパラメータ数を持つ標準的なANNモデルを実装し、
#   公平な性能比較の土台を築く。
#
# アーキテクチャ:
# - 事前学習済みモデルは使用せず、スクラッチで学習するシンプルなTransformerエンコーダを採用。
# - 単語埋め込み層 + Transformerエンコーダ層 + 分類ヘッドという標準的な構成。

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class ANNBaselineModel(nn.Module):
    """
    シンプルなTransformerベースのテキスト分類モデル。
    BreakthroughSNNとの比較用。
    """
    def __init__(self, vocab_size: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformerエンコーダ層を定義
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        # 分類ヘッド
        self.classifier = nn.Linear(d_model, 2) # positive / negative

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.classifier.bias.data.zero_()
        self.classifier.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): 入力シーケンス (batch_size, seq_len)
            src_padding_mask (torch.Tensor): パディングマスク (batch_size, seq_len)

        Returns:
            torch.Tensor: 分類ロジット (batch_size, 2)
        """
        embedded = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        
        # Transformerエンコーダに入力
        encoded = self.transformer_encoder(embedded, src_key_padding_mask=src_padding_mask)
        
        # [CLS]トークン（シーケンスの先頭）の出力を取得して分類
        # ここでは簡単化のため、シーケンス全体の平均プーリングで代用
        pooled = encoded.mean(dim=1)
        
        logits = self.classifier(pooled)
        return logits