# matsushibadenki/snn/snn_research/training/losses.py
# SNN学習で使用する損失関数
# 
# 機能:
# - snn_coreとknowledge_distillationから損失関数クラスを移動・集約。
# - 蒸留時にTokenizerを統一したため、DistillationLoss内の不整合対応ロジックを削除。
# - DIコンテナの依存関係解決を遅延させるため、pad_idではなくtokenizerを直接受け取るように変更。
# - ハードウェア実装を意識し、モデルのスパース性を促進するL1正則化項を追加。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from transformers import PreTrainedTokenizerBase

def _calculate_sparsity_loss(model: nn.Module) -> torch.Tensor:
    """モデルの重みのL1ノルムを計算し、スパース性を促進する。"""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return torch.tensor(0.0) # パラメータがない場合は0を返す

    device = params[0].device
    l1_norm = sum(
        (p.abs().sum() for p in params),
        start=torch.tensor(0.0, device=device)
    )
    return l1_norm

class CombinedLoss(nn.Module):
    """クロスエントロピー損失、スパイク発火率、スパース性の正則化を組み合わせた損失関数。"""
    def __init__(self, ce_weight: float, spike_reg_weight: float, sparsity_reg_weight: float, tokenizer: PreTrainedTokenizerBase, target_spike_rate: float = 0.02):
        super().__init__()
        pad_id = tokenizer.pad_token_id
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id if pad_id is not None else -100)
        self.weights = {'ce': ce_weight, 'spike_reg': spike_reg_weight, 'sparsity_reg': sparsity_reg_weight}
        self.target_spike_rate = target_spike_rate

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, spikes: torch.Tensor, model: nn.Module) -> dict:
        ce_loss = self.ce_loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        spike_rate = spikes.mean()
        spike_reg_loss = F.mse_loss(spike_rate, torch.tensor(self.target_spike_rate, device=spike_rate.device))
        
        sparsity_loss = _calculate_sparsity_loss(model)
        
        total_loss = (self.weights['ce'] * ce_loss + 
                      self.weights['spike_reg'] * spike_reg_loss +
                      self.weights['sparsity_reg'] * sparsity_loss)
        
        return {
            'total': total_loss, 'ce_loss': ce_loss,
            'spike_reg_loss': spike_reg_loss, 'sparsity_loss': sparsity_loss,
            'spike_rate': spike_rate
        }

class DistillationLoss(nn.Module):
    """知識蒸留のための損失関数（スパース性正則化付き）。"""
    def __init__(self, tokenizer: PreTrainedTokenizerBase, ce_weight: float, distill_weight: float,
                 spike_reg_weight: float, sparsity_reg_weight: float, temperature: float):
        super().__init__()
        student_pad_id = tokenizer.pad_token_id
        self.temperature = temperature
        self.weights = {'ce': ce_weight, 'distill': distill_weight, 'spike_reg': spike_reg_weight, 'sparsity_reg': sparsity_reg_weight}
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=student_pad_id if student_pad_id is not None else -100)
        self.distill_loss_fn = nn.KLDivLoss(reduction='batchmean', log_target=False)

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                targets: torch.Tensor, spikes: torch.Tensor, model: nn.Module) -> Dict[str, torch.Tensor]:

        assert student_logits.shape == teacher_logits.shape, \
            f"Shape mismatch! Student: {student_logits.shape}, Teacher: {teacher_logits.shape}"

        ce_loss = self.ce_loss_fn(student_logits.view(-1, student_logits.size(-1)), targets.view(-1))
        
        soft_student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        distill_loss = self.distill_loss_fn(soft_student_log_probs, soft_teacher_probs) * (self.temperature ** 2)

        spike_rate = spikes.mean()
        target_spike_rate = torch.tensor(0.02, device=spikes.device)
        spike_reg_loss = F.mse_loss(spike_rate, target_spike_rate)

        sparsity_loss = _calculate_sparsity_loss(model)

        total_loss = (self.weights['ce'] * ce_loss +
                      self.weights['distill'] * distill_loss +
                      self.weights['spike_reg'] * spike_reg_loss +
                      self.weights['sparsity_reg'] * sparsity_loss)

        return {
            'total': total_loss, 'ce_loss': ce_loss,
            'distill_loss': distill_loss, 'spike_reg_loss': spike_reg_loss,
            'sparsity_loss': sparsity_loss
        }
