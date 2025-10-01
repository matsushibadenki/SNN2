# matsushibadenki/snn/snn_research/deployment.py
# SNNの実用デプロイメントのための最適化、監視、継続学習システム
#
# 変更点:
# - AdaptiveQuantizationPruningクラスに、具体的なプルーニングと量子化のロジックを実装。
#   - nn.utils.prune を利用したMagnitudeプルーニングを導入。
#   - torch.quantization を利用した動的量子化を導入。
# - SNNInferenceEngineがモデルをロードする際に strict=False を使用するように変更。
# - mypyエラー解消のため、型ヒントを追加。
# - 独自Vocabularyを廃止し、Hugging Face Tokenizerを使用するようにSNNInferenceEngineを修正。
# - `generate` メソッドをストリーミング応答（ジェネレータ）に変更し、逐次的なテキスト生成を可能に。
# - `stop_sequences` のロジックを改善し、生成テキスト全体に含まれるかをチェックするようにした。
# - 推論時の総スパイク数を計測し、インスタンス変数 `last_inference_stats` に保存する機能を追加。
# - [改善] generateメソッドに、Top-KおよびTop-P (Nucleus)サンプリングのデコーディング戦略を追加。
# - [修正] mypyエラー解消のため、_sample_next_tokenの戻り値をintにキャストし、型ヒントを修正。

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import os
import copy
import time
from typing import Dict, Any, List, Optional, Iterator
from enum import Enum
from dataclasses import dataclass
from transformers import AutoTokenizer
import torch.nn.functional as F

# --- SNN 推論エンジン ---
class SNNInferenceEngine:
    """SNNモデルでテキスト生成を行う推論エンジン"""
    def __init__(self, model_path: str, device: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

        from .core.snn_core import BreakthroughSNN

        self.model_path = model_path
        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'config' in checkpoint:
            self.config: Dict[str, Any] = checkpoint['config']
            tokenizer_name = checkpoint.get('tokenizer_name', 'gpt2')
        else:
            print("⚠️ 古い形式のチェックポイントです。デフォルト設定を使用します。")
            self.config = {'d_model': 128, 'd_state': 64, 'num_layers': 4, 'time_steps': 20, 'n_head': 2}
            tokenizer_name = 'gpt2'

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = BreakthroughSNN(vocab_size=self.tokenizer.vocab_size, **self.config).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()
        self.last_inference_stats: Dict[str, Any] = {}

    def _sample_next_token(self, logits: torch.Tensor, top_k: int, top_p: float, temperature: float) -> int:
        """Top-K, Top-Pサンプリングを用いて次のトークンを決定する"""
        logits = logits / temperature

        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float("Inf")

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float("Inf")
        
        probs = F.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        return int(next_token_id.item())

    def generate(self, start_text: str, max_len: int, stop_sequences: Optional[List[str]] = None,
                 top_k: int = 50, top_p: float = 0.95, temperature: float = 0.8) -> Iterator[str]:
        """
        テキストをストリーミング形式で生成します。
        """
        self.last_inference_stats = {"total_spikes": 0, "total_mem": 0}

        bos_token = self.tokenizer.bos_token or ''
        prompt_ids = self.tokenizer.encode(f"{bos_token}{start_text}", return_tensors='pt').to(self.device)
        
        input_tensor = prompt_ids
        generated_text = ""
        
        with torch.no_grad():
            for _ in range(max_len):
                logits, spikes, mem = self.model(input_tensor)
                
                if spikes.numel() > 0: self.last_inference_stats["total_spikes"] += spikes.sum().item()
                if mem.numel() > 0: self.last_inference_stats["total_mem"] += mem.abs().sum().item()
                
                next_token_logits = logits[:, -1, :]
                
                next_token_id = self._sample_next_token(next_token_logits, top_k, top_p, temperature)
                
                if next_token_id == self.tokenizer.eos_token_id:
                    break

                new_token = self.tokenizer.decode([next_token_id])
                generated_text += new_token
                yield new_token
                
                if stop_sequences:
                    if any(stop_seq in generated_text for stop_seq in stop_sequences):
                        break
                    
                input_tensor = torch.cat([input_tensor, torch.tensor([[next_token_id]], device=self.device)], dim=1)


# --- ニューロモーフィック デプロイメント機能 ---
class NeuromorphicChip(Enum):
    INTEL_LOIHI = "intel_loihi"
    IBM_TRUENORTH = "ibm_truenorth"
    GENERIC_EDGE = "generic_edge"

@dataclass
class NeuromorphicProfile:
    chip_type: NeuromorphicChip
    num_cores: int
    power_budget_mw: float

class AdaptiveQuantizationPruning:
    """モデルのプルーニングと量子化を適用するクラス。"""
    def apply_pruning(self, model: nn.Module, pruning_ratio: float):
        """
        MagnitudeプルーニングをモデルのLinear層に適用する。
        """
        if pruning_ratio <= 0: return
        print(f"Applying magnitude pruning with ratio: {pruning_ratio}")
        for module in model.modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=pruning_ratio)
                prune.remove(module, 'weight')
    
    def apply_quantization(self, model: nn.Module, bits: int) -> nn.Module:
        """
        動的量子化をモデルに適用する。
        """
        if bits >= 32: return model
        print(f"Applying dynamic quantization to {bits}-bit...")
        model_to_quantize = copy.deepcopy(model).cpu()
        quantized_model = torch.quantization.quantize_dynamic(
            model_to_quantize, {nn.Linear}, dtype=torch.qint8
        )
        return quantized_model

class NeuromorphicDeploymentManager:
    def __init__(self, profile: NeuromorphicProfile):
        self.profile = profile
        self.adaptive_compression = AdaptiveQuantizationPruning()

    def deploy_model(self, model: nn.Module, name: str, optimization_target: str = "balanced"):
        print(f"🔧 ニューロモーフィックデプロイメント開始: {name}")
        if optimization_target == "balanced": sparsity, bit_width = 0.7, 8
        elif optimization_target == "ultra_low_power": sparsity, bit_width = 0.9, 8
        else: sparsity, bit_width = 0.5, 16
        
        optimized_model = copy.deepcopy(model)
        optimized_model.eval()
        
        print(f"  - プルーニング適用中 (スパース率: {sparsity})...")
        self.adaptive_compression.apply_pruning(optimized_model, float(sparsity))
        
        print(f"  - 量子化適用中 (ビット幅: {bit_width}-bit)...")
        optimized_model = self.adaptive_compression.apply_quantization(optimized_model, int(bit_width))

        print(f"✅ デプロイメント完了: {name}")
        return optimized_model