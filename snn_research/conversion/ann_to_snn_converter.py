# matsushibadenki/snn2/snn_research/conversion/ann_to_snn_converter.py
# GGUF/Safetensors形式のANNモデルからSNNへの変換・蒸留を行うコンバータ
#
# 機能:
# - 指定されたパスからSafetensorsまたはGGUFモデルの重みをロードする。
# - ANN-SNN変換: ANNの重みをSNNモデルに直接コピーする。
# - オンライン知識蒸留: ANNを教師モデルとして、SNNを学習させる。
# - 閾値キャリブレーション機能を追加し、変換後のSNNの活動を安定させる。
# - [改善] GGUFファイルの読み込み機能を正式に実装。

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from safetensors.torch import load_file
from tqdm import tqdm  # type: ignore
from typing import Dict, Any, Optional, Iterator
from gguf import GGUFReader

from snn_research.benchmark.ann_baseline import ANNBaselineModel
from snn_research.core.snn_core import AdaptiveLIFNeuron, BreakthroughSNN

def _load_gguf(path: str) -> Dict[str, torch.Tensor]:
    """GGUFファイルを読み込み、PyTorchのstate_dictを返す。"""
    print(f" GGUFファイルをロード中: {path}")
    reader = GGUFReader(path, 'r')
    state_dict = {}
    for tensor in reader.tensors:
        # GGUFのテンソルをPyTorchテンソルに変換
        # GGUFはリトルエンディアンで保存されているため、コピーしてメモリレイアウトを保証
        state_dict[tensor.name] = torch.from_numpy(tensor.data.copy())
    print(f"✅ GGUFから {len(state_dict)} 個のテンソルをロードしました。")
    return state_dict

class AnnToSnnConverter:
    """
    既存のANNモデルファイルからSNNモデルを生成するユーティリティ。
    """
    def __init__(self, snn_model: nn.Module, model_config: Dict[str, Any]):
        """
        コンバータを初期化します。

        Args:
            snn_model: 変換先となるSNNモデルのインスタンス。
            model_config: SNNモデルのアーキテクチャ設定。
        """
        self.snn_model = snn_model
        self.model_config = model_config
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.snn_model.to(self.device)

    def _load_ann_weights(self, ann_model_path: str) -> Dict[str, torch.Tensor]:
        """ANNモデルの重みをファイルから読み込む。"""
        print(f"💾 ANNモデルの重みをロード中: {ann_model_path}")
        if ann_model_path.endswith(".safetensors"):
            return load_file(ann_model_path, device=self.device)
        elif ann_model_path.endswith(".gguf"):
            # GGUFローダーを呼び出すように修正
            return _load_gguf(ann_model_path)
        else:
            raise ValueError("サポートされていないファイル形式です。.safetensorsまたは.ggufを指定してください。")

    def calibrate_thresholds(self, calibration_loader: Any, target_rate: float = 0.1, epochs: int = 1):
        """
        変換後のSNNモデルの発火閾値をキャリブレーションする。
        """
        print(f"⚙️ 発火閾値のキャリブレーションを開始します (目標発火率: {target_rate:.2f})...")
        self.snn_model.train() # trainモードで実行し、適応的閾値を更新させる

        # 適応的閾値を持つニューロン層のみを対象とする
        lif_layers = [m for m in self.snn_model.modules() if isinstance(m, AdaptiveLIFNeuron)]
        if not lif_layers:
            print("⚠️ 適応的閾値を持つLIFニューロンが見つからないため、キャリブレーションをスキップします。")
            return

        # 目標発火率を設定
        for layer in lif_layers:
            layer.target_spike_rate = target_rate

        with torch.no_grad():
            for epoch in range(epochs):
                for batch in tqdm(calibration_loader, desc=f"Calibration Epoch {epoch+1}"):
                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0].to(self.device)
                    else:
                        if isinstance(batch, (list, tuple)):
                            inputs = batch[0].to(self.device)
                        else:
                            inputs = batch.to(self.device)

                    # モデルを実行してAdaptiveLIFNeuron内の閾値更新ロジックをトリガー
                    self.snn_model(inputs)

        print("✅ キャリブレーションが完了しました。")
        for i, layer in enumerate(lif_layers):
            avg_threshold = layer.adaptive_threshold.mean().item()
            print(f"  - Layer {i+1} の平均閾値: {avg_threshold:.4f}")
        self.snn_model.eval() # 評価モードに戻す

    def convert_weights(
        self,
        ann_model_path: str,
        output_path: str,
        calibration_loader: Optional[Any] = None
    ) -> None:
        """
        ANN-SNN変換（重みコピー）を実行し、オプションで閾値キャリブレーションを行う。
        """
        ann_weights = self._load_ann_weights(ann_model_path)
        snn_state_dict = self.snn_model.state_dict()

        print("🔄 ANNの重みをSNNモデルにコピーしています...")
        
        # ANNとSNNでキー名が対応していると仮定してコピー
        # 実際にはモデル構造に合わせてマッピングロジックが必要
        copied_keys = 0
        for name, param in snn_state_dict.items():
            if name in ann_weights and param.shape == ann_weights[name].shape:
                snn_state_dict[name].copy_(ann_weights[name])
                copied_keys += 1
            else:
                # 形状やキーが一致しない場合はスキップ
                pass
        
        print(f"  - {copied_keys}個のパラメータをコピーしました。")
        self.snn_model.load_state_dict(snn_state_dict, strict=False)

        # 閾値キャリブレーションを実行
        if calibration_loader:
            self.calibrate_thresholds(calibration_loader)
        
        # 変換後のSNNモデルを保存
        torch.save({
            'model_state_dict': self.snn_model.state_dict(),
            'config': self.model_config
        }, output_path)
        print(f"✅ 重み変換が完了し、モデルを '{output_path}' に保存しました。")

    def run_online_distillation(
        self,
        ann_teacher_model: nn.Module,
        dummy_data_loader: Any, # 本来は学習データローダー
        output_path: str,
        epochs: int = 3
    ) -> None:
        """
        オンライン知識蒸留を実行する。
        """
        ann_teacher_model.to(self.device)
        ann_teacher_model.eval()

        optimizer = optim.AdamW(self.snn_model.parameters(), lr=1e-4)
        loss_fn = nn.KLDivLoss(reduction='batchmean', log_target=True)

        print("🔥 オンライン知識蒸留を開始します...")
        self.snn_model.train()

        for epoch in range(epochs):
            progress_bar = tqdm(dummy_data_loader, desc=f"Distillation Epoch {epoch+1}")
            for batch in progress_bar:
                # DataLoaderからの出力がタプルかチェック
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)
                
                optimizer.zero_grad()
                
                # SNN (生徒) の出力を取得
                snn_logits, _ = self.snn_model(inputs)
                
                # ANN (教師) の出力を取得
                with torch.no_grad():
                    if isinstance(ann_teacher_model, ANNBaselineModel):
                         teacher_logits = ann_teacher_model(inputs)
                    else:
                         teacher_logits = ann_teacher_model(inputs).logits

                # 損失を計算 (KLダイバージェンス)
                loss = loss_fn(
                    F.log_softmax(snn_logits / 2.0, dim=-1),
                    F.log_softmax(teacher_logits / 2.0, dim=-1)
                )
                
                loss.backward()
                optimizer.step()
                progress_bar.set_postfix({"loss": loss.item()})
        
        # 学習後のSNNモデルを保存
        torch.save({
            'model_state_dict': self.snn_model.state_dict(),
            'config': self.model_config
        }, output_path)
        print(f"✅ 知識蒸留が完了し、モデルを '{output_path}' に保存しました。")