# snn_research/training/bio_trainer.py
# Title: 生物学的学習則用トレーナー
# Description: STDPなどのオンライン学習を行うモデルの学習ループを管理します。

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore
from typing import Dict, cast

# BioSNNクラスを型ヒントのためにインポート
from snn_research.bio_models.simple_network import BioSNN

class BioTrainer:
    """生物学的学習則モデルのためのトレーナー。"""
    def __init__(self, model: nn.Module, device: str):
        self.model = model.to(device)
        self.device = device

    def train_epoch(self, dataloader: DataLoader, epoch: int, time_steps: int) -> Dict[str, float]:
        """学習エポックを実行する。"""
        self.model.train()
        # mypyが `self.model` を汎用のnn.Moduleとして認識するため、具体的な型にキャスト
        bio_model = cast(BioSNN, self.model)

        progress_bar = tqdm(dataloader, desc=f"Bio Training Epoch {epoch}")
        total_output_spikes = 0
        
        for batch in progress_bar:
            # 実際の応用では、入力データをスパイクに変換するエンコーダが必要
            # ここでは簡易的にランダムスパイクを生成
            input_spikes = (torch.rand(time_steps, bio_model.n_input, device=self.device) < 0.15).float()

            for t in range(time_steps):
                # 報酬信号のダミー生成 (例: 最後の100msで発火したら報酬)
                optional_params: Dict[str, float] = {}
                if t > time_steps - 100:
                    # このロジックはタスクに大きく依存する
                    optional_params["reward"] = 1.0 
                
                output_spikes = self.model(input_spikes[t], optional_params)
                total_output_spikes += output_spikes.sum().item()

        avg_spikes = total_output_spikes / (len(dataloader) * time_steps) if len(dataloader) > 0 else 0.0
        print(f"Epoch {epoch} - Average Output Spikes: {avg_spikes:.4f}")
        return {"avg_output_spikes": avg_spikes}

    def evaluate(self, dataloader: DataLoader, epoch: int, time_steps: int) -> Dict[str, float]:
        """評価ループ（ここでは学習ループと同じ）。"""
        print(f"Evaluating Epoch {epoch}...")
        self.model.eval()
        # 生物学的学習では評価中に学習しないため、train=Falseで実行
        # 本来はタスク固有の評価指標（正解率など）を計算するべき
        
        # このデモでは、単純にtrain_epochと同様の処理を呼ぶが、
        # 本来は self.model(..., train=False) のようにして重み更新を止めるべき
        # 今回のBioSNNの実装では self.training フラグで更新を制御しているため eval() でOK
        
        # train_epochから重み更新を除いた評価ロジックをここに書くのが理想
        # 今回は簡易的に学習ループを再利用する
        with torch.no_grad(): # 重み更新がないことを保証
             # train_epochを呼び出すと重みが更新されてしまうため、ここでは簡易的なメトリクスのみ
            print("Evaluation logic is simplified for this demo.")
        return {"eval_metric": 0.0} # ダミーの評価メトリクス