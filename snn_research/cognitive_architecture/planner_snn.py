# /snn_research/cognitive_architecture/planner_snn.py
# Phase 3: 学習可能な階層的思考プランナーSNN
#
# 機能:
# - 自然言語のタスク要求を入力として受け取る。
# - 利用可能な専門家スキル（サブタスク）の最適な実行順序を予測して出力する。
# - BreakthroughSNNをベースアーキテクチャとして使用する。

from snn_research.core.snn_core import BreakthroughSNN

class PlannerSNN(BreakthroughSNN):
    """
    タスク要求からサブタスクのシーケンスを生成することに特化したSNNモデル。
    """
    def __init__(self, vocab_size: int, d_model: int, d_state: int, num_layers: int, time_steps: int, n_head: int, num_skills: int, **kwargs):
        """
        Args:
            num_skills (int): 予測対象となるスキル（サブタスク）の総数。
        """
        super().__init__(vocab_size, d_model, d_state, num_layers, time_steps, n_head, **kwargs)
        
        # BreakthroughSNNの出力層を、スキルを予測するための分類層に置き換える
        self.output_projection = nn.Linear(d_model, num_skills)
        print(f"🧠 学習可能プランナーSNNが {num_skills} 個のスキルを認識して初期化されました。")

    def forward(self, input_ids: torch.Tensor, **kwargs):
        """
        フォワードパスを実行し、各タイムステップでのスキル予測ロジットを返す。
        """
        # BreakthroughSNNのフォワードパスを流用
        logits, spikes, mem = super().forward(input_ids, **kwargs)
        
        # プーリング処理（最終タイムステップの特徴量を使用）
        pooled_output = logits[:, -1, :]
        
        # スキル分類器を適用
        skill_logits = self.output_projection(pooled_output)
        
        return skill_logits, spikes, mem