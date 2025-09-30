# matsushibadenki/snn2/snn_research/cognitive_architecture/intrinsic_motivation.py
# Phase 6: 内発的動機付けシステム
#
# 機能:
# - 自由エネルギー原理に基づき、予測誤差の減少（＝世界の理解が進むこと）を
#   内在的な報酬としてモデル化する。
# - ドーパミンやセロトニンといった神経修飾物質の役割を抽象的に再現する。
# - エージェントが未知の情報を探求し、学習を続けるための「好奇心」の源となる。

import torch
from collections import deque
from typing import Deque

class IntrinsicMotivationSystem:
    """
    予測誤差の変化率を監視し、内的な報酬信号を生成するシステム。
    """
    def __init__(self, window_size: int = 100):
        # 最近の予測誤差（例: 損失関数の値）を記録するためのウィンドウ
        self.error_history: Deque[float] = deque(maxlen=window_size)
        self.window_size = window_size
        self.last_reward = 0.0

    def update_error(self, current_error: float) -> float:
        """
        新しい予測誤差を観測し、それに基づいて報酬を計算する。

        Args:
            current_error (float): 最新の学習ステップまたは推論で観測された予測誤差（損失）。

        Returns:
            float: 計算された内在的報酬（ドーパミン放出量のアナロジー）。
                   正の値は「好奇心が満たされた」ことを、負の値は「予測が外れた」ことを示す。
        """
        if len(self.error_history) < self.window_size:
            # 履歴が十分に溜まるまでは報酬を計算しない
            self.error_history.append(current_error)
            return 0.0

        # ウィンドウ内の平均誤差（過去の期待）
        past_average_error = sum(self.error_history) / len(self.error_history)
        
        # 報酬 = 期待される誤差の減少量 (期待 - 現実)
        # 誤差が過去の平均より「減少」した場合に、正の報酬が生成される
        reward = past_average_error - current_error
        
        # 履歴を更新
        self.error_history.append(current_error)
        
        self.last_reward = reward
        
        if reward > 0.01:
            print(f"🧠 内的報酬: {reward:+.4f} (予測が当たり、世界への理解が深まりました)")
        elif reward < -0.01:
            print(f"🤯 内的ペナルティ: {reward:+.4f} (予測が外れました。要注目です)")
            
        return reward

    def should_explore(self, reward_threshold: float = 0.001) -> bool:
        """
        現在の報酬レベルに基づき、エージェントが新しい探求活動を
        開始すべきかどうかを判断する。

        Returns:
            bool: 報酬が低く（＝退屈しており）、新しい刺激を求めるべきならTrue。
        """
        is_bored = abs(self.last_reward) < reward_threshold
        if is_bored and len(self.error_history) == self.window_size:
            print("🥱 システムは安定しており、退屈しています。新しい探求を推奨します。")
        return is_bored
