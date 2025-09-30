# matsushibadenki/snn2/snn_research/agent/digital_life_form.py
# Phase 6: 全ての認知機能を統合し、自律的に活動するデジタル生命体

import time
from typing import Dict, Any

from .self_evolving_agent import SelfEvolvingAgent
from snn_research.cognitive_architecture.emergent_system import EmergentSystem
from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem
from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner

class DigitalLifeForm(SelfEvolvingAgent):
    """
    自己の内部状態（好奇心や矛盾）に基づき、永続的に活動する
    デジタル生命体のコアロジック。
    """
    def __init__(self, project_root: str = "."):
        super().__init__(project_root)
        self.motivation_system = IntrinsicMotivationSystem(window_size=10)
        self.emergent_system = EmergentSystem()
        self.planner = HierarchicalPlanner()
        
        # 初期状態として、ダミーの性能指標とエラーを持つ
        self.current_metrics: Dict[str, Any] = {"accuracy": 0.8, "avg_spikes_per_sample": 1200.0}
        self.last_prediction_error = 0.1

    def awareness_loop(self, cycles: int = 5):
        """
        デジタル生命体の意識と活動のコアとなる自律ループ。
        """
        print("\n" + "="*20 + "🧬 デジタル生命体 意識ループ開始 🧬" + "="*20)
        
        for i in range(cycles):
            print(f"\n--- [意識サイクル {i+1}/{cycles}] ---")
            
            # 1. 内的状態の更新 (世界の理解度を評価)
            #    学習が進むと予測誤差が減り、報酬(喜び)が生まれる
            reward = self.motivation_system.update_error(self.last_prediction_error)
            
            # 2. 探求すべきか判断 (退屈しているか？)
            if self.motivation_system.should_explore():
                print("🤔 好奇心に基づき、新たな探求を開始します...")
                
                # 3. 探求テーマの決定 (何に興味を持つべきか？)
                #    専門家間の意見の対立(=未知)を探す
                synthesis_result = self.emergent_system.synthesize_responses(
                    prompt="現在、最も意見が分かれている概念は何か？", 
                    domain="文章要約" # 例としてドメインを指定
                )
                print(f"🔍 探求テーマ: {synthesis_result}")
                
                # 4. 探求計画の立案と実行
                #    発見したテーマを解決するための計画を立てる
                task_request = f"'{synthesis_result}' について、関連情報を調査し、新たな知見を要約せよ。"
                plan_result = self.planner.execute_task(task_request, context=synthesis_result)
                print(f"💡 探求結果: {plan_result}")
                
                # (シミュレーション) 探求によって世界の理解が少し進み、次の予測誤差が少し減る
                self.last_prediction_error *= 0.95
            
            else:
                print("😌 現在の理解に満足しています。活動は安定しています。")
                # 安定している場合でも、自己の性能に改善の余地がないか内省する
                if self.current_metrics["accuracy"] < 0.95:
                     print("...しかし、自己の性能にはまだ改善の余地があるようです。自己進化を試みます。")
                     self.run_evolution_cycle("汎用言語理解", self.current_metrics)
                     # (シミュレーション) 自己進化により性能が向上
                     self.current_metrics["accuracy"] *= 1.05


            time.sleep(1) # サイクル間の小休止

        print("\n" + "="*20 + "🧬 デジタル生命体 意識ループ終了 🧬" + "="*20)
