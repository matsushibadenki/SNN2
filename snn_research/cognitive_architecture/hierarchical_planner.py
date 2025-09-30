# matsushibadenki/snn2/snn_research/cognitive_architecture/hierarchical_planner.py
# Phase 3: 階層的思考プランナー
#
# 変更点:
# - ハードコードされたルールベースの計画立案を撤廃。
# - ModelRegistryと連携し、エージェントが利用可能なスキル（専門家モデル）に基づいて
#   動的に実行計画を生成するロジックに変更。
# - 将来的に学習可能な「プランナーSNN」への置き換えを見据えた設計にした。

from .global_workspace import GlobalWorkspace
from snn_research.agent.memory import Memory
from snn_research.distillation.model_registry import ModelRegistry
from typing import Optional, Dict, Any, List

class HierarchicalPlanner:
    """
    複雑なタスクをサブタスクに分解し、GlobalWorkspaceと連携して実行を管理する。
    自己の能力（利用可能な専門家モデル）に基づき、動的に計画を立案する。
    """
    def __init__(self):
        self.workspace = GlobalWorkspace()
        self.memory = Memory()
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        self.registry = ModelRegistry()
        # 将来的には、このプランナー自体が学習可能なSNNモデルになる
        # self.planner_snn = self.load_planner_model()
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
    def _create_plan(self, task_request: str) -> List[str]:
        """
        自然言語のタスク要求と、利用可能な専門家モデルのリストから、
        実行すべきサブタスクのシーケンスを動的に生成する。
        """
        print("📝 実行計画を立案中...")
        
        # 1. 現在利用可能な全てのスキル（専門家タスク）をモデル登録簿から取得
        available_skills = list(self.registry.registry.keys())
        if not available_skills:
            print("  - ⚠️ 利用可能な専門家モデルが一つも登録されていません。")
            return []
            
        print(f"  - 利用可能なスキル: {available_skills}")

        # 2. タスク要求の中に、利用可能なスキル名が含まれているかチェックし、計画を生成
        #    将来的に、この部分は意味的類似性検索や学習済みプランナーSNNに置き換えられる
        plan = []
        # ユーザーのリクエストの語順を尊重するため、単純なループでスキルを抽出
        for skill in available_skills:
            if skill in task_request:
                plan.append(skill)
        
        # 順序が重要になる場合があるため、現時点では抽出された順序を維持する
        # (例：「要約して、分析して」と「分析して、要約して」は意味が違う)
        
        self.memory.add_entry("PLAN_CREATED", {"request": task_request, "available_skills": available_skills, "plan": plan})
        return plan
    # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

    def execute_task(self, task_request: str, context: str) -> Optional[str]:
        """
        タスクの計画立案から実行までを統括する。
        """
        self.memory.add_entry("HIGH_LEVEL_TASK_RECEIVED", {"request": task_request, "context": context})
        
        plan = self._create_plan(task_request)
        if not plan:
            print(f"❌ タスク '{task_request}' に対する実行計画を立案できませんでした。")
            self.memory.add_entry("PLANNING_FAILED", {"request": task_request})
            return None

        print(f"✅ 実行計画が決定: {' -> '.join(plan)}")
        
        current_context = context
        for sub_task in plan:
            print(f"\n Fase de ejecución de la subtarea: '{sub_task}'...")
            
            # ワークスペースにサブタスクの実行を依頼
            result = self.workspace.process_sub_task(sub_task, current_context)
            
            if result is None:
                print(f"❌ サブタスク '{sub_task}' の実行に失敗しました。")
                self.memory.add_entry("SUB_TASK_FAILED", {"sub_task": sub_task})
                return None
            
            current_context = result # 次のサブタスクの入力として結果を渡す
        
        return current_context
