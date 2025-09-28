# matsushibadenki/snn2/snn_research/cognitive_architecture/hierarchical_planner.py
# Phase 3: 階層的思考プランナー

from .global_workspace import GlobalWorkspace
from .memory import Memory
from typing import Optional, Dict, Any, List

class HierarchicalPlanner:
    """
    複雑なタスクをサブタスクに分解し、GlobalWorkspaceと連携して実行を管理する。
    """
    def __init__(self):
        self.workspace = GlobalWorkspace()
        self.memory = Memory()
        # シンプルなキーワードベースのプランニングルール
        self.planning_rules = {
            "要約": "文章要約",
            "感情": "感情分析",
            "分析": "感情分析",
        }

    def _create_plan(self, task_request: str) -> List[str]:
        """
        自然言語のタスク要求から、実行すべきサブタスクのシーケンスを生成する。
        """
        plan = []
        # キーワードを順番にチェックし、計画を作成
        if "要約" in task_request and "感情" in task_request:
            plan = ["文章要約", "感情分析"]
        elif "要約" in task_request:
            plan = ["文章要約"]
        elif "感情" in task_request or "分析" in task_request:
            plan = ["感情分析"]
        
        self.memory.add_entry("PLAN_CREATED", {"request": task_request, "plan": plan})
        return plan

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

        print(f"📝 実行計画: {' -> '.join(plan)}")
        
        current_context = context
        for sub_task in plan:
            print(f"\n executing sub-task: '{sub_task}'...")
            
            # ワークスペースにサブタスクの実行を依頼
            result = self.workspace.process_sub_task(sub_task, current_context)
            
            if result is None:
                print(f"❌ サブタスク '{sub_task}' の実行に失敗しました。")
                self.memory.add_entry("SUB_TASK_FAILED", {"sub_task": sub_task})
                return None
            
            current_context = result # 次のサブタスクの入力として結果を渡す
        
        return current_context