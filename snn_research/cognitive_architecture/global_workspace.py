# matsushibadenki/snn2/snn_research/cognitive_architecture/global_workspace.py
# Phase 3: グローバル・ワークスペース

from snn_research.distillation.model_registry import ModelRegistry
from snn_research.deployment import SNNInferenceEngine
from .memory import Memory
from typing import Optional, Dict, Any
import torch

class GlobalWorkspace:
    """
    複数の専門家SNNモデルを管理し、思考の中核を担う。
    """
    def __init__(self):
        self.registry = ModelRegistry()
        self.memory = Memory()
        self.active_specialists: Dict[str, SNNInferenceEngine] = {}

    def _load_specialist(self, task_description: str) -> Optional[SNNInferenceEngine]:
        """
        指定されたタスクの専門家モデルを検索し、アクティブな推論エンジンとしてロードする。
        """
        if task_description in self.active_specialists:
            return self.active_specialists[task_description]

        # Registryから最適なモデルを検索 (現時点では最初のモデルを選択)
        candidate_models = self.registry.find_models_for_task(task_description)
        if not candidate_models:
            return None
        
        best_model_info = candidate_models[0] # ここでエネルギーベース選択も可能
        model_path = best_model_info['model_path']
        
        print(f"🧠 ワークスペースが専門家をロード中: {model_path}")
        self.memory.add_entry("SPECIALIST_LOADED", {"task": task_description, "model_path": model_path})

        engine = SNNInferenceEngine(
            model_path=model_path,
            device="mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.active_specialists[task_description] = engine
        return engine

    def process_sub_task(self, sub_task: str, context: str) -> Optional[str]:
        """
        単一のサブタスクを実行する。
        """
        specialist = self._load_specialist(sub_task)
        if not specialist:
            print(f"⚠️ タスク '{sub_task}' を実行できる専門家が見つかりません。")
            self.memory.add_entry("SPECIALIST_NOT_FOUND", {"task": sub_task})
            # ここでKnowledgeDistillationManagerを呼び出し、学習をトリガーすることも可能
            return None

        print(f"🤖 専門家 '{sub_task}' が応答を生成中...")
        self.memory.add_entry("SUB_TASK_STARTED", {"sub_task": sub_task, "context": context})
        
        full_response = ""
        for chunk in specialist.generate(context, max_len=150):
            full_response += chunk
        
        self.memory.add_entry("SUB_TASK_COMPLETED", {"sub_task": sub_task, "response": full_response})
        print(f"   - 応答: {full_response.strip()}")
        return full_response.strip()