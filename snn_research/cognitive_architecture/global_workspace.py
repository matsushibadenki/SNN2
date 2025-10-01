# matsushibadenki/snn2/snn_research/cognitive_architecture/global_workspace.py
# Phase 3: グローバル・ワークスペース

from snn_research.distillation.model_registry import ModelRegistry
from snn_research.deployment import SNNInferenceEngine
from snn_research.agent.memory import Memory
from .rag_snn import RAGSystem
from typing import Optional, Dict, Any
import torch

class GlobalWorkspace:
    """
    複数の専門家SNNモデルを管理し、思考の中核を担う。
    RAGシステムと連携して知識を活用する。
    """
    def __init__(self) -> None:
        self.registry = ModelRegistry()
        self.memory = Memory()
        self.rag_system = RAGSystem()
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
        専門家が見つからない場合は、RAGシステムに問い合わせる。
        """
        specialist = self._load_specialist(sub_task)
        if not specialist:
            print(f"⚠️ タスク '{sub_task}' を実行できる専門家が見つかりません。RAGシステムに問い合わせます...")
            self.memory.add_entry("SPECIALIST_NOT_FOUND", {"task": sub_task})
            
            # RAGで関連情報を検索
            rag_query = f"タスク「{sub_task}」の実行方法について"
            rag_results = self.rag_system.search(rag_query)
            
            knowledge = "\n\n".join(rag_results)
            print(f"🔍 RAGからの知識:\n---\n{knowledge}\n---")
            self.memory.add_entry("RAG_SEARCH_PERFORMED", {"query": rag_query, "results": knowledge})
            
            # 現状では学習をトリガーせず、知識を基にした応答を返す
            return f"専門家が見つかりませんでした。関連知識によると、タスク '{sub_task}' は以下のように処理される可能性があります: {knowledge}"


        print(f"🤖 専門家 '{sub_task}' が応答を生成中...")
        self.memory.add_entry("SUB_TASK_STARTED", {"sub_task": sub_task, "context": context})
        
        full_response = ""
        for chunk in specialist.generate(context, max_len=150):
            full_response += chunk
        
        self.memory.add_entry("SUB_TASK_COMPLETED", {"sub_task": sub_task, "response": full_response})
        print(f"   - 応答: {full_response.strip()}")
        return full_response.strip()