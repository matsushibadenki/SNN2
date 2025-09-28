# matsushibadenki/snn2/snn_research/agent/autonomous_agent.py
# Phase 0 の自律エージェント
#
# 変更点:
# - 長期記憶システム(Memory)を統合し、全ての意思決定プロセスを記録するようにした。
# - 選択した専門家モデルを使って推論を実行する `run_inference` メソッドを追加。
# - SNNInferenceEngineを動的にロードするロジックを実装。

import torch
from .memory import Memory
from snn_research.distillation.model_registry import ModelRegistry
from snn_research.distillation.knowledge_distillation_manager import KnowledgeDistillationManager
from snn_research.deployment import SNNInferenceEngine
from typing import Optional, Dict, Any

class AutonomousAgent:
    """
    タスクに応じて最適な専門家SNNを選択、またはオンデマンドで生成するエージェント。
    全ての行動はMemoryに記録される。
    """
    def __init__(self, accuracy_threshold: float = 0.6, energy_budget: float = 1000.0):
        self.registry = ModelRegistry()
        self.memory = Memory()
        self.distillation_manager = KnowledgeDistillationManager(
            base_config_path="configs/base_config.yaml",
            model_config_path="configs/models/small.yaml" # デフォルト
        )
        self.accuracy_threshold = accuracy_threshold
        self.energy_budget = energy_budget # 平均スパイク数の上限
        self.active_model: Optional[SNNInferenceEngine] = None

    def _select_best_model(self, task_description: str) -> Optional[Dict[str, Any]]:
        """
        エネルギー効率と性能に基づいて最適なモデルを選択する。
        """
        self.memory.add_entry("MODEL_SELECTION_STARTED", {"task": task_description})
        candidate_models = self.registry.find_models_for_task(task_description)
        if not candidate_models:
            self.memory.add_entry("MODEL_SELECTION_ENDED", {"result": "no_candidates_found"})
            return None

        best_model = None
        highest_score = -1

        print(f"🧠 {len(candidate_models)}個の候補モデルの中から最適な専門家を選定中...")
        for model_info in candidate_models:
            metrics = model_info['metrics']
            accuracy = metrics.get('accuracy', 0)
            spikes = metrics.get('avg_spikes_per_sample', float('inf'))

            if accuracy >= self.accuracy_threshold and spikes <= self.energy_budget:
                score = (accuracy * 100) - (spikes / self.energy_budget) 
                print(f"  - 候補: {model_info['model_path']} (Accuracy: {accuracy:.4f}, Spikes: {spikes:,.0f}, Score: {score:.2f})")
                if score > highest_score:
                    highest_score = score
                    best_model = model_info
        
        self.memory.add_entry("MODEL_SELECTION_ENDED", {
            "task": task_description,
            "best_model_found": best_model['model_path'] if best_model else "None",
            "score": highest_score
        })
        return best_model

    def handle_task(self, task_description: str, unlabeled_data_path: Optional[str], force_retrain: bool) -> Optional[Dict[str, Any]]:
        """
        タスクを処理するメインのメソッド。
        """
        self.memory.add_entry("TASK_RECEIVED", {
            "task": task_description,
            "unlabeled_data_path": unlabeled_data_path,
            "force_retrain": force_retrain
        })
        print(f"受け取ったタスク: '{task_description}'")

        if not force_retrain:
            selected_model = self._select_best_model(task_description)
            if selected_model:
                print(f"✅ 最適な専門家モデルを選定しました: {selected_model['model_path']}")
                return selected_model

        print(f"最適な専門家モデルが見つからないか、再学習が要求されました。")
        
        if not unlabeled_data_path:
            error_details = "新しいモデルを学習するためには --unlabeled_data_path が必要です。"
            print(f"❌ {error_details}")
            self.memory.add_entry("ERROR_ENCOUNTERED", {"reason": error_details})
            return None

        # 新しい専門家を育成
        self.memory.add_entry("TRAINING_INITIATED", {"task": task_description})
        self.distillation_manager.run_on_demand_pipeline(
            task_description=task_description,
            unlabeled_data_path=unlabeled_data_path,
            teacher_model_name="gpt2" # 将来的には動的に選択
        )
        self.memory.add_entry("TRAINING_COMPLETED", {"task": task_description})
        
        # 学習後、再度最適なモデルを選択して返す
        return self._select_best_model(task_description)
        
    def run_inference(self, model_info: Dict[str, Any], prompt: str):
        """
        指定されたモデルで推論を実行し、結果をストリーミング出力する。
        """
        model_path = model_info['model_path']
        self.memory.add_entry("INFERENCE_STARTED", {"model_path": model_path, "prompt": prompt})
        
        if not self.active_model or self.active_model.model_path != model_path:
            print(f"🧠 推論エンジンをロード中: {model_path}")
            self.active_model = SNNInferenceEngine(
                model_path=model_path,
                device="mps" if torch.backends.mps.is_available() else "cpu"
            )
        
        print("🤖 応答:")
        full_response = ""
        for chunk in self.active_model.generate(prompt, max_len=100):
            print(chunk, end="", flush=True)
            full_response += chunk
        print()
        
        self.memory.add_entry("INFERENCE_COMPLETED", {"model_path": model_path, "response": full_response})
        return full_response