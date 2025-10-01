# matsushibadenki/snn2/snn_research/cognitive_architecture/hierarchical_planner.py
# Phase 3: 階層的思考プランナー
#
# 変更点:
# - ハードコードされたルールベースの計画立案を撤廃。
# - ModelRegistryと連携し、エージェントが利用可能なスキル（専門家モデル）に基づいて
#   動的に実行計画を生成するロジックに変更。
# - [改善] 学習済みの「プランナーSNN」をロードし、計画立案を知能化。

import torch
import os
from transformers import AutoTokenizer

from .global_workspace import GlobalWorkspace
from snn_research.agent.memory import Memory
from snn_research.distillation.model_registry import ModelRegistry
from .planner_snn import PlannerSNN
from typing import Optional, Dict, Any, List

class HierarchicalPlanner:
    """
    複雑なタスクをサブタスクに分解し、GlobalWorkspaceと連携して実行を管理する。
    自己の能力（利用可能な専門家モデル）に基づき、動的に計画を立案する。
    """
    def __init__(self, planner_model_path: str = "runs/planner_snn.pth"):
        self.workspace = GlobalWorkspace()
        self.memory = Memory()
        self.registry = ModelRegistry()
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2") # コンフィグから取得するのが望ましい
        self.available_skills = list(self.registry.registry.keys())
        self.skill_to_id = {skill: i for i, skill in enumerate(self.available_skills)}
        self.id_to_skill = {i: skill for skill, i in self.skill_to_id.items()}

        self.planner_snn = self._load_planner_model(planner_model_path)

    def _load_planner_model(self, model_path: str) -> Optional[PlannerSNN]:
        """学習済みのプランナーSNNモデルをロードする。"""
        if not self.available_skills or not os.path.exists(model_path):
            print("⚠️ プランナーSNNモデルが見つからないか、利用可能なスキルがありません。ルールベースのフォールバックは現在ありません。")
            return None
        
        # モデル設定はダミー（本来はDIコンテナ経由で取得）
        model_config: Dict[str, int] = {'d_model': 128, 'd_state': 64, 'num_layers': 4, 'time_steps': 20, 'n_head': 2}
        
        planner_model = PlannerSNN(
            vocab_size=self.tokenizer.vocab_size,
            d_model=model_config['d_model'],
            d_state=model_config['d_state'],
            num_layers=model_config['num_layers'],
            time_steps=model_config['time_steps'],
            n_head=model_config['n_head'],
            num_skills=len(self.available_skills)
        ).to(self.device)
        
        planner_model.load_state_dict(torch.load(model_path, map_location=self.device))
        planner_model.eval()
        print("✅ 学習済みプランナーSNNを正常にロードしました。")
        return planner_model

    @torch.no_grad()
    def _create_plan(self, task_request: str) -> List[str]:
        """
        学習済みプランナーSNNを用いて、実行計画を動的に推論する。
        """
        print("📝 プランナーSNNが実行計画を推論中...")
        if not self.planner_snn or not self.available_skills:
            return []

        input_ids = self.tokenizer.encode(
            task_request, return_tensors='pt',
            max_length=self.planner_snn.time_steps,
            padding='max_length', truncation=True
        ).to(self.device)

        # モデルで推論
        skill_logits, _, _ = self.planner_snn(input_ids)
        
        # 最も確率の高いスキルを順番に選択 (複数スキルを予測する場合)
        # ここでは簡単のため、最も確率の高い2つを選択
        predicted_skill_ids = torch.topk(skill_logits, k=min(2, len(self.available_skills)), dim=-1).indices.squeeze().tolist()
        
        plan = [self.id_to_skill[skill_id] for skill_id in predicted_skill_ids if skill_id in self.id_to_skill]
        
        self.memory.add_entry("PLAN_CREATED", {"request": task_request, "available_skills": self.available_skills, "plan": plan})
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
