# matsushibadenki/snn2/snn_research/distillation/model_registry.py
# 学習済みの専門家SNNモデルの情報を管理する登録簿

import json
import os
from typing import Dict, Any, Optional, List

class ModelRegistry:
    """
    専門家SNNモデルのメタデータを管理する。
    これにより、システムは自己の能力を把握し、重複学習を避けることができる。
    """
    def __init__(self, registry_path: str = "runs/model_registry.json"):
        self.registry_path = registry_path
        # 1つのタスクに対して複数のモデルバージョンを保存できるようにリスト構造に変更
        self.registry: Dict[str, List[Dict[str, Any]]] = self._load()

    def _load(self) -> Dict[str, List[Dict[str, Any]]]:
        """レジストリファイルを読み込む。"""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {} # ファイルが空か壊れている場合
        return {}

    def _save(self):
        """レジストリをファイルに保存する。"""
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=4, ensure_ascii=False)

    def register_model(self, task_description: str, model_path: str, metrics: Dict[str, Any], config: Dict[str, Any]):
        """
        新しい専門家SNNモデルを登録する。同じタスクのモデルは追記される。
        """
        print(f"🏛️ モデル登録簿に新しい専門家を追加: '{task_description}'")
        
        new_entry = {
            "model_path": model_path,
            "metrics": metrics,
            "config": config
        }
        
        if task_description in self.registry:
            self.registry[task_description].append(new_entry)
        else:
            self.registry[task_description] = [new_entry]
            
        self._save()

    def find_models_for_task(self, task_description: str) -> List[Dict[str, Any]]:
        """
        指定されたタスクに対応する全てのモデルを検索する。

        Returns:
            List[Dict[str, Any]]: 見つかったモデル情報のリスト。
        """
        # 現状は完全一致で検索。将来的には意味的類似性で検索する。
        return self.registry.get(task_description, [])