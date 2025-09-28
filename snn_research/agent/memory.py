# matsushibadenki/snn2/snn_research/agent/memory.py
# 自律エージェントの長期記憶システム

import json
import os
from datetime import datetime
from typing import Dict, Any

class Memory:
    """
    エージェントの行動と思考のログを時系列で記録する長期記憶システム。
    全ての記録は、将来の内省と自己改良のための重要な経験となる。
    """
    def __init__(self, memory_path: str = "runs/agent_memory.jsonl"):
        self.memory_path = memory_path
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)

    def add_entry(self, event_type: str, details: Dict[str, Any]):
        """
        新しい記憶（イベント）を記録する。

        Args:
            event_type (str): イベントの種類 (例: "TASK_RECEIVED", "MODEL_SELECTED")
            details (Dict[str, Any]): イベントに関する詳細情報
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "details": details
        }
        
        with open(self.memory_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")