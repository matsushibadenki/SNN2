# /snn_research/agent/self_evolving_agent.py
# Phase 5: メタ進化 - AIによる自己開発を担うエージェント
#
# 機能:
# - AutonomousAgentを継承し、自己進化の能力を追加。
# - 自己参照RAG: 自身のソースコードを知識ベースとして参照する。
# - ベンチマーク駆動ループ: コード変更が性能に与える影響を予測・評価する。
# - 自律的コード修正: 性能向上が見込めるコードの修正案を生成し、適用する。
# - [改善] 修正案を構造化データとして生成し、実際にファイルを書き換える機能を実装。

import os
import subprocess
import fileinput
from typing import Dict, Any, Optional, List

from .autonomous_agent import AutonomousAgent
from snn_research.cognitive_architecture.rag_snn import RAGSystem

class SelfEvolvingAgent(AutonomousAgent):
    """
    自己のソースコードとパフォーマンスを監視し、
    自律的に自己改良を行うメタ進化エージェント。
    """
    def __init__(self, project_root: str = "."):
        super().__init__()
        self.project_root = project_root
        # 自身のソースコードを知識源とするRAGシステム
        self.self_reference_rag = RAGSystem(vector_store_path="runs/self_reference_vector_store")
        self._setup_self_reference()

    def _setup_self_reference(self):
        """自己参照用ベクトルストアのセットアップ"""
        if not os.path.exists(self.self_reference_rag.vector_store_path):
            print("🧠 自己参照用ナレッジベースを構築しています...")
            self.self_reference_rag.setup_vector_store(knowledge_dir=self.project_root)

    def reflect_on_performance(self, task_description: str, metrics: Dict[str, Any]) -> str:
        """
        特定のタスクの性能評価結果を分析し、改善の方向性を考察する。
        """
        self.memory.add_entry("PERFORMANCE_REFLECTION_STARTED", {"task": task_description, "metrics": metrics})
        
        reflection_prompt = (
            f"タスク「{task_description}」の性能が以下の通りでした: {metrics}。\n"
            f"性能向上のためのボトルネックはどこにあると考えられますか？\n"
            f"関連する可能性のあるソースコードの箇所を特定してください。"
        )
        
        # 自己のコードベースから関連情報を検索
        relevant_code_snippets = self.self_reference_rag.search(reflection_prompt, k=3)
        
        analysis = (
            f"考察: タスク「{task_description}」の性能指標は {metrics} でした。\n"
            f"関連するコード断片:\n" + "\n---\n".join(relevant_code_snippets)
        )
        
        self.memory.add_entry("PERFORMANCE_REFLECTION_ENDED", {"analysis": analysis})
        return analysis

    def generate_code_modification_proposal(self, analysis: str) -> Optional[Dict[str, str]]:
        """
        分析結果に基づき、具体的なコード修正案を構造化データとして生成する。
        """
        self.memory.add_entry("CODE_MODIFICATION_PROPOSAL_STARTED", {"analysis": analysis})
        
        proposal = None
        # スパイク数が多すぎる場合、正則化を強める提案
        if "avg_spikes_per_sample" in analysis and "1000.0" in analysis: # ダミー条件
            proposal = {
                "file_path": "configs/base_config.yaml",
                "action": "replace",
                "target": "    spike_reg_weight: 0.01",
                "new_content": "    spike_reg_weight: 0.05 # Increased by agent"
            }
        # 精度が低い場合、モデルを大きくする提案
        elif "accuracy" in analysis and "0.75" in analysis: # ダミー条件
             proposal = {
                "file_path": "configs/models/medium.yaml",
                "action": "replace",
                "target": "  num_layers: 8",
                "new_content": "  num_layers: 10 # Increased by agent"
            }
            
        self.memory.add_entry("CODE_MODIFICATION_PROPOSAL_ENDED", {"proposal": proposal})
        return proposal

    def apply_code_modification(self, proposal: Dict[str, str]) -> bool:
        """
        提案されたコード修正をファイルシステムに適用する。
        """
        self.memory.add_entry("CODE_MODIFICATION_APPLY_STARTED", {"proposal": proposal})
        file_path = proposal["file_path"]
        
        if not os.path.exists(file_path):
            print(f"❌ 修正対象ファイルが見つかりません: {file_path}")
            self.memory.add_entry("CODE_MODIFICATION_APPLY_FAILED", {"reason": "file_not_found"})
            return False

        try:
            print(f"📝 ファイルを修正中: {file_path}")
            # fileinputを使ってファイルをインプレースで置換
            with fileinput.FileInput(file_path, inplace=True, backup='.bak') as file:
                for line in file:
                    if proposal["target"] in line:
                        print(proposal["new_content"], end='\n')
                    else:
                        print(line, end='')
            
            print("✅ ファイルの修正が完了しました。バックアップが `.bak` として作成されました。")
            self.memory.add_entry("CODE_MODIFICATION_APPLY_ENDED", {"file_path": file_path})
            return True
        except Exception as e:
            print(f"❌ ファイル修正中にエラーが発生しました: {e}")
            self.memory.add_entry("CODE_MODIFICATION_APPLY_FAILED", {"reason": str(e)})
            return False

    def run_evolution_cycle(self, task_description: str, initial_metrics: Dict[str, Any]):
        """
        単一の自己進化サイクル（内省→提案→適用→検証）を実行する。
        """
        print("\n" + "="*20 + "🧬 自己進化サイクル開始 🧬" + "="*20)
        
        # 1. 内省
        analysis = self.reflect_on_performance(task_description, initial_metrics)
        print(f"【内省結果】\n{analysis}")

        # 2. 修正案の生成
        proposal = self.generate_code_modification_proposal(analysis)
        if not proposal:
            print("【結論】現時点では有効な改善案を生成できませんでした。")
            print("="*65)
            return

        print(f"【改善提案】\n{proposal}")
        
        # 3. 修正の適用
        if not self.apply_code_modification(proposal):
            print("【結論】コード修正の適用に失敗したため、サイクルを中断します。")
            print("="*65)
            return
        
        # 4. 検証 (この部分はシミュレーションまたは実際の学習・評価が必要)
        print("【検証】提案された変更を適用し、性能が向上するかを検証する必要があります。(このステップは現在シミュレーションです)")
        
        # (将来的な実装)
        # 1. `run_benchmark.py` などをサブプロセスで実行し、新しい性能メトリクスを取得
        # 2. 新しいメトリクスが `initial_metrics` を上回っているか評価
        # 3. 性能が向上した場合、変更を恒久的なものにする (例: git commit)
        # 4. 向上しなかった場合、.bakファイルから変更を元に戻す
        
        print("="*65)
