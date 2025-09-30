# /snn_research/agent/self_evolving_agent.py
# Phase 5: メタ進化 - AIによる自己開発を担うエージェント
#
# 機能:
# - AutonomousAgentを継承し、自己進化の能力を追加。
# - 自己参照RAG: 自身のソースコードを知識ベースとして参照する。
# - ベンチマーク駆動ループ: コード変更が性能に与える影響を予測・評価する。
# - 自律的コード修正: 性能向上が見込めるコードの修正案を生成し、適用する。
# - [改善] 修正案を構造化データとして生成し、実際にファイルを書き換える機能を実装。
# - [改善] 修正後の性能をベンチマークで検証し、性能が向上しない場合は変更を元に戻すロールバック機能を追加。

import os
import re
import subprocess
import fileinput
import shutil
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
        if "avg_spikes_per_sample" in analysis and "1500.0" in analysis: # ダミー条件
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

    def revert_code_modification(self, proposal: Dict[str, str]) -> bool:
        """
        バックアップファイルを使って、適用した修正を元に戻す。
        """
        self.memory.add_entry("CODE_REVERT_STARTED", {"proposal": proposal})
        file_path = proposal["file_path"]
        backup_path = file_path + ".bak"

        if not os.path.exists(backup_path):
            print(f"❌ バックアップファイルが見つかりません: {backup_path}")
            self.memory.add_entry("CODE_REVERT_FAILED", {"reason": "backup_not_found"})
            return False
        
        try:
            print(f"⏪ 変更を元に戻しています: {file_path}")
            shutil.move(backup_path, file_path)
            print("✅ 変更を元に戻しました。")
            self.memory.add_entry("CODE_REVERT_ENDED", {"file_path": file_path})
            return True
        except Exception as e:
            print(f"❌ ファイルの復元中にエラーが発生しました: {e}")
            self.memory.add_entry("CODE_REVERT_FAILED", {"reason": str(e)})
            return False

    def verify_performance_improvement(self, initial_metrics: Dict[str, Any]) -> bool:
        """
        ベンチマークを実行し、性能が向上したかを確認する。
        """
        self.memory.add_entry("PERFORMANCE_VERIFICATION_STARTED", {})
        print("📊 変更後の性能をベンチマークで検証します...")

        try:
            # 実際のプロジェクトでは `scripts/run_benchmark.py` を実行する想定
            # ここではダミーの出力でシミュレートする
            # output = subprocess.check_output(["python", "scripts/run_benchmark.py"]).decode('utf-8')
            
            # --- シミュレーション ---
            # 性能が向上したダミー出力
            output = "SNN      0.80     1200.5     1450.0"
            print("  - (シミュレーション) ベンチマーク実行完了。")
            # --------------------
            
            new_metrics = {}
            snn_results_str = re.search(r"SNN\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.,NA/]+)", output, re.IGNORECASE)
            if snn_results_str:
                new_metrics["accuracy"] = float(snn_results_str.group(1))
                new_metrics["avg_spikes_per_sample"] = float(snn_results_str.group(3).replace(',', ''))
            
            if not new_metrics:
                print("  - ⚠️ ベンチマーク結果の解析に失敗しました。")
                self.memory.add_entry("PERFORMANCE_VERIFICATION_FAILED", {"reason": "parsing_failed"})
                return False

            print(f"  - 修正前の性能: {initial_metrics}")
            print(f"  - 修正後の性能: {new_metrics}")

            # 性能評価ロジック (精度が向上し、スパイク数が悪化していないか)
            improved = (new_metrics["accuracy"] > initial_metrics["accuracy"] and
                        new_metrics["avg_spikes_per_sample"] <= initial_metrics["avg_spikes_per_sample"] * 1.1)
            
            self.memory.add_entry("PERFORMANCE_VERIFICATION_ENDED", {"new_metrics": new_metrics, "improved": improved})
            return improved
        except Exception as e:
            print(f"  - ❌ ベンチマーク実行中にエラー: {e}")
            self.memory.add_entry("PERFORMANCE_VERIFICATION_FAILED", {"reason": str(e)})
            return False


    def run_evolution_cycle(self, task_description: str, initial_metrics: Dict[str, Any]):
        """
        単一の自己進化サイクル（内省→提案→適用→検証→ロールバック）を実行する。
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
        
        # 4. 検証
        performance_improved = self.verify_performance_improvement(initial_metrics)

        # 5. 結論と後処理
        if performance_improved:
            print("【結論】✅ 性能が向上しました。変更を維持します。")
            # バックアップファイルを削除
            backup_path = proposal["file_path"] + ".bak"
            if os.path.exists(backup_path):
                os.remove(backup_path)
        else:
            print("【結論】❌ 性能が向上しなかったため、変更を元に戻します。")
            self.revert_code_modification(proposal)
        
        print("="*65)
