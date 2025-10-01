# matsushibadenki/snn2/snn_research/distillation/knowledge_distillation_manager.py
# 自律的な知識蒸留プロセスを管理するクラス
#
# 変更点:
# - ModelRegistryと連携し、重複学習の回避と学習結果の自動登録を行うようにした。
# - ベンチマーク結果の出力を正規表現でパースする機能を追加。
# - 学習済みモデルをタスク固有のパスに保存するように変更。
# - [改善] 評価時に、学習済みのモデルパスをベンチマークスクリプトに渡すように修正。
# - [改善] run_on_demand_pipelineにforce_retrain引数を追加し、モデル登録簿のチェックを行うように修正。

import os
import re
import subprocess
import yaml
from typing import Dict, Any
from .model_registry import ModelRegistry

class KnowledgeDistillationManager:
    """
    Phase 0 の中核となる、オンデマンドの知識蒸留パイプラインを管理する。
    """
    def __init__(self, base_config_path: str, model_config_path: str):
        self.base_config_path = base_config_path
        self.model_config_path = model_config_path
        
        with open(base_config_path, 'r') as f:
            self.base_config: Dict[str, Any] = yaml.safe_load(f)
        with open(model_config_path, 'r') as f:
            self.model_config: Dict[str, Any] = yaml.safe_load(f)
            
        self.registry = ModelRegistry()

    def _run_command(self, command: list[str]) -> str:
        """サブプロセスでコマンドを実行し、標準出力を返す。"""
        # (既存のコード)
        print("\n" + "="*20 + f" 🚀 EXECUTING: {' '.join(command)} " + "="*20)
        try:
            result = subprocess.run(
                command, check=True, capture_output=True, encoding='utf-8', text=True
            )
            print(result.stdout)
            if result.stderr:
                print("--- STDERR ---")
                print(result.stderr)
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"❌ コマンドの実行に失敗しました: {e}")
            print("--- STDOUT ---")
            print(e.stdout)
            print("--- STDERR ---")
            print(e.stderr)
            raise
        finally:
            print("="*60 + "\n")

    def _parse_benchmark_results(self, output: str) -> Dict[str, float]:
        """ベンチマークスクリプトの出力からSNNの性能指標を抽出する。"""
        # (既存のコード)
        metrics = {}
        try:
            # SNNの結果行を見つける (より柔軟な正規表現)
            snn_results_str = re.search(r"SNN\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.,NA/]+)", output, re.IGNORECASE)
            if snn_results_str:
                accuracy = float(snn_results_str.group(1))
                avg_latency_ms = float(snn_results_str.group(2))
                spikes_str = snn_results_str.group(3).replace(',', '')
                avg_spikes = float(spikes_str) if 'n/a' not in spikes_str.lower() else 0.0
                
                metrics = {
                    "accuracy": accuracy,
                    "avg_latency_ms": avg_latency_ms,
                    "avg_spikes_per_sample": avg_spikes
                }
        except (AttributeError, IndexError, ValueError) as e:
            print(f"⚠️ ベンチマーク結果のパースに失敗しました: {e}\nOutput:\n{output}")
        return metrics

    def _evaluate_and_register_model(self, task_description: str, task_run_dir: str):
        """学習済みモデルを評価し、結果を登録簿に登録する。"""
        # (既存のコード)
        print("📊 学習済みSNNモデルの性能評価を開始します...")
        
        best_model_src = os.path.join(task_run_dir, 'best_model.pth')
        
        if not os.path.exists(best_model_src):
             print(f"⚠️ ベストモデルが見つかりません: {best_model_src}")
             return

        # 学習したモデルのパスを指定してベンチマークを実行
        benchmark_output = self._run_command([
            "python", "scripts/run_benchmark.py",
            "--model_path", best_model_src
        ])
        metrics = self._parse_benchmark_results(benchmark_output)
        
        if not metrics:
            print("⚠️ 性能指標を取得できなかったため、モデル登録をスキップします。")
            return

        self.registry.register_model(
            task_description=task_description,
            model_path=best_model_src,
            metrics=metrics,
            config=self.model_config['model']
        )
        print("🏆 性能評価とモデル登録が完了しました。")

    def run_on_demand_pipeline(self, task_description: str, unlabeled_data_path: str, teacher_model_name: str, force_retrain: bool = False):
        """未知のタスクに対し、自律的に専門家SNNを生成するパイプライン。"""
        
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↓修正開始◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️
        # Step 0: Check model registry unless retraining is forced
        if not force_retrain:
            existing_models = self.registry.find_models_for_task(task_description)
            if existing_models:
                print(f"✅ タスク '{task_description}' の学習済みモデルが既に存在します。学習をスキップします。")
                return
        # ◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️↑修正終わり◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️◾️

        task_id = task_description.replace(' ', '_').lower()
        distillation_data_dir = f"precomputed_data/{task_id}"
        task_run_dir = f"runs/specialists/{task_id}" # タスク固有のログ/モデル保存ディレクトリ

        # --- ステップ1: 知識蒸留データの準備 ---
        self._run_command([
            "python", "scripts/prepare_distillation_data.py",
            "--input_file", unlabeled_data_path,
            "--output_dir", distillation_data_dir,
            "--teacher_model", teacher_model_name
        ])

        # --- ステップ2: 専門家SNNの学習 ---
        self._run_command([
            "python", "train.py",
            "--config", self.base_config_path,
            "--model_config", self.model_config_path,
            "--data_path", distillation_data_dir,
            "--override_config", f"training.type=distillation",
            "--override_config", f"training.log_dir={task_run_dir}"
        ])
        
        print("✅ 専門家SNNモデルの学習が完了しました。")

        # --- ステップ3: 評価とモデル登録 ---
        self._evaluate_and_register_model(task_description, task_run_dir)

        print("🎉 全てのパイプラインが正常に完了しました。")