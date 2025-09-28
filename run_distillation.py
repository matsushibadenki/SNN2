# matsushibadenki/snn2/run_distillation.py
# Phase 0: Neuromorphic Knowledge Distillation を実行するためのメインスクリプト
#
# 変更点:
# - --force_retrain フラグを追加し、モデル登録簿のチェックをスキップできるようにした。

import argparse
from snn_research.distillation.knowledge_distillation_manager import KnowledgeDistillationManager

def main():
    """
    自律的な知識蒸留プロセスを開始します。
    """
    parser = argparse.ArgumentParser(
        description="自律的ニューロモーフィック知識蒸留フレームワーク"
    )
    parser.add_argument(
        "--task_description", 
        type=str, 
        required=True, 
        help="解決したいタスクの自然言語による説明 (例: '感情分析')。"
    )
    parser.add_argument(
        "--input_data_path", 
        type=str, 
        required=True, 
        help="タスクに関連する入力データへのパス (例: 'data/sentiment_analysis_unlabeled.jsonl')。"
    )
    parser.add_argument(
        "--teacher_model", 
        type=str, 
        default="gpt2", 
        help="知識の蒸留元となる教師モデル。"
    )
    parser.add_argument(
        "--student_model_config", 
        type=str, 
        default="configs/models/small.yaml", 
        help="学習させる生徒SNNモデルのアーキテクチャ設定。"
    )
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        help="このフラグを立てると、既に学習済みのモデルが存在する場合でも強制的に再学習します。"
    )

    args = parser.parse_args()

    manager = KnowledgeDistillationManager(
        base_config_path="configs/base_config.yaml",
        model_config_path=args.student_model_config
    )

    manager.run_on_demand_pipeline(
        task_description=args.task_description,
        unlabeled_data_path=args.input_data_path,
        teacher_model_name=args.teacher_model,
        force_retrain=args.force_retrain
    )

if __name__ == "__main__":
    main()