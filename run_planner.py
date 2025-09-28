# matsushibadenki/snn2/run_planner.py
# Phase 3: 高次認知アーキテクチャの実行インターフェース

import argparse
from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner

def main():
    """
    階層的思考プランナーに複雑なタスクを依頼し、
    複数の専門家SNNを連携させた問題解決を実行させる。
    """
    parser = argparse.ArgumentParser(
        description="SNN高次認知アーキテクチャ 実行フレームワーク",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--task_request",
        type=str,
        required=True,
        help="エージェントに解決させたい、自然言語による複雑なタスク要求。\n例: 'この記事を要約して、その内容の感情を分析してください。'"
    )
    parser.add_argument(
        "--context_data",
        type=str,
        required=True,
        help="タスク処理の対象となる文脈データ（文章や質問など）。"
    )

    args = parser.parse_args()

    # 階層的プランナーを初期化
    planner = HierarchicalPlanner()

    # プランナーにタスク処理を依頼
    final_result = planner.execute_task(
        task_request=args.task_request,
        context=args.context_data
    )

    if final_result:
        print("\n" + "="*20 + " ✅ FINAL RESULT " + "="*20)
        print(final_result)
        print("="*56)
    else:
        print("\n" + "="*20 + " ❌ TASK FAILED " + "="*20)
        print("タスクを完了できませんでした。必要な専門家モデルが不足している可能性があります。")

if __name__ == "__main__":
    main()