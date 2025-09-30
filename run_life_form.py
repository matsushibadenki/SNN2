# matsushibadenki/snn2/run_life_form.py
# Phase 6: デジタル生命体の自律的意識ループを開始する

import argparse
from snn_research.agent.digital_life_form import DigitalLifeForm

def main():
    """
    デジタル生命体を起動し、指定されたサイクル数だけ
    自律的な思考と学習のループを実行させる。
    """
    parser = argparse.ArgumentParser(
        description="デジタル生命体 実行フレームワーク",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=5,
        help="実行する意識サイクルの回数。"
    )
    args = parser.parse_args()

    # デジタル生命体をインスタンス化
    life_form = DigitalLifeForm(project_root=".")
    
    # 意識ループを開始
    life_form.awareness_loop(cycles=args.cycles)

if __name__ == "__main__":
    main()
