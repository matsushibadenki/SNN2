# matsushibadenki/snn2/scripts/convert_model.py
# ANNモデルからSNNモデルへの変換・蒸留を実行するためのスクリプト
#
# 変更点:
# - [改善] オンライン知識蒸留で、ダミーではなく指定された教師モデルをロードするように修正。

import argparse
import sys
from pathlib import Path
import torch

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.containers import TrainingContainer
from snn_research.conversion import AnnToSnnConverter
from snn_research.benchmark.ann_baseline import ANNBaselineModel # 蒸留の教師役ダミーとして使用

def main():
    parser = argparse.ArgumentParser(
        description="ANNモデルからSNNへの変換・蒸留ツール",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["convert", "distill"],
        help="実行する手法を選択:\n"
             " - convert: ANNの重みをSNNに直接コピーします。\n"
             " - distill: ANNを教師役としてSNNをオンラインで蒸留学習させます。"
    )
    parser.add_argument(
        "--ann_model_path",
        type=str,
        required=True,
        help="変換元となるANNモデルのパス (.safetensorsまたは.gguf)。"
    )
    parser.add_argument(
        "--snn_model_config",
        type=str,
        default="configs/models/small.yaml",
        help="変換先となるSNNモデルのアーキテクチャ設定ファイル。"
    )
    parser.add_argument(
        "--output_snn_path",
        type=str,
        required=True,
        help="変換・学習後にSNNモデルを保存するパス。"
    )
    args = parser.parse_args()

    # DIコンテナからSNNモデルのインスタンスと設定を取得
    container = TrainingContainer()
    container.config.from_yaml(args.snn_model_config)
    snn_model = container.snn_model()
    snn_config = container.config.model.to_dict()

    # コンバータを初期化
    converter = AnnToSnnConverter(snn_model=snn_model, model_config=snn_config)

    if args.method == "convert":
        converter.convert_weights(
            ann_model_path=args.ann_model_path,
            output_path=args.output_snn_path
        )
    elif args.method == "distill":
        # 知識蒸留のデモンストレーション
        print(f"教師ANNモデルを {args.ann_model_path} からロードします。")

        # ANNのアーキテクチャをSNNと合わせる（この例ではANNBaselineModelを使用）
        # 注: 実際のユースケースでは、変換元ANNのアーキテクチャに合わせたモデルクラスが必要です
        teacher_model = ANNBaselineModel(
            vocab_size=container.tokenizer.provided.vocab_size(),
            d_model=snn_config['d_model'],
            nhead=snn_config['n_head'],
            d_hid=snn_config['d_model'] * 2,
            nlayers=snn_config['num_layers'],
            num_classes=container.tokenizer.provided.vocab_size()
        )
        
        # コンバータの内部メソッドを使って重みをロード
        try:
            ann_weights = converter._load_ann_weights(args.ann_model_path)
            # SNNと互換性のあるキーのみをロード
            teacher_model.load_state_dict(ann_weights, strict=False)
            print("✅ 教師モデルの重みを正常にロードしました。")
        except Exception as e:
            print(f"❌ 教師モデルの重みロードに失敗しました: {e}")
            print("   SNNモデルとANNモデルのアーキテクチャが一致しているか確認してください。")
            return

        # ダミーの学習データローダーを作成
        dummy_dataset = [torch.randint(0, container.tokenizer.provided.vocab_size(), (snn_config['time_steps'],)) for _ in range(32)]
        dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=4)

        converter.run_online_distillation(
            ann_teacher_model=teacher_model,
            dummy_data_loader=dummy_loader,
            output_path=args.output_snn_path,
            epochs=3
        )

if __name__ == "__main__":
    main()