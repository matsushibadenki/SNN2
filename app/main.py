# matsushibadenki/snn/app/main.py
# DIコンテナを利用した、Gradioリアルタイム対話UIの起動スクリプト
#
# 機能:
# - DIコンテナを初期化し、設定を読み込む。
# - コンテナから完成品のChatServiceを取得してGradioに渡す。
# - 共通UIビルダー関数を呼び出してUIを構築・起動する。
# - --model_config 引数を追加し、ベース設定とモデル設定を分けて読み込めるようにした。

import gradio as gr
import argparse
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from app.containers import AppContainer
from app.utils import build_gradio_ui

def main():
    parser = argparse.ArgumentParser(description="SNNベース リアルタイム対話AI プロトタイプ")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="設定ファイルのパス")
    parser.add_argument("--model_config", type=str, default="configs/models/small.yaml", help="モデルアーキテクチャ設定ファイルのパス")
    parser.add_argument("--model_path", type=str, help="モデルのパス (設定ファイルを上書き)")
    args = parser.parse_args()

    container = AppContainer()
    # ベース設定とモデル設定を両方読み込む
    container.config.from_yaml(args.config)
    container.config.from_yaml(args.model_config)

    # コマンドラインからモデルパスが指定された場合は、設定を上書き
    if args.model_path:
        container.config.model.path.from_value(args.model_path)

    chat_service = container.chat_service()

    print(f"Loading SNN model from: {container.config.model.path()}")
    print("✅ SNN model loaded successfully via DI Container.")
    
    # 共通UIビルダーを使用してUIを構築
    demo = build_gradio_ui(
        stream_fn=chat_service.stream_response,
        title="🤖 SNN-based AI Chat Prototype",
        description="""
        進化したBreakthroughSNNモデルとのリアルタイム対話。
        右側のパネルには、推論時間や総スパイク数（エネルギー効率の代理指標）などの統計情報がリアルタイムで表示されます。
        """,
        chatbot_label="SNN Chat",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky")
    )

    # Webアプリケーションの起動
    print("\nStarting Gradio web server...")
    print(f"Please open http://{container.config.app.server_name()}:{container.config.app.server_port()} in your browser.")
    demo.launch(
        server_name=container.config.app.server_name(),
        server_port=container.config.app.server_port(),
    )

if __name__ == "__main__":
    main()

