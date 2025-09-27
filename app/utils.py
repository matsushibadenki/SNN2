# matsushibadenki/snn/app/utils.py
# Gradioアプリケーション用の共通ユーティリティ
#
# 機能:
# - アプリケーション間で共有される定数や関数を定義する。
# - UI用のアバターSVGアイコンを一元管理する。
# - 共通のGradio UIレイアウトを構築する関数を提供する。

import gradio as gr
from typing import Callable, Iterator, Tuple, List

def get_avatar_svgs():
    """
    Gradioチャットボット用のアバターSVGアイコンのタプルを返す。

    Returns:
        tuple[str, str]: ユーザー用とアシスタント用のSVGアイコン文字列のタプル。
    """
    user_avatar_svg = r"""
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-user"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
    """
    assistant_avatar_svg = r"""
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-zap"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
    """
    return user_avatar_svg, assistant_avatar_svg

def build_gradio_ui(
    stream_fn: Callable[[str, List[List[str]]], Iterator[Tuple[List[List[str]], str]]],
    title: str,
    description: str,
    chatbot_label: str,
    theme: gr.themes.Base
) -> gr.Blocks:
    """
    共通のGradio Blocks UIを構築する。

    Args:
        stream_fn: チャットメッセージを処理し、応答をストリーミングする関数。
        title: UIのメインタイトル。
        description: UIの説明文。
        chatbot_label: チャットボットコンポーネントのラベル。
        theme:適用するGradioテーマ。

    Returns:
        gr.Blocks: 構築されたGradio UIオブジェクト。
    """
    user_avatar, assistant_avatar = get_avatar_svgs()

    with gr.Blocks(theme=theme) as demo:
        gr.Markdown(f"# {title}\n{description}")
        
        initial_stats_md = """
        **Inference Time:** `N/A`
        **Tokens/Second:** `N/A`
        ---
        **Total Spikes:** `N/A`
        **Spikes/Second:** `N/A`
        """

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label=chatbot_label, height=500, avatar_images=(user_avatar, assistant_avatar))
            with gr.Column(scale=1):
                stats_display = gr.Markdown(value=initial_stats_md, label="📊 Inference Stats")

        with gr.Row():
            msg_textbox = gr.Textbox(
                show_label=False,
                placeholder="メッセージを入力...",
                container=False,
                scale=6,
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear", scale=1)

        gr.Markdown("<footer><p>© 2025 SNN System Design Project. All rights reserved.</p></footer>")

        def clear_all():
            return [], "", initial_stats_md

        # `submit` アクションの定義
        submit_event = msg_textbox.submit(
            fn=stream_fn,
            inputs=[msg_textbox, chatbot],
            outputs=[chatbot, stats_display]
        )
        submit_event.then(fn=lambda: "", inputs=None, outputs=msg_textbox)
        
        button_submit_event = submit_btn.click(
            fn=stream_fn,
            inputs=[msg_textbox, chatbot],
            outputs=[chatbot, stats_display]
        )
        button_submit_event.then(fn=lambda: "", inputs=None, outputs=msg_textbox)

        # `clear` アクションの定義
        clear_btn.click(
            fn=clear_all,
            inputs=None,
            outputs=[chatbot, msg_textbox, stats_display],
            queue=False
        )
    
    return demo

