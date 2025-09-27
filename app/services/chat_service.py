# matsushibadenki/snn/app/services/chat_service.py
# チャット機能のビジネスロジックを担うサービス
#
# 機能:
# - DIコンテナから推論エンジンを受け取る。
# - Gradioからの入力を処理し、整形して推論エンジンに渡す。
# - 推論結果をGradioが扱える形式で返す。
# - ストリーミング応答をサポート。
# - 推論完了後に総スパイク数をコンソールに出力。
# - UI表示用に、リアルタイムの統計情報も生成する。

import time
from snn_research.deployment import SNNInferenceEngine
from typing import Iterator, Tuple, List

class ChatService:
    def __init__(self, snn_engine: SNNInferenceEngine, max_len: int):
        """
        ChatServiceを初期化します。

        Args:
            snn_engine: テキスト生成に使用するSNN推論エンジン。
            max_len: 生成するテキストの最大長。
        """
        self.snn_engine = snn_engine
        self.max_len = max_len

    def stream_response(self, message: str, history: List[List[str]]) -> Iterator[Tuple[List[List[str]], str]]:
        """
        GradioのBlocks UIのために、チャット履歴と統計情報をストリーミング生成する。
        """
        history.append([message, ""])
        
        prompt = ""
        for user_msg, bot_msg in history[:-1]:
            if bot_msg is not None:
                prompt += f"User: {user_msg}\nAssistant: {bot_msg}\n"
        prompt += f"User: {message}\nAssistant:"

        print("-" * 30)
        print(f"Input prompt to SNN:\n{prompt}")

        start_time = time.time()
        
        full_response = ""
        token_count = 0
        for chunk in self.snn_engine.generate(prompt, max_len=self.max_len):
            full_response += chunk
            token_count += 1
            history[-1][1] = full_response
            
            duration = time.time() - start_time
            stats = self.snn_engine.last_inference_stats
            total_spikes = stats.get("total_spikes", 0)
            spikes_per_second = total_spikes / duration if duration > 0 else 0
            tokens_per_second = token_count / duration if duration > 0 else 0

            stats_md = f"""
            **Inference Time:** `{duration:.2f} s`
            **Tokens/Second:** `{tokens_per_second:.2f}`
            ---
            **Total Spikes:** `{total_spikes:,.0f}`
            **Spikes/Second:** `{spikes_per_second:,.0f}`
            """
            
            yield history, stats_md

        # Final log to console
        duration = time.time() - start_time
        stats = self.snn_engine.last_inference_stats
        total_spikes = stats.get("total_spikes", 0)
        print(f"\nGenerated response: {full_response.strip()}")
        print(f"Inference time: {duration:.4f} seconds")
        print(f"Total spikes: {total_spikes:,.0f}")
        print("-" * 30)

