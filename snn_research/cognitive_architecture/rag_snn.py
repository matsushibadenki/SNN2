# matsushibadenki/snn2/snn_research/cognitive_architecture/rag_snn.py
# Phase 3: RAG-SNN (Retrieval-Augmented Generation) システム

import os
from typing import List, Optional
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RAGSystem:
    """
    外部知識（ドキュメント）と内部記憶（エージェントログ）を検索し、
    思考のための文脈を提供するRAGシステム。
    """
    def __init__(self, vector_store_path: str = "runs/vector_store"):
        self.vector_store_path = vector_store_path
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store: Optional[FAISS] = self._load_vector_store()

    def _load_vector_store(self) -> Optional[FAISS]:
        """ベクトルストアをディスクから読み込む。"""
        if os.path.exists(self.vector_store_path):
            print(f"📚 既存のベクトルストアをロード中: {self.vector_store_path}")
            return FAISS.load_local(self.vector_store_path, self.embedding_model)
        return None

    def setup_vector_store(self, knowledge_dir: str = "doc", memory_file: str = "runs/agent_memory.jsonl"):
        """
        知識源からドキュメントを読み込み、ベクトルストアを構築・保存する。
        """
        print("🛠️ ベクトルストアの構築を開始します...")
        
        # 1. ドキュメント（.md, .txt）を読み込む
        doc_loader = DirectoryLoader(
            knowledge_dir, glob="**/*.md", loader_cls=TextLoader, silent_errors=True
        )
        txt_loader = DirectoryLoader(
            knowledge_dir, glob="**/*.txt", loader_cls=TextLoader, silent_errors=True
        )
        docs = doc_loader.load() + txt_loader.load()

        # 2. エージェントの記憶ファイルを読み込む
        if os.path.exists(memory_file):
            memory_loader = TextLoader(memory_file)
            docs.extend(memory_loader.load())
        
        if not docs:
            print("⚠️ 知識源となるドキュメントが見つかりませんでした。")
            return

        # 3. ドキュメントを分割
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(docs)

        # 4. ベクトル化してFAISSインデックスを作成
        print(f"📄 {len(split_docs)}個のドキュメントチャンクをベクトル化しています...")
        self.vector_store = FAISS.from_documents(split_docs, self.embedding_model)
        
        # 5. ベクトルストアをディスクに保存
        os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
        self.vector_store.save_local(self.vector_store_path)
        print(f"✅ ベクトルストアの構築が完了し、'{self.vector_store_path}' に保存しました。")

    def search(self, query: str, k: int = 3) -> List[str]:
        """
        クエリに最も関連するドキュメントチャンクを検索する。
        """
        if self.vector_store is None:
            print("ベクトルストアがセットアップされていません。先に setup_vector_store() を実行してください。")
            # ライブでセットアップを試みる
            self.setup_vector_store()
            if self.vector_store is None:
                 return ["エラー: ベクトルストアを構築できませんでした。"]

        results = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in results]