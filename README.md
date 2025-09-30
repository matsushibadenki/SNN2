# **SNNベース AIチャットシステム (v2.7 \- 高次認知アーキテクチャ)**

## **1\. 概要**

本プロジェクトは、スパイキングニューラルネットワーク（SNN）技術を基盤とした、次世代のAIチャットシステムです。最終目標は、自ら思考し、学習し、自己を改良する**自律的デジタル生命体**を創造することにあります。

現在のバージョンでは、システムは以下の主要な能力を持つ**自律エージェント**として動作します。

* **オンデマンド学習:** 未知のタスクに直面した際、大規模言語モデルから知識を蒸留し、タスクに特化した超省エネルギーな「専門家SNN」を自律的に生成します。  
* **自己認識とモデル選択:** 自身の能力（学習済み専門家モデルのリスト）を把握しており、与えられたタスクに対し、エネルギー効率と性能を天秤にかけて最適な専門家を自律的に選択します。  
* **長期記憶:** 全ての思考、判断、行動を時系列で記録する長期記憶システムを備えており、この記憶は将来の自己改良と、後述するRAGシステムの知識源となります。  
* **高次認知:** 複数の専門家SNNとRAG（Retrieval-Augmented Generation）システムを連携させ、複雑なタスクを複数のサブタスクに分解・計画し、実行する高次の問題解決能力を持ちます。

アーキテクチャ全体は、依存性の注入（DI）コンテナを用いて構築されており、研究開発からアプリケーション化までをシームレスに繋ぐ、高い保守性と拡張性を実現しています。

## **2\. システムの実行方法**

### **ステップ1: 環境設定**

まず、必要なライブラリをインストールします。

pip install \-r requirements.txt

### **2.1. 主要な実行スクリリプトの役割**

このプロジェクトには、目的別にいくつかの実行スクリプトが用意されています。

| スクリプト | 役割 | 主な利用者 |
| :---- | :---- | :---- |
| run\_agent.py | **【推奨】自律エージェントの操作**。単一のタスクを解決させたい場合に利用。 | 一般利用者 |
| run\_planner.py | **高次認知プランナーの操作**。複数の手順を要する複雑なタスクを解決させたい場合に利用。 | 一般利用者 |
| run\_evolution.py | **【最先端】自己進化サイクルの実行**。AIに自己のコードを改善させたい場合に利用。 | 開発者 |
| app/main.py | **対話UIの起動**。学習済みモデルとチャットするためのWeb UIを起動します。 | 一般利用者 |
| scripts/convert\_model.py | **【開発者向け】ANN-SNN変換・蒸留**。既存のモデルファイルからSNNを生成します。 | 開発者 |
| train.py | **【開発者向け】学習プロセスの手動実行**。モデルの学習を細かく制御したい場合に利用。 | 開発者 |
| run\_distillation.py | 知識蒸留を手動で実行する旧来のスクリプト。現在はrun\_agent.pyの利用を推奨。 | 開発者 |

### **2.2. 基本操作: 自律エージェントによるタスク処理 (run\_agent.py)**

run\_agent.py は、エージェントに単一のタスクを依頼するための基本的なインターフェースです。

#### **例1: 既存の専門家モデルを選択して推論を実行**

事前に「感情分析」の専門家モデルが学習済みであると仮定します。

python run\_agent.py \\  
    \--task\_description "感情分析" \\  
    \--prompt "この映画は本当に素晴らしかった！"

エージェントはモデル登録簿（runs/model\_registry.json）を調べ、最適な「感情分析」モデルを選択してプロンプトに対する応答を生成します。

#### **例2: 新しい専門家モデルをオンデマンドで学習**

「文章要約」の専門家が存在しない場合、エージェントは自動的に学習を開始します。

python run\_agent.py \\  
    \--task\_description "文章要約" \\  
    \--unlabeled\_data\_path data/sample\_data.jsonl \\  
    \--prompt "SNNは、生物の神経系における情報の伝達と処理のメカニズムを模倣したニューラルネットワークの一種である。"

エージェントは「文章要約」モデルが存在しないことを検知し、data/sample\_data.jsonlを使って新しい専門家の学習を自動的に開始します。学習完了後、そのモデルを使ってプロンプトに対する要約を生成します。

### **2.3. 応用操作: 高次認知プランナーによる複雑なタスク処理 (run\_planner.py)**

run\_planner.py は、エージェントに複数のステップを要する複雑なタスクを依頼するためのインターフェースです。プランナーは内部でRAGシステムを利用し、自己の知識ベース（ドキュメントや過去の記憶）を参照します。

#### **2.3.1. 知識ベースの構築 (初回のみ)**

プランナーを実行する前に、RAGシステムが検索するためのベクトル化された知識ベースを構築する必要があります。

python scripts/build\_knowledge\_base.py

このコマンドは doc ディレクトリと runs/agent\_memory.jsonl の内容を読み込み、検索可能なインデックスを runs/vector\_store に作成します。

#### **2.3.2. 階層的プランナーによる複雑なタスクの実行**

知識ベースの準備ができたら、プランナーに複雑なタスクを依頼できます。

python run\_planner.py \\  
    \--task\_request "この記事を要約して、その内容の感情を分析してください。" \\  
    \--context\_data "SNNは非常にエネルギー効率が高いことで知られているが、その性能はまだANNに及ばない点もある。"

プランナーはタスクを「文章要約」と「感情分析」の2つのサブタスクに分解し、それぞれに対応する専門家を呼び出して、段階的に問題を解決します。もし専門家が見つからない場合は、RAGシステムが検索した関連知識を返します。

### **2.4. 開発者向け: 学習プロセスの手動実行 (train.py)**

train.pyは、モデルの学習プロセスを直接的かつ詳細に制御したい開発者向けの低レベルインターフェースです。自律エージェントも内部でこのスクリプトに類する処理を呼び出しています。

#### **例1: 標準的な事前学習**

大規模なテキストデータセット（wikitext-103など）を用いて、汎用的な言語モデルをゼロから学習させます。

python train.py \\  
    \--config configs/base\_config.yaml \\  
    \--model\_config configs/models/medium.yaml \\  
    \--data\_path data/wikitext-103\_train.jsonl \\  
    \--data\_format simple\_text

#### **例2: 知識蒸留**

特定のタスク（例: 感情分析）のために事前計算された教師モデルのデータ（ロジット）を用いて、小型の専門家SNNを学習させます。

python train.py \\  
    \--config configs/base\_config.yaml \\  
    \--model\_config configs/models/small.yaml \\  
    \--data\_path precomputed\_data/sentiment\_analysis \\  
    \--override\_config "training.type=distillation"

*\*--override\_config 引数で、base\_config.yamlの設定を動的に上書きできます。*

#### **例3: 分散学習 (マルチGPU)**

複数のGPUを使用して大規模モデルの学習を高速化します。torchrunユーティリティを使用します。scripts/run\_distributed\_training.sh はそのための便利なラッパースクリプトです。

\# 推奨: ラッパースクリプトを使用  
bash scripts/run\_distributed\_training.sh

\# もしくは直接torchrunを使用 (2GPUの場合)  
torchrun \--nproc\_per\_node=2 train.py \\  
    \--distributed \\  
    \--model\_config configs/models/medium.yaml

### **2.5. 対話アプリケーションの起動 (Gradio UI)**

学習済みのモデルと直接対話するためのWeb UIも用意されています。

#### **標準チャットUI**

python \-m app.main \--model\_config configs/models/medium.yaml

#### **LangChain連携チャットUI**

python \-m app.langchain\_main \--model\_config configs/models/medium.yaml

ブラウザで http://0.0.0.0:7860 (LangChain版は 7861\) を開いてください。

### **2.6. 【開発者向け】既存ANNモデルからの直接変換・蒸留 (scripts/convert\_model.py)**

scripts/convert\_model.py は、手元にあるANNモデルファイル（.safetensors, .gguf）から直接SNNモデルを生成するためのツールです。

#### **例1: ANN-SNN 重み変換**

llama.safetensors というモデルの重みを、small.yaml で定義されたアーキテクチャを持つSNNに直接コピーし、converted\_snn.pth として保存します。

python scripts/convert\_model.py \\  
    \--method convert \\  
    \--ann\_model\_path path/to/llama.safetensors \\  
    \--snn\_model\_config configs/models/small.yaml \\  
    \--output\_snn\_path runs/converted\_snn.pth

**注意:** この方法は、ANNとSNNの層の名前や構造がある程度一致している必要があります。

#### **例2: オンライン知識蒸留**

llama.safetensors を教師役としてメモリにロードし、SNNモデルをオンラインで模倣学習させ、distilled\_snn.pth として保存します。

（注: 以下のコマンドは、教師モデルのロード部分がダミー実装です。実際の動作にはann\_to\_snn\_converter.py内のモデルロード部分の調整が必要です。）

python scripts/convert\_model.py \\  
    \--method distill \\  
    \--ann\_model\_path path/to/llama.safetensors \\  
    \--snn\_model\_config configs/models/small.yaml \\  
    \--output\_snn\_path runs/distilled\_snn.pth

### **2.7. 【最先端】自己進化サイクルの実行 (run\_evolution.py)**

run\_evolution.py は、Phase 5の中核機能である自己進化エージェントを起動するためのインターフェースです。エージェントは、与えられたタスクの初期性能を分析し、改善のためのコード修正案を自律的に生成・適用・検証します。

#### **実行例：精度の改善サイクル**

現在の感情分析タスクの精度が0.75であると仮定し、エージェントに改善を促します。

python run\_evolution.py \\  
    \--task\_description "感情分析" \\  
    \--initial\_accuracy 0.75 \\  
    \--initial\_spikes 1500.0

エージェントは以下のサイクルを実行します。

1. **内省**: initial\_accuracyが低いことを認識し、関連する可能性のある設定ファイル（configs/base\_config.yamlなど）を特定します。  
2. **提案**: 学習率を下げることで学習が安定し、精度が向上する可能性があるという仮説を立て、learning\_rateの値を修正するコード変更案を生成します。  
3. **適用**: configs/base\_config.yamlを直接書き換えます（元のファイルは.bakとしてバックアップされます）。  
4. **検証**: scripts/run\_benchmark.pyを再実行し、変更後のモデルの性能を評価します。  
5. **結論**: 性能が向上した場合、変更を維持します。向上しなかった場合、バックアップからファイルを復元し、変更を元に戻します。

## **3\. プロジェクト構造**

* /app: Gradio UIやLangChain連携など、アプリケーション層のコード。  
* /configs: モデルのアーキテクチャや学習パラメータなどの設定ファイル。  
* /data: 学習や評価に使用するサンプルデータ。  
* /doc: プロジェクトの設計書、ロードマップなどのドキュメント。  
* /runs: 学習済みモデル、ログ、モデル登録簿、ベクトルストアなど、実行時に生成されるファイル。  
* /scripts: データ準備や知識ベース構築など、補助的なスクリプト。  
* /snn\_research: SNNのコアロジック、学習パイプライン、認知アーキテクチャなど、研究開発の中核をなすコード。

## **4\. ロードマップ**

本プロジェクトは、**自律的デジタル生命体の創造**という壮大な目標に向け、以下のロードマップに沿って開発を進めています。詳細は doc/ROADMAP.md を参照してください。

* **Phase 0-2 (完了):** 基礎となるSNNアーキテクチャ、学習基盤、自律エージェントの枠組みを構築。  
* **Phase 3 (完了):** 階層的プランナー、グローバル・ワークスペース、RAG-SNNを統合し、高次認知能力を獲得。  
* **Phase 4 (完了):** 経験を通じて自己の構造を動的に変化させる、パフォーマンス駆動型の自己進化メカニズムを実装。  
* **Phase 5 (進行中):** 自己参照によるコードの自動修正、内的な動機付け、そして「心」の創発を目指す。  
* **Phase 6以降:** 未知への問い、仮想世界での自律的生存、そして意識の工学的探求へ。
