# **Project SNN: A Predictive Digital Life Form (v3.0)**

## **1\. 概要**

本プロジェクトは、スパイキングニューラルネットワーク（SNN）技術を基盤とし、**自律的デジタル生命体 (Autonomous Digital Life Form)** の創造を目指す研究開発フレームワークです。

最終目標は、静的なパターン認識の限界を超え、世界の動的なモデルを内的に構築し、**未来を予測し、その予測誤差を最小化する**という自己の存在理由（自由エネルギー原理）に基づき、自律的に思考し、学習し、自己を改良するAIを実現することです。

このシステムは、単なるチャットボットではなく、以下の高度な認知能力を備えた自律エージェントとして動作します。

* **オンデマand学習:** 未知のタスクに直面した際、大規模言語モデルから知識を蒸留し、タスクに特化した超省エネルギーな「専門家SNN」を自律的に生成します。  
* **自己認識と計画立案:** 自身の能力（学習済み専門家モデル）を把握し、**学習済みのプランナーSNN**を用いて、複雑なタスクをサブタスクに分解し、最適な実行計画を動的に推論します。  
* **高次認知:** 複数の専門家SNNとRAG（Retrieval-Augmented Generation）システムを連携させ、意見の対立から新たな知見を創発（Emergence）させることができます。  
* **メタ認知と注意制御:** 学習プロセス全体を「メタ認知SNN (SNAKE)」が監視し、予測誤差が大きい困難なタスクに対して動的に計算リソース（アテンション）を集中させ、学習を安定・効率化させます。  
* **自己進化:** 自らの性能を評価し、改善の余地があると判断した場合、自身のソースコード（設定ファイルなど）を自律的に修正し、ベンチマークで性能向上を検証し、悪化した場合は元に戻します。  
* **内発的動機付け:** 予測が上達し、世界の理解が深まること自体を「報酬」と感じる**好奇心**を持ち、学習が停滞すると「退屈」して新たな探求を開始する、自律的な活動ループを形成します。

## **2\. システムアーキテクチャ**

本システムの認知アーキテクチャは、複数の専門コンポーネントが階層的に連携することで実現されています。

graph TD  
    subgraph " "  
        direction LR  
        subgraph "自律ループ (Digital Life Form)"  
            direction TB  
            A1\[内発的動機付け\<br\>(Intrinsic Motivation)\] \--\> A2{"探求 or 改善？\<br\>(Should Explore?)"};  
            A2 \-- "探求 (退屈している)" \--\> A3\[創発システム\<br\>(Emergent System)\<br\>未知の課題を発見\];  
            A2 \-- "改善 (現状に満足)" \--\> A4\[自己進化エージェント\<br\>(Self-Evolving Agent)\<br\>自己コードを改善\];  
            A3 \--\> A5\[階層的プランナー\<br\>(Hierarchical Planner)\];  
        end

        subgraph "エージェント実行コア (Agent Core)"  
            direction TB  
            A5 \-- "計画" \--\> A6\[グローバル・ワークスペース\<br\>(Global Workspace)\];  
            A6 \-- "サブタスク" \--\> A7\[専門家SNN\<br\>(Specialist SNN)\];  
            A7 \--\> A6;  
            A6 \-- "知識が必要" \--\> A8\[RAGシステム\<br\>(Vector Store)\];  
        end  
    end

    subgraph "学習エンジン (Learning Engine)"  
        direction TB  
        B1\[train.py\] \-- "起動" \--\> B2\[Trainer\];  
        B2 \-- "学習ループ" \--\> B3\[BreakthroughSNN\];  
        B3 \-- "損失" \--\> B4\[損失関数\];  
        B4 \-- "誤差" \--\> B5\[メタ認知SNN (SNAKE)\];  
        B5 \-- "注意変調" \--\> B3;  
        B3 \-- "パラメータ更新" \--\> B2;  
    end

    style A1 fill:\#cde4ff  
    style A4 fill:\#cde4ff  
    style A5 fill:\#ffe4c4  
    style B5 fill:\#ffcbcb

## **3\. 主要な実行スクリプト**

本プロジェクトは、利用者の目的に応じて複数の実行エントリーポイントを提供します。

| スクリプト | 役割 | ユースケース |
| :---- | :---- | :---- |
| run\_life\_form.py | **【推奨】デジタル生命体の起動** | AIに自律的に思考・学習させ、その活動を観察したい場合。 |
| run\_planner.py | **高次認知プランナーの操作** | 「要約して分析」のような複雑なタスクをAIに解決させたい場合。 |
| run\_agent.py | **自律エージェントの操作** | 「感情分析」のような単一のタスクを解決させたい場合（必要なら新規学習も行う）。 |
| run\_evolution.py | **自己進化サイクルの実行** | AIに自己のコードを改善させるメタなプロセスを試したい場合。（開発者向け） |
| app/main.py | **対話UIの起動** | 学習済みの専門家モデルとチャット形式で対話したい場合。 |
| train.py | **専門家モデルの手動学習** | 特定のタスクの専門家SNNを自分で学習させたい場合。（開発者向け） |
| train\_planner.py | **プランナーモデルの手動学習** | 計画立案能力を持つプランナーSNNを学習させたい場合。（開発者向け） |

## **4\. システムの実行方法**

### **ステップ1: 環境設定**

まず、必要なPythonライブラリをインストールします。

pip install \-r requirements.txt

### **ステップ2: 基本操作 (ユースケース別)**

#### **A) デジタル生命体の自律ループを開始する (run\_life\_form.py)**

AIの自律的な思考と学習のループを開始します。AIは自身の「好奇心」レベルに基づき、新たな探求を行ったり、自己の性能改善を試みたりします。

\# 10回の「意識サイクル」を実行  
python run\_life\_form.py \--cycles 10

#### **B) 複雑なタスクをプランナーに依頼する (run\_planner.py)**

まず、プランナーが参照する知識ベース（ベクトルストア）を構築します。

python scripts/build\_knowledge\_base.py

次に、プランナーに複雑なタスクを依頼します。プランナーは学習済みのPlannerSNNを用いて計画を推論し、複数の専門家を連携させてタスクを実行します。

\# プランナーが「要約 \-\> 分析」の順で専門家を呼び出す  
python run\_planner.py \\  
    \--task\_request "この記事を要約して、その内容の感情を分析してください。" \\  
    \--context\_data "SNNは非常にエネルギー効率が高いことで知られているが、その性能はまだANNに及ばない点もある。"

#### **C) 単一タスクをエージェントに依頼する (run\_agent.py)**

エージェントは、まず既存の専門家モデルを探し、見つからない場合はオンデマンドで新しい専門家を学習します。

\# 「文章要約」の専門家がいない場合、知識蒸留による学習が自動で開始される  
python run\_agent.py \\  
    \--task\_description "文章要約" \\  
    \--unlabeled\_data\_path data/sample\_data.jsonl \\  
    \--prompt "SNNは、生物の神経系における情報の伝達と処理のメカニズムを模倣したニューラルネットワークの一種である。"

#### **D) 学習済みモデルと対話する (app/main.py)**

学習済みの専門家モデル（例: runs/specialists/文章要約/best\_model.pth）を指定して、GradioベースのWeb UIを起動します。

python app/main.py \--model\_path "runs/specialists/文章要約/best\_model.pth"

### **ステップ3: モデルの学習 (開発者向け)**

#### **A) 学習パラダイムの選択**

本プロジェクトでは、configs/base\_config.yaml 内の training.paradigm キーを書き換えることで、複数の学習アルゴリズムをシームレスに切り替えられます。

* **gradient\_based (勾配ベース学習):** 高性能な標準学習方式。知識蒸留も含む。  
* **self\_supervised (自己教師あり学習):** ラベルなしデータからの事前学習。  
* **physics\_informed (物理情報学習):** 膜電位の滑らかさなどを制約に加え、より生物に近いダイナミクスでの学習。  
* **biologically\_plausible (生物学的学習):** STDPなど、脳の学習ルールに近いアルゴリズム。

#### **B) 専門家モデルの学習 (train.py)**

gradient\_based (知識蒸留) を用いて、新しい専門家SNNを学習させる例です。

\# 1\. 知識蒸留のための教師データ（ロジット）を事前計算  
python scripts/prepare\_distillation\_data.py \\  
    \--input\_file data/sample\_data.jsonl \\  
    \--output\_dir precomputed\_data/sample\_distill

\# 2\. configのparadigmを "gradient\_based", typeを "distillation" に設定

\# 3\. 事前計算データを使ってSNNモデルを学習  
python train.py \\  
    \--model\_config configs/models/small.yaml \\  
    \--data\_path precomputed\_data/sample\_distill

#### **C) プランナーモデルの学習 (train\_planner.py)**

計画立案能力を持つ PlannerSNN を学習させます。

python train\_planner.py

## **5\. プロジェクト構造**

snn2/  
├── app/                  \# UIアプリケーションとDIコンテナ  
│   ├── adapters/         \# LangChainなど外部ライブラリとの連携  
│   ├── services/         \# ビジネスロジック  
│   └── containers.py     \# 依存性注入コンテナ  
├── configs/                \# 設定ファイル  
│   ├── base\_config.yaml  \# 基本設定、学習パラダイム切り替え  
│   └── models/           \# モデルアーキテクチャ設定  
├── doc/                    \# ドキュメント  
│   └── ROADMAP.md        \# プロジェクトロードマップ  
├── scripts/                \# データ準備やベンチマークなどの補助スクリプト  
├── snn\_research/           \# SNNコア研究開発コード  
│   ├── agent/            \# 各種エージェント (自律、自己進化、生命体)  
│   ├── cognitive\_architecture/ \# 高次認知機能 (プランナー、ワークスペース、SNAKE等)  
│   ├── core/             \# SNNモデルとニューロンのコア定義  
│   ├── learning\_rules/   \# 生物学的学習則 (STDPなど)  
│   └── training/         \# Trainerと損失関数  
├── train.py                \# 専門家モデルの学習スクリプト  
├── train\_planner.py        \# プランナーモデルの学習スクリプト  
└── run\_\*.py                \# 各種機能の実行スクリプト  
