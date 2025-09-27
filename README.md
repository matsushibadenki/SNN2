# **SNNベース AIチャットシステム (v2.4 \- 大規模分散学習環境)**

## **1\. 概要**

本プロジェクトは、スパイキングニューラルネットワーク（SNN）技術を基盤とした、次世代のAIチャットシステムです。DIコンテナの導入により、研究開発からアプリケーション化までをシSeamlessに繋ぐ、高い保守性と拡張性を持つアーキテクチャに刷新されました。

### **1.1. 設計思想**

* **関心の分離:** snn\_research（SNNコア技術の研究開発）と app（モデルを利用するアプリケーション）を明確に分離。  
* **依存性の注入 (DI):** dependency-injector を用い、クラス間の依存関係を外部コンテナで管理することで、疎結合でテスト容易性の高い設計を実現。  
* **設定のモジュール化:** 学習設定（configs/base\_config.yaml）とモデルアーキテクチャ設定（configs/models/\*.yaml）を分離し、実験の組み合わせを容易に。  
* **スケーラブルな学習環境:** torchrun を活用し、複数GPUを用いた大規模な分散学習を堅牢かつ容易に実行できる環境を構築。

## **2\. 使い方 (How to Use)**

### **ステップ1: 環境設定**

まず、プロジェクトに必要なライブラリをインストールします。

pip install \-r requirements.txt

### **ステップ2: データ準備**

#### **2.1. 通常学習用データ (オプション)**

WikiTextのような大規模データセットを準備する場合、以下のスクリプトを実行します。

python \-m scripts.data\_preparation

#### **2.2. 知識蒸留用データ (推奨)**

知識蒸留を行う前に、教師モデルのロジットを事前計算する必要があります。

\# 例: sample\_data.jsonl から蒸留用データを作成  
python \-m scripts.prepare\_distillation\_data \\  
    \--input\_file data/sample\_data.jsonl \\  
    \--output\_dir precomputed\_data/

### **ステップ3: モデルの学習**

#### **3.1. 単一デバイスでの学習**

train.py を直接実行します。--configでベース設定を、--model\_configでモデルのアーキテクチャを指定します。

\# smallモデルのアーキテクチャで学習を開始  
python train.py \\  
    \--config configs/base\_config.yaml \\  
    \--model\_config configs/models/small.yaml \\  
    \--data\_path data/sample\_data.jsonl

\# WikiTextデータセットで学習を開始  
python train.py \\  
    \--config configs/base\_config.yaml \\  
    \--model\_config configs/models/small.yaml \\  
    \--data\_path data/wikitext-103\_train.jsonl

#### **3.2. 大規模分散学習 (Multi-GPU)**

新しく追加された run\_distributed\_training.sh を使用します。**このスクリプトは、利用可能な全てのGPUを自動検出し、分散学習を開始します。**

\# スクリプトに実行権限を付与  
chmod \+x scripts/run\_distributed\_training.sh

\# 分散学習を開始 (スクリプト内の設定が使用されます)  
./scripts/run\_distributed\_training.sh

### **ステップ4: 対話アプリケーションの起動**

学習済みのモデルを使って、GradioベースのチャットUIを起動します。**学習時に使用したモデル設定ファイルを指定してください。**

\# 例: mediumモデルを起動  
python \-m app.main \\  
    \--config configs/base\_config.yaml \\  
    \--model\_config configs/models/medium.yaml

http://0.0.0.0:7860 を開いてください。

### **ステップ5: ベンチマークによる性能評価**

SST-2データセットを用いて、SNNモデルとANNベースラインモデルの性能を比較評価します。

python \-m scripts.run\_benchmark  
