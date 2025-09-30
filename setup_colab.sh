#!/bin/bash
# matsushibadenki/snn2/setup_colab.sh
# Google Colab環境でプロジェクトを実行するためのセットアップスクリプト

# エラーが発生した場合はスクリプトを終了
set -e

echo " clonando el repositorio del proyecto SNN..."
# (Colabは通常プロジェクトルートから始まるため、リポジトリをクローンする想定)
# git clone https://github.com/your-repo/SNN2-project.git
# cd SNN2-project

echo " instalando las dependencias necesarias desde requirements.txt..."
# Colabの基本環境に含まれないライブラリをインストール
pip install -q -r requirements.txt

echo " comprobando la configuración del dispositivo..."
DEVICE=$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")

echo "✅ ¡Configuración completa!"
echo "--------------------------------------------------"
echo " Entorno listo para usar."
echo " Dispositivo detectado: $DEVICE"
echo " Para empezar, abre el notebook 'SNN_Project_Colab_Quickstart.ipynb'."
echo "--------------------------------------------------"