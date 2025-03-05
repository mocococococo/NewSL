# NewSL
- [デジタルカーリング第3世代](https://github.com/digitalcurling/DigitalCurling3)用の思考エンジン
- Windows、WSL2 対応
- Mac 対応
- Linux 不明

# 環境
事前に揃えてある必要がある環境です。()内のものは必須ではありません。
- Python 3.12.8
- (cuda)
- (cuDNN)


- [使用するパッケージ](#requirements)

# 外部パッケージ
|使用するパッケージ|用途|
|---|---|
|click|コマンドライン引数の実装|
|numpy|行列計算|
|pytorch|ニューラルネットワークの構成と学習の実装|

# セットアップ方法
```
git clone https://github.com/mocococococo/NewSL.git
cd NewSL
pip3 install -r requirements.txt
```

# 対戦方法
```
python sample.py --host="localhost" --port=10000 --model="Default.bin" --gpu_use=True --name="AI_Name"
```