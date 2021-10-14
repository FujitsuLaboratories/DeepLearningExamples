# pytorch + oneDNNサンプルプログラム

## このプログラムについて
- oneDNNの効果を確認するためのサンプルプログラムです。
- 学習モデルとしてAlexNet, 学習データはCIFAR-10を使用します。

## ファイル構成
- README.md：README(英語)
- READMEjp.md：このファイル
- main.py：メイン
- alexnet.py：学習モデル(AlexNet)

## 使用コマンド・ライブラリ
- python 3.x
- torch
- torchvision
- argparse

## 環境構築
- pipなどを使って必要なライブラリをインストール
```
$ pip install torch torchvision argparse tqdm
```

## 使い方(例)
```
$ python3 main.py --onednn -e 10
```

## 引数
- -e, --epochs num(int): 実行するepoch数(default=2)
- -o, --onednn(bool): oneDNN適用
- -h, --help: ヘルプ

## 入力ファイル
- CIFAR-10 (自動取得)

## 出力ファイル
- なし(標準出力)

## Copyright
COPYRIGHT Fujitsu Limited 2021
