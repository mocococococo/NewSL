#Residual Blockの実装

import torch
from torch import nn  #必要なモジュールのインポート


class ResidualBlock(nn.Module):
    #Residual Blockの実装クラス

    def __init__(self, channels: int, momentum: float=0.01):
        #channels (int): 畳み込み層のチャネル数。
        #momentum (float, optional): バッチ正則化層のモーメンタムパラメータ. Defaults to 0.01.

        super().__init__()      #継承したnn.Moduleの初期化関数(nn.Moduleの中の__init__())を起動

        #正方形フィルタ(カーネル)の1辺のサイズを3として畳み込み
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, \
            kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, \
            kernel_size=3, padding=1, bias=False)

        #バッチ正則化
        self.bn1 = nn.BatchNorm2d(num_features=channels, eps=2e-5, momentum=momentum)
        self.bn2 = nn.BatchNorm2d(num_features=channels, eps=2e-5, momentum=momentum)

        #ReLU
        self.relu = nn.ReLU()

    def forward(self, input_plane: torch.Tensor) -> torch.Tensor:
        #前向き伝搬処理を実行する。
        #input_plane (torch.Tensor): 入力テンソル。
        #torch.Tensor: ブロックの出力テンソル。

        hidden_1 = self.relu(self.bn1(self.conv1(input_plane)))
        hidden_2 = self.bn2(self.conv2(hidden_1))

        return self.relu(input_plane + hidden_2)