#polocy headの実装  途中なの注意

import torch
from torch import nn

class PolicyHead(nn.Module):
    #Policy headの実装クラス。
    def __init__(self, channels: int, momentum: float=0.01):
        #Policy headの初期化処理。

        #channels (int): 共通ブロック部の畳み込み層のチャネル数。
        #momentum (float, optional): バッチ正則化層のモーメンタムパラメータ. Defaults to 0.01.
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=2, \
            kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=2, \
            kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=2, eps=2e-5, momentum=momentum)
        self.bn2 = nn.BatchNorm2d(num_features=2, eps=2e-5, momentum=momentum)
        self.relu = nn.ReLU()

    def forward(self, input_plane: torch.Tensor) -> torch.Tensor:
        #前向き伝播処理を実行する
        hidden1 = self.relu(self.bn1(self.conv1(input_plane)))
        policy_out = self.relu(self.bn2(self.conv2(hidden1)))
        return policy_out