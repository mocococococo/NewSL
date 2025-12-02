#polocy headの実装

import torch
from torch import nn

from board.constant import BOARD_SIZE_X, BOARD_SIZE_Y, VX_SIZE, VY_SIZE

class PolicyHead(nn.Module):
    #Policy headの実装クラス。
    def __init__(self, channels: int, momentum: float=0.01):
        #Policy headの初期化処理。

        #channels (int): 共通ブロック部の畳み込み層のチャネル数。
        #momentum (float, optional): バッチ正則化層のモーメンタムパラメータ. Defaults to 0.01.
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=2, \
            kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=1, \
            kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=2, eps=2e-5, momentum=momentum)
        self.bn2 = nn.BatchNorm2d(num_features=1, eps=2e-5, momentum=momentum)
        self.fc_layer1 = nn.Linear(BOARD_SIZE_X * BOARD_SIZE_Y, 2048)
        self.fc_layer2 = nn.Linear(2048, 2 * VX_SIZE * VY_SIZE)
        self.relu = nn.ReLU()

    def forward(self, input_plane: torch.Tensor) -> torch.Tensor:
        #前向き伝播処理を実行する
        hidden1 = self.relu(self.bn1(self.conv1(input_plane)))
        hidden2 = self.relu(self.bn2(self.conv2(hidden1)))
        batch_size, _, height, width = hidden2.shape
        reshape = hidden2.reshape(batch_size, height * width)
        fc1 = self.relu(self.fc_layer1(reshape))
        policy_out = self.fc_layer2(fc1)
        
        return policy_out