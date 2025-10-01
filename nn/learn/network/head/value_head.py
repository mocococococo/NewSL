"""Value headの実装。
"""
import torch
from torch import nn

from board.constant import BOARD_SIZE


class ValueHead(nn.Module):
    """Value headの実装クラス。
    """
    def __init__(self, channels: int, momentum: float=0.01):
        """Value headの初期化処理。

        Args:
            channels (int): 共通ブロック部の畳み込み層のチャネル数。
            momentum (float, optional): バッチ正則化層のモーメンタムパラメータ. Defaults to 0.01.
        """
        super().__init__()

        self.conv_layer = nn.Conv2d(in_channels=channels, out_channels=1, \
            kernel_size=3, padding=1, bias=False)
        self.bn_layer = nn.BatchNorm2d(num_features=1, eps=2e-5, momentum=momentum)
        self.fc_layer1 = nn.Linear(BOARD_SIZE ** 2, 256)
        self.fc_layer2 = nn.Linear(256, 17)
        self.relu = nn.ReLU()

    def forward(self, input_plane: torch.Tensor) -> torch.Tensor:
        """前向き伝搬処理を実行する。

        Args:
            input_plane (torch.Tensor): Value headへの入力テンソル。

        Returns:
            torch.Tensor: ValueのLogit出力
        """
        hidden = self.relu(self.bn_layer(self.conv_layer(input_plane)))
        batch_size, _, height, width = hidden.shape
        reshape = hidden.reshape(batch_size, height * width)
        fc1 = self.fc_layer1(reshape)
        value_out = self.fc_layer2(fc1)

        return value_out