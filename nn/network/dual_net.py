#Dual Networkの実装

from typing import Tuple
from torch import nn
import torch

from board.constant import BOARD_SIZE, PLANES_SIZE
from nn.network.res_block import ResidualBlock
from nn.network.head.policy_head import PolicyHead
from nn.network.head.value_head import ValueHead


class DualNet(nn.Module):
    def __init__(self, device: torch.device):
        #Dual Networkの実装クラス

        super().__init__()
        filters = 32  
        blocks = 9    

        self.device = device

        self.conv_layer = nn.Conv2d(in_channels=PLANES_SIZE, out_channels=filters, \
            kernel_size=3, padding=1, bias=False)
        self.bn_layer = nn.BatchNorm2d(num_features=filters)
        self.relu = nn.ReLU()
        self.blocks = make_common_blocks(blocks, filters)
        self.policy_head = PolicyHead(filters)
        self.value_head = ValueHead(filters)

        self.softmax0 = nn.Softmax(dim=0)
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)

    def forward(self, input_plane: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        #前向き伝播処理を実行する。
        blocks_out = self.blocks(self.relu(self.bn_layer(self.conv_layer(input_plane))))

        return self.policy_head(blocks_out), self.value_head(blocks_out)

    def forward_for_sl(self, input_plane: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        #前向き伝搬処理を実行する。教師有り学習で利用する。
        policy, value = self.forward(input_plane)
        batch_size = input_plane.shape[0]
        policy_size = 2 * BOARD_SIZE * BOARD_SIZE
        #print("batch_size: ", batch_size, "policy_size: ", policy_size)
        policy = policy.view(batch_size, policy_size)
        return policy, value


    def forward_with_softmax(self, input_plane: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        #前向き伝搬処理を実行する。
        policy, value = self.forward(input_plane)
        batch_size = input_plane.shape[0]
        policy_size = 2 * BOARD_SIZE * BOARD_SIZE
        #print("batch_size: ", batch_size, "policy_size: ", policy_size)
        policy = policy.view(batch_size, policy_size)
        return self.softmax(policy), self.softmax(value)

    def forward_with_softmax2(self, input_plane: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        #前向き伝搬処理を実行する。
        policy, value = self.forward(input_plane)
        policy_size = 2 * BOARD_SIZE * BOARD_SIZE
        policy = policy.view(1, policy_size)
        return self.softmax(policy), self.softmax(value)

    def inference(self, input_plane: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        #前向き伝搬処理を実行する。探索用に使うメソッドのため、デバイス間データ転送も内部処理する。
        policy, value = self.forward(input_plane.to(self.device))
        return self.softmax(policy).cpu(), self.softmax(value).cpu()


    def inference_with_policy_logits(self, input_plane: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        #前向き伝搬処理を実行する。Gumbel AlphaZero用の探索に使うメソッドのため、
        #デバイス間データ転送も内部処理する。
        policy, value = self.forward(input_plane.to(self.device))
        return policy.cpu(), self.softmax(value).cpu()

    '''def forward_original(self, input_plane: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        #自分で作った前向き伝播処理
        policy, value = self.forward(input_plane) '''


def make_common_blocks(num_blocks: int, num_filters: int) -> torch.nn.Sequential:
    #DualNetで用いる残差ブロックの塊を構成して返す。
    
    blocks = [ResidualBlock(num_filters) for _ in range(num_blocks)]
    return nn.Sequential(*blocks)