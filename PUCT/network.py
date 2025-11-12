import torch
from typing import List, Tuple

from board.constant import BOARD_SIZE, PLANES_SIZE
from nn.mcts.network.dual_net import DualNet
from nn.mcts.feature import generate_input_planes

from PUCT.state import State

class PolicyNetwork:
    """
    ポリシーネットワーク。

    使い方:
        nn = PolicyNetwork(device)
        policy = nn.evaluate(state)
    """

    def __init__(self, network: DualNet):
        """
        構築子。

        Args:
            device (torch.device): ネットワークを配置するデバイス。
        """
        self.network = network
        self.network.eval()

    def evaluate(self, state: State) -> List[float]:
        """
        状態に対するポリシーを推論する。

        Args:
            state (State): 対象状態。

        Returns:
            List[float]: 各アクションの確率（長さ2048）。
        """
        input_planes = generate_input_planes(
            stones=state.stones,
            scores=state.score_diff,
            end=state.end,
            shot=state.throw_index,
        )
        input_tensor = torch.tensor(input_planes.reshape(1, PLANES_SIZE, BOARD_SIZE, BOARD_SIZE)).to(self.network.device)
        policy, value = self.network.forward_with_softmax2(input_tensor)
        policy = policy.reshape(BOARD_SIZE * BOARD_SIZE * 2).cpu()

        return policy