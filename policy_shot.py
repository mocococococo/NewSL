import torch
import numpy as np
from typing import Tuple

from nn.network.dual_net import DualNet
from dc3client.models import StoneRotation
from board.constant import BOARD_SIZE, PLANES_SIZE, VX_MIN, VX_MAX, VY_MIN, VY_MAX

def generate_move_from_policy(network: DualNet, input, shot_index: int) -> Tuple[float, float, StoneRotation]:
    input_data = torch.tensor(input.reshape(1, PLANES_SIZE, BOARD_SIZE, BOARD_SIZE)).to(network.device)
    policy, value = network.forward_with_softmax2(input_data)
    policy = policy.reshape(BOARD_SIZE * BOARD_SIZE * 2).cpu()
    
    selected_index = np.argmax(policy)

    return index_to_shot(selected_index)

def index_to_shot(index) -> Tuple[float, float, StoneRotation]:
    rotation = StoneRotation.clockwise
    if(index >= (BOARD_SIZE * BOARD_SIZE)):
        rotation = StoneRotation.counterclockwise
        index = index - (BOARD_SIZE * BOARD_SIZE)
    
    x_index = index % BOARD_SIZE
    y_index = index // BOARD_SIZE
    x_interval = (VX_MAX - VX_MIN) / (BOARD_SIZE - 1)
    y_interval = (VY_MAX - VY_MIN) / (BOARD_SIZE - 1)
    x = float(x_index * x_interval + VX_MIN)
    y = float(y_index * y_interval + VY_MIN)
    return x, y, rotation
