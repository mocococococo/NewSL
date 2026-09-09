import torch
import numpy as np
from typing import Tuple

from nn.network.dual_net import DualNet
from dc3client.models import StoneRotation
from board.constant import BOARD_SIZE_X, BOARD_SIZE_Y, PLANES_SIZE, \
                            VX_MIN, VX_MAX, VY_MIN, VY_MAX, \
                            VX_SIZE, VY_SIZE, VY_SHEET_MAX, \
                            VY_SHEET_SIZE, VY_EXTRA_SIZE

def generate_move_from_policy(network: DualNet, input, shot_index: int) -> Tuple[float, float, StoneRotation]:
    input_data = torch.tensor(input.reshape(1, PLANES_SIZE, BOARD_SIZE_Y, BOARD_SIZE_X)).to(network.device)
    policy, value = network.forward_with_softmax2(input_data)
    policy = policy.reshape(VX_SIZE * VY_SIZE * 2).cpu()
    
    selected_index = np.argmax(policy)

    return index_to_shot(selected_index)

def index_to_shot(index) -> Tuple[float, float, StoneRotation]:
    """
    1次元index（= 2回転 × VX_SIZE × VY_SIZE）から
    セル中心の (vx, vy) と回転方向を復元する
    """
    board_len = VX_SIZE * VY_SIZE

    # 回転（0: cw, 1: ccw）
    if index >= board_len:
        rotation = StoneRotation.counterclockwise
        cell = index - board_len
    else:
        rotation = StoneRotation.clockwise
        cell = index

    # セル座標 (vxi, vyi)
    vxi = cell % VX_SIZE
    vyi = cell // VX_SIZE

    # セル幅
    dvx = (VX_MAX - VX_MIN) / VX_SIZE
    dvy = (VY_SHEET_MAX - VY_MIN) / VY_SHEET_SIZE
    dvy_extra = (VY_MAX - VY_SHEET_MAX) / VY_EXTRA_SIZE

    # vx は均等：セル中心
    vx = VX_MIN + (vxi + 0.5) * dvx

    # vy は区間で分岐：セル中心
    if vyi < VY_SHEET_SIZE:
        vy = VY_MIN + (vyi + 0.5) * dvy
    else:
        vy_idx2 = vyi - VY_SHEET_SIZE  # 高速域内のインデックス
        vy = VY_SHEET_MAX + (vy_idx2 + 0.5) * dvy_extra

    return float(vx), float(vy), rotation
