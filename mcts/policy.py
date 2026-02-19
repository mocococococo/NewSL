# policy.py
from __future__ import annotations
from typing import List, Optional, Tuple

import torch

from nn.feature import generate_input_planes
from nn.network.dual_net import DualNet
from board.constant import VX_SIZE, VY_SIZE, PLANES_SIZE, BOARD_SIZE_X, BOARD_SIZE_Y
from .state import State

# 32*32*2=2048
N_ACTIONS = VX_SIZE * VY_SIZE * 2
N_VALUE_CLASSES = 17

# 外から設定する（search開始前に1回だけセット）
_DUAL_NET = None
_SCORE_DIFF: int = None


def set_policy_context(dual_net: DualNet, score_diff: int) -> None:
    """探索で使うNNとscoresをセットする（開始前に1回だけ呼ぶ）"""
    global _DUAL_NET, _SCORE_DIFF
    _DUAL_NET = dual_net
    _SCORE_DIFF = score_diff


def _state_stones_to_feature_stones(state: State) -> List[Optional[dict]]:
    """State.stones (x,y or None) -> feature.pyが期待するdict形式に変換"""
    out: List[Optional[dict]] = []
    for p in state.stones:
        if p is None:
            out.append(None)
        else:
            x, y = float(p[0]), float(p[1])
            out.append({"position": {"x": x, "y": y}})
    return out


def get_policy_and_value(state: State) -> Tuple[List[float], List[float]]:
    """
    Node.expand_if_needed() から呼ばれる想定の関数。
    返り値: 長さ2048の確率分布（list[float]）
    """
    if _DUAL_NET is None:
        raise RuntimeError("dual_net is not set. Call set_policy_context() first.")
    if _SCORE_DIFF is None:
        raise RuntimeError("score_diff is not set. Call set_policy_context() first.")

    stones_for_feature = _state_stones_to_feature_stones(state)
    
    # debug用表示
    # print("[POLICY] ------ Before Generate Input Planes -----")
    # for i, p in enumerate(stones_for_feature):
    #     if p is None:
    #         print(f"stone pos: None")
    #     else:
    #         if i < 8:
    #             print(f"stone pos [team0]: x={p['position']['x']} y={p['position']['y']}")
    #         else:
    #             print(f"stone pos [team1]: x={p['position']['x']} y={p['position']['y']}")

    planes_np = generate_input_planes(
        stones=stones_for_feature,
        end=state.end,
        shot=state.shot_index,
        hammer=state.hammer_team,
        score_diff_for_team0=_SCORE_DIFF
    )  # (PLANES_SIZE, 32, 32) float32 :contentReference[oaicite:4]{index=4}

    input_data = torch.tensor(planes_np.reshape(1, PLANES_SIZE, BOARD_SIZE_Y, BOARD_SIZE_X))

    _DUAL_NET.eval()
    with torch.no_grad():
        device = getattr(_DUAL_NET, "device", None)
        if device is not None:
            input_data = input_data.to(device)

        policy_t, value_t = _DUAL_NET.forward_with_softmax2(input_plane=input_data)
        policy = policy_t.squeeze(0).detach().cpu().tolist()
        value = value_t.squeeze(0).detach().cpu().tolist()

    if len(policy) != N_ACTIONS:
        raise RuntimeError(f"policy length mismatch: {len(policy)} != {N_ACTIONS}")
    if len(value) != N_VALUE_CLASSES:
        raise RuntimeError(f"value length mismatch: {len(value)} != {N_VALUE_CLASSES}")

    return policy, value


def get_policy(state: State) -> List[float]:
    policy, _ = get_policy_and_value(state)
    return policy
