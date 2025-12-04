# simulator_step.py
from __future__ import annotations
from typing import List, Optional, Tuple

from . import fast_simulator
from .state import State
from board.constant import BOARD_SIZE, VX_MIN, VX_MAX, VY_MIN, VY_MAX

N_ACTIONS = BOARD_SIZE * BOARD_SIZE * 2
VEC_SIZE = BOARD_SIZE * BOARD_SIZE  # 1024

def _idx_to_value(i: int, vmin: float, vmax: float, size: int) -> float:
    # feature.py の discretization と逆対応になるように（size-1 で割る）
    step = (vmax - vmin) / (size - 1)
    return vmin + step * i

def decode_action(action: int) -> Tuple[float, float, int]:
    """
    action(0..2047) -> (vx, vy, spin)
    dual_net の (2,32,32) を view(2048) した並び（= spin面が先）に合わせる。
    """
    if not (0 <= action < N_ACTIONS):
        raise ValueError(f"action out of range: {action}")

    spin = action // VEC_SIZE           # 0 or 1
    vindex = action % VEC_SIZE          # 0..1023
    vx_i = vindex % BOARD_SIZE          # 0..31
    vy_i = vindex // BOARD_SIZE         # 0..31

    vx = _idx_to_value(vx_i, VX_MIN, VX_MAX, BOARD_SIZE)
    vy = _idx_to_value(vy_i, VY_MIN, VY_MAX, BOARD_SIZE)
    return vx, vy, spin

def simulator_step(state: State, action: int) -> State:
    """
    1手進める。スコア計算やエンド更新はここではしない（shot_indexだけ進める）。
    """
    if state.is_end_terminal():
        return state

    vx, vy, spin = decode_action(action)

    stones_in: List[Tuple[float, float]] = []
    for p in state.stones:
        if p is None:
            stones_in.append((0.0, 0.0))
        else:
            stones_in.append((float(p[0]), float(p[1])))

    results = fast_simulator.simulate(stones_in, state.shot_index, (vx, vy, spin))
    # C++側は y>0 を「石あり」としている:contentReference[oaicite:3]{index=3}
    stones_out: List[Optional[Tuple[float, float]]] = []
    for x, y in results:
        if y > 0:
            stones_out.append((float(x), float(y)))
        else:
            stones_out.append(None)

    return State(
        stones=tuple(stones_out),
        hammer=state.hammer,
        shot_index=state.shot_index + 1,
        end=state.end,
        score_diff=state.score_diff,
    )
