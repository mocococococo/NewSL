# simulator_step.py
from __future__ import annotations
from typing import List, Optional, Tuple

from . import fast_simulator
from .state import State
from board.constant import BOARD_SIZE, VX_MIN, VX_MAX, VY_MIN, VY_MAX, DCL2_YPOS_DIFF

N_ACTIONS = BOARD_SIZE * BOARD_SIZE * 2
VEC_SIZE = BOARD_SIZE * BOARD_SIZE  # 1024
StonePos = Tuple[float, float]
Stones16 = List[Optional[StonePos]]

DEBUG_SIM_INDEX = True

def _shot_to_teamblock_index(shot_index: int, hammer: bool) -> int:
    """
    shot_index(0..15) が「投球順」のとき、その石が対応する team-block の index を返す。
    team-block は 0..7 が team0, 8..15 が team1。
    hammer=True なら hammerチームは team1、False なら hammerチームは team0。
    """
    h = 1 if hammer else 0
    nh = 1 - h
    k = shot_index // 2
    team = nh if (shot_index % 2 == 0) else h
    return k if team == 0 else 8 + k

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
    
    if DEBUG_SIM_INDEX:
        tb = _shot_to_teamblock_index(state.shot_index, state.hammer)
        tm = state.to_move()
        before_none = (state.stones[tb] is None)
        print(f"[SIMIDX] BEFORE shot_index={state.shot_index} to_move={tm} teamblock_index={tb} is_none={before_none}")
    
    # print("------ DEBUG simulate_step Before Simulate -----")
    # for i, p in enumerate(state.stones):
    #     print(f"stone pos: x={p[0]} y={p[1]}" if p is not None else f"stone pos: None")
    
    stones_shotorder = _teamblock_to_shotorder(list(state.stones), state.hammer)

    stones_in: List[Tuple[float, float]] = []
    for p in stones_shotorder:
        if p is None:
            stones_in.append((0.0, 0.0))
        else:
            stones_in.append((float(p[0]), float(p[1]) + DCL2_YPOS_DIFF))

    results = fast_simulator.simulate(stones_in, state.shot_index, (vx, vy, spin))
    # C++側は y>0 を「石あり」としている:contentReference[oaicite:3]{index=3}
    stones_out_shotorder: Stones16 = []
    for x, y in results:
        if y > 0:        
            y -= DCL2_YPOS_DIFF
            stones_out_shotorder.append((x, y))
        else:
            stones_out_shotorder.append(None)
            
    stones_out_teamblock = _shotorder_to_teamblock(stones_out_shotorder, state.hammer)
    
    if DEBUG_SIM_INDEX:
        tb = _shot_to_teamblock_index(state.shot_index, state.hammer)
        tm = state.to_move()
        after_none = (stones_out_teamblock[tb] is None)
        if after_none:
            print(f"[SIMIDX] AFTER  shot_index={state.shot_index} to_move={tm} teamblock_index={tb} is_none=True")
        else:
            x, y = stones_out_teamblock[tb]
            print(f"[SIMIDX] AFTER  shot_index={state.shot_index} to_move={tm} teamblock_index={tb} is_none=False pos=({x:.3f},{y:.3f})")
    
    # print("------ DEBUG simulate_step After Simulate -----")
    # for i, p in enumerate(stones_out_teamblock):
    #     print(f"stone pos: x={p[0]} y={p[1]}" if p is not None else f"stone pos: None")

    return State(
        stones=tuple(stones_out_teamblock),
        hammer=state.hammer,
        shot_index=state.shot_index + 1,
        end=state.end,
        score_diff=state.score_diff,
    )

def _teamblock_to_shotorder(stones_teamblock, hammer) -> Stones16:
    # stones_teamblock: [0..7 team0][8..15 team1]
    # return: [0..15 shot order] where even=nonhammer, odd=hammer
    h = 1 if hammer else 0
    nh = 1 - h
    out = [None] * 16
    for team in (0, 1):
        for k in range(8):
            tb = team * 8 + k
            p = stones_teamblock[tb]
            shot = 2 * k if team == nh else 2 * k + 1
            out[shot] = p
    return out

def _shotorder_to_teamblock(stones_shotorder, hammer) -> Stones16:
    h = 1 if hammer else 0
    nh = 1 - h
    out = [None] * 16
    for shot in range(16):
        p = stones_shotorder[shot]
        k = shot // 2
        team = nh if (shot % 2 == 0) else h
        tb = k if team == 0 else 8 + k
        out[tb] = p
    return out

def _to_move_team(shot_index: int, hammer: bool) -> int:
    """
    今この shot_index を投げるチームを返す
    hammer=True なら team1 が後攻(ハンマー)
    shot_index 偶数: 先攻(ハンマーではない側)
    shot_index 奇数: 後攻(ハンマー側)
    """
    h = 1 if hammer else 0
    nh = 1 - h
    if (shot_index % 2) == 0:
        return nh
    return h


def _teamblock_index_for_current_shot(shot_index: int, hammer: bool) -> int:
    """
    今回の投球 shot_index が、teamblock(0-7 team0, 8-15 team1) のどこに入るべきか
    例: shot_index=0 は先攻1投目なので (先攻チームの0番) に入る
    """
    team = _to_move_team(shot_index, hammer)
    k = shot_index // 2  # そのチームの何投目か(0..7)
    if team == 0:
        return k
    return 8 + k


def _stone_is_none(stones_teamblock, idx: int) -> bool:
    p = stones_teamblock[idx]
    return p is None