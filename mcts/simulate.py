# simulator_step.py
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

from . import fast_simulator
from .state import State
from board.constant import VX_SIZE, VY_SIZE, VX_MIN, VX_MAX, VY_MIN, VY_MAX, VY_SHEET_MAX
from .params import STDDV_SPEED, STDDV_ANGLE
from policy_shot import index_to_shot

N_ACTIONS = VX_SIZE * VY_SIZE * 2  # 2048
VEC_SIZE = VX_SIZE * VY_SIZE  # 1024
StonePos = Tuple[float, float]
Stones16 = List[Optional[StonePos]]
ShotNoise = Tuple[float, float]

DEBUG_SIM_INDEX = False  # True にすると、simulate_step 内で詳細ログを出す

def _shot_to_teamblock_index(shot_index: int, hammer_team: int) -> int:
    """
    shot_index(0..15) が「投球順」のとき、その石が対応する team-block の index を返す。
    team-block は 0..7 が team0, 8..15 が team1。
    hammer_team は後攻チーム番号 (0 or 1)。
    例: hammer_team=1 (team1が後攻) のとき
        shot_index=0 -> team0の0番 -> return 0
        shot_index=1 -> team1の0番 -> return 8
    """
    team = _to_move_team(shot_index, hammer_team)
    k = shot_index // 2  # そのチームの何投目か(0..7)
    if team == 0:
        return k
    return 8 + k

def _vx_idx_to_value(vxi: int) -> float:
    dvx = (VX_MAX - VX_MIN) / VX_SIZE
    return VX_MIN + (vxi + 0.5) * dvx

def _vy_idx_to_value(vyi: int) -> float:
    # policy_shot と同じ：前半(VY_SIZE-5)は VY_MIN..VY_SHEET_MAX、後半5binは VY_SHEET_MAX..VY_MAX
    dvy = (VY_SHEET_MAX - VY_MIN) / (VY_SIZE - 5)
    dvy_extra = (VY_MAX - VY_SHEET_MAX) / 5

    if vyi < (VY_SIZE - 5):
        return VY_MIN + (vyi + 0.5) * dvy
    else:
        vy2 = vyi - (VY_SIZE - 5)  # 0..4
        return VY_SHEET_MAX + (vy2 + 0.5) * dvy_extra

def decode_action(action: int) -> Tuple[float, float, int]:
    if not (0 <= action < N_ACTIONS):
        raise ValueError(f"action out of range: {action}")

    board_len = VX_SIZE * VY_SIZE  # 1024

    # spin: 0=cw, 1=ccw（policy_shot と同じ並び）
    if action >= board_len:
        spin = 1
        cell = action - board_len
    else:
        spin = 0
        cell = action

    vxi = cell % VX_SIZE
    vyi = cell // VX_SIZE

    vx = _vx_idx_to_value(vxi)
    vy = _vy_idx_to_value(vyi)
    return vx, vy, spin

def _add_noise_to_vector(
    x: float,
    y: float,
    stddev_speed: float,
    stddev_angle: float,
    noise: Optional[ShotNoise] = None,
) -> Tuple[float, float]:
    magnitude = np.sqrt(x ** 2 + y ** 2)
    angle = np.arctan2(y, x)

    if noise is None:
        speed_noise = np.random.normal(0.0, stddev_speed)
        angle_noise = np.random.normal(0.0, stddev_angle)
    else:
        speed_noise, angle_noise = noise

    noisy_magnitude = magnitude + float(speed_noise)
    noisy_angle = angle + float(angle_noise)

    new_x = noisy_magnitude * np.cos(noisy_angle)
    new_y = noisy_magnitude * np.sin(noisy_angle)
    return float(new_x), float(new_y)

def simulator_step(state: State, action: int) -> State:
    """
    1手進める。スコア計算やエンド更新はここではしない（shot_indexだけ進める）。
    """
    if state.is_end_terminal():
        return state

    vx, vy, spin = decode_action(action)
    vx, vy = _add_noise_to_vector(vx, vy, STDDV_SPEED, STDDV_ANGLE)
    debug_vx, debug_vy, debug_spin = index_to_shot(action)
    
    if DEBUG_SIM_INDEX:
        print(f"[SIMULATOR_STEP] action={action} -> vx={vx:.3f} vy={vy:.3f} spin={spin}")
        print(f"[SIMULATOR_STEP] DEBUG_SHOT   -> vx={debug_vx:.3f} vy={debug_vy:.3f} spin={debug_spin}")

    if DEBUG_SIM_INDEX:
        tb = _shot_to_teamblock_index(state.shot_index, state.hammer_team)
        tm = state.to_move()
        before_none = (state.stones[tb] is None)
        print("------ DEBUG simulate_step Before Simulate -----")
        print(f"[SIMIDX] BEFORE shot_index={state.shot_index} to_move={tm} teamblock_index={tb} is_none={before_none}")
        
    stones_shotorder = _teamblock_to_shotorder(list(state.stones), state.hammer_team)
    
    if DEBUG_SIM_INDEX:
        print("[BEFORE] ------ DEBUG simulate_step Shotorder Before Simulate -----")
        for i, p in enumerate(stones_shotorder):
            print(f"stone pos [shot{i}]: x={p[0]} y={p[1]}" if p is not None else f"stone pos: None")

    stones_in: List[Tuple[float, float]] = []
    for p in stones_shotorder:
        if p is None:
            stones_in.append((0.0, 0.0))
        else:
            stones_in.append((float(p[0]), float(p[1])))
            
    freeguard = (state.shot_index < 5)  # 先攻後攻合わせて最初の5投はフリーガードゾーンルール適用
    results = fast_simulator.simulate(stones_in, state.shot_index, (vx, vy, spin), freeguard)
    # C++側は y>0 を「石あり」としている:contentReference[oaicite:3]{index=3}
    stones_out_shotorder: Stones16 = []
    for x, y in results:
        if y > 0:
            stones_out_shotorder.append((x, y))
        else:
            stones_out_shotorder.append(None)
            
    stones_out_teamblock = _shotorder_to_teamblock(stones_out_shotorder, state.hammer_team)
    
    if DEBUG_SIM_INDEX:
        print("[AFTER] ------ DEBUG simulate_step Teamblock After Simulate -----")
        for i, p in enumerate(stones_out_teamblock):
            if i < 8:
                print(f"stone pos [team0]: x={p[0]} y={p[1]}" if p is not None else f"stone pos: None")
            else:
                print(f"stone pos [team1]: x={p[0]} y={p[1]}" if p is not None else f"stone pos: None")
        tb = _shot_to_teamblock_index(state.shot_index, state.hammer_team)
        tm = state.to_move()
        after_none = (stones_out_teamblock[tb] is None)
        if after_none:
            print(f"[SIMIDX] AFTER  shot_index={state.shot_index} to_move={tm} teamblock_index={tb} is_none=True")
        else:
            x, y = stones_out_teamblock[tb]
            print(f"[SIMIDX] AFTER  shot_index={state.shot_index} to_move={tm} teamblock_index={tb} is_none=False pos=({x:.3f},{y:.3f})")
    

    return State(
        stones=tuple(stones_out_teamblock),
        end=state.end,
        hammer_team=state.hammer_team,
        shot_index=state.shot_index + 1,
        score_diff=state.score_diff,
    )

def simulator_step_continuous(
    state: State,
    vx: float,
    vy: float,
    spin: int,
    noise: Optional[ShotNoise] = None,
) -> State:
    """
    Continuous-shot variant of simulator_step.

    The action-grid decode is skipped, but the same shot noise, freeguard
    handling, team-block conversion, and fast simulator are used.
    """
    if state.is_end_terminal():
        return state

    vx, vy = _add_noise_to_vector(
        float(vx),
        float(vy),
        STDDV_SPEED,
        STDDV_ANGLE,
        noise=noise,
    )
    spin = 1 if int(spin) == 1 else 0

    if DEBUG_SIM_INDEX:
        print(f"[SIMULATOR_STEP] continuous -> vx={vx:.3f} vy={vy:.3f} spin={spin}")

    if DEBUG_SIM_INDEX:
        tb = _shot_to_teamblock_index(state.shot_index, state.hammer_team)
        tm = state.to_move()
        before_none = (state.stones[tb] is None)
        print("------ DEBUG simulate_step Before Simulate -----")
        print(f"[SIMIDX] BEFORE shot_index={state.shot_index} to_move={tm} teamblock_index={tb} is_none={before_none}")

    stones_shotorder = _teamblock_to_shotorder(list(state.stones), state.hammer_team)

    if DEBUG_SIM_INDEX:
        print("[BEFORE] ------ DEBUG simulate_step Shotorder Before Simulate -----")
        for i, p in enumerate(stones_shotorder):
            print(f"stone pos [shot{i}]: x={p[0]} y={p[1]}" if p is not None else f"stone pos: None")

    stones_in: List[Tuple[float, float]] = []
    for p in stones_shotorder:
        if p is None:
            stones_in.append((0.0, 0.0))
        else:
            stones_in.append((float(p[0]), float(p[1])))

    freeguard = (state.shot_index < 5)
    results = fast_simulator.simulate(stones_in, state.shot_index, (vx, vy, spin), freeguard)
    stones_out_shotorder: Stones16 = []
    for x, y in results:
        if y > 0:
            stones_out_shotorder.append((x, y))
        else:
            stones_out_shotorder.append(None)

    stones_out_teamblock = _shotorder_to_teamblock(stones_out_shotorder, state.hammer_team)

    if DEBUG_SIM_INDEX:
        print("[AFTER] ------ DEBUG simulate_step Teamblock After Simulate -----")
        for i, p in enumerate(stones_out_teamblock):
            if i < 8:
                print(f"stone pos [team0]: x={p[0]} y={p[1]}" if p is not None else f"stone pos: None")
            else:
                print(f"stone pos [team1]: x={p[0]} y={p[1]}" if p is not None else f"stone pos: None")
        tb = _shot_to_teamblock_index(state.shot_index, state.hammer_team)
        tm = state.to_move()
        after_none = (stones_out_teamblock[tb] is None)
        if after_none:
            print(f"[SIMIDX] AFTER  shot_index={state.shot_index} to_move={tm} teamblock_index={tb} is_none=True")
        else:
            x, y = stones_out_teamblock[tb]
            print(f"[SIMIDX] AFTER  shot_index={state.shot_index} to_move={tm} teamblock_index={tb} is_none=False pos=({x:.3f},{y:.3f})")

    return State(
        stones=tuple(stones_out_teamblock),
        end=state.end,
        hammer_team=state.hammer_team,
        shot_index=state.shot_index + 1,
        score_diff=state.score_diff,
    )

def _teamblock_to_shotorder(stones_teamblock, hammer_team: int) -> Stones16:
    """    
    team-block 形式の stones を 投球順形式に変換する。
    つまり、先攻・後攻が交互に並ぶ形に変換する。
    hammer_team は後攻チーム番号 (0 or 1)。
        例: hammer_team=1 (team1が後攻) のとき
            shot_index=0 -> team0の0番
            shot_index=1 -> team1の0番
            shot_index=2 -> team0の1番
            shot_index=3 -> team1の1番
            ...
    """
    # stones_teamblock: [0..7 team0][8..15 team1]
    # return: [0..15 shot order] where even=nonhammer, odd=hammer
    h = hammer_team
    nh = 1 - h
    out: Stones16 = [None] * 16
    for shot in range(16):
        k = shot // 2
        team = nh if (shot % 2 == 0) else h
        tb = k if team == 0 else 8 + k
        out[shot] = stones_teamblock[tb]
        
    return out

def _shotorder_to_teamblock(stones_shotorder, hammer_team: int) -> Stones16:
    """
    投球順形式の stones を team-block 形式に変換する。
    hammer_team は後攻チーム番号 (0 or 1)。
        例: hammer_team=0 でも、1 でも同じ結果になるように
             つまり、どちらのチームが後攻でも同じ team-block になる。
            shot_index=0 -> team0の0番
            shot_index=1 -> team0の1番
            shot_index=2 -> team0の2番
            shot_index=3 -> team0の3番
            ...
            shot_index=8 -> team1の0番
            shot_index=9 -> team1の1番
            shot_index=10 -> team1の2番
            shot_index=11 -> team1の3番
            ...
    """
    # stones_shotorder: [0..15 shot order] where even=nonhammer, odd=hammer
    # return: [0..7 team0][8..15 team1]
    h = hammer_team
    nh = 1 - h
    out: Stones16 = [None] * 16
    for shot in range(16):
        k = shot // 2
        team = nh if (shot % 2 == 0) else h
        tb = k if team == 0 else 8 + k
        out[tb] = stones_shotorder[shot]
        
    return out

def _to_move_team(shot_index: int, hammer_team: int) -> int:
    """
    今この shot_index を投げるチームを返す
    hammer_team は後攻チーム番号 (0 or 1)。
    例: hammer_team=1 (team1が後攻) のとき
        shot_index=0 -> team0の番 -> return 0
        shot_index=1 -> team1の番 -> return 1
        hammer_team=0 (team0が後攻) のとき
        shot_index=0 -> team1の番 -> return 1
        shot_index=1 -> team0の番 -> return 0
    """
    h = hammer_team
    nh = 1 - h
    if (shot_index % 2) == 0:
        return nh
    return h

