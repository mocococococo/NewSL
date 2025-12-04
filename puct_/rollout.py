# rollout.py
import math

from board.constant import Y_TEE, R_HOUSE, STONE_RADIUS
from .state import is_end_terminal
from .simulate import simulator_step
from .policy import get_policy

def _end_score_diff_team0_minus_team1(stones) -> int:
    """終端石配置から、このエンドの得点差(= team0 - team1)を返す"""
    in_house = []
    for i, p in enumerate(stones):
        if p is None:
            continue
        x, y = float(p[0]), abs(float(p[1]))
        d_2 = x * x + (y - Y_TEE) ** 2
        if d_2 <= (R_HOUSE + STONE_RADIUS) ** 2:
            team = 0 if i < 8 else 1
            in_house.append((math.sqrt(d_2), team))

    if not in_house:
        return 0

    in_house.sort(key=lambda t: t[0])
    scoring_team = in_house[0][1]
    score = 0
    for _, team in in_house:
        if team == scoring_team:
            score += 1
        else:
            break

    return score if scoring_team == 0 else -score

def _to_move_defined_even_if_terminal(state) -> int:
    """終端でも「次に手番になるはずだったチーム」を定義して返す"""
    if not state.is_end_terminal():
        return state.to_move()
    # shot_index==16 の想定。最後に投げたのは (shot_index-1)。
    last_shot = state.shot_index - 1
    h = state.hammer_team()
    nh = 1 - h
    last_mover = nh if (last_shot % 2 == 0) else h
    return 1 - last_mover

def rollout_to_end_score(state) -> float:
    """終端までプレイアウトし、返り値は「stateの手番視点」のスカラー"""
    leaf_view_team = _to_move_defined_even_if_terminal(state)

    s = state
    while not is_end_terminal(s):
        policy = get_policy(s)          # len=2048
        a = max(range(len(policy)), key=lambda i: policy[i])  # まずは貪欲でOK
        s = simulator_step(s, a)

    # スコアを計算して手番視点に変換して 8点 で割って正規化
    diff01 = _end_score_diff_team0_minus_team1(s.stones) / 8.0
    return float(diff01 if leaf_view_team == 0 else -diff01)
