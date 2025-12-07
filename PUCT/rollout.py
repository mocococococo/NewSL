# rollout.py
import math

from .state import State, is_end_terminal
from .simulate import simulator_step
from .policy import get_policy
from board.constant import Y_TEE, R_HOUSE, STONE_RADIUS
from .params import PUCT_DEBUG_SCORE, PUCT_DEBUG_SCORE_LIMIT


_SCORE_DEBUG_ENV = PUCT_DEBUG_SCORE
_SCORE_DEBUG = (_SCORE_DEBUG_ENV == "1")
_SCORE_DEBUG_LIMIT = PUCT_DEBUG_SCORE_LIMIT
_SCORE_DEBUG_COUNT = 0


def _dbg(msg: str) -> None:
    global _SCORE_DEBUG_COUNT
    if not _SCORE_DEBUG:
        return
    if _SCORE_DEBUG_COUNT >= _SCORE_DEBUG_LIMIT:
        return
    _SCORE_DEBUG_COUNT += 1
    print(msg)


def _end_score_diff_team0_minus_team1(stones) -> int:
    """終端石配置から、このエンドの得点差(= team0 - team1)を返す"""
    in_house = []
    all_d = []  # (d, i, team, x, y, d2)

    thr2 = (R_HOUSE + STONE_RADIUS) ** 2

    for i, p in enumerate(stones):
        if p is None:
            continue
        x, y = float(p[0]), float(p[1])
        d2 = x * x + (y - Y_TEE) ** 2
        team = 0 if i < 8 else 1
        d = math.sqrt(d2)

        all_d.append((d, i, team, x, y, d2))

        if d2 <= thr2:
            in_house.append((d, team))

    if not in_house:
        if all_d:
            all_d.sort(key=lambda t: t[0])
            dmin, imin, tmin, xmin, ymin, d2min = all_d[0]
            _dbg(
                "[SCORE] in_house=0  thr2={:.6f}  Y_TEE={:.6f} R_HOUSE={:.6f} STONE_RADIUS={:.6f}   "
                "min_d={:.6f} (i={} team={} x={:.6f} y={:.6f} d2={:.6f})".format(
                    thr2, Y_TEE, R_HOUSE, STONE_RADIUS,
                    dmin, imin, tmin, xmin, ymin, d2min
                )
            )
            top = all_d[:5]
            _dbg(
                "[SCORE] nearest5: " + " | ".join(
                    "d={:.4f} i={} t={} x={:.3f} y={:.3f}".format(d, i, t, x, y)
                    for (d, i, t, x, y, _) in top
                )
            )
        else:
            _dbg(
                "[SCORE] in_house=0 and no stones exist  thr2={:.6f}  Y_TEE={:.6f} R_HOUSE={:.6f} STONE_RADIUS={:.6f}".format(
                    thr2, Y_TEE, R_HOUSE, STONE_RADIUS
                )
            )
        return 0

    in_house.sort(key=lambda t: t[0])
    scoring_team = in_house[0][1]
    score = 0
    for _, team in in_house:
        if team == scoring_team:
            score += 1
        else:
            break

    out = score if scoring_team == 0 else -score
    _dbg("[SCORE] in_house={} scoring_team={} raw_score={}".format(len(in_house), scoring_team, out))
    return out


def _to_move_defined_even_if_terminal(state: State) -> int:
    """終端でも「次に手番になるはずだったチーム」を定義して返す"""
    if not state.is_end_terminal():
        return state.to_move()
    # shot_index==16 の想定。最後に投げたのは (shot_index-1)。
    last_shot = state.shot_index - 1
    h = state.hammer_team
    nh = 1 - h
    last_mover = nh if (last_shot % 2 == 0) else h
    return 1 - last_mover


def rollout_to_end_score(state: State, debug: bool = True) -> float:
    """終端までプレイアウトし、返り値は「stateの手番視点」のスカラー"""
    global _SCORE_DEBUG
    if debug:
        _SCORE_DEBUG = True

    leaf_view_team = _to_move_defined_even_if_terminal(state)

    s = state
    depth = 0
    while not is_end_terminal(s):
        policy = get_policy(s)  # len=2048
        a = max(range(len(policy)), key=lambda i: policy[i])  # まずは貪欲でOK
        s = simulator_step(s, a)
        depth += 1

    raw = _end_score_diff_team0_minus_team1(s.stones)
    diff01 = raw / 8.0
    v = float(diff01 if leaf_view_team == 0 else -diff01)

    if _SCORE_DEBUG:
        _dbg("[ROLLOUT] depth={} terminal_shot_index={} leaf_view_team={} raw={} v={}".format(
            depth, getattr(s, "shot_index", None), leaf_view_team, raw, v
        ))
        
    return v
