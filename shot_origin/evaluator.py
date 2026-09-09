from __future__ import annotations

from typing import List

from mcts.rollout import _end_score_diff_team0_minus_team1, score_to_winvalue
from mcts.state import State

VALUE_CLASS_OFFSET = 8


def value_probs_to_winvalue(state: State, value_probs: List[float]) -> float:
    if len(value_probs) != 17:
        raise ValueError(f"value_probs length must be 17, got {len(value_probs)}")

    to_move_team = state.to_move()
    if to_move_team not in (0, 1):
        raise ValueError(f"state.to_move() must be 0/1 for non-terminal state, got {to_move_team}")

    score_diff_leaf = state.score_diff if to_move_team == 0 else -state.score_diff
    had_hammer_this_end = to_move_team == state.hammer_team

    expected_v = 0.0
    for cls, prob in enumerate(value_probs):
        score_end = cls - VALUE_CLASS_OFFSET
        win_v = score_to_winvalue(state.end, score_end, score_diff_leaf, had_hammer_this_end)
        expected_v += float(prob) * float(win_v)
    return expected_v


def terminal_score_class_from_root_view(root_state: State, terminal_state: State) -> int:
    raw_score = _end_score_diff_team0_minus_team1(terminal_state.stones)
    root_team = root_state.to_move()
    score = raw_score if root_team == 0 else -raw_score
    score = max(-VALUE_CLASS_OFFSET, min(VALUE_CLASS_OFFSET, score))
    return score + VALUE_CLASS_OFFSET