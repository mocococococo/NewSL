from typing import List, Optional

from .rollout import _end_score_diff_team0_minus_team1
from .state import State, is_end_terminal

VALUE_CLASS_COUNT = 17
VALUE_CLASS_OFFSET = 8


def _terminal_score_class_from_root_view(root_state: State, terminal_state: State) -> int:
    raw_score = _end_score_diff_team0_minus_team1(terminal_state.stones)
    root_team = root_state.to_move()
    score = raw_score if root_team == 0 else -raw_score
    score = max(-VALUE_CLASS_OFFSET, min(VALUE_CLASS_OFFSET, score))
    return score + VALUE_CLASS_OFFSET


def _value_probs_to_root_histogram(
    root_state: State,
    leaf_state: State,
    value_probs: List[float],
) -> List[float]:
    """
    leaf_state.to_move() 視点のvalue分布を root_state.to_move() 視点へ直す。
    """
    if len(value_probs) != VALUE_CLASS_COUNT:
        raise ValueError(f"value_probs length must be {VALUE_CLASS_COUNT}, got {len(value_probs)}")

    hist = [float(p) for p in value_probs]
    if leaf_state.to_move() != root_state.to_move():
        hist = hist[::-1]
    return hist


def _add_value_histogram(value_list: List[float], hist: List[float]) -> None:
    if len(value_list) != VALUE_CLASS_COUNT:
        raise ValueError(f"value_list length must be {VALUE_CLASS_COUNT}, got {len(value_list)}")
    if len(hist) != VALUE_CLASS_COUNT:
        raise ValueError(f"hist length must be {VALUE_CLASS_COUNT}, got {len(hist)}")

    for i, v in enumerate(hist):
        value_list[i] += float(v)


def record_value_histogram(
    value_list: List[float],
    root_state: State,
    leaf_state: State,
    value_probs: Optional[List[float]],
) -> None:
    if is_end_terminal(leaf_state):
        hist = [0.0 for _ in range(VALUE_CLASS_COUNT)]
        score_class = _terminal_score_class_from_root_view(root_state, leaf_state)
        hist[score_class] = 1.0
    else:
        if value_probs is None:
            raise ValueError("value_probs must not be None for non-terminal leaf")
        hist = _value_probs_to_root_histogram(root_state, leaf_state, value_probs)

    _add_value_histogram(value_list, hist)
