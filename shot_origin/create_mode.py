from typing import Any, Dict, List, Optional

from mcts.state import State, is_end_terminal

from .evaluator import terminal_score_class_from_root_view
from .node import Node

RootCandidateStat = Dict[str, Any]
ScoreHistograms = Dict[int, List[float]]

VALUE_CLASS_COUNT = 17
VALUE_CLASS_OFFSET = 8


def _one_hot_score_histogram(score_class: int) -> List[float]:
    hist = [0.0 for _ in range(VALUE_CLASS_COUNT)]
    hist[score_class] = 1.0
    return hist


def _value_probs_to_root_histogram(
    root_state: State,
    leaf_state: State,
    value_probs: List[float],
) -> List[float]:
    if len(value_probs) != VALUE_CLASS_COUNT:
        raise ValueError(f"value_probs length must be {VALUE_CLASS_COUNT}, got {len(value_probs)}")

    hist = [float(p) for p in value_probs]
    if leaf_state.to_move() != root_state.to_move():
        hist = hist[::-1]
    return hist


def _add_score_histogram(
    score_histograms: ScoreHistograms,
    action_id: int,
    hist: List[float],
) -> None:
    if action_id not in score_histograms:
        score_histograms[action_id] = [0.0 for _ in range(VALUE_CLASS_COUNT)]
    dst = score_histograms[action_id]
    for i, v in enumerate(hist):
        dst[i] += float(v)


def _expected_score_from_histogram(hist: List[float]) -> float:
    total = sum(hist)
    if total <= 0.0:
        return 0.0
    return sum((i - VALUE_CLASS_OFFSET) * float(v) for i, v in enumerate(hist)) / total


def record_root_score_histogram(
    score_histograms: ScoreHistograms,
    root_action_id: int,
    root_state: State,
    leaf_state: State,
    value_probs: Optional[List[float]],
) -> None:
    if is_end_terminal(leaf_state):
        score_class = terminal_score_class_from_root_view(root_state, leaf_state)
        hist = _one_hot_score_histogram(score_class)
    else:
        if value_probs is None:
            return
        hist = _value_probs_to_root_histogram(root_state, leaf_state, value_probs)

    _add_score_histogram(score_histograms, root_action_id, hist)


def build_root_candidate_stats(
    root: Node,
    score_histograms: ScoreHistograms,
) -> List[RootCandidateStat]:
    assert root.P is not None and root.Q is not None and root.Nsa is not None

    survivor_actions = set(root.actions)
    stats: List[RootCandidateStat] = []

    for action_id, visit_count in enumerate(root.Nsa):
        if visit_count <= 0:
            continue

        score_histogram = score_histograms.get(
            action_id,
            [0.0 for _ in range(VALUE_CLASS_COUNT)],
        )
        stats.append(
            {
                "action_id": int(action_id),
                "visit_count": int(visit_count),
                "score_histogram": [float(v) for v in score_histogram],
                "expected_score": float(_expected_score_from_histogram(score_histogram)),
                "q": float(root.Q[action_id]),
                "prior": float(root.P[action_id]),
                "is_survivor": action_id in survivor_actions,
            }
        )

    return stats