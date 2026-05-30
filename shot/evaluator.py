from __future__ import annotations

from typing import List, Tuple

from mcts.hybrid_policy import get_policy_and_value
from mcts.rollout import (
    _end_score_diff_team0_minus_team1,
    rollout_to_end_score,
    score_to_winvalue,
)
from mcts.simulate import simulator_step
from mcts.state import State, is_end_terminal

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


def terminal_score_class_from_actor_view(actor_team: int, terminal_state: State) -> int:
    raw_score = _end_score_diff_team0_minus_team1(terminal_state.stones)
    score = raw_score if actor_team == 0 else -raw_score
    score = max(-VALUE_CLASS_OFFSET, min(VALUE_CLASS_OFFSET, score))
    return score + VALUE_CLASS_OFFSET


def one_hot_score_distribution(score_class: int) -> List[float]:
    dist = [0.0] * 17
    dist[score_class] = 1.0
    return dist


def value_distribution_from_actor_view(
    state: State,
    value_probs: List[float],
    actor_team: int,
) -> List[float]:
    if len(value_probs) != 17:
        raise ValueError(f"value_probs length must be 17, got {len(value_probs)}")
    if state.to_move() == actor_team:
        return [float(p) for p in value_probs]
    return [float(p) for p in reversed(value_probs)]


def evaluate_action(
    state: State,
    action: int,
    max_depth: int,
) -> Tuple[float, List[float], int]:
    actor_team = state.to_move()
    if actor_team not in (0, 1):
        raise ValueError(f"state.to_move() must be 0/1, got {actor_team}")

    child_state = simulator_step(state, action)
    if is_end_terminal(child_state):
        score_class = terminal_score_class_from_actor_view(actor_team, child_state)
        return rollout_to_end_score(child_state), one_hot_score_distribution(score_class), 1

    if max_depth > 1:
        child_policy, _ = get_policy_and_value(child_state)
        reply_action = max(range(len(child_policy)), key=lambda a: child_policy[a])
        child_value, child_value_dist, child_sims = evaluate_action(
            child_state,
            reply_action,
            max_depth=max_depth - 1,
        )
        return -child_value, [float(p) for p in reversed(child_value_dist)], child_sims + 1

    _, value_probs = get_policy_and_value(child_state)
    leaf_value = value_probs_to_winvalue(child_state, value_probs)
    value = leaf_value if child_state.to_move() == actor_team else -leaf_value
    value_dist = value_distribution_from_actor_view(child_state, value_probs, actor_team)
    return value, value_dist, 1

