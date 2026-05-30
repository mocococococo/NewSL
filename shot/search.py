from __future__ import annotations

import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from common.translate_state import stones_listdict_to_xy16
from nn.network.dual_net import DualNet
from transformer.network import TransformerNetwork

from board.constant import VX_SIZE, VY_SIZE
from mcts.hybrid_policy import (
    get_policy_and_value,
    reset_policy_selection_log,
    set_policy_context,
)
from mcts.rollout import (
    _end_score_diff_team0_minus_team1,
    rollout_to_end_score,
    score_to_winvalue,
)
from mcts.simulate import decode_action, simulator_step
from mcts.state import State, is_end_terminal

from .node import ShotActionStats, make_action_stats
from .params import (
    DEFAULT_SHOT_INITIAL_CANDIDATES,
    DEFAULT_SHOT_MAX_DEPTH,
    DEFAULT_SHOT_MAX_SIMULATIONS,
    DEFAULT_SHOT_TIME_LIMIT_SEC,
    SHOT_KEEP_RATIO,
    SHOT_MIN_VISITS_PER_ACTION,
)

VALUE_CLASS_OFFSET = 8
N_ACTIONS = VX_SIZE * VY_SIZE * 2

SearchAction = Tuple[float, float, int]
SearchDataResult = Tuple[SearchAction, List[int], List[float]]


def _emit_lines(lines: List[str], log_path: Optional[str]) -> None:
    if not log_path:
        return
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def _value_probs_to_winvalue(state: State, value_probs: List[float]) -> float:
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


def _terminal_score_class_from_actor_view(actor_team: int, terminal_state: State) -> int:
    raw_score = _end_score_diff_team0_minus_team1(terminal_state.stones)
    score = raw_score if actor_team == 0 else -raw_score
    score = max(-VALUE_CLASS_OFFSET, min(VALUE_CLASS_OFFSET, score))
    return score + VALUE_CLASS_OFFSET


def _one_hot_score_distribution(score_class: int) -> List[float]:
    dist = [0.0] * 17
    dist[score_class] = 1.0
    return dist


def _value_distribution_from_actor_view(
    state: State,
    value_probs: List[float],
    actor_team: int,
) -> List[float]:
    if len(value_probs) != 17:
        raise ValueError(f"value_probs length must be 17, got {len(value_probs)}")
    if state.to_move() == actor_team:
        return [float(p) for p in value_probs]
    return [float(p) for p in reversed(value_probs)]


def _evaluate_action(
    state: State,
    action: int,
    max_depth: int,
) -> Tuple[float, List[float], int]:
    actor_team = state.to_move()
    if actor_team not in (0, 1):
        raise ValueError(f"state.to_move() must be 0/1, got {actor_team}")

    child_state = simulator_step(state, action)
    if is_end_terminal(child_state):
        score_class = _terminal_score_class_from_actor_view(actor_team, child_state)
        return rollout_to_end_score(child_state), _one_hot_score_distribution(score_class), 1

    if max_depth > 1:
        child_policy, _ = get_policy_and_value(child_state)
        reply_action = max(range(len(child_policy)), key=lambda a: child_policy[a])
        child_value, child_value_dist, child_sims = _evaluate_action(
            child_state,
            reply_action,
            max_depth=max_depth - 1,
        )
        return -child_value, [float(p) for p in reversed(child_value_dist)], child_sims + 1

    _, value_probs = get_policy_and_value(child_state)
    leaf_value = _value_probs_to_winvalue(child_state, value_probs)
    value = leaf_value if child_state.to_move() == actor_team else -leaf_value
    value_dist = _value_distribution_from_actor_view(child_state, value_probs, actor_team)
    return value, value_dist, 1


def _top_k_actions(policy: List[float], k: int) -> List[int]:
    if len(policy) != N_ACTIONS:
        raise RuntimeError(f"policy length mismatch: {len(policy)} != {N_ACTIONS}")
    k = max(1, min(int(k), len(policy)))
    return sorted(range(len(policy)), key=lambda a: policy[a], reverse=True)[:k]


def _ceil_log2(n: int) -> int:
    if n <= 1:
        return 1
    return int(math.ceil(math.log2(n)))


def _rank_actions(
    actions: List[int],
    stats: List[ShotActionStats],
    policy: List[float],
) -> List[int]:
    return sorted(
        actions,
        key=lambda a: (stats[a].mean_value, stats[a].visits, policy[a]),
        reverse=True,
    )


def _format_top_actions(
    actions: List[int],
    stats: List[ShotActionStats],
    policy: List[float],
    topk: int,
) -> str:
    parts = []
    for action in _rank_actions(actions, stats, policy)[:topk]:
        vx, vy, spin = decode_action(action)
        parts.append(
            f"{action}:N={stats[action].visits},Q={stats[action].mean_value:.4f},"
            f"P={policy[action]:.4g},shot=({vx:.4f},{vy:.4f},{spin})"
        )
    return " | ".join(parts)


def shot_search(
    root_state: State,
    initial_candidates: int = DEFAULT_SHOT_INITIAL_CANDIDATES,
    max_simulations: int = DEFAULT_SHOT_MAX_SIMULATIONS,
    time_limit_sec: Optional[float] = DEFAULT_SHOT_TIME_LIMIT_SEC,
    max_depth: int = DEFAULT_SHOT_MAX_DEPTH,
    keep_ratio: float = SHOT_KEEP_RATIO,
    min_visits_per_action: int = SHOT_MIN_VISITS_PER_ACTION,
    debug: bool = False,
    debug_topk: int = 5,
    stats_log_path: Optional[str] = None,
    is_create_data: bool = False,
) -> Union[SearchAction, SearchDataResult]:
    reset_policy_selection_log()

    if max_simulations <= 0:
        raise ValueError(f"max_simulations must be positive, got {max_simulations}")
    if initial_candidates <= 0:
        raise ValueError(f"initial_candidates must be positive, got {initial_candidates}")
    if not (0.0 < keep_ratio < 1.0):
        raise ValueError(f"keep_ratio must be in (0, 1), got {keep_ratio}")
    if max_depth <= 0:
        raise ValueError(f"max_depth must be positive, got {max_depth}")

    policy, _ = get_policy_and_value(root_state)
    active_actions = _top_k_actions(policy, initial_candidates)
    stats = make_action_stats(N_ACTIONS)
    value_list = [0.0] * 17

    start_time = time.perf_counter()
    simulations = 0
    round_index = 0

    if debug:
        print(
            "[SHOT] start "
            f"end={root_state.end} shot_index={root_state.shot_index} "
            f"hammer={root_state.hammer_team} shot_team={root_state.to_move()} "
            f"score_diff={root_state.score_diff} candidates={len(active_actions)}"
        )

    while len(active_actions) > 1 and simulations < max_simulations:
        if time_limit_sec is not None and (time.perf_counter() - start_time) >= time_limit_sec:
            break

        remaining = max_simulations - simulations
        rounds_left = _ceil_log2(len(active_actions))
        estimated_sims_per_eval = max(1, max_depth)
        visits_per_action = remaining // max(
            1,
            len(active_actions) * rounds_left * estimated_sims_per_eval,
        )
        visits_per_action = max(int(min_visits_per_action), visits_per_action)

        evaluated_this_round = 0
        for action in list(active_actions):
            for _ in range(visits_per_action):
                if simulations >= max_simulations:
                    break
                if time_limit_sec is not None and (time.perf_counter() - start_time) >= time_limit_sec:
                    break

                depth_budget = min(max_depth, max_simulations - simulations)
                value, value_dist, sims_used = _evaluate_action(
                    root_state,
                    action,
                    max_depth=depth_budget,
                )
                stats[action].update(value, value_dist)
                simulations += sims_used
                evaluated_this_round += 1

                if is_create_data:
                    for cls, prob in enumerate(value_dist):
                        value_list[cls] += prob

            if simulations >= max_simulations:
                break
            if time_limit_sec is not None and (time.perf_counter() - start_time) >= time_limit_sec:
                break

        if evaluated_this_round <= 0:
            break

        ranked = _rank_actions(active_actions, stats, policy)
        keep_count = max(1, int(math.ceil(len(ranked) * keep_ratio)))
        if keep_count >= len(ranked):
            keep_count = len(ranked) - 1
        active_actions = ranked[:keep_count]

        if debug:
            print(
                f"[SHOT] round={round_index} sims={simulations} "
                f"active={len(active_actions)} top: "
                f"{_format_top_actions(active_actions, stats, policy, debug_topk)}"
            )
        round_index += 1

    visited_actions = [a for a in range(N_ACTIONS) if stats[a].visits > 0]
    if active_actions:
        best_action_id = _rank_actions(active_actions, stats, policy)[0]
    elif visited_actions:
        best_action_id = _rank_actions(visited_actions, stats, policy)[0]
    else:
        best_action_id = _top_k_actions(policy, 1)[0]

    best_action = decode_action(best_action_id)
    elapsed_time = time.perf_counter() - start_time

    lines = [
        "-----------------------------------------------------",
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]",
        f"shot={root_state.shot_index} end={root_state.end} "
        f"hammer={root_state.hammer_team} score_diff={root_state.score_diff}",
        f"simulations={simulations} elapsed={elapsed_time:.2f}sec",
        f"initial_candidates={min(initial_candidates, N_ACTIONS)} visited={len(visited_actions)} "
        f"remaining={len(active_actions)} best={best_action_id}",
        "-----------------------------------------------------",
    ]
    if is_create_data:
        _emit_lines(lines, stats_log_path)

    print("-----------------------------------------------------")
    print(f"SHOT search simulations: {simulations}, time: {elapsed_time:.2f} sec")
    print(
        "SHOT candidates: "
        f"initial={min(initial_candidates, N_ACTIONS)}, "
        f"visited={len(visited_actions)}, remaining={len(active_actions)}"
    )
    print("-----------------------------------------------------")

    if is_create_data:
        visits = [s.visits for s in stats]
        return best_action, visits, value_list

    return best_action


def set_root_state(
    network: DualNet,
    stones: List[Optional[Dict]],
    score_diff: int,
    end: int,
    shot_index: int,
    hammer_team: int,
    transformer_network: Optional[TransformerNetwork] = None,
    debug: bool = False,
    use_transformer: bool = False,
    transformer_target_end: Tuple[int, ...] = (9, 10),
    transformer_target_shot: Tuple[int, ...] = (15,),
) -> State:
    stones16 = stones_listdict_to_xy16(stones)

    if debug:
        print("------ DEBUG set_root_state -----")
        for p in stones16:
            if p is None:
                print("root_state stone: None")
            else:
                print(f"root_state stone: x={p[0]} y={p[1]}")

    set_policy_context(
        network,
        score_diff,
        transformer_net=transformer_network,
        use_transformer=use_transformer,
        transformer_target_end=transformer_target_end,
        transformer_target_shot=transformer_target_shot,
    )

    return State.initial(
        stones=stones16,
        end=end,
        hammer_team=hammer_team,
        shot_index=shot_index,
        score_diff=score_diff,
    )
