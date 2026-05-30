from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

from common.translate_state import stones_listdict_to_xy16
from nn.network.dual_net import DualNet
from transformer.network import TransformerNetwork

from mcts.hybrid_policy import (
    get_policy_and_value,
    reset_policy_selection_log,
    set_policy_context,
)
from mcts.simulate import decode_action
from mcts.state import State

from .candidates import (
    N_ACTIONS,
    ceil_log2,
    keep_count_after_halving,
    rank_actions,
    top_k_actions,
    visited_actions,
)
from .debugger import emit_lines, format_topk_action_stats
from .evaluator import evaluate_action
from .node import make_action_stats
from .params import (
    DEFAULT_SHOT_INITIAL_CANDIDATES,
    DEFAULT_SHOT_MAX_DEPTH,
    DEFAULT_SHOT_MAX_SIMULATIONS,
    DEFAULT_SHOT_TIME_LIMIT_SEC,
    SHOT_KEEP_RATIO,
    SHOT_MIN_VISITS_PER_ACTION,
)

SearchAction = Tuple[float, float, int]
SearchDataResult = Tuple[SearchAction, List[int], List[float]]


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
    active_actions = top_k_actions(policy, initial_candidates)
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
        rounds_left = ceil_log2(len(active_actions))
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
                value, value_dist, sims_used = evaluate_action(
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

        ranked = rank_actions(active_actions, stats, policy)
        keep_count = keep_count_after_halving(len(ranked), keep_ratio)
        active_actions = ranked[:keep_count]

        if debug:
            print(
                f"[SHOT] round={round_index} sims={simulations} "
                f"active={len(active_actions)} top: "
                f"{format_topk_action_stats(active_actions, stats, policy, debug_topk)}"
            )
        round_index += 1

    evaluated_actions = visited_actions(stats)
    if active_actions:
        best_action_id = rank_actions(active_actions, stats, policy)[0]
    elif evaluated_actions:
        best_action_id = rank_actions(evaluated_actions, stats, policy)[0]
    else:
        best_action_id = top_k_actions(policy, 1)[0]

    best_action = decode_action(best_action_id)
    elapsed_time = time.perf_counter() - start_time

    lines = [
        "-----------------------------------------------------",
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]",
        f"shot={root_state.shot_index} end={root_state.end} "
        f"hammer={root_state.hammer_team} score_diff={root_state.score_diff}",
        f"simulations={simulations} elapsed={elapsed_time:.2f}sec",
        f"initial_candidates={min(initial_candidates, N_ACTIONS)} visited={len(evaluated_actions)} "
        f"remaining={len(active_actions)} best={best_action_id}",
        "-----------------------------------------------------",
    ]
    if is_create_data:
        emit_lines(lines, stats_log_path)

    print("-----------------------------------------------------")
    print(f"SHOT search simulations: {simulations}, time: {elapsed_time:.2f} sec")
    print(
        "SHOT candidates: "
        f"initial={min(initial_candidates, N_ACTIONS)}, "
        f"visited={len(evaluated_actions)}, remaining={len(active_actions)}"
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
