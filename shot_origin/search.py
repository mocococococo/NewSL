import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from common.translate_state import stones_listdict_to_xy16
from nn.network.dual_net import DualNet
from transformer.network import TransformerNetwork
from transformer.params import TRANSFORMER_VY_MODE, TransformerVyMode, get_transformer_action_dim

from mcts.hybrid_policy import get_policy_and_value, reset_policy_selection_log, set_policy_context
from mcts.rollout import _end_score_diff_team0_minus_team1, score_to_winvalue
from mcts.simulate import decode_action, simulator_step
from mcts.state import State, is_end_terminal

from .create_mode import (
    RootCandidateStat,
    ScoreHistograms,
    build_root_candidate_stats,
    record_root_score_histogram,
)
from .debugger import Debugger, format_topk_root_shot, summarize_stones
from .evaluator import value_probs_to_winvalue
from .node import Node, argmax_over_actions, tree_size
from .params import (
    DEFAULT_SHOT_ORIGIN_MAX_DEPTH,
    DEFAULT_SHOT_ORIGIN_MAX_SIMULATIONS,
    DEFAULT_SHOT_ORIGIN_TIE_BREAK_SEED,
)

SearchAction = Tuple[float, float, int]
SearchDataResult = Tuple[int, List[RootCandidateStat]]


def _emit_lines(lines: List[str], log_path: Optional[str]) -> None:
    if not log_path:
        return
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def _to_move_defined_even_if_terminal(state: State) -> int:
    if state.is_end_terminal():
        last_shot = state.shot_index - 1
        hammer = state.hammer_team
        non_hammer = 1 - hammer
        return non_hammer if (last_shot % 2 == 0) else hammer
    return 1 - state.to_move()


def _policy_free_rollout_to_end_score(
    state: State,
    action_type: TransformerVyMode,
) -> float:
    leaf_view_team = _to_move_defined_even_if_terminal(state)
    s = state
    action_dim = get_transformer_action_dim(action_type)
    while not is_end_terminal(s):
        a = random.randrange(action_dim)
        s = simulator_step(s, a, action_type=action_type)

    raw = _end_score_diff_team0_minus_team1(s.stones)
    score_end = raw if leaf_view_team == 0 else -raw
    score_diff_leaf = s.score_diff if leaf_view_team == 0 else -s.score_diff
    had_hammer_this_end = leaf_view_team == s.hammer_team
    return score_to_winvalue(s.end, score_end, score_diff_leaf, had_hammer_this_end)


def shot_origin_search(
    root_state: State,
    max_simulations: int = DEFAULT_SHOT_ORIGIN_MAX_SIMULATIONS,
    max_depth: int = DEFAULT_SHOT_ORIGIN_MAX_DEPTH,
    debug: bool = False,
    debug_every: int = 10,
    debug_topk: int = 5,
    stats_log_path: Optional[str] = None,
    is_create_data: bool = False,
    use_value: bool = True,
    action_type: TransformerVyMode = TRANSFORMER_VY_MODE,
    tie_break_seed: int = DEFAULT_SHOT_ORIGIN_TIE_BREAK_SEED,
) -> Union[SearchAction, SearchDataResult]:
    """Policy-free SHOT using round extra visits 1, 2, 3, ... ."""
    reset_policy_selection_log()

    dbg = Debugger(debug, every=debug_every)
    decode_search_action = lambda a: decode_action(a, action_type=action_type)
    dbg.log(
        f"[SHOT_ORIGIN] start end={root_state.end} shot_index={root_state.shot_index} "
        f"hammer={root_state.hammer_team} shot_team={root_state.to_move()}, "
        f"score_diff={root_state.score_diff}"
    )
    dbg.log("[SHOT_ORIGIN] " + summarize_stones(root_state.stones))

    root = Node(root_state, action_type=action_type, tie_break_seed=tie_break_seed)
    root.expand_if_needed()
    dbg.log("[SHOT_ORIGIN] root " + root.round_info())

    start_time = time.perf_counter()
    sims = 0
    root_score_histograms: ScoreHistograms = {}

    while sims < max_simulations:
        root.halve_actions_if_needed()
        if len(root.actions) <= 1:
            break

        path: List[Tuple[Node, int]] = []
        node: Node = root
        state: State = root_state
        select_depth = 0

        while node.is_expanded() and (not is_end_terminal(state)) and select_depth < max_depth:
            assert node.Q is not None and node.Nsa is not None

            node.halve_actions_if_needed()
            a = node.select_action()
            path.append((node, a))
            state = simulator_step(state, a, action_type=action_type)
            node = node.child_for(a, state)
            select_depth += 1

        value_probs = None
        if not is_end_terminal(state) and use_value:
            _, value_probs = get_policy_and_value(state, action_type=action_type)
            v_to_move = value_probs_to_winvalue(state, value_probs)
            v = -float(v_to_move)
        else:
            v = _policy_free_rollout_to_end_score(state, action_type=action_type)

        if not is_end_terminal(state) and select_depth < max_depth and not node.is_expanded():
            node.expand_if_needed()

        if is_create_data and path:
            record_root_score_histogram(
                root_score_histograms,
                path[0][1],
                root_state,
                state,
                value_probs,
            )

        for n, a in reversed(path):
            assert n.Nsa is not None and n.W is not None and n.Q is not None
            n.N += 1
            n.Nsa[a] += 1
            n.W[a] += v
            n.Q[a] = n.W[a] / n.Nsa[a]
            v = -v

        if dbg.on(sims):
            dbg.log(
                f"[SHOT_ORIGIN] sim={sims} depth={select_depth} "
                f"leaf_shot={state.shot_index} leaf_terminal={is_end_terminal(state)}"
            )
            dbg.log("[SHOT_ORIGIN] root " + root.round_info())

        sims += 1

    root.halve_actions_if_needed()
    assert root.Nsa is not None and root.Q is not None and root.P is not None

    visited_children = sum(1 for n in root.Nsa if n > 0)
    expanded_children = 0
    for action_children in root.children.values():
        if any(child.is_expanded() for child in action_children):
            expanded_children += 1

    best_actions = [a for a in root.actions if root.Nsa[a] > 0]
    if not best_actions:
        best_actions = [a for a, n in enumerate(root.Nsa) if n > 0]
    if not best_actions:
        best_actions = root.actions

    best_action_id = argmax_over_actions(
        best_actions,
        key=lambda a: (root.Q[a], root.Nsa[a], root.tie_break_value(a)),
    )
    best_action = decode_action(best_action_id, action_type=action_type)

    if dbg.enabled:
        elapsed = time.perf_counter() - start_time
        dbg.log(f"[SHOT_ORIGIN] done sims={sims} elapsed={elapsed:.3f}s")
        dbg.log("[SHOT_ORIGIN] root topk: " + format_topk_root_shot(root, debug_topk, decode_search_action))
        dbg.log(f"[SHOT_ORIGIN] best a={best_action_id} -> {best_action}")

    elapsed_time = time.perf_counter() - start_time
    lines = [
        "-----------------------------------------------------",
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]",
        f"shot={root_state.shot_index} end={root_state.end} hammer={root_state.hammer_team} score_diff={root_state.score_diff}",
        f"simulations={sims} elapsed={elapsed_time:.2f}sec nodes={tree_size(root)}",
        f"root_children visited={visited_children} expanded={expanded_children} active={len(root.actions)} candidates={root.n_actions}",
        "-----------------------------------------------------",
    ]
    if is_create_data:
        _emit_lines(lines, stats_log_path)

    print("-----------------------------------------------------")
    print(f"SHOT_ORIGIN search simulations: {sims}, time: {elapsed_time:.2f} sec, nodes: {tree_size(root)}")
    print(
        "SHOT_ORIGIN root children: "
        f"visited={visited_children}, expanded={expanded_children} "
        f"active={len(root.actions)} (candidates={root.n_actions})"
    )
    print("-----------------------------------------------------")

    if is_create_data:
        return best_action_id, build_root_candidate_stats(root, root_score_histograms)

    return best_action


def set_root_state(
    network: Optional[Union[DualNet, TransformerNetwork]],
    stones: List[Optional[Dict]],
    score_diff: int,
    end: int,
    shot_index: int,
    hammer_team: int,
    transformer_network: Optional[Union[TransformerNetwork, Dict[int, TransformerNetwork]]] = None,
    sl_model_is_cnn: bool = True,
    debug: bool = False,
    use_transformer: bool = False,
    transformer_target_end: Tuple[int, ...] = (9, 10),
    transformer_target_shot: Tuple[int, ...] = (15,),
) -> State:
    stones16 = stones_listdict_to_xy16(stones)

    if debug:
        print("------ DEBUG set_root_state -----")
        for p in stones16:
            print(f"root_state stone: x={p[0]} y={p[1]}" if p is not None else "root_state stone: None")

    set_policy_context(
        sl_model=network,
        score_diff=score_diff,
        sl_model_is_cnn=sl_model_is_cnn,
        search_based_model=transformer_network,
        use_search_based_model=use_transformer,
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