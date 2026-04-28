import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from common.translate_state import stones_listdict_to_xy16
from nn.network.dual_net import DualNet

from .debugger import (
    Debugger,
    format_topk_policy,
    format_topk_root_visits,
    policy_stats,
    summarize_stones,
)
from .node import (
    Node,
    argmax_over_actions,
    clear_node_table,
    count_reachable_nodes,
    get_child_node,
    get_node,
)
from .params import (
    DEFAULT_CPUCT,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_SIMULATIONS,
    DEFAULT_TIME_LIMIT_SEC,
)
from .policy import get_policy_and_value, set_policy_context
from .rollout import rollout_to_end_score, score_to_winvalue
from .simulate import decode_action, simulator_step
from .state import State, is_end_terminal

VALUE_CLASS_OFFSET = 8


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


def mcts_search(
    root_state: State,
    max_simulations: int = DEFAULT_MAX_SIMULATIONS,
    cpuct: float = DEFAULT_CPUCT,
    use_progressive_widening: bool = True,
    use_transposition_table: bool = True,
    max_depth: int = DEFAULT_MAX_DEPTH,
    debug: bool = False,
    debug_every: int = 10,
    debug_topk: int = 5,
    stats_log_path: Optional[str] = None,
) -> Tuple[float, float, int]:
    clear_node_table()

    time_limit_sec = DEFAULT_TIME_LIMIT_SEC

    dbg = Debugger(debug, every=debug_every)
    dbg.log(
        f"[PUCT] start end={root_state.end} shot_index={root_state.shot_index} "
        f"hammer={root_state.hammer_team} shot_team={root_state.to_move()} "
        f"score_diff={root_state.score_diff}"
    )
    dbg.log(
        f"[PUCT] options progressive_widening={use_progressive_widening} "
        f"transposition_table={use_transposition_table}"
    )
    dbg.log("[PUCT] " + summarize_stones(root_state.stones))

    if use_transposition_table:
        root = get_node(root_state, use_progressive_widening=use_progressive_widening)
    else:
        root = Node(root_state, use_progressive_widening=use_progressive_widening)

    dbg.tic("root_expand")
    root.expand_if_needed()
    dbg.toc("root_expand")

    if dbg.enabled and root.P is not None:
        dbg.log("[PUCT] " + policy_stats(root.P))
        dbg.log("[PUCT] root policy topk: " + format_topk_policy(root.P, debug_topk, decode_action))

    start_time = time.perf_counter()
    sims = 0

    while sims < max_simulations:
        if time_limit_sec is not None and (time.perf_counter() - start_time) >= time_limit_sec:
            break

        path: List[Tuple[Node, int]] = []
        node = root
        state = root_state

        dbg.tic("selection")
        select_depth = 0

        while node.is_expanded() and (not is_end_terminal(state)) and select_depth < max_depth:
            assert node.P is not None and node.Q is not None and node.Nsa is not None

            node.maybe_widen()

            a = argmax_over_actions(
                node.actions,
                key=lambda action: node.Q[action]
                + cpuct * node.P[action] * ((node.N ** 0.5) / (1 + node.Nsa[action])),
            )
            path.append((node, a))
            state = simulator_step(state, a)
            node = get_child_node(
                node,
                a,
                state,
                use_transposition_table=use_transposition_table,
            )
            select_depth += 1

        dbg.toc("selection")

        dbg.tic("expansion")
        if not is_end_terminal(state):
            pi, value_probs = get_policy_and_value(state)
            v_to_move = _value_probs_to_winvalue(state, value_probs)
            if not node.is_expanded():
                node.expand(pi)
        else:
            pi = None
            v_to_move = None
        dbg.toc("expansion")

        dbg.tic("rollout")
        if is_end_terminal(state):
            v = rollout_to_end_score(state)
        else:
            v = -float(v_to_move)
        dbg.toc("rollout")

        dbg.tic("backprop")
        for n, a in reversed(path):
            n.N += 1
            n.Nsa[a] += 1
            n.W[a] += v
            n.Q[a] = n.W[a] / n.Nsa[a]
            v = -v
        dbg.toc("backprop")

        if dbg.on(sims):
            dbg.log(
                f"[PUCT] sim={sims} depth={select_depth} leaf_shot={state.shot_index} "
                f"leaf_terminal={is_end_terminal(state)} v={v:.4g}"
            )
            if pi is not None:
                dbg.log("[PUCT] leaf " + policy_stats(pi))
                dbg.log("[PUCT] leaf policy topk: " + format_topk_policy(pi, debug_topk, decode_action))

        sims += 1

    assert root.Nsa is not None

    visited_children = 0
    expanded_children = 0

    for a in root.actions:
        if root.Nsa[a] <= 0:
            continue
        visited_children += 1

        child_node = root.children.get(a)
        if child_node is not None and child_node.is_expanded():
            expanded_children += 1

    best_action_id = argmax_over_actions(root.actions, key=lambda action: root.Nsa[action])
    best_action = decode_action(best_action_id)

    if dbg.enabled:
        elapsed_for_dbg = time.perf_counter() - start_time
        dbg.log(
            f"[PUCT] done sims={sims} elapsed={elapsed_for_dbg:.3f}s "
            f"({(sims / (elapsed_for_dbg + 1e-12)):.3f} sims/s)"
        )
        dbg.log("[PUCT] timing: " + dbg.summary())
        if root.P is not None and root.Q is not None and root.Nsa is not None:
            dbg.log("[PUCT] root Nsa topk: " + format_topk_root_visits(root, debug_topk, decode_action))
        dbg.log(f"[PUCT] best a={best_action_id} -> {best_action}")

    elapsed_time = time.perf_counter() - start_time
    nodes = count_reachable_nodes(root)
    lines = [
        "-----------------------------------------------------",
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]",
        f"shot={root_state.shot_index} end={root_state.end} hammer={root_state.hammer_team} score_diff={root_state.score_diff}",
        f"simulations={sims} elapsed={elapsed_time:.2f}sec nodes={nodes}",
        f"root_children visited={visited_children} expanded={expanded_children} candidates={len(root.actions)}",
        f"progressive_widening={use_progressive_widening} transposition_table={use_transposition_table}",
        "-----------------------------------------------------",
    ]
    _emit_lines(lines, stats_log_path)
    print("-----------------------------------------------------")
    print(f"PUCT search simulations: {sims}, time: {elapsed_time:.2f} sec, nodes: {nodes}")
    print(f"PUCT root children: visited={visited_children}, expanded={expanded_children} (candidates={len(root.actions)})")
    print(
        "PUCT options: "
        f"progressive_widening={use_progressive_widening}, "
        f"transposition_table={use_transposition_table}"
    )
    print("-----------------------------------------------------")

    return best_action


def set_root_state(
    network: DualNet,
    stones: List[Optional[Dict]],
    score_diff: int,
    end: int,
    shot_index: int,
    hammer_team: int,
    debug: bool = False,
) -> State:
    stones16 = stones_listdict_to_xy16(stones)

    if debug:
        print("------ DEBUG set_root_state -----")
        for p in stones16:
            print(f"root_state stone: x={p[0]} y={p[1]}" if p is not None else "root_state stone: None")

    set_policy_context(network, score_diff)

    return State.initial(
        stones=stones16,
        end=end,
        hammer_team=hammer_team,
        shot_index=shot_index,
        score_diff=score_diff,
    )
