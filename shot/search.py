import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from common.translate_state import stones_listdict_to_xy16
from nn.network.dual_net import DualNet
from transformer.network import TransformerNetwork

from mcts.hybrid_policy import get_policy_and_value, reset_policy_selection_log, set_policy_context
from mcts.rollout import rollout_to_end_score
from mcts.simulate import decode_action, simulator_step
from mcts.state import State, is_end_terminal

from .create_mode import (
    RootCandidateStat,
    ScoreHistograms,
    build_root_candidate_stats,
    record_root_score_histogram,
)
from .debugger import Debugger, format_topk_policy, format_topk_root_shot, policy_stats, summarize_stones
from .evaluator import value_probs_to_winvalue
from .node import Node, argmax_over_actions, tree_size
from .params import DEFAULT_SHOT_MAX_DEPTH, DEFAULT_SHOT_MAX_SIMULATIONS, DEFAULT_SHOT_TIME_LIMIT_SEC

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


def shot_search(
    root_state: State,
    max_simulations: int = DEFAULT_SHOT_MAX_SIMULATIONS,
    max_depth: int = DEFAULT_SHOT_MAX_DEPTH,
    debug: bool = False,
    debug_every: int = 10,
    debug_topk: int = 5,
    stats_log_path: Optional[str] = None,
    is_create_data: bool = False,
) -> Union[SearchAction, SearchDataResult]:
    """
    SHOTで探索して最善手(action_id: 0..N_ACTIONS-1)を返す。
    - max_simulations: シミュレーション回数上限
    - time_limit_sec: 時間上限（秒）。Noneなら時間制限なし
    ※ どちらかの上限に達したら終了
    """
    # 最初に探索木の root を作る
    reset_policy_selection_log()

    time_limit_sec = DEFAULT_SHOT_TIME_LIMIT_SEC
        # if root_state.shot_index % 2 == 0 \
        # else DEFAULT_SHOT_TIME_LIMIT_SEC_LIST[root_state.shot_index]
    if is_create_data:
        time_limit_sec = None  # データ生成時は時間制限なしでシミュレーション回数で制御する

    dbg = Debugger(debug, every=debug_every)
    dbg.log(f"[SHOT] start end={root_state.end} shot_index={root_state.shot_index} hammer={root_state.hammer_team} shot_team={root_state.to_move()}, score_diff={root_state.score_diff}")
    dbg.log("[SHOT] " + summarize_stones(root_state.stones))

    root = Node(root_state)
    root.set_shot_budget(max_simulations)
    dbg.tic("root_expand")
    root.expand_if_needed()  # P(s,a) を入れ、policy上位から初期候補を作る
    dbg.toc("root_expand")

    if dbg.enabled and root.P is not None:
        dbg.log("[SHOT] " + policy_stats(root.P))
        dbg.log("[SHOT] root policy topk: " + format_topk_policy(root.P, debug_topk, decode_action))
        dbg.log("[SHOT] root " + root.round_info())

    start_time = time.perf_counter()
    sims = 0
    root_score_histograms: ScoreHistograms = {}

    while sims < max_simulations:
        if time_limit_sec is not None and (time.perf_counter() - start_time) >= time_limit_sec:
            break

        path: List[Tuple[Node, int]] = []
        node: Node = root
        state: State = root_state

        # 1) Selection
        dbg.tic("selection")
        select_depth = 0

        while node.is_expanded() and (not is_end_terminal(state)) and select_depth < max_depth:
            assert node.P is not None and node.Q is not None and node.Nsa is not None

            node.halve_actions_if_needed()
            a = node.select_action()
            path.append((node, a))
            state = simulator_step(state, a)     # 1投進める
            node = node.child_for(a, state)
            select_depth += 1

        dbg.toc("selection")

        # 2) Expansion
        dbg.tic("expansion")
        if not is_end_terminal(state):
            pi, value_probs = get_policy_and_value(state)
            v_to_move = value_probs_to_winvalue(state, value_probs)
            if not node.is_expanded():
                node.expand(pi)
        else:
            pi = None
            value_probs = None
            v_to_move = None
        dbg.toc("expansion")

        # 3) Evaluation
        dbg.tic("evaluation")
        if is_end_terminal(state):
            v = rollout_to_end_score(state)  # state の手番視点で返す
        else:
            v = -float(v_to_move)  # v_to_move は leaf の手番視点なので、直前手番の視点へ反転する
        dbg.toc("evaluation")

        if is_create_data and path:
            record_root_score_histogram(
                root_score_histograms,
                path[0][1],
                root_state,
                state,
                value_probs,
            )

        # 4) Backprop
        dbg.tic("backprop")
        for (n, a) in reversed(path):
            n.N += 1
            n.Nsa[a] += 1
            n.W[a] += v
            n.Q[a] = n.W[a] / n.Nsa[a]
            v = -v
        dbg.toc("backprop")

        if dbg.on(sims):
            dbg.log(f"[SHOT] sim={sims} depth={select_depth} leaf_shot={state.shot_index} leaf_terminal={is_end_terminal(state)} v={v:.4g}")
            if pi is not None:
                dbg.log("[SHOT] leaf " + policy_stats(pi))
                dbg.log("[SHOT] leaf policy topk: " + format_topk_policy(pi, debug_topk, decode_action))
            dbg.log("[SHOT] root " + root.round_info())

        sims += 1

    assert root.Nsa is not None

    root.halve_actions_if_needed()

    visited_children = 0
    expanded_children = 0

    for a in root.actions:
        if root.Nsa[a] <= 0:
            continue
        visited_children += 1

        child_nodes = root.children_for_action(a)
        if any(child.is_expanded() for child in child_nodes):
            expanded_children += 1

    best_actions = [a for a in root.actions if root.Nsa[a] > 0]
    if not best_actions:
        best_actions = root.actions

    best_action_id = argmax_over_actions(
        best_actions,
        key=lambda a: (root.Q[a], root.Nsa[a], root.P[a]),
    )
    best_action = decode_action(best_action_id)

    if dbg.enabled:
        dbg.log(f"[SHOT] done sims={sims} elapsed={time.perf_counter()-start_time:.3f}s ({(sims/(time.perf_counter()-start_time+1e-12)):.3f} sims/s)")
        dbg.log("[SHOT] timing: " + dbg.summary())
        if root.P is not None and root.Q is not None and root.Nsa is not None:
            dbg.log("[SHOT] root topk: " + format_topk_root_shot(root, debug_topk, decode_action))
        dbg.log(f"[SHOT] best a={best_action_id} -> {best_action}")

    # シミュレーション回数と、シミュレーション時間を表示する
    elapsed_time = time.perf_counter() - start_time
    lines = [
        "-----------------------------------------------------",
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]",
        f"shot={root_state.shot_index} end={root_state.end} hammer={root_state.hammer_team} score_diff={root_state.score_diff}",
        f"simulations={sims} elapsed={elapsed_time:.2f}sec nodes={tree_size(root)}",
        f"root_children visited={visited_children} expanded={expanded_children} candidates={len(root.actions)}",
        "-----------------------------------------------------",
    ]
    if is_create_data:
        _emit_lines(lines, stats_log_path)
    print("-----------------------------------------------------")
    print(f"SHOT search simulations: {sims}, time: {elapsed_time:.2f} sec, nodes: {tree_size(root)}")
    print(f"SHOT root children: visited={visited_children}, expanded={expanded_children} (candidates={len(root.actions)})")
    print("-----------------------------------------------------")

    if is_create_data:
        return best_action_id, build_root_candidate_stats(root, root_score_histograms)

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
    """
    プレイヤーがSHOT前に最初に呼ぶ想定。
    - network: dual_net（policy/value用）
    - stones: list[(x,y)|None] 16要素
    - score_diff: team0 から見た得点差
    - end: 現在のエンド数
    - shot_index: 現在のショット番号
    - hammer_team: 後攻チーム番号

    戻り値: SHOT用 root_state
    """
    stones16 = stones_listdict_to_xy16(stones)

    if debug:
        print("------ DEBUG set_root_state -----")
        for i, p in enumerate(stones16):
            print(f"root_state stone: x={p[0]} y={p[1]}" if p is not None else "root_state stone: None")

    # policy側のグローバルに network と scores_dict をセット
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
        score_diff=score_diff
    )
