import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from search_config import require_search_mode

from common.translate_state import stones_listdict_to_xy16
from nn.network.dual_net import DualNet
from transformer.network import TransformerNetwork
from transformer.params import TRANSFORMER_VY_MODE, TransformerVyMode

from mcts.hybrid_policy import (
    get_policy_and_value, get_value_probs, get_value_probs_batch,
    reset_policy_selection_log, set_policy_context,
)
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
from .params import (
    DEFAULT_SHOT_MAX_DEPTH, DEFAULT_SHOT_MAX_SIMULATIONS,
    DEFAULT_SHOT_TIME_LIMIT_SEC,
    DEFAULT_SHOT_INFERENCE_BATCH_SIZE,
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


def shot_search(
    root_state: State,
    max_simulations: int = DEFAULT_SHOT_MAX_SIMULATIONS,
    max_depth: int = DEFAULT_SHOT_MAX_DEPTH,
    debug: bool = False,
    debug_every: int = 10,
    debug_topk: int = 5,
    stats_log_path: Optional[str] = None,
    is_create_data: bool = False,
    use_value: bool = True,
    action_type: TransformerVyMode = TRANSFORMER_VY_MODE,
    inference_batch_size: int = DEFAULT_SHOT_INFERENCE_BATCH_SIZE,
) -> Union[SearchAction, SearchDataResult]:
    """
    SHOTで探索して最善手(action_id: 0..N_ACTIONS-1)を返す。
    - max_simulations: シミュレーション回数上限
    - time_limit_sec: 時間上限（秒）。Noneなら時間制限なし
    ※ どちらかの上限に達したら終了
    - inference_batch_size: 深さ1・value評価の教師生成でまとめる推論数の上限。
      1は逐次推論。2以上では浮動小数点の丸め差が生じる可能性がある。
    """
    require_search_mode("shot", action_type)
    if not isinstance(max_simulations, int) or isinstance(max_simulations, bool) or max_simulations < 1:
        raise ValueError("max_simulations must be a positive integer")
    if not isinstance(inference_batch_size, int) or inference_batch_size < 1:
        raise ValueError("inference_batch_size must be a positive integer")
    # 最初に探索木の root を作る
    reset_policy_selection_log()

    time_limit_sec = DEFAULT_SHOT_TIME_LIMIT_SEC
    if is_create_data:
        time_limit_sec = None  # データ生成時は時間制限なしでシミュレーション回数で制御する

    # 深さ1の教師生成では、末端から行動選択しないため子ノードと policy は不要。
    value_only_leaf = is_create_data and max_depth == 1 and use_value
    batch_value_leaf = value_only_leaf and inference_batch_size > 1 and not is_end_terminal(root_state)

    dbg = Debugger(debug, every=debug_every)
    decode_search_action = lambda a: decode_action(a, action_type=action_type)
    dbg.log(f"[SHOT] start end={root_state.end} shot_index={root_state.shot_index} hammer={root_state.hammer_team} shot_team={root_state.to_move()}, score_diff={root_state.score_diff}")
    dbg.log("[SHOT] " + summarize_stones(root_state.stones))

    root = Node(root_state, action_type=action_type)
    root.set_shot_budget(max_simulations)
    dbg.tic("root_expand")
    root.expand_if_needed()  # P(s,a) を入れ、policy上位から初期候補を作る
    dbg.toc("root_expand")

    if dbg.enabled and root.P is not None:
        dbg.log("[SHOT] " + policy_stats(root.P))
        dbg.log("[SHOT] root policy topk: " + format_topk_policy(root.P, debug_topk, decode_search_action))
        dbg.log("[SHOT] root " + root.round_info())

    start_time = time.perf_counter()
    sims = 0
    root_score_histograms: ScoreHistograms = {}
    pending_leaves: deque[Tuple[int, State, Optional[List[float]]]] = deque()

    while sims < max_simulations:
        if time_limit_sec is not None and (time.perf_counter() - start_time) >= time_limit_sec:
            break

        path: List[Tuple[Node, int]] = []
        node: Node = root
        state: State = root_state

        # 1) Selection
        dbg.tic("selection")
        select_depth = 0

        if batch_value_leaf:
            if not pending_leaves:
                root.halve_actions_if_needed()
                actions = root.select_action_batch(min(inference_batch_size, max_simulations - sims))
                # 逐次実行と同じ順番で乱数を消費し、各候補を1回だけ評価する。
                leaves = [simulator_step(root_state, a, action_type=action_type) for a in actions]
                nonterminal_indices = [i for i, leaf in enumerate(leaves) if not is_end_terminal(leaf)]
                dbg.toc("selection")
                dbg.tic("batch_inference")
                predictions = get_value_probs_batch(
                    [leaves[i] for i in nonterminal_indices], action_type=action_type,
                )
                dbg.toc("batch_inference")
                dbg.tic("selection")
                values_by_index = dict(zip(nonterminal_indices, predictions))
                pending_leaves.extend(
                    (a, leaf, values_by_index.get(i))
                    for i, (a, leaf) in enumerate(zip(actions, leaves))
                )
            a, state, batch_value_probs = pending_leaves.popleft()
            path.append((root, a))
            select_depth = 1
        else:
            while node.is_expanded() and (not is_end_terminal(state)) and select_depth < max_depth:
                assert node.P is not None and node.Q is not None and node.Nsa is not None

                node.halve_actions_if_needed()
                a = node.select_action()
                path.append((node, a))
                state = simulator_step(state, a, action_type=action_type)     # 1投進める
                if not value_only_leaf:
                    node = node.child_for(a, state)
                select_depth += 1

        dbg.toc("selection")

        # 2) Expansion
        dbg.tic("expansion")
        if not is_end_terminal(state):
            if value_only_leaf:
                pi = None
                if batch_value_leaf:
                    assert batch_value_probs is not None
                    value_probs = batch_value_probs
                else:
                    value_probs = get_value_probs(state, action_type=action_type)
            else:
                pi, value_probs = get_policy_and_value(state, action_type=action_type)
            v_to_move = (
                value_probs_to_winvalue(state, value_probs)
                if use_value
                else None
            )
            if not value_only_leaf and not node.is_expanded():
                node.expand(pi)
        else:
            pi = None
            value_probs = None
            v_to_move = None
        dbg.toc("expansion")

        # 3) Evaluation
        dbg.tic("evaluation")
        if is_end_terminal(state) or not use_value:
            v = rollout_to_end_score(state, action_type=action_type)  # state の手番視点で返す
        else:
            assert v_to_move is not None
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
                dbg.log("[SHOT] leaf policy topk: " + format_topk_policy(pi, debug_topk, decode_search_action))
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
    best_action = decode_action(best_action_id, action_type=action_type)

    if dbg.enabled:
        dbg.log(f"[SHOT] done sims={sims} elapsed={time.perf_counter()-start_time:.3f}s ({(sims/(time.perf_counter()-start_time+1e-12)):.3f} sims/s)")
        dbg.log("[SHOT] timing: " + dbg.summary())
        if root.P is not None and root.Q is not None and root.Nsa is not None:
            dbg.log("[SHOT] root topk: " + format_topk_root_shot(root, debug_topk, decode_search_action))
        dbg.log(f"[SHOT] best a={best_action_id} -> {best_action}")

    # シミュレーション回数と、シミュレーション時間を表示する
    elapsed_time = time.perf_counter() - start_time
    if value_only_leaf:
        children_summary = (
            f"root_actions visited={visited_children} candidates={len(root.actions)} "
            "leaf_nodes=omitted"
        )
        if batch_value_leaf:
            children_summary += f" inference_batch_size={inference_batch_size}"
    else:
        children_summary = (
            f"root_children visited={visited_children} expanded={expanded_children} "
            f"candidates={len(root.actions)}"
        )
    lines = [
        "-----------------------------------------------------",
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]",
        f"shot={root_state.shot_index} end={root_state.end} hammer={root_state.hammer_team} score_diff={root_state.score_diff}",
        f"simulations={sims} elapsed={elapsed_time:.2f}sec nodes={tree_size(root)}",
        children_summary,
        "-----------------------------------------------------",
    ]
    if is_create_data:
        _emit_lines(lines, stats_log_path)
    print("-----------------------------------------------------")
    print(f"SHOT search simulations: {sims}, time: {elapsed_time:.2f} sec, nodes: {tree_size(root)}")
    if value_only_leaf:
        print(f"SHOT {children_summary}")
    else:
        print(f"SHOT root children: visited={visited_children}, expanded={expanded_children} (candidates={len(root.actions)})")
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
        score_diff=score_diff
    )
