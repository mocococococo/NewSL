import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

from common.translate_state import stones_listdict_to_xy16, scores_dict_to_list
from nn.network.dual_net import DualNet
from transformer.network import TransformerNetwork
from transformer.params import TransformerVyMode
from .node import Node, get_node, get_child_node, argmax_over_actions, clear_node_table, node_table_size, peek_node, count_reachable_nodes, reset_tt_stats, get_tt_stats
from .state import State, is_end_terminal, score_diff_from_scores
from .simulate import simulator_step, decode_action
from .hybrid_policy import get_policy_and_value, reset_policy_selection_log, set_policy_context
from .rollout import rollout_to_end_score, score_to_winvalue
from .params import DEFAULT_MAX_SIMULATIONS, DEFAULT_CPUCT, \
    DEFAULT_TIME_LIMIT_SEC, DEFAULT_TIME_LIMIT_SEC_LIST, DEFAULT_MAX_DEPTH
from .create_mode import VALUE_CLASS_COUNT, record_value_histogram

from .debugger import Debugger, summarize_stones, policy_stats, format_topk_policy, format_topk_root_visits

VALUE_CLASS_OFFSET = 8

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
    """
    NNの17クラスvalue分布(shot_team/to_move視点)を、winvalue期待値[-1,1]に変換する。
    """
    if len(value_probs) != 17:
        raise ValueError(f"value_probs length must be 17, got {len(value_probs)}")

    to_move_team = state.to_move()
    if to_move_team not in (0, 1):
        raise ValueError(f"state.to_move() must be 0/1 for non-terminal state, got {to_move_team}")

    score_diff_leaf = state.score_diff if to_move_team == 0 else -state.score_diff
    had_hammer_this_end = (to_move_team == state.hammer_team)

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
    max_depth: int = DEFAULT_MAX_DEPTH,  # 追加
    debug: bool = False,              # 追加
    debug_every: int = 10,            # 追加（10回に1回出力）
    debug_topk: int = 5,              # 追加（上位k手を表示）    
    stats_log_path: Optional[str] = None,
    is_create_data: bool = False,
    use_value: bool = True,
    use_progressive_widening: bool = True,
    use_transposition_table: bool = True,
    return_stats: bool = False,
    action_type: TransformerVyMode = "default",
) -> Union[SearchAction, SearchDataResult]:
    """
    PUCTで探索して最善手を返す。
    - max_simulations: シミュレーション回数上限
    - time_limit_sec: 時間上限（秒）。Noneなら時間制限なし
    ※ どちらかの上限に達したら終了
    """
    # 最初にノードテーブルをクリア
    clear_node_table()
    reset_tt_stats()
    reset_policy_selection_log()

    def decode_search_action(action: int) -> SearchAction:
        return decode_action(action, action_type=action_type)
    
    time_limit_sec = DEFAULT_TIME_LIMIT_SEC
        # if root_state.shot_index % 2 == 0 \
        # else DEFAULT_TIME_LIMIT_SEC_LIST[root_state.shot_index]
    if is_create_data:
        value_list = [0.0 for _ in range(VALUE_CLASS_COUNT)]
        time_limit_sec = None  # データ生成時は時間制限なしでシミュレーション回数で制御する

    dbg = Debugger(debug, every=debug_every)
    dbg.log(f"[PUCT] start end={root_state.end} shot_index={root_state.shot_index} hammer={root_state.hammer_team} shot_team={root_state.to_move()}, score_diff={root_state.score_diff}")
    dbg.log(
        f"[PUCT] options progressive_widening={use_progressive_widening} "
        f"transposition_table={use_transposition_table}"
    )
    dbg.log("[PUCT] " + summarize_stones(root_state.stones))
    
    if use_transposition_table:
        root: Node = get_node(
            root_state,
            use_progressive_widening=use_progressive_widening,
            action_type=action_type,
        )
    else:
        root = Node(
            root_state,
            use_progressive_widening=use_progressive_widening,
            action_type=action_type,
        )
    dbg.tic("root_expand")
    root.expand_if_needed()  # P(s,a) を入れる
    dbg.toc("root_expand")

    # ルートに探索ノイズ（任意。探索多様化に効く）
    # root.P = (1-ε)*P + ε*Dir(α)
    # ε=0.25, αは候補数に応じて調整（AlphaZero流）
    
    if dbg.enabled and root.P is not None:
        dbg.log("[PUCT] " + policy_stats(root.P))
        dbg.log(
            "[PUCT] root policy topk: "
            + format_topk_policy(root.P, debug_topk, decode_search_action)
        )

    
    start_time = time.perf_counter()
    sims = 0

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
            
            node.maybe_widen()
            
            a = argmax_over_actions(
                node.actions,
                key=lambda a: node.Q[a] + cpuct * node.P[a] * ( (node.N ** 0.5) / (1 + node.Nsa[a]) )
            )
            path.append((node, a))
            state = simulator_step(
                state,
                a,
                action_type=action_type,
            )  # 1投進める
            node = get_child_node(
                node,
                a,
                state,
                use_transposition_table=use_transposition_table,
            )
            select_depth += 1

        dbg.toc("selection")

        # 2) Expansion
        dbg.tic("expansion")
        if not is_end_terminal(state):
            pi, value_probs = get_policy_and_value(
                state,
                action_type=action_type,
            )
            v_to_move = (
                _value_probs_to_winvalue(state, value_probs)
                if use_value
                else None
            )
            if not node.is_expanded():
                node.expand(pi)
        else:
            pi = None
            value_probs = None
            v_to_move = None
        dbg.toc("expansion")

        # 3) Rollout (エンド終端まで)
        dbg.tic("rollout")
        if is_end_terminal(state) or not use_value:
            v = rollout_to_end_score(
                state,
                action_type=action_type,
            )  # 「stateの手番視点」で返すのが楽
        else:
            assert v_to_move is not None
            # rolloutは簡易版のpolicyでやる（高速化のため）
            v = -float(v_to_move)  # rolloutは相手視点でやる（v_to_moveはstateの手番視点なので符号反転）
        dbg.toc("rollout")

        if is_create_data:
            record_value_histogram(value_list, root_state, state, value_probs)

        # 4) Backprop (手番反転のため符号反転)
        dbg.tic("backprop")
        for (n, a) in reversed(path):
            n.N += 1
            n.Nsa[a] += 1
            n.W[a] += v
            n.Q[a] = n.W[a] / n.Nsa[a]
            v = -v
        dbg.toc("backprop")
        
        # たまに状況を出す（1行で済む形）
        if dbg.on(sims):
            dbg.log(f"[PUCT] sim={sims} depth={select_depth} leaf_shot={state.shot_index} leaf_terminal={is_end_terminal(state)} v={v:.4g}")
            if pi is not None:
                dbg.log("[PUCT] leaf " + policy_stats(pi))
                dbg.log(
                    "[PUCT] leaf policy topk: "
                    + format_topk_policy(pi, debug_topk, decode_search_action)
                )
            
        sims += 1

    # 最終手
    assert root.Nsa is not None
    
    visited_children = 0
    expanded_children = 0

    for a in root.actions:
        if root.Nsa[a] <= 0:
            continue
        visited_children += 1

        child_nodes = root.children.get(a, [])
        if any(child.is_expanded() for child in child_nodes):
            expanded_children += 1
            
    best_action_id = argmax_over_actions(root.actions, key=lambda a: root.Nsa[a])
    best_action = decode_search_action(best_action_id)
    
    if dbg.enabled:
        dbg.log(f"[PUCT] done sims={sims} elapsed={time.perf_counter()-start_time:.3f}s ({(sims/(time.perf_counter()-start_time+1e-12)):.3f} sims/s)")
        dbg.log("[PUCT] timing: " + dbg.summary())
        if root.P is not None and root.Q is not None and root.Nsa is not None:
            dbg.log(
                "[PUCT] root Nsa topk: "
                + format_topk_root_visits(
                    root,
                    debug_topk,
                    decode_search_action,
                )
            )
        dbg.log(f"[PUCT] best a={best_action_id} -> {best_action}")
    
    # シミュレート回数と、シミュレート時間を表示する
    elapsed_time = time.perf_counter() - start_time
    nodes = count_reachable_nodes(root)
    tt_stats = get_tt_stats()
    tt_requests = int(tt_stats["requests"])
    tt_hits = int(tt_stats["hits"])
    tt_misses = int(tt_stats["misses"])
    tt_hit_rate = float(tt_hits / tt_requests) if tt_requests > 0 else 0.0
    search_stats = {
        "simulations": int(sims),
        "elapsed": float(elapsed_time),
        "nodes": int(nodes),
        "root_visited": int(visited_children),
        "root_expanded": int(expanded_children),
        "root_candidates": int(len(root.actions)),
        "tt_requests": tt_requests,
        "tt_hits": tt_hits,
        "tt_misses": tt_misses,
        "tt_hit_rate": tt_hit_rate,
        "use_progressive_widening": bool(use_progressive_widening),
        "use_transposition_table": bool(use_transposition_table),
    }
    lines = [
        "-----------------------------------------------------",
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]",
        f"shot={root_state.shot_index} end={root_state.end} hammer={root_state.hammer_team} score_diff={root_state.score_diff}",
        f"simulations={sims} elapsed={elapsed_time:.2f}sec nodes={nodes}",
        f"root_children visited={visited_children} expanded={expanded_children} candidates={len(root.actions)}",
        f"tt requests={tt_requests} hits={tt_hits} misses={tt_misses} hit_rate={tt_hit_rate:.6f}",
        f"progressive_widening={use_progressive_widening} transposition_table={use_transposition_table}",
        "-----------------------------------------------------",
    ]
    if stats_log_path:
        _emit_lines(lines, stats_log_path)
    print("-----------------------------------------------------")
    print(f"MCTS search simulations: {sims}, time: {elapsed_time:.2f} sec, nodes: {nodes}")
    print(f"MCTS root children: visited={visited_children}, expanded={expanded_children} (candidates={len(root.actions)})")
    print(f"MCTS TT: requests={tt_requests}, hits={tt_hits}, misses={tt_misses}, hit_rate={tt_hit_rate:.6f}")
    print(
        "MCTS options: "
        f"progressive_widening={use_progressive_widening}, "
        f"transposition_table={use_transposition_table}"
    )
    print("-----------------------------------------------------")

    if is_create_data:
        return best_action, root.Nsa.copy(), value_list
    if return_stats:
        return best_action, search_stats
    
    return best_action

def set_root_state(
    sl_model: Union[DualNet, TransformerNetwork],
    stones: List[Optional[Dict]],
    score_diff: int,
    end: int,
    shot_index: int,
    hammer_team: int,
    debug: bool = False,
    transformer_target_end: Tuple[int, ...] = (9, 10),  # transformerのターゲットとするエンド（複数指定可）
    transformer_target_shot: Tuple[int, ...] = (15,),  # transformerのターゲットとするショット（複数指定可）
    sl_model_is_cnn: bool = True,
    search_based_model: Optional[Union[TransformerNetwork, Dict[int, TransformerNetwork]]] = None,
    use_search_based_model: bool = False,
) -> State:
    """
    プレイヤーがPUCT前に最初に呼ぶ想定。
    - sl_model: 教師あり学習モデル
    - stones: list[(x,y)|None] 16要素
    - score_diff: team0から見た得点差
    - end: 現在のエンド数
    - shot_index: 現在のショット番号
    - hammer: Trueならteam1が後攻(ハンマー)、Falseならteam0が後攻(ハンマー)

    戻り値: PUCT用 root_state
    """
    stones16 = stones_listdict_to_xy16(stones)
    
    if debug:
        print("------ DEBUG set_root_state -----")
        for i, p in enumerate(stones16):
            print(f"root_state stone: x={p[0]} y={p[1]}" if p is not None else f"root_state stone: None")
        

    # policy側のグローバルにモデルと得点差をセット
    set_policy_context(
        sl_model=sl_model,
        score_diff=score_diff,
        sl_model_is_cnn=sl_model_is_cnn,
        search_based_model=search_based_model,
        use_search_based_model=use_search_based_model,
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
