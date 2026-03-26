import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from common.translate_state import stones_listdict_to_xy16, scores_dict_to_list
from nn.network.dual_net import DualNet
from .node import Node, get_node, argmax_over_actions, clear_node_table, node_table_size, peek_node
from .state import State, is_end_terminal, score_diff_from_scores
from .simulate import simulator_step, decode_action
from .policy import get_policy_and_value, set_policy_context
from .rollout import rollout_to_end_score, score_to_winvalue
from .params import DEFAULT_MAX_SIMULATIONS, DEFAULT_CPUCT, \
    DEFAULT_TIME_LIMIT_SEC, DEFAULT_TIME_LIMIT_SEC_LIST, DEFAULT_MAX_DEPTH

from .debugger import Debugger, summarize_stones, policy_stats, format_topk_policy, format_topk_root_visits

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
) -> Tuple[float, float, int]:
    """
    PUCTで探索して最善手(action_id: 0..2047)を返す。
    - max_simulations: シミュレーション回数上限
    - time_limit_sec: 時間上限（秒）。Noneなら時間制限なし
    ※ どちらかの上限に達したら終了
    """
    # 最初にノードテーブルをクリア
    clear_node_table()
    
    time_limit_sec = DEFAULT_TIME_LIMIT_SEC #\
        # if root_state.shot_index % 2 == 0 \
        # else DEFAULT_TIME_LIMIT_SEC_LIST[root_state.shot_index]

    dbg = Debugger(debug, every=debug_every)
    dbg.log(f"[PUCT] start end={root_state.end} shot_index={root_state.shot_index} hammer={root_state.hammer_team} shot_team={root_state.to_move()}, score_diff={root_state.score_diff}")
    dbg.log("[PUCT] " + summarize_stones(root_state.stones))
    
    root: Node = get_node(root_state)
    dbg.tic("root_expand")
    root.expand_if_needed()  # P(s,a) を入れる
    dbg.toc("root_expand")

    # ルートに探索ノイズ（任意。探索多様化に効く）
    # root.P = (1-ε)*P + ε*Dir(α)
    # ε=0.25, αは候補数に応じて調整（AlphaZero流）
    
    if dbg.enabled and root.P is not None:
        dbg.log("[PUCT] " + policy_stats(root.P))
        dbg.log("[PUCT] root policy topk: " + format_topk_policy(root.P, debug_topk, decode_action))

    
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
            state = simulator_step(state, a)     # 1投進める
            node = get_node(state)
            select_depth += 1

        dbg.toc("selection")

        # 2) Expansion
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

        # 3) Rollout (エンド終端まで)
        dbg.tic("rollout")
        if is_end_terminal(state):
            v = rollout_to_end_score(state)  # 「stateの手番視点」で返すのが楽
        else:
            # rolloutは簡易版のpolicyでやる（高速化のため）
            v = -float(v_to_move)  # rolloutは相手視点でやる（v_to_moveはstateの手番視点なので符号反転）
        dbg.toc("rollout")

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
                dbg.log("[PUCT] leaf policy topk: " + format_topk_policy(pi, debug_topk, decode_action))
            
        sims += 1

    # 最終手
    assert root.Nsa is not None
    
    visited_children = 0
    expanded_children = 0

    for a in root.actions:
        if root.Nsa[a] <= 0:
            continue
        visited_children += 1

        child_state = simulator_step(root_state, a)   # 1投進めた局面
        child_node = peek_node(child_state)           # ※新規作成しない
        if child_node is not None and child_node.is_expanded():
            expanded_children += 1
            
    best_action_id = argmax_over_actions(root.actions, key=lambda a: root.Nsa[a])
    best_action = decode_action(best_action_id)
    
    if dbg.enabled:
        dbg.log(f"[PUCT] done sims={sims} elapsed={time.perf_counter()-start_time:.3f}s ({(sims/(time.perf_counter()-start_time+1e-12)):.3f} sims/s)")
        dbg.log("[PUCT] timing: " + dbg.summary())
        if root.P is not None and root.Q is not None and root.Nsa is not None:
            dbg.log("[PUCT] root Nsa topk: " + format_topk_root_visits(root, debug_topk, decode_action))
        dbg.log(f"[PUCT] best a={best_action_id} -> {best_action}")
    
    # シミュレート回数と、シミュレート時間を表示する
    elapsed_time = time.perf_counter() - start_time
    lines = [
        "-----------------------------------------------------",
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]",
        f"shot={root_state.shot_index} end={root_state.end} hammer={root_state.hammer_team} score_diff={root_state.score_diff}",
        f"simulations={sims} elapsed={elapsed_time:.2f}sec nodes={node_table_size()}",
        f"root_children visited={visited_children} expanded={expanded_children} candidates={len(root.actions)}",
        "-----------------------------------------------------",
    ]
    _emit_lines(lines, stats_log_path)
    print("-----------------------------------------------------")
    print(f"PUCT search simulations: {sims}, time: {elapsed_time:.2f} sec, nodes: {node_table_size()}")
    print(f"PUCT root children: visited={visited_children}, expanded={expanded_children} (candidates={len(root.actions)})")
    print("-----------------------------------------------------")
    
    return best_action

def set_root_state(
    network: DualNet,
    stones: List[Optional[Dict]],
    score_diff: int,
    end: int,
    shot_index: int,
    hammer_team: int,
    debug: bool = False
) -> State:
    """
    プレイヤーがPUCT前に最初に呼ぶ想定。
    - network: dual_net（policy用）
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
        

    # policy側のグローバルに network と scores_dict をセット
    set_policy_context(network, score_diff)

    return State.initial(
        stones=stones16,
        end=end,
        hammer_team=hammer_team,
        shot_index=shot_index,
        score_diff=score_diff
    )
