# node.py
from __future__ import annotations
from typing import Dict, List, Optional, Iterable, Any

from transformer.params import TransformerVyMode, get_transformer_action_dim
from .state import State
from .hybrid_policy import get_policy
from .params import TOPK_INIT, TOPK_MAX, PW_C, PW_ALPHA

# 木（transposition用）
_NODE_TABLE: Dict[tuple, "Node"] = {}
_TT_STATS = {
    "requests": 0,
    "hits": 0,
    "misses": 0,
}

def clear_node_table() -> None:
    """puct_search() 1回分の探索が終わったら木を破棄したい用途。"""
    _NODE_TABLE.clear()
    
def reset_tt_stats() -> None:
    _TT_STATS["requests"] = 0
    _TT_STATS["hits"] = 0
    _TT_STATS["misses"] = 0

def get_tt_stats() -> Dict[str, int]:
    return dict(_TT_STATS)

def node_table_size() -> int:
    """現在 _NODE_TABLE に保持されているノード数（=到達した局面数）"""
    return len(_NODE_TABLE)

def peek_node(state):
    """
    _NODE_TABLE に既に存在するノードだけ返す（無ければNone）。
    ※ get_node(state) と違って新規作成しない
    """
    return _NODE_TABLE.get(state.key(), None)

def argmax_over_actions(actions: Iterable[int], key):
    """actions の中で key(a) が最大の a を返す（同値は先に出た方）"""
    best_a = None
    best_v = None
    for a in actions:
        v = key(a)
        if best_a is None or v > best_v:
            best_a = a
            best_v = v
    return best_a


class Node:
    """
    PUCT用ノード（value無しrolloutの統計）。
    search.py が参照するフィールド/メソッドをそのまま用意する。
    """

    def __init__(
        self,
        state: State,
        use_progressive_widening: bool = True,
        action_type: TransformerVyMode = "default",
    ):
        self.state = state
        self.use_progressive_widening = use_progressive_widening
        self.action_type = action_type
        self.n_actions = get_transformer_action_dim(action_type)

        self.N: int = 0
        self.actions: List[int] = list(range(self.n_actions))
        self.children: Dict[int, List["Node"]] = {}

        # 展開後に埋まる
        self.P: Optional[List[float]] = None
        self.Nsa: Optional[List[int]] = None
        self.W: Optional[List[float]] = None
        self.Q: Optional[List[float]] = None
        
        self._policy_order: Optional[List[int]] = None  # Policyの降順 action list
        
    def maybe_widen(self) -> None:
        """
        Progressive Widening:
        ノード訪問回数Nに応じて、actions（候補手）を方策順に増やす。
        """
        if not self.use_progressive_widening:
            return
        if self._policy_order is None:
            return

        # 例: target = TOPK_INIT + PW_C * N^PW_ALPHA
        target = TOPK_INIT + int(PW_C * (self.N ** PW_ALPHA))
        if target > TOPK_MAX:
            target = TOPK_MAX
        if target <= len(self.actions):
            return

        self.actions = self._policy_order[:target]

    def is_expanded(self) -> bool:
        return self.P is not None

    def expand_if_needed(self) -> None:
        """
        search.py から root.expand_if_needed() と呼ばれる前提。
        policy(state) は search.py 側で後で実装される想定なので、ここでは参照だけする。
        """
        if self.is_expanded():
            return
        # get_policy はグローバル関数としてどこかに定義されている前提（search.py想定）
        policy = get_policy(self.state, action_type=self.action_type)
        self.expand(policy)

    def expand(self, policy) -> None:
        """
        policy は以下どちらでもOKにする:
          - List[float]（len=self.n_actions）
          - Dict[int, float]（一部のみでもOK）
        """
        P = [0.0] * self.n_actions

        if isinstance(policy, dict):
            for a, p in policy.items():
                if 0 <= a < self.n_actions:
                    P[a] = float(p)
        else:
            policy_list = list(policy)
            if len(policy_list) != self.n_actions:
                raise ValueError(
                    "policy policy must have length "
                    f"{self.n_actions}, got {len(policy_list)}"
                )
            for i, p in enumerate(policy_list):
                P[i] = float(p)

        s = sum(P)
        if s <= 0.0:
            # もし policy が壊れてても探索が止まらないように一様にする
            u = 1.0 / self.n_actions
            P = [u] * self.n_actions
        else:
            inv = 1.0 / s
            P = [p * inv for p in P]

        self.P = P
        self.Nsa = [0] * self.n_actions
        self.W = [0.0] * self.n_actions
        self.Q = [0.0] * self.n_actions
        
        # Pの大きい順に action を並べる
        self._policy_order = sorted(
            range(self.n_actions),
            key=lambda a: self.P[a],
            reverse=True,
        )
        
        if self.use_progressive_widening:
            k0 = TOPK_INIT if TOPK_INIT < TOPK_MAX else TOPK_MAX
            self.actions = self._policy_order[:k0]
        else:
            self.actions = list(self._policy_order)


def get_node(
    state,
    use_progressive_widening: bool = True,
    action_type: TransformerVyMode = "default",
) -> Node:
    """
    同一局面（state.key()が同じ）なら同じNodeを返す。
    """
    k = state.key()
    n = _NODE_TABLE.get(k)
    if n is None:
        n = Node(
            state,
            use_progressive_widening=use_progressive_widening,
            action_type=action_type,
        )
        _NODE_TABLE[k] = n
    elif n.use_progressive_widening != use_progressive_widening:
        raise ValueError("Node for the same state was requested with conflicting Progressive Widening settings.")
    elif n.action_type != action_type:
        raise ValueError(
            "同じ局面に異なる行動種類のNodeが要求されました: "
            f"{n.action_type} != {action_type}"
        )
    return n


def get_child_node(
    parent: Node,
    action: int,
    child_state: State,
    use_transposition_table: bool = True,
    measure_tt_stats: bool = True,
) -> Node:
    if use_transposition_table:
        if measure_tt_stats:
            child_key = child_state.key()
            _TT_STATS["requests"] += 1
            if child_key in _NODE_TABLE:
                _TT_STATS["hits"] += 1
            else:
                _TT_STATS["misses"] += 1

        child = get_node(
            child_state,
            use_progressive_widening=parent.use_progressive_widening,
            action_type=parent.action_type,
        )
    else:
        child = Node(
            child_state,
            use_progressive_widening=parent.use_progressive_widening,
            action_type=parent.action_type,
        )

    action_children = parent.children.setdefault(action, [])
    if not any(existing is child for existing in action_children):
        action_children.append(child)
    return child


def count_reachable_nodes(root: Node) -> int:
    seen = set()
    stack = [root]

    while stack:
        node = stack.pop()
        node_id = id(node)
        if node_id in seen:
            continue
        seen.add(node_id)
        for action_children in node.children.values():
            stack.extend(action_children)

    return len(seen)
