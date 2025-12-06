# node.py
from __future__ import annotations
from typing import Dict, List, Optional, Iterable, Any

from board.constant import BOARD_SIZE
from .state import State
from .policy import get_policy

# 行動空間： (vx, vy, spin) の全組合せを 0..2047 に潰す
VX_SIZE = BOARD_SIZE
VY_SIZE = BOARD_SIZE
SPIN_SIZE = 2
N_ACTIONS = VX_SIZE * VY_SIZE * SPIN_SIZE  # 2048

# どのノードでも同じ actions を使うので、1回だけ作る
ALL_ACTIONS: List[int] = list(range(N_ACTIONS))

# 木（transposition用）
_NODE_TABLE: Dict[tuple, "Node"] = {}


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

    def __init__(self, state: State):
        self.state = state

        self.N: int = 0
        self.actions: List[int] = ALL_ACTIONS

        # 展開後に埋まる
        self.P: Optional[List[float]] = None   # len=2048
        self.Nsa: Optional[List[int]] = None   # len=2048
        self.W: Optional[List[float]] = None   # len=2048
        self.Q: Optional[List[float]] = None   # len=2048

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
        policy = get_policy(self.state)  # type: ignore[name-defined]
        self.expand(policy)

    def expand(self, policy) -> None:
        """
        policy は以下どちらでもOKにする:
          - List[float]（len=2048）
          - Dict[int, float]（一部のみでもOK）
        """
        P = [0.0] * N_ACTIONS

        if isinstance(policy, dict):
            for a, p in policy.items():
                if 0 <= a < N_ACTIONS:
                    P[a] = float(p)
        else:
            policy_list = list(policy)
            if len(policy_list) != N_ACTIONS:
                raise ValueError(f"policy policy must have length {N_ACTIONS}, got {len(policy_list)}")
            for i, p in enumerate(policy_list):
                P[i] = float(p)

        s = sum(P)
        if s <= 0.0:
            # もし policy が壊れてても探索が止まらないように一様にする
            u = 1.0 / N_ACTIONS
            P = [u] * N_ACTIONS
        else:
            inv = 1.0 / s
            P = [p * inv for p in P]

        self.P = P
        self.Nsa = [0] * N_ACTIONS
        self.W = [0.0] * N_ACTIONS
        self.Q = [0.0] * N_ACTIONS


def get_node(state) -> Node:
    """
    同一局面（state.key()が同じ）なら同じNodeを返す。
    """
    k = state.key()
    n = _NODE_TABLE.get(k)
    if n is None:
        n = Node(state)
        _NODE_TABLE[k] = n
    return n
