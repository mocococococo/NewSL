from __future__ import annotations

import math
from typing import Dict, Iterable, Iterator, List, Optional

from mcts.hybrid_policy import get_policy
from mcts.state import State
from transformer.params import TRANSFORMER_ACTION_DIM

from .params import (
    DEFAULT_SHOT_INITIAL_CANDIDATES,
    SHOT_KEEP_RATIO,
    SHOT_MIN_VISITS_PER_ACTION,
)

# 行動空間: (vx, vy, spin) の全組み合わせを 0..N_ACTIONS-1 に畳む
N_ACTIONS = TRANSFORMER_ACTION_DIM

ALL_ACTIONS: List[int] = list(range(N_ACTIONS))

def argmax_over_actions(actions: Iterable[int], key):
    """actions の中で key(a) が最大の a を返す。同値は先に出た方。"""
    best_a = None
    best_v = None
    for a in actions:
        v = key(a)
        if best_a is None or v > best_v:
            best_a = a
            best_v = v
    return best_a


def _ceil_log2(n: int) -> int:
    if n <= 1:
        return 1
    return int(math.ceil(math.log2(n)))


def _keep_count_after_halving(action_count: int) -> int:
    keep_count = max(1, int(math.ceil(action_count * SHOT_KEEP_RATIO)))
    if keep_count >= action_count:
        keep_count = action_count - 1
    return keep_count


class Node:
    """
    SHOT用ノード。
    P/Nsa/W/Q は mcts.Node と同じ意味を持ち、actions だけを
    Sequential Halving の生存候補として更新する。
    """

    def __init__(self, state: State):
        self.state = state
        self.key = state.key()

        self.N: int = 0
        self.actions: List[int] = ALL_ACTIONS
        self.children: Dict[int, List["Node"]] = {}

        self.P: Optional[List[float]] = None
        self.Nsa: Optional[List[int]] = None
        self.W: Optional[List[float]] = None
        self.Q: Optional[List[float]] = None

        self._policy_order: Optional[List[int]] = None
        self._shot_budget: Optional[int] = None
        self._round_index: int = 0
        self._round_visit_target: int = 0

    def set_shot_budget(self, shot_budget: Optional[int]) -> None:
        """
        このノードに割り当てる探索回数の目安を設定する。
        root は max_simulations を持ち、内部ノードは未指定なら到達回数に応じて育つ。
        """
        if shot_budget is None:
            self._shot_budget = None
        else:
            self._shot_budget = max(1, int(shot_budget))
        if self.is_expanded():
            self._start_next_round()

    def child_for(self, action: int, state: State) -> "Node":
        """
        このノードの子として state に対応する Node を返す。
        同じ局面が別の親から現れても共有しないので、transposition table にはしない。
        """
        state_key = state.key()
        action_children = self.children.setdefault(action, [])
        for child in action_children:
            if child.key == state_key:
                return child

        child = Node(state)
        action_children.append(child)
        return child

    def children_for_action(self, action: int) -> List["Node"]:
        return self.children.get(action, [])

    def iter_children(self) -> Iterator["Node"]:
        for action_children in self.children.values():
            for child in action_children:
                yield child

    def is_expanded(self) -> bool:
        return self.P is not None

    def expand_if_needed(self) -> None:
        """
        search.py から root.expand_if_needed() と呼ばれる前提。
        policy(state) を取得し、SHOTの初期候補を作る。
        """
        if self.is_expanded():
            return
        pi = get_policy(self.state)
        self.expand(pi)

    def expand(self, pi) -> None:
        """
        policy は以下のどちらでも受け付ける:
          - List[float]: len=N_ACTIONS
          - Dict[int, float]: 一部の行動だけを持つ sparse 形式
        """
        P = [0.0] * N_ACTIONS

        if isinstance(pi, dict):
            for a, p in pi.items():
                if 0 <= a < N_ACTIONS:
                    P[a] = float(p)
        else:
            pi_list = list(pi)
            if len(pi_list) != N_ACTIONS:
                raise ValueError(f"policy must have length {N_ACTIONS}, got {len(pi_list)}")
            for i, p in enumerate(pi_list):
                P[i] = float(p)

        s = sum(P)
        if s <= 0.0:
            u = 1.0 / N_ACTIONS
            P = [u] * N_ACTIONS
        else:
            inv = 1.0 / s
            P = [p * inv for p in P]

        self.P = P
        self.Nsa = [0] * N_ACTIONS
        self.W = [0.0] * N_ACTIONS
        self.Q = [0.0] * N_ACTIONS

        self._policy_order = sorted(range(N_ACTIONS), key=lambda a: self.P[a], reverse=True)

        k0 = min(DEFAULT_SHOT_INITIAL_CANDIDATES, N_ACTIONS)
        self.actions = self._policy_order[:k0]
        self._round_index = 0
        self._round_visit_target = 0
        self._start_next_round()

    def _round_extra_visits(self) -> int:
        if self._shot_budget is None:
            return max(1, int(SHOT_MIN_VISITS_PER_ACTION))

        remaining_budget = max(0, self._shot_budget - self.N)
        rounds_left = _ceil_log2(len(self.actions))
        denom = max(1, len(self.actions) * rounds_left)
        extra = remaining_budget // denom
        return max(int(SHOT_MIN_VISITS_PER_ACTION), int(extra))

    def _start_next_round(self) -> None:
        assert self.Nsa is not None
        if not self.actions:
            self._round_visit_target = 0
            return

        base_visits = min(self.Nsa[a] for a in self.actions)
        self._round_visit_target = base_visits + self._round_extra_visits()

    def _can_halve(self) -> bool:
        assert self.Nsa is not None
        if len(self.actions) <= 1:
            return False
        return all(self.Nsa[a] >= self._round_visit_target for a in self.actions)

    def halve_actions_if_needed(self) -> None:
        """
        現在のラウンドで全候補が必要回数だけ評価されたら、Qで並べて候補を半減する。
        """
        assert self.P is not None and self.Q is not None and self.Nsa is not None

        while self._can_halve():
            keep_count = _keep_count_after_halving(len(self.actions))
            self.actions = sorted(
                self.actions,
                key=lambda a: (self.Q[a], self.Nsa[a], self.P[a]),
                reverse=True,
            )[:keep_count]
            self._round_index += 1
            self._start_next_round()

    def select_action(self) -> int:
        """
        SHOTの現在ラウンドで次に評価する行動を返す。
        ラウンド内では訪問回数が少ない候補を優先し、同数ならpolicyを使う。
        """
        assert self.P is not None and self.Q is not None and self.Nsa is not None

        need_actions = [a for a in self.actions if self.Nsa[a] < self._round_visit_target]

        if need_actions:
            return argmax_over_actions(
                need_actions,
                key=lambda a: (-self.Nsa[a], self.P[a], self.Q[a]),
            )

        return argmax_over_actions(
            self.actions,
            key=lambda a: (self.Q[a], self.Nsa[a], self.P[a]),
        )

    def round_info(self) -> str:
        return (
            f"round={self._round_index} active={len(self.actions)} "
            f"target={self._round_visit_target} N={self.N}"
        )


def tree_size(root: Node) -> int:
    """root から辿れる探索木のノード数を返す。"""
    seen = set()
    stack = [root]
    count = 0

    while stack:
        node = stack.pop()
        node_id = id(node)
        if node_id in seen:
            continue
        seen.add(node_id)
        count += 1
        stack.extend(node.iter_children())

    return count
