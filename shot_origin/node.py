from __future__ import annotations

import math
from typing import Dict, Iterable, Iterator, List, Optional

from mcts.state import State
from transformer.params import TRANSFORMER_VY_MODE, TransformerVyMode, get_transformer_action_dim

from .params import SHOT_ORIGIN_KEEP_RATIO


def argmax_over_actions(actions: Iterable[int], key):
    best_a = None
    best_v = None
    for a in actions:
        v = key(a)
        if best_a is None or v > best_v:
            best_a = a
            best_v = v
    return best_a


def _keep_count_after_halving(action_count: int) -> int:
    keep_count = max(1, int(math.ceil(action_count * SHOT_ORIGIN_KEEP_RATIO)))
    if keep_count >= action_count:
        keep_count = action_count - 1
    return keep_count


class Node:
    """Policy-free SHOT node using the full discrete action space."""

    def __init__(
        self,
        state: State,
        action_type: TransformerVyMode = TRANSFORMER_VY_MODE,
    ):
        self.state = state
        self.key = state.key()
        self.action_type = action_type
        self.n_actions = get_transformer_action_dim(action_type)

        self.N: int = 0
        self.actions: List[int] = list(range(self.n_actions))
        self.children: Dict[int, List["Node"]] = {}

        self.P: Optional[List[float]] = None
        self.Nsa: Optional[List[int]] = None
        self.W: Optional[List[float]] = None
        self.Q: Optional[List[float]] = None

        self._round_index: int = 0
        self._round_visit_target: int = 0

    def child_for(self, action: int, state: State) -> "Node":
        state_key = state.key()
        action_children = self.children.setdefault(action, [])
        for child in action_children:
            if child.key == state_key:
                return child

        child = Node(state, action_type=self.action_type)
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
        if self.is_expanded():
            return
        self.expand()

    def expand(self) -> None:
        prior = 1.0 / self.n_actions
        self.P = [prior] * self.n_actions
        self.Nsa = [0] * self.n_actions
        self.W = [0.0] * self.n_actions
        self.Q = [0.0] * self.n_actions
        self.actions = list(range(self.n_actions))
        self._round_index = 0
        self._round_visit_target = 0
        self._start_next_round()

    def _round_extra_visits(self) -> int:
        return self._round_index + 1

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
        assert self.Q is not None and self.Nsa is not None

        while self._can_halve():
            keep_count = _keep_count_after_halving(len(self.actions))
            self.actions = sorted(
                self.actions,
                key=lambda a: (self.Q[a], self.Nsa[a], -a),
                reverse=True,
            )[:keep_count]
            self._round_index += 1
            self._start_next_round()

    def select_action(self) -> int:
        assert self.Q is not None and self.Nsa is not None

        need_actions = [a for a in self.actions if self.Nsa[a] < self._round_visit_target]
        if need_actions:
            return argmax_over_actions(
                need_actions,
                key=lambda a: (-self.Nsa[a], -a),
            )

        return argmax_over_actions(
            self.actions,
            key=lambda a: (self.Q[a], self.Nsa[a], -a),
        )

    def round_info(self) -> str:
        return (
            f"round={self._round_index} active={len(self.actions)} "
            f"target={self._round_visit_target} extra={self._round_extra_visits()} N={self.N}"
        )


def tree_size(root: Node) -> int:
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