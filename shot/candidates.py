from __future__ import annotations

import math
from typing import List

from board.constant import VX_SIZE, VY_SIZE

from .node import ShotActionStats

N_ACTIONS = VX_SIZE * VY_SIZE * 2


def top_k_actions(policy: List[float], k: int) -> List[int]:
    if len(policy) != N_ACTIONS:
        raise RuntimeError(f"policy length mismatch: {len(policy)} != {N_ACTIONS}")
    k = max(1, min(int(k), len(policy)))
    return sorted(range(len(policy)), key=lambda a: policy[a], reverse=True)[:k]


def ceil_log2(n: int) -> int:
    if n <= 1:
        return 1
    return int(math.ceil(math.log2(n)))


def rank_actions(
    actions: List[int],
    stats: List[ShotActionStats],
    policy: List[float],
) -> List[int]:
    return sorted(
        actions,
        key=lambda a: (stats[a].mean_value, stats[a].visits, policy[a]),
        reverse=True,
    )


def keep_count_after_halving(action_count: int, keep_ratio: float) -> int:
    keep_count = max(1, int(math.ceil(action_count * keep_ratio)))
    if keep_count >= action_count:
        keep_count = action_count - 1
    return keep_count


def visited_actions(stats: List[ShotActionStats]) -> List[int]:
    return [a for a, action_stats in enumerate(stats) if action_stats.visits > 0]

