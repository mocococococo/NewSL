from __future__ import annotations

import math
from typing import List

from .node import N_ACTIONS


def top_k_actions(pi: List[float], k: int) -> List[int]:
    if len(pi) != N_ACTIONS:
        raise RuntimeError(f"policy length mismatch: {len(pi)} != {N_ACTIONS}")
    k = max(1, min(int(k), len(pi)))
    return sorted(range(len(pi)), key=lambda a: pi[a], reverse=True)[:k]


def ceil_log2(n: int) -> int:
    if n <= 1:
        return 1
    return int(math.ceil(math.log2(n)))


def keep_count_after_halving(action_count: int, keep_ratio: float) -> int:
    keep_count = max(1, int(math.ceil(action_count * keep_ratio)))
    if keep_count >= action_count:
        keep_count = action_count - 1
    return keep_count
