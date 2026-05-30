from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ShotActionStats:
    visits: int = 0
    value_sum: float = 0.0
    value_dist_sum: List[float] = field(default_factory=lambda: [0.0] * 17)

    @property
    def mean_value(self) -> float:
        if self.visits <= 0:
            return float("-inf")
        return self.value_sum / self.visits

    def update(self, value: float, value_distribution: Optional[List[float]] = None) -> None:
        self.visits += 1
        self.value_sum += float(value)

        if value_distribution is None:
            return
        if len(value_distribution) != len(self.value_dist_sum):
            raise ValueError(
                f"value_distribution length mismatch: "
                f"{len(value_distribution)} != {len(self.value_dist_sum)}"
            )
        for i, p in enumerate(value_distribution):
            self.value_dist_sum[i] += float(p)

    def value_distribution(self) -> List[float]:
        total = sum(self.value_dist_sum)
        if total <= 0.0:
            return [0.0] * len(self.value_dist_sum)
        return [p / total for p in self.value_dist_sum]


def make_action_stats(size: int) -> List[ShotActionStats]:
    return [ShotActionStats() for _ in range(size)]

