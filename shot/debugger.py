from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from mcts.simulate import decode_action

from .candidates import rank_actions
from .node import ShotActionStats


def emit_lines(lines: List[str], log_path: Optional[str]) -> None:
    if not log_path:
        return
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def format_topk_action_stats(
    actions: List[int],
    stats: List[ShotActionStats],
    policy: List[float],
    k: int,
) -> str:
    parts = []
    for a in rank_actions(actions, stats, policy)[:k]:
        vx, vy, sp = decode_action(a)
        parts.append(
            f"a={a} N={stats[a].visits} Q={stats[a].mean_value:.4g} "
            f"P={policy[a]:.4g} -> (vx={vx:.4g}, vy={vy:.4g}, sp={sp})"
        )
    return " | ".join(parts)
