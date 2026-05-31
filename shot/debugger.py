from __future__ import annotations

from typing import Any

from mcts.debugger import Debugger, format_topk_policy, policy_stats, summarize_stones


def format_topk_root_shot(root: Any, k: int, decode_action) -> str:
    """rootのSHOT候補を Q/Nsa/P 順で表示する。"""
    if getattr(root, "Nsa", None) is None:
        return ""

    actions = list(getattr(root, "actions", []))
    actions = sorted(
        actions,
        key=lambda a: (
            root.Nsa[a] > 0,
            root.Q[a] if root.Nsa[a] > 0 else float("-inf"),
            root.Nsa[a],
            root.P[a],
        ),
        reverse=True,
    )[:k]

    parts = []
    for a in actions:
        vx, vy, sp = decode_action(a)
        parts.append(
            f"a={a} Nsa={root.Nsa[a]} Q={root.Q[a]:.4g} "
            f"P={root.P[a]:.4g} -> (vx={vx:.4g}, vy={vy:.4g}, sp={sp})"
        )
    return " | ".join(parts)
