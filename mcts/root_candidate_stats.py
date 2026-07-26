from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from transformer.params import TRANSFORMER_VY_MODE, TransformerVyMode

from . import fast_simulator
from .node import Node
from .simulate import decode_action

RootCandidateStat = Dict[str, Any]


@lru_cache(maxsize=None)
def _nominal_target_for_action(
    action_id: int,
    action_type: TransformerVyMode,
) -> Optional[Tuple[float, float]]:
    vx, vy, spin = decode_action(action_id, action_type=action_type)
    x, y = fast_simulator.shot2dest((vx, vy, spin))
    if y <= 0:
        return None
    return float(x), float(y)


def build_root_candidate_stats(
    root: Node,
    best_action_id: int,
    action_type: TransformerVyMode = TRANSFORMER_VY_MODE,
) -> List[RootCandidateStat]:
    assert root.P is not None and root.Q is not None and root.Nsa is not None

    active_actions = set(root.actions)
    stats: List[RootCandidateStat] = []

    for action_id, visit_count in enumerate(root.Nsa):
        if visit_count <= 0:
            continue

        vx, vy, spin = decode_action(action_id, action_type=action_type)
        target = _nominal_target_for_action(
            action_id,
            action_type,
        )
        target_x = None if target is None else float(target[0])
        target_y = None if target is None else float(target[1])

        stats.append(
            {
                "action_id": int(action_id),
                "visit_count": int(visit_count),
                "q": float(root.Q[action_id]),
                "prior": float(root.P[action_id]),
                "vx": float(vx),
                "vy": float(vy),
                "spin": int(spin),
                "target_x": target_x,
                "target_y": target_y,
                "is_best": int(action_id) == int(best_action_id),
                "is_active": action_id in active_actions,
            }
        )

    stats.sort(
        key=lambda stat: (
            int(stat["visit_count"]),
            float(stat["q"]),
            float(stat["prior"]),
        ),
        reverse=True,
    )
    return stats
