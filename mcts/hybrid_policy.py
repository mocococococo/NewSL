# hybrid_policy.py
from __future__ import annotations

from typing import List, Tuple

from nn.network.dual_net import DualNet
from transformer.network import TransformerNetwork

from . import policy as cnn_policy
from . import transformer_policy
from .state import State


_USE_TRANSFORMER_FOR_SHOT15 = False


def set_policy_context(
    dual_net: DualNet,
    score_diff: int,
    transformer_net: TransformerNetwork | None = None,
    use_transformer_for_shot15: bool = False,
) -> None:
    """CNN / Transformer の推論 context をまとめてセットする。"""

    global _USE_TRANSFORMER_FOR_SHOT15

    if use_transformer_for_shot15 and transformer_net is None:
        raise RuntimeError(
            "transformer_net must be provided when use_transformer_for_shot15 is True."
        )

    cnn_policy.set_policy_context(dual_net, score_diff)
    transformer_policy.set_policy_context(transformer_net, score_diff)
    _USE_TRANSFORMER_FOR_SHOT15 = use_transformer_for_shot15


def _should_use_transformer(state: State) -> bool:
    """この局面で Transformer を使うかどうかを返す。"""

    return _USE_TRANSFORMER_FOR_SHOT15 and state.shot_index == 15


def get_policy_and_value(state: State) -> Tuple[List[float], List[float]]:
    """局面に応じて CNN / Transformer の推論を切り替える。"""

    if _should_use_transformer(state):
        return transformer_policy.get_policy_and_value(state)
    return cnn_policy.get_policy_and_value(state)


def get_policy(state: State) -> List[float]:
    """局面に応じて CNN / Transformer の policy 推論を切り替える。"""

    if _should_use_transformer(state):
        return transformer_policy.get_policy(state)
    return cnn_policy.get_policy(state)
