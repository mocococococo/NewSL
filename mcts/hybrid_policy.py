# hybrid_policy.py
from __future__ import annotations

from typing import List, Set, Tuple

from nn.network.dual_net import DualNet
from transformer.network import TransformerNetwork

from . import policy as cnn_policy
from . import transformer_policy
from .state import State


_USE_TRANSFORMER = False
_TRANSFORMER_TARGET_END = (9,)
_TRANSFORMER_TARGET_SHOT = (15,)
_LOGGED_END_SHOTS: Set[Tuple[int, int]] = set()


def set_policy_context(
    dual_net: DualNet,
    score_diff: int,
    transformer_net: TransformerNetwork | None = None,
    use_transformer: bool = False,
    transformer_target_end: Tuple[int, ...] = (9, 10),  # transformerのターゲットとするエンド（複数指定可）
    transformer_target_shot: Tuple[int, ...] = (15,),  # transformerのターゲットとするショット（複数指定可）
) -> None:
    """CNN / Transformer の推論 context をまとめてセットする。"""

    global _USE_TRANSFORMER, _TRANSFORMER_TARGET_END, _TRANSFORMER_TARGET_SHOT

    if use_transformer and transformer_net is None:
        raise RuntimeError(
            "transformer_net must be provided when use_transformer is True."
        )

    cnn_policy.set_policy_context(dual_net, score_diff)
    transformer_policy.set_policy_context(transformer_net, score_diff)
    _USE_TRANSFORMER = use_transformer
    _TRANSFORMER_TARGET_END = transformer_target_end
    _TRANSFORMER_TARGET_SHOT = transformer_target_shot


def _should_use_transformer(state: State) -> bool:
    """この局面で Transformer を使うかどうかを返す。"""

    return (
        _USE_TRANSFORMER
        and state.end in _TRANSFORMER_TARGET_END
        and state.shot_index in _TRANSFORMER_TARGET_SHOT
    )


def reset_policy_selection_log() -> None:
    """mcts_search ごとにネットワーク選択ログをリセットする。"""

    _LOGGED_END_SHOTS.clear()


def _log_selected_policy_once(state: State, selected_policy: str) -> None:
    """同じ end / shot に対するログを 1 回だけ出す。"""

    key = (state.end, state.shot_index)
    if key in _LOGGED_END_SHOTS:
        return

    _LOGGED_END_SHOTS.add(key)
    print(
        f"[HYBRID] Using {selected_policy} "
        f"(end={state.end}, shot={state.shot_index})"
    )


def get_policy_and_value(state: State) -> Tuple[List[float], List[float]]:
    """局面に応じて CNN / Transformer の推論を切り替える。"""

    if _should_use_transformer(state):
        _log_selected_policy_once(state, "Transformer")
        return transformer_policy.get_policy_and_value(state)
    _log_selected_policy_once(state, "CNN")
    return cnn_policy.get_policy_and_value(state)


def get_policy(state: State) -> List[float]:
    """局面に応じて CNN / Transformer の policy 推論を切り替える。"""

    if _should_use_transformer(state):
        _log_selected_policy_once(state, "Transformer")
        return transformer_policy.get_policy(state)
    _log_selected_policy_once(state, "CNN")
    return cnn_policy.get_policy(state)
