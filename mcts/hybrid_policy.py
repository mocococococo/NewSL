# hybrid_policy.py
from __future__ import annotations

from typing import Dict, List, Set, Tuple

from board.constant import VX_SIZE, VY_SIZE
from nn.network.dual_net import DualNet
from transformer.network import TransformerNetwork
from transformer.params import (
    TRANSFORMER_VY_MODE,
    TransformerVyMode,
    get_transformer_action_dim,
)

from . import policy as cnn_policy
from . import transformer_policy
from .state import State


_SL_MODEL_IS_CNN = True
_SL_TRANSFORMER_MODEL: TransformerNetwork | None = None
_USE_SEARCH_BASED_MODEL = False
_SEARCH_BASED_TARGET_END = (9,)
_SEARCH_BASED_TARGET_SHOT = (15,)
_SEARCH_BASED_MODEL: TransformerNetwork | None = None
_SEARCH_BASED_MODEL_BY_SHOT: Dict[int, TransformerNetwork] = {}
_SCORE_DIFF = 0
_LOGGED_END_SHOTS: Set[Tuple[int, int]] = set()
_CNN_ACTION_DIM = 2 * VX_SIZE * VY_SIZE


def set_policy_context(
    sl_model: DualNet | TransformerNetwork,
    score_diff: int,
    sl_model_is_cnn: bool = True,
    search_based_model: TransformerNetwork | Dict[int, TransformerNetwork] | None = None,
    use_search_based_model: bool = False,
    transformer_target_end: Tuple[int, ...] = (9, 10),
    transformer_target_shot: Tuple[int, ...] = (15,),
) -> None:
    """教師ありモデルと探索統計学習モデルの推論 context をセットする。"""

    global _SL_MODEL_IS_CNN, _SL_TRANSFORMER_MODEL
    global _USE_SEARCH_BASED_MODEL
    global _SEARCH_BASED_TARGET_END, _SEARCH_BASED_TARGET_SHOT
    global _SEARCH_BASED_MODEL, _SEARCH_BASED_MODEL_BY_SHOT, _SCORE_DIFF

    if use_search_based_model and search_based_model is None:
        raise RuntimeError(
            "探索統計学習モデルを使用する場合はsearch_based_modelが必要です。"
        )

    _SL_MODEL_IS_CNN = sl_model_is_cnn
    if sl_model_is_cnn:
        cnn_policy.set_policy_context(sl_model, score_diff)
        _SL_TRANSFORMER_MODEL = None
    else:
        _SL_TRANSFORMER_MODEL = sl_model

    if isinstance(search_based_model, dict):
        _SEARCH_BASED_MODEL = None
        _SEARCH_BASED_MODEL_BY_SHOT = {
            int(k): v for k, v in search_based_model.items()
        }
    else:
        _SEARCH_BASED_MODEL = search_based_model
        _SEARCH_BASED_MODEL_BY_SHOT = {}

    _SCORE_DIFF = int(score_diff)
    _USE_SEARCH_BASED_MODEL = use_search_based_model
    _SEARCH_BASED_TARGET_END = transformer_target_end
    _SEARCH_BASED_TARGET_SHOT = transformer_target_shot


def _should_use_search_based_model(state: State) -> bool:
    """この局面で探索統計学習モデルを使うかどうかを返す。"""

    return (
        _USE_SEARCH_BASED_MODEL
        and state.end in _SEARCH_BASED_TARGET_END
        and state.shot_index in _SEARCH_BASED_TARGET_SHOT
        and (
            _SEARCH_BASED_MODEL is not None
            or state.shot_index in _SEARCH_BASED_MODEL_BY_SHOT
        )
    )


def _select_search_based_model(state: State) -> TransformerNetwork:
    if _SEARCH_BASED_MODEL_BY_SHOT:
        return _SEARCH_BASED_MODEL_BY_SHOT[state.shot_index]
    return _SEARCH_BASED_MODEL


def _use_transformer(model: TransformerNetwork) -> None:
    transformer_policy.set_policy_context(model, _SCORE_DIFF)


def _check_cnn_action_space(action_type: TransformerVyMode) -> None:
    action_dim = get_transformer_action_dim(action_type)
    if action_dim != _CNN_ACTION_DIM:
        raise RuntimeError(
            "指定された行動種類とCNNでは行動数が異なります: "
            f"{action_dim} != {_CNN_ACTION_DIM}"
        )


def _check_transformer_action_space(
    model: TransformerNetwork,
    action_type: TransformerVyMode,
) -> None:
    action_dim = get_transformer_action_dim(action_type)
    network_action_dim = model.config.action_dim
    if action_dim != network_action_dim:
        raise RuntimeError(
            "指定された行動種類とTransformerでは行動数が異なります: "
            f"{action_dim} != {network_action_dim}"
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


def get_policy_and_value(
    state: State,
    action_type: TransformerVyMode = TRANSFORMER_VY_MODE,
) -> Tuple[List[float], List[float]]:
    """局面に応じて教師ありモデルと探索統計学習モデルを切り替える。"""

    if _should_use_search_based_model(state):
        model = _select_search_based_model(state)
        _check_transformer_action_space(model, action_type)
        _log_selected_policy_once(state, "Search-based Transformer")
        _use_transformer(model)
        return transformer_policy.get_policy_and_value(state)

    if _SL_MODEL_IS_CNN:
        _check_cnn_action_space(action_type)
        _log_selected_policy_once(state, "Supervised CNN")
        return cnn_policy.get_policy_and_value(state)

    _check_transformer_action_space(_SL_TRANSFORMER_MODEL, action_type)
    _log_selected_policy_once(state, "Supervised Transformer")
    _use_transformer(_SL_TRANSFORMER_MODEL)
    return transformer_policy.get_policy_and_value(state)


def get_value_probs(
    state: State,
    action_type: TransformerVyMode = TRANSFORMER_VY_MODE,
) -> List[float]:
    """通常の推論と同じモデルを選び、末端評価用の value 分布だけを返す。"""

    if _should_use_search_based_model(state):
        model = _select_search_based_model(state)
        _check_transformer_action_space(model, action_type)
        _log_selected_policy_once(state, "Search-based Transformer")
        _use_transformer(model)
        return transformer_policy.get_value_probs(state)

    if _SL_MODEL_IS_CNN:
        _check_cnn_action_space(action_type)
        _log_selected_policy_once(state, "Supervised CNN")
        return cnn_policy.get_value_probs(state)

    _check_transformer_action_space(_SL_TRANSFORMER_MODEL, action_type)
    _log_selected_policy_once(state, "Supervised Transformer")
    _use_transformer(_SL_TRANSFORMER_MODEL)
    return transformer_policy.get_value_probs(state)


def get_value_probs_batch(
    states: List[State],
    action_type: TransformerVyMode = TRANSFORMER_VY_MODE,
) -> List[List[float]]:
    """同じ推論context内の局面をモデル別にまとめ、入力順にvalueを返す。"""
    if not states:
        return []
    if len(states) == 1:
        return [get_value_probs(states[0], action_type=action_type)]

    groups: Dict[TransformerNetwork | None, List[Tuple[int, State]]] = {}
    for index, state in enumerate(states):
        if _should_use_search_based_model(state):
            model = _select_search_based_model(state)
            _check_transformer_action_space(model, action_type)
            label = "Search-based Transformer"
        elif _SL_MODEL_IS_CNN:
            model = None
            _check_cnn_action_space(action_type)
            label = "Supervised CNN"
        else:
            model = _SL_TRANSFORMER_MODEL
            _check_transformer_action_space(model, action_type)
            label = "Supervised Transformer"
        _log_selected_policy_once(state, label)
        groups.setdefault(model, []).append((index, state))

    results: List[List[float]] = [[] for _ in states]
    for model, indexed_states in groups.items():
        batch = [state for _, state in indexed_states]
        if model is None:
            values = cnn_policy.get_value_probs_batch(batch)
        else:
            _use_transformer(model)
            values = transformer_policy.get_value_probs_batch(batch)
        for (index, _), value in zip(indexed_states, values):
            results[index] = value
    return results


def get_policy(
    state: State,
    action_type: TransformerVyMode = TRANSFORMER_VY_MODE,
) -> List[float]:
    """局面に応じて教師ありモデルと探索統計学習モデルを切り替える。"""

    if _should_use_search_based_model(state):
        model = _select_search_based_model(state)
        _check_transformer_action_space(model, action_type)
        _log_selected_policy_once(state, "Search-based Transformer")
        _use_transformer(model)
        return transformer_policy.get_policy(state)

    if _SL_MODEL_IS_CNN:
        _check_cnn_action_space(action_type)
        _log_selected_policy_once(state, "Supervised CNN")
        return cnn_policy.get_policy(state)

    _check_transformer_action_space(_SL_TRANSFORMER_MODEL, action_type)
    _log_selected_policy_once(state, "Supervised Transformer")
    _use_transformer(_SL_TRANSFORMER_MODEL)
    return transformer_policy.get_policy(state)
