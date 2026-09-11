# transformer_policy.py
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from transformer.feature import generate_input_features
from transformer.network import TransformerNetwork
from .state import State

_TRANSFORMER_NET: TransformerNetwork | None = None
_SCORE_DIFF: int | None = None


def set_policy_context(transformer_net: TransformerNetwork, score_diff: int) -> None:
    """探索で使う Transformer ネットワークと score_diff をセットする。"""

    global _TRANSFORMER_NET, _SCORE_DIFF
    _TRANSFORMER_NET = transformer_net
    _SCORE_DIFF = score_diff


def _state_stones_to_feature_stones(state: State) -> List[Optional[dict]]:
    """State.stones (x, y or None) を feature.py 用の stones 形式へ変換する。"""

    out: List[Optional[dict]] = []
    for stone in state.stones:
        if stone is None:
            out.append(None)
        else:
            x, y = float(stone[0]), float(stone[1])
            out.append({"position": {"x": x, "y": y}})
    return out


def _get_network_device(transformer_net: TransformerNetwork) -> torch.device:
    """ネットワークが載っている device を取得する。"""

    try:
        return next(transformer_net.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _prepare_inputs(
    state: State, *, to_device: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """両 head 推論と value 専用推論で同じ入力を使う。"""

    if _TRANSFORMER_NET is None:
        raise RuntimeError("transformer_net is not set. Call set_policy_context() first.")
    if _SCORE_DIFF is None:
        raise RuntimeError("score_diff is not set. Call set_policy_context() first.")

    stones_for_feature = _state_stones_to_feature_stones(state)
    stones_feature, game_feature, stone_mask = generate_input_features(
        stones=stones_for_feature,
        end=state.end,
        shot=state.shot_index,
        hammer=state.hammer_team,
        score_diff_for_team0=_SCORE_DIFF,
    )

    stones_tensor = torch.tensor(stones_feature, dtype=torch.float32).unsqueeze(0)
    game_tensor = torch.tensor(game_feature, dtype=torch.float32).unsqueeze(0)
    stone_mask_tensor = torch.tensor(stone_mask, dtype=torch.bool).unsqueeze(0)

    if not to_device:
        return stones_tensor, game_tensor, stone_mask_tensor
    device = _get_network_device(_TRANSFORMER_NET)
    return stones_tensor.to(device), game_tensor.to(device), stone_mask_tensor.to(device)


def get_policy_and_value(state: State) -> Tuple[List[float], List[float]]:
    """State から Transformer を使って policy / value 分布を得る。"""

    stones_tensor, game_tensor, stone_mask_tensor = _prepare_inputs(state)
    _TRANSFORMER_NET.eval()
    with torch.no_grad():
        policy_t, value_t = _TRANSFORMER_NET.inference(
            stones_tensor,
            game_tensor,
            stone_mask_tensor,
        )
        policy = policy_t.squeeze(0).detach().cpu().tolist()
        value = value_t.squeeze(0).detach().cpu().tolist()

    if len(policy) != _TRANSFORMER_NET.config.action_dim:
        raise RuntimeError(
            "policy length mismatch: "
            f"{len(policy)} != {_TRANSFORMER_NET.config.action_dim}"
        )
    if len(value) != _TRANSFORMER_NET.config.value_dim:
        raise RuntimeError(
            "value length mismatch: "
            f"{len(value)} != {_TRANSFORMER_NET.config.value_dim}"
        )

    return policy, value


def get_value_probs(state: State) -> List[float]:
    """policy の計算・転送を省き、17クラスの value 分布を返す。"""

    stones_tensor, game_tensor, stone_mask_tensor = _prepare_inputs(state)
    _TRANSFORMER_NET.eval()
    with torch.no_grad():
        value_t = _TRANSFORMER_NET.inference_value(
            stones_tensor, game_tensor, stone_mask_tensor,
        )
        value = value_t.squeeze(0).detach().cpu().tolist()

    if len(value) != _TRANSFORMER_NET.config.value_dim:
        raise RuntimeError(
            "value length mismatch: "
            f"{len(value)} != {_TRANSFORMER_NET.config.value_dim}"
        )
    return value


def get_value_probs_batch(states: List[State]) -> List[List[float]]:
    """入力をCPU上でまとめ、1回の推論・転送で各局面のvalue分布を返す。"""
    if not states:
        return []
    if len(states) == 1:
        return [get_value_probs(states[0])]

    inputs = [_prepare_inputs(state, to_device=False) for state in states]
    device = _get_network_device(_TRANSFORMER_NET)
    stones, games, masks = (
        torch.cat([sample[i] for sample in inputs], dim=0).to(device)
        for i in range(3)
    )
    _TRANSFORMER_NET.eval()
    with torch.no_grad():
        values = _TRANSFORMER_NET.inference_value(stones, games, masks)
        if tuple(values.shape) != (len(states), _TRANSFORMER_NET.config.value_dim):
            raise RuntimeError(f"value batch shape mismatch: {tuple(values.shape)}")
        return values.detach().cpu().tolist()


def get_policy(state: State) -> List[float]:
    """State から policy 分布だけを返す。"""

    policy, _ = get_policy_and_value(state)
    return policy
