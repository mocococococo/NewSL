"""Transformer ネットワーク用のハイパーパラメータ定義。

`network.py` と `feature.py` が同じ値を参照することで、入力特徴量の
shape とネットワークの期待 shape がずれないようにする。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from board.constant import VX_SIZE, VY_SIZE


ActivationName = Literal["relu", "gelu"]


MAX_STONES = 16
STONE_FEAT_DIM = 5
GAME_FEAT_DIM = 4
END_NORM_MAX = 9.0
SHOT_NORM_MAX = 15.0
SCORE_DIFF_CLIP = 8.0


@dataclass(frozen=True)
class TransformerNetworkConfig:
    """TransformerNetwork の設定値。"""

    # 入出力特徴量
    stone_feat_dim: int = STONE_FEAT_DIM
    game_feat_dim: int = GAME_FEAT_DIM
    action_dim: int = 2 * VX_SIZE * VY_SIZE
    value_dim: int = 17
    max_stones: int = MAX_STONES

    # Transformer 本体
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    activation: ActivationName = "gelu"

    # embedding と head
    embedding_hidden_dim: int | None = None
    head_hidden_dim: int | None = None

    # 石 token は集合に近い扱いにするため、位置埋め込みは初期状態では使わない。
    use_type_embedding: bool = True
    use_positional_embedding: bool = False


DEFAULT_TRANSFORMER_CONFIG = TransformerNetworkConfig()


def make_transformer_config(**overrides: object) -> TransformerNetworkConfig:
    """デフォルト設定の一部だけを上書きして config を作る。"""

    return replace(DEFAULT_TRANSFORMER_CONFIG, **overrides)


__all__ = [
    "ActivationName",
    "MAX_STONES",
    "STONE_FEAT_DIM",
    "GAME_FEAT_DIM",
    "END_NORM_MAX",
    "SHOT_NORM_MAX",
    "SCORE_DIFF_CLIP",
    "TransformerNetworkConfig",
    "DEFAULT_TRANSFORMER_CONFIG",
    "make_transformer_config",
]
