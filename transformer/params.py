"""Transformer ネットワーク用のハイパーパラメータ定義。

ネットワーク本体では値を直接持たず、このファイルの
TransformerNetworkConfig を import して使う。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from board.constant import VX_SIZE, VY_SIZE


ActivationName = Literal["relu", "gelu"]


@dataclass(frozen=True)
class TransformerNetworkConfig:
    """TransformerNetwork の設定値。

    別の学習スクリプトや実験用パラメータファイルから、この config を作って
    TransformerNetwork に渡す想定。
    """

    # 入出力特徴量
    stone_feat_dim: int = 5
    game_feat_dim: int = 3
    action_dim: int = 2 * VX_SIZE * VY_SIZE
    value_dim: int = 17
    max_stones: int = 16

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
    "TransformerNetworkConfig",
    "DEFAULT_TRANSFORMER_CONFIG",
    "make_transformer_config",
]
