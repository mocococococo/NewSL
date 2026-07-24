"""Transformer ネットワーク用のハイパーパラメータ定義。

`network.py` と `feature.py` が同じ値を参照することで、入力特徴量の
shape とネットワークの期待 shape がずれないようにする。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from board.constant import (
    TRANSFORMER_HIGH_RESOLUTION_VY_EXTRA_SIZE,
    TRANSFORMER_HIGH_RESOLUTION_VY_SHEET_SIZE,
    TRANSFORMER_HIGH_RESOLUTION_VY_SIZE,
    VX_SIZE,
    VY_EXTRA_SIZE,
    VY_SHEET_SIZE,
    VY_SIZE,
)


ActivationName = Literal["relu", "gelu"]
TransformerVyMode = Literal["default", "high_resolution"]


# Transformerで使用するY方向の分割構成を指定する。
TRANSFORMER_VY_MODE: TransformerVyMode = "default"

if TRANSFORMER_VY_MODE == "default":
    TRANSFORMER_VY_SHEET_SIZE = VY_SHEET_SIZE
    TRANSFORMER_VY_EXTRA_SIZE = VY_EXTRA_SIZE
    TRANSFORMER_VY_SIZE = VY_SIZE
    TRANSFORMER_SUPERVISED_DATA_DIRECTORY = "supervised"
    TRANSFORMER_MODEL_NAME_SUFFIX = ""
elif TRANSFORMER_VY_MODE == "high_resolution":
    TRANSFORMER_VY_SHEET_SIZE = TRANSFORMER_HIGH_RESOLUTION_VY_SHEET_SIZE
    TRANSFORMER_VY_EXTRA_SIZE = TRANSFORMER_HIGH_RESOLUTION_VY_EXTRA_SIZE
    TRANSFORMER_VY_SIZE = TRANSFORMER_HIGH_RESOLUTION_VY_SIZE
    TRANSFORMER_SUPERVISED_DATA_DIRECTORY = (
        f"supervised_vy{TRANSFORMER_VY_SIZE}"
    )
    TRANSFORMER_MODEL_NAME_SUFFIX = f"-vy{TRANSFORMER_VY_SIZE}"
else:
    raise ValueError(
        f"未対応のTransformer Y分割構成です: {TRANSFORMER_VY_MODE}"
    )

TRANSFORMER_ACTION_DIM = 2 * VX_SIZE * TRANSFORMER_VY_SIZE


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
    action_dim: int = TRANSFORMER_ACTION_DIM
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
    "TransformerVyMode",
    "TRANSFORMER_VY_MODE",
    "TRANSFORMER_VY_SHEET_SIZE",
    "TRANSFORMER_VY_EXTRA_SIZE",
    "TRANSFORMER_VY_SIZE",
    "TRANSFORMER_ACTION_DIM",
    "TRANSFORMER_SUPERVISED_DATA_DIRECTORY",
    "TRANSFORMER_MODEL_NAME_SUFFIX",
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
