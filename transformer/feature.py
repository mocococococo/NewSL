"""Transformer 用の入力特徴量生成。

CNN 版 `nn/feature.py::generate_input_planes()` と同じ raw state を受け取り、
Transformer 版 network に渡すための単一局面特徴を返す。
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from board.constant import (
    STONE_RADIUS,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    Y_TEE,
    R_HOUSE,
)
from transformer.params import (
    END_NORM_MAX,
    GAME_FEAT_DIM,
    MAX_STONES,
    SCORE_DIFF_CLIP,
    SHOT_NORM_MAX,
    STONE_FEAT_DIM,
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _shot_team(shot: int, hammer: int) -> int:
    """CNN 版と同じ規則で、現在投げるチームを求める。"""

    if hammer not in (0, 1):
        raise ValueError(f"hammer must be 0 or 1, got {hammer}")
    return hammer if (shot % 2) == 1 else 1 - hammer


def _stone_team(stone_index: int) -> int:
    """raw stones の index から team0/team1 を求める。"""

    return 0 if stone_index < 8 else 1


def _stone_position(stone: dict) -> Tuple[float, float]:
    """CNN 版と同じ stone['position']['x/y'] 形式から座標を取り出す。"""

    position = stone["position"]
    return float(position["x"]), float(position["y"])


def _distance_to_tee(x: float, y: float) -> float:
    return math.sqrt(x * x + (y - Y_TEE) * (y - Y_TEE))


def _is_house_from_distance(distance: float) -> bool:
    return distance <= (R_HOUSE + STONE_RADIUS)


def _make_game_feature(
    end: int,
    shot: int,
    hammer: int,
    score_diff_for_team0: int,
) -> np.ndarray:
    """game token 用の 4 次元特徴を作る。"""

    shot_team = _shot_team(shot, hammer)
    has_hammer = 1.0 if shot_team == hammer else 0.0

    if shot_team == 0:
        score_diff_for_current_team = float(score_diff_for_team0)
    else:
        score_diff_for_current_team = -float(score_diff_for_team0)

    end_norm = _clamp(float(end), 0.0, END_NORM_MAX) / END_NORM_MAX
    shot_norm = _clamp(float(shot), 0.0, SHOT_NORM_MAX) / SHOT_NORM_MAX
    score_diff_norm = (
        _clamp(score_diff_for_current_team, -SCORE_DIFF_CLIP, SCORE_DIFF_CLIP)
        / SCORE_DIFF_CLIP
    )

    return np.array(
        [end_norm, shot_norm, has_hammer, score_diff_norm],
        dtype=np.float32,
    )


def generate_input_features(
    stones: List[Optional[dict]],
    end: int,
    shot: int,
    hammer: int,
    score_diff_for_team0: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """単一局面から Transformer 用入力特徴を作る。

    戻り値:
        stones_feature: (16, 5) float32
        game_feature: (4,) float32
        stone_mask: (16,) bool。False が有効石、True が padding。
    """

    if len(stones) != MAX_STONES:
        raise ValueError(f"stones must have length {MAX_STONES}, got {len(stones)}")

    shot_team = _shot_team(shot, hammer)
    coord_scale = Y_MAX - Y_MIN

    valid_stones = []
    for stone_index, stone in enumerate(stones):
        if stone is None:
            continue

        x_raw, y_raw = _stone_position(stone)
        x = _clamp(x_raw, X_MIN, X_MAX)
        y = _clamp(y_raw, Y_MIN, Y_MAX)

        distance = _distance_to_tee(x, y)
        stone_team = _stone_team(stone_index)

        x_feature = x / coord_scale
        y_feature = (y - Y_MIN) / coord_scale
        is_own_stone = 1.0 if stone_team == shot_team else 0.0
        dist_feature = distance / coord_scale
        in_house = 1.0 if _is_house_from_distance(distance) else 0.0

        feature = np.array(
            [
                x_feature,
                y_feature,
                is_own_stone,
                dist_feature,
                in_house,
            ],
            dtype=np.float32,
        )
        valid_stones.append((distance, feature))

    valid_stones.sort(key=lambda item: item[0])

    stones_feature = np.zeros((MAX_STONES, STONE_FEAT_DIM), dtype=np.float32)
    stone_mask = np.ones((MAX_STONES,), dtype=np.bool_)

    for output_index, (_, feature) in enumerate(valid_stones[:MAX_STONES]):
        stones_feature[output_index] = feature
        stone_mask[output_index] = False

    game_feature = _make_game_feature(
        end=end,
        shot=shot,
        hammer=hammer,
        score_diff_for_team0=score_diff_for_team0,
    )

    return stones_feature, game_feature, stone_mask


__all__ = [
    "MAX_STONES",
    "STONE_FEAT_DIM",
    "GAME_FEAT_DIM",
    "generate_input_features",
]
