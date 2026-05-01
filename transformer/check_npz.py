"""Transformer 用 npz 学習データの簡易チェック。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from transformer.params import DEFAULT_TRANSFORMER_CONFIG


def _print_array_summary(name: str, array: np.ndarray) -> None:
    """配列の shape, dtype, min, max を表示する。"""

    print(
        f"{name:11}: shape={array.shape}, dtype={array.dtype}, "
        f"min={array.min()}, max={array.max()}"
    )


def _check_required_keys(data: np.lib.npyio.NpzFile) -> None:
    """新形式 npz に必要なキーが揃っているか確認する。"""

    required_keys = {"stones", "games", "stone_masks", "policy", "value", "log_count"}
    missing_keys = required_keys.difference(data.files)
    if missing_keys:
        raise KeyError(f"missing keys: {sorted(missing_keys)}")


def _check_shapes(
    stones: np.ndarray,
    games: np.ndarray,
    stone_masks: np.ndarray,
    policy: np.ndarray,
    value: np.ndarray,
) -> None:
    """Transformer 入力と target の shape を確認する。"""

    sample_count = len(value)
    config = DEFAULT_TRANSFORMER_CONFIG
    expected_shapes = {
        "stones": (sample_count, config.max_stones, config.stone_feat_dim),
        "games": (sample_count, config.game_feat_dim),
        "stone_masks": (sample_count, config.max_stones),
        "policy": (sample_count, config.action_dim),
        "value": (sample_count, config.value_dim),
    }
    actual_shapes = {
        "stones": stones.shape,
        "games": games.shape,
        "stone_masks": stone_masks.shape,
        "policy": policy.shape,
        "value": value.shape,
    }

    for name, expected_shape in expected_shapes.items():
        if actual_shapes[name] != expected_shape:
            raise ValueError(
                f"{name} shape mismatch: expected {expected_shape}, got {actual_shapes[name]}"
            )


def _print_target_checks(name: str, target: np.ndarray) -> None:
    """target 分布の合計と不正行を表示する。"""

    target_sums = target.sum(axis=1)
    bad_rows = np.where(
        (~np.isfinite(target_sums))
        | (target_sums <= 0)
        | (~np.isclose(target_sums, 1.0, atol=1e-4, rtol=1e-4))
    )[0]

    print(f"{name} sums:", target_sums[:10])
    print(f"bad {name} rows:", bad_rows[:20], "count=", len(bad_rows))


def main(path: str | Path | None = None) -> None:
    """npz ファイルの中身を確認する。"""

    if path is None:
        path = ROOT_DIR / "data" / "sl_data_0.npz"
    path = Path(path)

    with np.load(path) as data:
        print("file:", path)
        print("keys:", data.files)
        _check_required_keys(data)

        stones = data["stones"]
        games = data["games"]
        stone_masks = data["stone_masks"]
        policy = data["policy"]
        value = data["value"]

        _check_shapes(stones, games, stone_masks, policy, value)

        _print_array_summary("stones", stones)
        _print_array_summary("games", games)
        _print_array_summary("stone_masks", stone_masks)
        _print_array_summary("policy", policy)
        _print_array_summary("value", value)
        print("log_count :", data["log_count"])

        print("valid stone counts:", (~stone_masks.astype(bool)).sum(axis=1)[:10])
        print("padding counts    :", stone_masks.astype(bool).sum(axis=1)[:10])

        _print_target_checks("policy", policy)
        _print_target_checks("value", value)

        for i in range(min(3, len(policy))):
            top = np.argsort(policy[i])[-10:][::-1]
            valid_indices = np.where(~stone_masks[i].astype(bool))[0]
            print(f"\nsample {i}")
            print("valid stone indices:", valid_indices)
            print("stones valid rows  :", stones[i][valid_indices])
            print("game feature       :", games[i])
            print("policy top actions :", top)
            print("policy top probs   :", policy[i][top])
            print("value distribution :", value[i])
            print("value classes      :", np.nonzero(value[i])[0] - 8)


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) >= 2 else None
    main(input_path)
