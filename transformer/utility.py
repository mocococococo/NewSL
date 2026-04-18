"""Transformer 学習用のユーティリティ。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from transformer.params import (
    DEFAULT_TRANSFORMER_CONFIG,
    GAME_FEAT_DIM,
    MAX_STONES,
    STONE_FEAT_DIM,
)


TransformerDataSet = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _calculate_losses(
    loss: dict[str, float],
    iteration: int,
) -> Tuple[float, float, float]:
    """各 loss の平均値を計算する。"""

    return (
        loss["loss"] / iteration,
        loss["policy"] / iteration,
        loss["value"] / iteration,
    )


def print_learning_process(
    loss_data: dict[str, float],
    epoch: int,
    index: int,
    iteration: int,
    start_time: float,
) -> None:
    """学習中の loss 情報を表示する。"""

    loss, policy_loss, value_loss = _calculate_losses(loss_data, iteration)
    training_time = time.time() - start_time

    print(f"epoch {epoch}, data-{index} : loss = {loss:6f}, time = {training_time:3f} seconds.")
    print(f"\tpolicy loss : {policy_loss:6f}")
    print(f"\tvalue loss  : {value_loss:6f}")


def print_evaluation_information(
    loss_data: dict[str, float],
    epoch: int,
    iteration: int,
    start_time: float,
) -> None:
    """評価用データの loss 情報を表示する。"""

    loss, policy_loss, value_loss = _calculate_losses(loss_data, iteration)
    testing_time = time.time() - start_time

    print(f"Test {epoch} : loss = {loss:6f}, time = {testing_time:3f} seconds.")
    print(f"\tpolicy loss : {policy_loss:6f}")
    print(f"\tvalue loss  : {value_loss:6f}")


def get_torch_device(use_gpu: bool) -> torch.device:
    """学習に使う torch.device を返す。"""

    if use_gpu:
        if not torch.cuda.is_available():
            print("CUDA が使えないため CPU で学習します。")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cpu")


def load_transformer_data_set(path: str | Path) -> TransformerDataSet:
    """Transformer 用 npz を読み込み、サンプル順をシャッフルして返す。

    戻り値:
        stones:      (N, 16, 5) float32
        games:       (N, 4) float32
        stone_masks: (N, 16) bool
        policy:      (N, action_dim) float32
        value:       (N, value_dim) float32
    """

    with np.load(path) as data:
        required_keys = {"stones", "games", "stone_masks", "policy", "value"}
        missing_keys = required_keys.difference(data.files)
        if missing_keys:
            raise KeyError(f"{path} is missing keys: {sorted(missing_keys)}")

        sample_count = len(data["value"])
        permutation = np.random.permutation(sample_count)

        stones = np.asarray(data["stones"][permutation], dtype=np.float32)
        games = np.asarray(data["games"][permutation], dtype=np.float32)
        stone_masks = np.asarray(data["stone_masks"][permutation], dtype=np.bool_)
        policy = np.asarray(data["policy"][permutation], dtype=np.float32)
        value = np.asarray(data["value"][permutation], dtype=np.float32)

    _validate_transformer_data_set(stones, games, stone_masks, policy, value, path)
    return stones, games, stone_masks, policy, value


def _validate_transformer_data_set(
    stones: np.ndarray,
    games: np.ndarray,
    stone_masks: np.ndarray,
    policy: np.ndarray,
    value: np.ndarray,
    path: str | Path,
) -> None:
    """npz 内の shape と target 分布を検証する。"""

    sample_count = len(value)
    expected_action_dim = DEFAULT_TRANSFORMER_CONFIG.action_dim
    expected_value_dim = DEFAULT_TRANSFORMER_CONFIG.value_dim

    expected_shapes = {
        "stones": (sample_count, MAX_STONES, STONE_FEAT_DIM),
        "games": (sample_count, GAME_FEAT_DIM),
        "stone_masks": (sample_count, MAX_STONES),
        "policy": (sample_count, expected_action_dim),
        "value": (sample_count, expected_value_dim),
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
                f"{path}: {name} must have shape {expected_shape}, "
                f"got {actual_shapes[name]}"
            )

    for name, target in (("policy", policy), ("value", value)):
        if not np.all(np.isfinite(target)):
            raise ValueError(f"{path}: {name} contains non-finite values")
        if np.any(target < 0):
            raise ValueError(f"{path}: {name} contains negative values")
        if not np.allclose(target.sum(axis=1), 1.0, atol=1e-4, rtol=1e-4):
            raise ValueError(f"{path}: {name} rows must sum to 1.0")


def split_train_test_set(
    file_list: List[str],
    train_data_ratio: float,
) -> Tuple[List[str], List[str]]:
    """npz ファイル一覧を train/test に分ける。"""

    if not file_list:
        raise ValueError("file_list is empty")
    if not (0.0 < train_data_ratio <= 1.0):
        raise ValueError(f"train_data_ratio must be in (0, 1], got {train_data_ratio}")

    split_index = int(len(file_list) * train_data_ratio)
    if len(file_list) > 1:
        split_index = min(max(split_index, 1), len(file_list) - 1)
    else:
        split_index = 1

    train_data_set = file_list[:split_index]
    test_data_set = file_list[split_index:]

    print(f"Training data set : {train_data_set}")
    print(f"Testing data set  : {test_data_set}")
    return train_data_set, test_data_set


def save_model(network: torch.nn.Module, path: str | Path) -> None:
    """モデルの state_dict を保存する。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), path)


def make_loss_history_path(program_dir: str | Path, model_name: str) -> Path:
    """loss history を保存する JSON ファイルパスを作る。"""

    record_dir = Path(program_dir) / "record"
    record_dir.mkdir(parents=True, exist_ok=True)
    path = record_dir / f"{model_name}.json"
    if not path.exists():
        path.write_text("{}", encoding="utf-8")
    return path


def save_loss_history(loss_history: dict[str, list[float]], path: str | Path) -> None:
    """loss history を JSON として保存する。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(loss_history, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )


__all__ = [
    "TransformerDataSet",
    "get_torch_device",
    "load_transformer_data_set",
    "print_learning_process",
    "print_evaluation_information",
    "split_train_test_set",
    "save_model",
    "make_loss_history_path",
    "save_loss_history",
]
