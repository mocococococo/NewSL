"""
dcl2 の局面から shot16 の policy , value を学習するためのデータ生成のためのコード

1. 15投目まで終了時の dcl2 の局面を読み込む
2. 盤面を特徴平面に変換する
3. 16投目を選択するための探索を行う
"""
import numpy as np
import copy
import os
import sys
import json
import random
from pathlib import Path
from typing import List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.translate_state import convert_team_stoi, scores_to_scorediff_for_team0
from nn.utility import load_network, get_torch_device
from transformer.feature import generate_input_features
from transformer.params import (
    DEFAULT_TRANSFORMER_CONFIG,
    GAME_FEAT_DIM,
    MAX_STONES,
    STONE_FEAT_DIM,
)
from mcts.search import mcts_search, set_root_state
from board.constant import VX_SIZE, VY_SIZE
from learning_param import BATCH_SIZE, DATA_SET_SIZE

N_ACTIONS = DEFAULT_TRANSFORMER_CONFIG.action_dim
N_VALUE_CLASSES = DEFAULT_TRANSFORMER_CONFIG.value_dim


def _flip_policy_target(policy_target):
    policy = np.asarray(policy_target)
    if policy.size != N_ACTIONS:
        raise ValueError(f"policy_target size must be {N_ACTIONS}, got {policy.size}")

    policy_3d = policy.reshape(2, VY_SIZE, VX_SIZE)
    flipped = policy_3d[::-1, :, ::-1]
    return flipped.reshape(N_ACTIONS).copy()


def _normalize_distribution(target, expected_size: int, name: str) -> np.ndarray:
    """KLD 用に、MCTS のカウント列を合計 1.0 の確率分布へ変換する。"""

    distribution = np.asarray(target, dtype=np.float32)
    if distribution.shape != (expected_size,):
        raise ValueError(f"{name} must have shape ({expected_size},), got {distribution.shape}")
    if not np.all(np.isfinite(distribution)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(distribution < 0):
        raise ValueError(f"{name} contains negative values")

    total = float(distribution.sum())
    if total <= 0.0:
        raise ValueError(f"{name} sum must be positive, got {total}")

    return (distribution / total).astype(np.float32, copy=False)


def _mirror_stones_x(stones):
    """左右反転用に、raw stones の x 座標だけ符号反転する。"""

    mirrored_stones = []
    for stone in stones:
        if stone is None:
            mirrored_stones.append(None)
            continue

        mirrored_stone = copy.deepcopy(stone)
        try:
            mirrored_stone["position"]["x"] = -float(mirrored_stone["position"]["x"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("stone must have position.x for mirroring") from exc
        mirrored_stones.append(mirrored_stone)

    return mirrored_stones


def _save_data(
    save_file_path: str,
    stones_data: np.ndarray,
    games_data: np.ndarray,
    stone_masks_data: np.ndarray,
    policy_data: np.ndarray,
    value_data: np.ndarray,
    log_counter: int,
) -> None:
    """学習データをnpzファイルとして出力する。

    Args:
        save_file_path (str): 保存するファイルパス。
        stones_data (np.ndarray): stone token の特徴量。
        games_data (np.ndarray): game token の特徴量。
        stone_masks_data (np.ndarray): padding stone を True にする mask。
        policy_data (np.ndarray): Policyのデータ。
        value_data (np.ndarray): Valueのデータ
        log_counter (int): データセットにある棋譜データの個数。
    """
    Path(save_file_path).parent.mkdir(parents=True, exist_ok=True)

    stones = np.asarray(stones_data[0:DATA_SET_SIZE], dtype=np.float32)
    games = np.asarray(games_data[0:DATA_SET_SIZE], dtype=np.float32)
    stone_masks = np.asarray(stone_masks_data[0:DATA_SET_SIZE], dtype=np.bool_)
    policy = np.asarray(policy_data[0:DATA_SET_SIZE], dtype=np.float32)
    value = np.asarray(value_data[0:DATA_SET_SIZE], dtype=np.float32)

    if stones.ndim != 3 or stones.shape[1:] != (MAX_STONES, STONE_FEAT_DIM):
        raise ValueError(f"stones must have shape (N, {MAX_STONES}, {STONE_FEAT_DIM}), got {stones.shape}")
    if games.ndim != 2 or games.shape[1:] != (GAME_FEAT_DIM,):
        raise ValueError(f"games must have shape (N, {GAME_FEAT_DIM}), got {games.shape}")
    if stone_masks.ndim != 2 or stone_masks.shape[1:] != (MAX_STONES,):
        raise ValueError(f"stone_masks must have shape (N, {MAX_STONES}), got {stone_masks.shape}")
    if policy.ndim != 2 or policy.shape[1:] != (N_ACTIONS,):
        raise ValueError(f"policy must have shape (N, {N_ACTIONS}), got {policy.shape}")
    if value.ndim != 2 or value.shape[1:] != (N_VALUE_CLASSES,):
        raise ValueError(f"value must have shape (N, {N_VALUE_CLASSES}), got {value.shape}")

    save_data = {
        "stones": stones,
        "games": games,
        "stone_masks": stone_masks,
        "policy": policy,
        "value": value,
        "log_count": np.array(log_counter)
    }
    print(f"Saving data to {save_file_path}")
    np.savez_compressed(save_file_path, **save_data)


def generate_data(
    log_path: str = "path/to/dcl2/records",
    save_path: str = "path/to/save/data",
    data_size: int = 1000,
    target_end: List[int] = [i for i in range(10)],
    target_shot: List[int] = [15],
    model: str = "path/to/shot16/model",
    max_simulations: int = 20000,
    use_gpu: bool = True,
    shuffle_seed: int = 0,
    chunk_index: int = 0,
    chunk_size: Optional[int] = None,
) -> None:
    log_size = 0
    log_counter = 1
    data_counter = 0
    stones_data = []
    games_data = []
    stone_masks_data = []
    policy_data = []
    value_data = []
    
    device = get_torch_device(use_gpu=use_gpu)
    model_path = Path(__file__).resolve().parents[1] / "model" / model
    network = load_network(model_path, use_gpu=use_gpu)
    network.to(device)

    log_files = os.listdir(log_path)

    rng = random.Random(shuffle_seed)
    shuffled_log_files = rng.sample(log_files, len(log_files))

    if chunk_size is not None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_index < 0:
            raise ValueError(f"chunk_index must be non-negative, got {chunk_index}")

        chunk_start = chunk_index * chunk_size
        chunk_end = min(chunk_start + chunk_size, len(shuffled_log_files))
        target_log_files = shuffled_log_files[chunk_start:chunk_end]

        print(
            f"chunk {chunk_index}: "
            f"logs[{chunk_start}:{chunk_end}] "
            f"= {len(target_log_files)} files"
        )
    else:
        target_log_files = shuffled_log_files

    for one_log in target_log_files:
        if not os.path.isdir(os.path.join(log_path, one_log)):
            continue
        if log_size >= data_size:
            break
        
        dcl2_path = os.path.join(log_path, one_log, "game.dcl2")
        if not os.path.exists(dcl2_path):
            continue
        with open(dcl2_path) as dclfile:
            dcl2_data = dclfile.readlines()
        try:
            if json.loads(dcl2_data[-2])['log']['state'] :
                log_size += 1
                print(f"Processing log: {one_log}, total processed logs: {log_size}")
        except KeyError:
            continue
        
        for i in range(9, len(dcl2_data)-2, 2):
            try:
                dcl2_state = json.loads(dcl2_data[i])['log']['state']
                stones = dcl2_state['stones']['team0'] + dcl2_state['stones']['team1']
                scores_for_scorediff = dcl2_state['scores']
                end = dcl2_state['end']
                scorediff_for_team0 = scores_to_scorediff_for_team0(scores_for_scorediff)
                # print(f"scores: {scores}, end: {end}, scorediff_for_team0: {scorediff_for_team0}")
                shot = dcl2_state['shot']
                hammer = convert_team_stoi(dcl2_state['hammer'])
                if not ((end in target_end) and (shot in target_shot)):
                    continue
            except KeyError:
                continue

            root = set_root_state(
                network=network,
                stones=stones,
                score_diff=scorediff_for_team0,
                end=end,
                shot_index=shot,
                hammer_team=hammer
            )
            # Transformer 用の stone/game/mask 特徴量を生成する。
            stones_feature, game_feature, stone_mask = generate_input_features(
                stones=stones,
                end=end,
                shot=shot,
                hammer=hammer,
                score_diff_for_team0=scorediff_for_team0
            )
            # 探索して、action, policy_target, value_target を得る
            _, policy_target, value_target = mcts_search(
                root_state=root,
                max_simulations=max_simulations,
                is_create_data=True,
            )
            policy_distribution = _normalize_distribution(policy_target, N_ACTIONS, "policy_target")
            value_distribution = _normalize_distribution(value_target, N_VALUE_CLASSES, "value_target")

            # 生成した特徴量と target 分布を保存する。
            stones_data.append(stones_feature)
            games_data.append(game_feature)
            stone_masks_data.append(stone_mask)
            policy_data.append(policy_distribution)
            value_data.append(value_distribution)

            mirrored_stones = _mirror_stones_x(stones)
            flipped_stones_feature, flipped_game_feature, flipped_stone_mask = generate_input_features(
                stones=mirrored_stones,
                end=end,
                shot=shot,
                hammer=hammer,
                score_diff_for_team0=scorediff_for_team0
            )
            flipped_policy_target = _flip_policy_target(policy_target)
            flipped_policy_distribution = _normalize_distribution(
                flipped_policy_target,
                N_ACTIONS,
                "flipped_policy_target",
            )
            stones_data.append(flipped_stones_feature)
            games_data.append(flipped_game_feature)
            stone_masks_data.append(flipped_stone_mask)
            policy_data.append(flipped_policy_distribution)
            value_data.append(value_distribution.copy())

            # 生成したデータを保存するコード
            if len(value_data) >= DATA_SET_SIZE:
                print(f"sl_data{data_counter}")
                _save_data(os.path.join
                        (
                            save_path,
                            f"sl_data_chunk{chunk_index}_{data_counter}"
                        ),
                    stones_data,
                    games_data,
                    stone_masks_data,
                    policy_data,
                    value_data,
                    log_counter
                )
                stones_data = stones_data[DATA_SET_SIZE:]
                games_data = games_data[DATA_SET_SIZE:]
                stone_masks_data = stone_masks_data[DATA_SET_SIZE:]
                policy_data = policy_data[DATA_SET_SIZE:]
                value_data = value_data[DATA_SET_SIZE:]
                log_counter = 1
                data_counter += 1
                print("data counter: ", data_counter)
            
            log_counter += 1
    
    # 端数データの保存
    n_batches = len(value_data) // BATCH_SIZE
    print("n_batches: ", n_batches)
    if n_batches > 0:
        _save_data(os.path.join(save_path, f"sl_data_chunk{chunk_index}_{data_counter}"), \
            stones_data[0:n_batches*BATCH_SIZE], games_data[0:n_batches*BATCH_SIZE], \
            stone_masks_data[0:n_batches*BATCH_SIZE], policy_data[0:n_batches*BATCH_SIZE], \
            value_data[0:n_batches*BATCH_SIZE], log_counter)
    
if __name__ == "__main__":
    generate_data(
        log_path=Path("D:/all"),
        save_path=Path(__file__).resolve().parents[1] / "data",
        data_size=1000,
        target_end=[i for i in range(10)],
        target_shot=[15],
        model=Path(__file__).resolve().parents[1] / "model" / "js20000CP-32-9-LeaRate1000-vx32-vy25-batchsize1024.bin",
        max_simulations=1000,
        use_gpu=True,
        shuffle_seed=12345,
        chunk_index=0,
        chunk_size=BATCH_SIZE
    )
