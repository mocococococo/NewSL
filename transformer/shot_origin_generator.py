from __future__ import annotations

import copy
import gc
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Optional

import click
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from board.constant import VX_SIZE
from common.translate_state import convert_team_stoi, scores_dict_to_list
from learning_param import BATCH_SIZE, DATA_SET_SIZE
from mcts.state import score_diff_from_scores
from nn.utility import get_torch_device, load_network
from shot_origin.params import DEFAULT_SHOT_ORIGIN_MAX_SIMULATIONS
from shot_origin.search import set_root_state, shot_origin_search
from transformer.feature import _shot_team, generate_input_features
from transformer.params import (
    DEFAULT_TRANSFORMER_CONFIG,
    GAME_FEAT_DIM,
    MAX_STONES,
    STONE_FEAT_DIM,
    TRANSFORMER_VY_MODE,
    TRANSFORMER_VY_SIZE,
    get_transformer_action_dim,
)
from transformer.shot_target import (
    build_policy_target_from_shot_stats,
    build_value_target_from_shot_stats,
    build_win_value_target_from_shot_stats,
)
from transformer.utility import load_transformer_network

N_ACTIONS = DEFAULT_TRANSFORMER_CONFIG.action_dim
N_VALUE_CLASSES = DEFAULT_TRANSFORMER_CONFIG.value_dim
SPLIT_NAMES = ("train", "valid", "test")


class SplitBuffer:
    def __init__(self) -> None:
        self.data_counter = 0
        self.log_counter = 1
        self.stones_data = []
        self.games_data = []
        self.stone_masks_data = []
        self.policy_data = []
        self.value_data = []
        self.win_value_data = []


def _flip_policy_target(policy_target):
    policy = np.asarray(policy_target)
    if policy.size != N_ACTIONS:
        raise ValueError(f"policy_target size must be {N_ACTIONS}, got {policy.size}")

    policy_3d = policy.reshape(2, TRANSFORMER_VY_SIZE, VX_SIZE)
    flipped = policy_3d[::-1, :, ::-1]
    return flipped.reshape(N_ACTIONS).copy()


def _normalize_distribution(target, expected_size: int, name: str) -> np.ndarray:
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


def _validate_split_ratios(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[float, float, float]:
    ratios = (float(train_ratio), float(validation_ratio), float(test_ratio))
    if any(ratio < 0.0 for ratio in ratios):
        raise ValueError(f"split ratios must be non-negative, got {ratios}")
    total = sum(ratios)
    if total <= 0.0:
        raise ValueError("at least one split ratio must be positive")
    return tuple(ratio / total for ratio in ratios)


def _split_bounds(
    total_size: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[int, int]:
    train_ratio, validation_ratio, test_ratio = _validate_split_ratios(
        train_ratio,
        validation_ratio,
        test_ratio,
    )
    train_end = int(total_size * train_ratio)
    validation_end = train_end + int(total_size * validation_ratio)
    return train_end, validation_end


def _split_name_for_global_index(
    global_index: int,
    train_end: int,
    validation_end: int,
) -> str:
    if global_index < train_end:
        return "train"
    if global_index < validation_end:
        return "valid"
    return "test"


def _count_team_stones_on_sheet(stones, team: int) -> int:
    if len(stones) != MAX_STONES:
        raise ValueError(f"stones must have length {MAX_STONES}, got {len(stones)}")
    if team == 0:
        team_stones = stones[:8]
    elif team == 1:
        team_stones = stones[8:16]
    else:
        raise ValueError(f"team must be 0 or 1, got {team}")
    return sum(stone is not None for stone in team_stones)


def _cleanup_after_save(use_gpu: bool) -> None:
    gc.collect()
    if use_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_model_path(model: str | Path) -> Path:
    model_path = Path(model)
    if model_path.is_absolute():
        return model_path
    return ROOT_DIR / "model" / model_path


def _load_primary_model(model: str | Path, use_gpu: bool, sl_model_is_cnn: bool):
    model_path = _resolve_model_path(model)
    if sl_model_is_cnn:
        device = get_torch_device(use_gpu=use_gpu)
        network = load_network(model_path, use_gpu=use_gpu)
        network.to(device)
        return network
    return load_transformer_network(model_path, use_gpu=use_gpu)


def _mirror_stones_x(stones):
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
    save_file_path: str | Path,
    stones_data: np.ndarray,
    games_data: np.ndarray,
    stone_masks_data: np.ndarray,
    policy_data: np.ndarray,
    value_data: np.ndarray,
    win_value_data: np.ndarray,
    log_counter: int,
) -> None:
    save_file_path = Path(save_file_path)
    save_file_path.parent.mkdir(parents=True, exist_ok=True)

    stones = np.asarray(stones_data[0:DATA_SET_SIZE], dtype=np.float32)
    games = np.asarray(games_data[0:DATA_SET_SIZE], dtype=np.float32)
    stone_masks = np.asarray(stone_masks_data[0:DATA_SET_SIZE], dtype=np.bool_)
    policy = np.asarray(policy_data[0:DATA_SET_SIZE], dtype=np.float32)
    value = np.asarray(value_data[0:DATA_SET_SIZE], dtype=np.float32)
    win_value = np.asarray(win_value_data[0:DATA_SET_SIZE], dtype=np.float32)

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
    if win_value.ndim != 1:
        raise ValueError(f"win_value must have shape (N,), got {win_value.shape}")
    if not np.all(np.isfinite(win_value)):
        raise ValueError("win_value contains non-finite values")

    save_data = {
        "stones": stones,
        "games": games,
        "stone_masks": stone_masks,
        "policy": policy,
        "value": value,
        "win_value": win_value,
        "log_count": np.array(log_counter),
    }
    print(f"Saving data to {save_file_path}")
    np.savez_compressed(save_file_path, **save_data)


def _save_buffer(
    save_path: str | Path,
    split_name: str,
    chunk_index: int,
    buffer: SplitBuffer,
    sample_count: int,
    use_gpu: bool,
) -> None:
    _save_data(
        Path(save_path)
        / "raw"
        / split_name
        / f"sl_data_origin_raw_chunk{chunk_index}_{split_name}_{buffer.data_counter}",
        buffer.stones_data[:sample_count],
        buffer.games_data[:sample_count],
        buffer.stone_masks_data[:sample_count],
        buffer.policy_data[:sample_count],
        buffer.value_data[:sample_count],
        buffer.win_value_data[:sample_count],
        buffer.log_counter,
    )
    buffer.data_counter += 1
    buffer.log_counter = 1
    del buffer.stones_data[:sample_count]
    del buffer.games_data[:sample_count]
    del buffer.stone_masks_data[:sample_count]
    del buffer.policy_data[:sample_count]
    del buffer.value_data[:sample_count]
    del buffer.win_value_data[:sample_count]
    _cleanup_after_save(use_gpu)
    print(f"{split_name} data counter: {buffer.data_counter}")


def _flush_full_buffer(
    save_path: str | Path,
    split_name: str,
    chunk_index: int,
    buffer: SplitBuffer,
    use_gpu: bool,
) -> None:
    while len(buffer.value_data) >= DATA_SET_SIZE:
        _save_buffer(save_path, split_name, chunk_index, buffer, DATA_SET_SIZE, use_gpu)


def _flush_remainder_samples(
    save_path: str | Path,
    split_name: str,
    chunk_index: int,
    buffer: SplitBuffer,
    use_gpu: bool,
) -> None:
    sample_count = len(buffer.value_data)
    print(f"{split_name} remaining_samples: {sample_count}")
    if sample_count <= 0:
        return
    _save_buffer(save_path, split_name, chunk_index, buffer, sample_count, use_gpu)


def _append_sample(
    buffer: SplitBuffer,
    stones_feature,
    game_feature,
    stone_mask,
    policy_distribution,
    value_distribution,
    win_value_target: float,
) -> None:
    buffer.stones_data.append(stones_feature)
    buffer.games_data.append(game_feature)
    buffer.stone_masks_data.append(stone_mask)
    buffer.policy_data.append(policy_distribution)
    buffer.value_data.append(value_distribution)
    buffer.win_value_data.append(float(win_value_target))


def generate_data(
    log_path: str | Path = "path/to/dcl2/records",
    save_path: str | Path = "path/to/save/data",
    data_size: int = 1000,
    target_end: int = 9,
    target_shot: List[int] = [15],
    use_end_augmentation: bool = True,
    use_score_diff_augmentation: bool = True,
    model: str | Path = "path/to/value/model",
    sl_model_is_cnn: bool = False,
    use_transformer: bool = False,
    transformer_model: Optional[str | Path] = None,
    transformer_target_end: Optional[List[int]] = None,
    transformer_target_shot: Optional[List[int]] = None,
    max_simulations: int = DEFAULT_SHOT_ORIGIN_MAX_SIMULATIONS,
    use_gpu: bool = True,
    use_value: bool = True,
    shuffle_seed: int = 0,
    chunk_start: int = 0,
    chunk_end: Optional[int] = None,
    chunk_size: Optional[int] = None,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    policy_min_visit: int = 3,
    policy_delta_q: float = 1.0,
    policy_alpha_visit: float = 0.2,
    policy_beta_q: float = 0.3,
    policy_lambda_best: float = 0.5,
    value_min_visit: int = 3,
    value_delta_q: float = 0.0,
    value_alpha_visit: float = 0.5,
    value_beta_q: float = 0.5,
    value_lambda_best: float = 0.5,
) -> None:
    if chunk_end is None:
        chunk_end = chunk_start
    if chunk_start < 0:
        raise ValueError(f"chunk_start must be non-negative, got {chunk_start}")
    if chunk_end < chunk_start:
        raise ValueError(f"chunk_end must be greater than or equal to chunk_start, got {chunk_end}")
    if chunk_start != chunk_end:
        if chunk_size is None:
            raise ValueError("chunk_size must be specified when running multiple chunks")
        for current_chunk_index in range(chunk_start, chunk_end + 1):
            generate_data(
                log_path=log_path,
                save_path=save_path,
                data_size=data_size,
                target_end=target_end,
                use_end_augmentation=use_end_augmentation,
                use_score_diff_augmentation=use_score_diff_augmentation,
                target_shot=target_shot,
                model=model,
                sl_model_is_cnn=sl_model_is_cnn,
                use_transformer=use_transformer,
                transformer_model=transformer_model,
                transformer_target_end=transformer_target_end,
                transformer_target_shot=transformer_target_shot,
                max_simulations=max_simulations,
                use_gpu=use_gpu,
                use_value=use_value,
                shuffle_seed=shuffle_seed,
                chunk_start=current_chunk_index,
                chunk_end=current_chunk_index,
                chunk_size=chunk_size,
                train_ratio=train_ratio,
                validation_ratio=validation_ratio,
                test_ratio=test_ratio,
                policy_min_visit=policy_min_visit,
                policy_delta_q=policy_delta_q,
                policy_alpha_visit=policy_alpha_visit,
                policy_beta_q=policy_beta_q,
                policy_lambda_best=policy_lambda_best,
                value_min_visit=value_min_visit,
                value_delta_q=value_delta_q,
                value_alpha_visit=value_alpha_visit,
                value_beta_q=value_beta_q,
                value_lambda_best=value_lambda_best,
            )
        return

    action_type = "default" if sl_model_is_cnn else TRANSFORMER_VY_MODE
    action_dim = get_transformer_action_dim(action_type)
    if action_dim != N_ACTIONS:
        raise ValueError(
            "shot_origin_generator currently writes Transformer data with "
            f"action_dim={N_ACTIONS}, but action_type={action_type!r} has {action_dim}."
        )

    chunk_index = chunk_start
    processed_logs = 0
    split_buffers = {split_name: SplitBuffer() for split_name in SPLIT_NAMES}

    network = _load_primary_model(model, use_gpu=use_gpu, sl_model_is_cnn=sl_model_is_cnn)

    transformer_network = None
    transformer_target_end_tuple = (
        tuple(transformer_target_end) if transformer_target_end is not None else (target_end,)
    )
    transformer_target_shot_tuple = (
        tuple(transformer_target_shot) if transformer_target_shot is not None else ()
    )
    if use_transformer:
        if transformer_model is None:
            raise ValueError("transformer_model must be specified when use_transformer is True")
        if not transformer_target_shot_tuple:
            raise ValueError("transformer_target_shot must be specified when use_transformer is True")
        transformer_network = load_transformer_network(_resolve_model_path(transformer_model), use_gpu=use_gpu)

    log_path = Path(log_path)
    log_files = os.listdir(log_path)
    rng = random.Random(shuffle_seed)
    shuffled_log_files = rng.sample(log_files, len(log_files))

    if chunk_size is not None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        chunk_start_pos = chunk_index * chunk_size
        chunk_end_pos = min(chunk_start_pos + chunk_size, len(shuffled_log_files))
        target_indexed_log_files = list(enumerate(shuffled_log_files))[chunk_start_pos:chunk_end_pos]
        print(
            f"chunk {chunk_index}: "
            f"logs[{chunk_start_pos}:{chunk_end_pos}] "
            f"= {len(target_indexed_log_files)} files"
        )
    else:
        target_indexed_log_files = list(enumerate(shuffled_log_files))

    train_end, validation_end = _split_bounds(
        len(shuffled_log_files),
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )
    target_split_counts = {split_name: 0 for split_name in SPLIT_NAMES}
    for global_index, _ in target_indexed_log_files:
        target_split_counts[
            _split_name_for_global_index(global_index, train_end, validation_end)
        ] += 1
    print(
        "global split log bounds: "
        f"train=[0:{train_end}), valid=[{train_end}:{validation_end}), "
        f"test=[{validation_end}:{len(shuffled_log_files)})"
    )
    print(
        "chunk split logs: "
        + ", ".join(f"{name}={count}" for name, count in target_split_counts.items())
    )

    for global_index, one_log in target_indexed_log_files:
        split_name = _split_name_for_global_index(global_index, train_end, validation_end)
        buffer = split_buffers[split_name]
        if not (log_path / one_log).is_dir():
            continue
        if processed_logs >= data_size:
            break

        dcl2_path = log_path / one_log / "game.dcl2"
        if not dcl2_path.exists():
            continue
        with dcl2_path.open() as dclfile:
            dcl2_data = dclfile.readlines()
        try:
            if json.loads(dcl2_data[-2])["log"]["state"]:
                processed_logs += 1
                print(
                    f"Processing {split_name} log: {one_log}, "
                    f"global index: {global_index}, "
                    f"total processed logs: {processed_logs}"
                )
        except KeyError:
            continue

        for i in range(9, len(dcl2_data) - 2, 2):
            try:
                dcl2_log = json.loads(dcl2_data[i])["log"]
                dcl2_state = dcl2_log["state"]
                stones = dcl2_state["stones"]["team0"] + dcl2_state["stones"]["team1"]
                logged_end = int(dcl2_state["end"])
                logged_score_diff = score_diff_from_scores(
                    scores_dict_to_list(dcl2_state["scores"])
                )
                shot = int(dcl2_state["shot"])
                next_team = dcl2_log["next_team"]
                hammer = convert_team_stoi(dcl2_state["hammer"])
                if shot not in target_shot:
                    continue
                if (not use_end_augmentation) and logged_end != target_end:
                    continue
            except KeyError:
                continue

            score_diff_candidates = (
                range(-8, 9) if use_score_diff_augmentation else [logged_score_diff]
            )

            for expanded_score_diff in score_diff_candidates:
                end = target_end if use_end_augmentation else logged_end
                try:
                    shot_team = convert_team_stoi(next_team)
                    expected_shot_team = _shot_team(shot, hammer)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid next_team {next_team!r}, shot {shot}, or hammer {hammer} "
                        f"in log {one_log}"
                    ) from exc

                if shot_team != expected_shot_team:
                    raise ValueError(
                        f"next_team mismatch in log {one_log}: "
                        f"next_team={next_team!r}, shot_team={shot_team}, "
                        f"expected_shot_team={expected_shot_team}, shot={shot}, hammer={hammer}"
                    )

                scorediff_for_shot_team = expanded_score_diff if shot_team == 0 else -expanded_score_diff
                print(f"split={split_name} shot={shot}, expanded_score_diff={scorediff_for_shot_team}")

                max_possible_end_score = _count_team_stones_on_sheet(stones, shot_team) + 1
                if max_possible_end_score + scorediff_for_shot_team < 0:
                    print(
                        f"Skip at shot {shot} due to unWinnable position: "
                        f"max_possible_end_score={max_possible_end_score}, "
                        f"scorediff_for_shot_team={scorediff_for_shot_team}"
                    )
                    continue

                root = set_root_state(
                    network=network,
                    stones=stones,
                    score_diff=expanded_score_diff,
                    end=end,
                    shot_index=shot,
                    hammer_team=hammer,
                    transformer_network=transformer_network,
                    sl_model_is_cnn=sl_model_is_cnn,
                    use_transformer=use_transformer,
                    transformer_target_end=transformer_target_end_tuple,
                    transformer_target_shot=transformer_target_shot_tuple,
                )
                stones_feature, game_feature, stone_mask = generate_input_features(
                    stones=stones,
                    end=end,
                    shot=shot,
                    hammer=hammer,
                    score_diff_for_team0=expanded_score_diff,
                )
                best_action_id, root_candidate_stats = shot_origin_search(
                    root_state=root,
                    max_simulations=max_simulations,
                    is_create_data=True,
                    use_value=use_value,
                    action_type=action_type,
                )
                policy_distribution = build_policy_target_from_shot_stats(
                    root_candidate_stats,
                    best_action_id,
                    action_dim=N_ACTIONS,
                    min_visit=policy_min_visit,
                    delta_q=policy_delta_q,
                    alpha_visit=policy_alpha_visit,
                    beta_q=policy_beta_q,
                    lambda_best=policy_lambda_best,
                )
                value_distribution = build_value_target_from_shot_stats(
                    root_candidate_stats,
                    best_action_id,
                    value_dim=N_VALUE_CLASSES,
                    min_visit=value_min_visit,
                    delta_q=value_delta_q,
                    alpha_visit=value_alpha_visit,
                    beta_q=value_beta_q,
                    lambda_best=value_lambda_best,
                )
                win_value_target = build_win_value_target_from_shot_stats(
                    root_candidate_stats,
                    best_action_id,
                    min_visit=policy_min_visit,
                    delta_q=policy_delta_q,
                    alpha_visit=policy_alpha_visit,
                    beta_q=policy_beta_q,
                    lambda_best=policy_lambda_best,
                )

                max_possible_end_score = 0
                for k, prob in enumerate(value_distribution):
                    if prob > 0:
                        max_possible_end_score = k - (N_VALUE_CLASSES // 2)

                if max_possible_end_score + scorediff_for_shot_team < 0:
                    print(
                        f"Skip at shot {shot} after SHOT_ORIGIN due to unWinnable position: "
                        f"max_possible_end_score={max_possible_end_score}, "
                        f"scorediff_for_shot_team={scorediff_for_shot_team}"
                    )
                    continue

                _append_sample(
                    buffer,
                    stones_feature,
                    game_feature,
                    stone_mask,
                    policy_distribution,
                    value_distribution,
                    float(win_value_target),
                )

                mirrored_stones = _mirror_stones_x(stones)
                flipped_stones_feature, flipped_game_feature, flipped_stone_mask = generate_input_features(
                    stones=mirrored_stones,
                    end=end,
                    shot=shot,
                    hammer=hammer,
                    score_diff_for_team0=expanded_score_diff,
                )
                flipped_policy_target = _flip_policy_target(policy_distribution)
                flipped_policy_distribution = _normalize_distribution(
                    flipped_policy_target,
                    N_ACTIONS,
                    "flipped_policy_target",
                )
                _append_sample(
                    buffer,
                    flipped_stones_feature,
                    flipped_game_feature,
                    flipped_stone_mask,
                    flipped_policy_distribution,
                    value_distribution.copy(),
                    float(win_value_target),
                )

                _flush_full_buffer(save_path, split_name, chunk_index, buffer, use_gpu)
                buffer.log_counter += 1

        if processed_logs >= data_size:
            break

    for split_name, buffer in split_buffers.items():
        _flush_remainder_samples(save_path, split_name, chunk_index, buffer, use_gpu)


@click.command()
@click.option("--start", type=int, required=True, help="chunk index to process")
@click.option("--end", type=int, required=True, help="chunk index to process")
def main(start: int, end: int) -> None:
    generate_data(
        log_path=ROOT_DIR / "LearnLog" / "all",
        save_path=ROOT_DIR / "data" / "shot_origin",
        data_size=20000,
        target_end=9,
        target_shot=[15],
        use_end_augmentation=True,
        use_score_diff_augmentation=False,
        model=ROOT_DIR / "model" / "transformer-supervised-model-AdamW-vy56.bin",
        sl_model_is_cnn=False,
        use_transformer=False,
        transformer_model=None,
        transformer_target_end=[9],
        transformer_target_shot=[15],
        max_simulations=DEFAULT_SHOT_ORIGIN_MAX_SIMULATIONS,
        use_gpu=True,
        use_value=True,
        shuffle_seed=12345,
        chunk_start=start,
        chunk_end=end,
        chunk_size=BATCH_SIZE,
        train_ratio=0.8,
        validation_ratio=0.1,
        test_ratio=0.1,
        policy_min_visit=3,
        policy_delta_q=1.0,
        policy_alpha_visit=0.2,
        policy_beta_q=0.3,
        policy_lambda_best=0.5,
        value_min_visit=3,
        value_delta_q=0.0,
        value_alpha_visit=0.5,
        value_beta_q=0.5,
        value_lambda_best=0.5,
    )


if __name__ == "__main__":
    main()
