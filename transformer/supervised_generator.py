"""Transformer用の教師データ生成処理。"""

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import NoReturn

import numpy as np

from common.translate_state import scores_to_scorediff_for_team0, convert_team_stoi
from learning_param import BATCH_SIZE, DATA_SET_SIZE
from nn.feature import generate_target_data, generate_value_data
from transformer.feature import generate_input_features


TRAIN_DATA_RATIO = 0.9
VALIDATION_DATA_RATIO = 0.05
TEST_DATA_RATIO = 0.05
SPLIT_METHOD = "sha256"


def create_file_if_not_exist(file_path: str) -> NoReturn:
    """データ保存先のディレクトリを作成する。

    Args:
        file_path (str): 保存するファイルパス。
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    return


def _save_data(
    save_file_path: str,
    stones_data: np.ndarray,
    games_data: np.ndarray,
    stone_masks_data: np.ndarray,
    policy_data: np.ndarray,
    value_data: np.ndarray,
    log_counter: int,
) -> NoReturn:
    """学習データをnpzファイルとして出力する。

    Args:
        save_file_path (str): 保存するファイルパス。
        stones_data (np.ndarray): ストーン特徴量。
        games_data (np.ndarray): ゲーム特徴量。
        stone_masks_data (np.ndarray): ストーンマスク。
        policy_data (np.ndarray): Policyのデータ。
        value_data (np.ndarray): Valueのデータ。
        log_counter (int): データセットにある棋譜データの個数。
    """
    save_data = {
        "stones": np.array(stones_data[0:DATA_SET_SIZE]),
        "games": np.array(games_data[0:DATA_SET_SIZE]),
        "stone_masks": np.array(stone_masks_data[0:DATA_SET_SIZE]),
        "policy": np.array(policy_data[0:DATA_SET_SIZE]),
        "value": np.array(value_data[0:DATA_SET_SIZE], dtype=np.int32),
        "log_count": np.array(log_counter),
    }
    print(f"Saving data to {save_file_path}")
    create_file_if_not_exist(save_file_path)
    np.savez_compressed(save_file_path, **save_data)


def _get_split_name(game_id: str) -> str:
    """試合IDのSHA-256から所属するデータ集合を決定する。"""
    digest = hashlib.sha256(game_id.encode("utf-8")).digest()
    hash_value = int.from_bytes(digest[:8], byteorder="big")
    train_threshold = int(TRAIN_DATA_RATIO * (1 << 64))
    validation_threshold = int(
        (TRAIN_DATA_RATIO + VALIDATION_DATA_RATIO) * (1 << 64)
    )
    if hash_value < train_threshold:
        return "train"
    if hash_value < validation_threshold:
        return "validation"
    return "test"


def _create_split_data() -> dict:
    """一つのデータ集合を生成するための作業領域を作成する。"""
    return {
        "stones": [],
        "games": [],
        "stone_masks": [],
        "policy": [],
        "value": [],
        "game_ids": [],
        "files": [],
        "log_counter": 1,
        "data_counter": 0,
    }


def _save_full_data_set(
    program_dir: str,
    split_name: str,
    split_data: dict,
) -> None:
    """規定件数に達したデータを一つのNPZへ保存する。"""
    data_counter = split_data["data_counter"]
    save_file_path = os.path.join(
        program_dir,
        "data",
        "transformer",
        "supervised",
        split_name,
        f"sl_data_{data_counter}",
    )
    _save_data(
        save_file_path,
        split_data["stones"],
        split_data["games"],
        split_data["stone_masks"],
        split_data["policy"],
        split_data["value"],
        split_data["log_counter"],
    )
    split_data["files"].append(
        (Path(split_name) / f"sl_data_{data_counter}.npz").as_posix()
    )
    split_data["stones"] = split_data["stones"][DATA_SET_SIZE:]
    split_data["games"] = split_data["games"][DATA_SET_SIZE:]
    split_data["stone_masks"] = split_data["stone_masks"][DATA_SET_SIZE:]
    split_data["policy"] = split_data["policy"][DATA_SET_SIZE:]
    split_data["value"] = split_data["value"][DATA_SET_SIZE:]
    split_data["log_counter"] = 1
    split_data["data_counter"] += 1
    print("data counter: ", split_data["data_counter"])


def _save_remaining_data(
    program_dir: str,
    split_name: str,
    split_data: dict,
) -> None:
    """ミニバッチを構成できる端数データをNPZへ保存する。"""
    n_batches = len(split_data["value"]) // BATCH_SIZE
    print(f"{split_name} n_batches: ", n_batches)
    if n_batches <= 0:
        return

    data_counter = split_data["data_counter"]
    save_file_path = os.path.join(
        program_dir,
        "data",
        "transformer",
        "supervised",
        split_name,
        f"sl_data_{data_counter}",
    )
    data_count = n_batches * BATCH_SIZE
    _save_data(
        save_file_path,
        split_data["stones"][0:data_count],
        split_data["games"][0:data_count],
        split_data["stone_masks"][0:data_count],
        split_data["policy"][0:data_count],
        split_data["value"][0:data_count],
        split_data["log_counter"],
    )
    split_data["files"].append(
        (Path(split_name) / f"sl_data_{data_counter}.npz").as_posix()
    )


def _save_split_manifest(
    program_dir: str,
    train_data: dict,
    validation_data: dict,
    test_data: dict,
) -> None:
    """試合の分割結果と今回生成したファイル一覧を保存する。"""
    supervised_dir = (
        Path(program_dir)
        / "data"
        / "transformer"
        / "supervised"
    )
    supervised_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = supervised_dir / "split_manifest.json"
    manifest = {
        "train_data_ratio": TRAIN_DATA_RATIO,
        "validation_data_ratio": VALIDATION_DATA_RATIO,
        "test_data_ratio": TEST_DATA_RATIO,
        "split_method": SPLIT_METHOD,
        "hash_input": "game_directory_name",
        "train_games": train_data["game_ids"],
        "validation_games": validation_data["game_ids"],
        "test_games": test_data["game_ids"],
        "train_files": train_data["files"],
        "validation_files": validation_data["files"],
        "test_files": test_data["files"],
    }
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=4)
    print(f"分割マニフェストを保存しました: {manifest_path}")


def generate_supervised_learning_data(
    program_dir: str,
    log_dir: str,
    data_size: int,
):
    """教師データを試合単位で分割し、npzファイルとして生成する。

    Args:
        program_dir (str): 保存するファイルパス。
        log_dir (str): ログデータのディレクトリパス。
        data_size (int): 読み込む試合数。
    """
    print(f"start generate {data_size} data from {log_dir} to {program_dir}!")

    split_data_map = {
        "train": _create_split_data(),
        "validation": _create_split_data(),
        "test": _create_split_data(),
    }
    game_size = 0

    for one_log in os.listdir(log_dir):
        if game_size > data_size:
            break
        if not os.path.isdir(os.path.join(log_dir, one_log)):
            continue

        dcl2_path = os.path.join(log_dir, one_log, "game.dcl2")
        if not os.path.exists(dcl2_path):
            continue

        with open(dcl2_path, encoding="utf-8") as dclfile:
            dcl2_data = dclfile.readlines()
            dcl2_json_data = json.loads(dcl2_data[-2])

        try:
            if not dcl2_json_data["log"]["state"]:
                continue
        except KeyError:
            continue

        print("one_log", game_size, "=", one_log)
        game_size += 1

        split_name = _get_split_name(one_log)
        split_data = split_data_map[split_name]
        split_data["game_ids"].append(one_log)

        for i in range(9, len(dcl2_data) - 2, 2):
            dcl2_state = json.loads(dcl2_data[i])["log"]["state"]
            dcl2_log2 = json.loads(dcl2_data[i + 1])["log"]
            stones = (
                dcl2_state["stones"]["team0"]
                + dcl2_state["stones"]["team1"]
            )
            scores = dcl2_json_data["log"]["state"]["scores"]
            scores_for_scorediff = dcl2_state["scores"]
            end = dcl2_state["end"]
            scorediff_for_team0 = scores_to_scorediff_for_team0(
                scores_for_scorediff
            )
            shot = dcl2_state["shot"]
            shot_team = convert_team_stoi(dcl2_log2["team"])
            hammer = convert_team_stoi(dcl2_state["hammer"])
            selected_move = dcl2_log2["move"]

            try:
                if end < 10:
                    stones_feature, game_feature, stone_mask = (
                        generate_input_features(
                            stones=stones,
                            end=end,
                            shot=shot,
                            hammer=hammer,
                            score_diff_for_team0=scorediff_for_team0,
                        )
                    )
                    split_data["stones"].append(stones_feature)
                    split_data["games"].append(game_feature)
                    split_data["stone_masks"].append(stone_mask)

                    policy = generate_target_data(selected_move)
                    split_data["policy"].append(policy)

                    value = generate_value_data(
                        scores=scores,
                        end=end,
                        shot_team=shot_team,
                    )
                    split_data["value"].append(value)

                    if split_name == "train":
                        # 左右反転データの生成
                        flipped_stones_feature = stones_feature.copy()
                        flipped_stones_feature[:, 0] *= -1
                        split_data["stones"].append(
                            flipped_stones_feature
                        )
                        split_data["games"].append(game_feature.copy())
                        split_data["stone_masks"].append(
                            stone_mask.copy()
                        )

                        flipped_move = copy.deepcopy(selected_move)
                        flipped_move["velocity"]["x"] = (
                            -1 * flipped_move["velocity"]["x"]
                        )
                        if flipped_move["rotation"] == "cw":
                            flipped_move["rotation"] = "ccw"
                        else:
                            flipped_move["rotation"] = "cw"

                        flipped_policy = generate_target_data(flipped_move)
                        split_data["policy"].append(flipped_policy)

                        # Valueは盤面を反転しても変わらない
                        split_data["value"].append(value)
            except Exception as error:  # pylint: disable=W0718
                print(f"Error processing log: {dcl2_data[i]}")
                print(f"Error message: {error}")

            if len(split_data["value"]) >= DATA_SET_SIZE:
                print(
                    f"{split_name}/sl_data"
                    f"{split_data['data_counter']}"
                )
                _save_full_data_set(
                    program_dir,
                    split_name,
                    split_data,
                )

            split_data["log_counter"] += 1

    for split_name, split_data in split_data_map.items():
        _save_remaining_data(program_dir, split_name, split_data)

    _save_split_manifest(
        program_dir,
        split_data_map["train"],
        split_data_map["validation"],
        split_data_map["test"],
    )
