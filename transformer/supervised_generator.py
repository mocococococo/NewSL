"""Transformer用の教師データ生成処理。"""

import copy
import json
import os
from pathlib import Path
from typing import NoReturn

import numpy as np

from common.translate_state import scores_to_scorediff_for_team0, convert_team_stoi
from learning_param import BATCH_SIZE, DATA_SET_SIZE
from nn.feature import generate_target_data, generate_value_data
from transformer.feature import generate_input_features


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


def generate_supervised_learning_data(
    program_dir: str,
    log_dir: str,
    data_size: int,
):
    """教師データをnpzファイルとして生成する。

    Args:
        program_dir (str): 保存するファイルパス。
        log_dir (str): ログデータのディレクトリパス。
        data_size (int): 読み込む試合数。
    """
    stones_data = []
    games_data = []
    stone_masks_data = []
    policy_data = []
    value_data = []

    log_counter = 1
    data_counter = 0
    print(f"start generate {data_size} data from {log_dir} to {program_dir}!")

    game_size = 0

    for one_log in os.listdir(log_dir):
        if os.path.isdir(os.path.join(log_dir, one_log)):
            if game_size > data_size:
                break
            dcl2_path = os.path.join(log_dir, one_log, "game.dcl2")
            if not os.path.exists(dcl2_path):
                continue
            with open(dcl2_path, encoding="utf-8") as dclfile:
                dcl2_data = dclfile.readlines()
                dcl2_json_data = json.loads(dcl2_data[-2])
            try:
                if json.loads(dcl2_data[-2])["log"]["state"]:
                    print("one_log", game_size, "=", one_log)
                    game_size += 1
            except KeyError:
                continue
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
                        stones_data.append(stones_feature)
                        games_data.append(game_feature)
                        stone_masks_data.append(stone_mask)

                        policy = generate_target_data(selected_move)
                        policy_data.append(policy)

                        value = generate_value_data(
                            scores=scores,
                            end=end,
                            shot_team=shot_team,
                        )
                        value_data.append(value)

                        # 左右反転データの生成
                        flipped_stones_feature = stones_feature.copy()
                        flipped_stones_feature[:, 0] *= -1
                        stones_data.append(flipped_stones_feature)
                        games_data.append(game_feature.copy())
                        stone_masks_data.append(stone_mask.copy())

                        flipped_move = copy.deepcopy(selected_move)
                        flipped_move["velocity"]["x"] = (
                            -1 * flipped_move["velocity"]["x"]
                        )
                        if flipped_move["rotation"] == "cw":
                            flipped_move["rotation"] = "ccw"
                        else:
                            flipped_move["rotation"] = "cw"

                        flipped_policy = generate_target_data(flipped_move)
                        policy_data.append(flipped_policy)

                        # Valueは盤面を反転しても変わらない
                        value_data.append(value)
                except Exception as error:  # pylint: disable=W0718
                    print(f"Error processing log: {dcl2_data[i]}")
                    print(f"Error message: {error}")

                if len(value_data) >= DATA_SET_SIZE:
                    print(f"sl_data{data_counter}")
                    _save_data(
                        os.path.join(
                            program_dir,
                            "data",
                            "transformer",
                            "supervised",
                            f"sl_data_{data_counter}",
                        ),
                        stones_data,
                        games_data,
                        stone_masks_data,
                        policy_data,
                        value_data,
                        log_counter,
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

    # 端数の出力
    n_batches = len(value_data) // BATCH_SIZE
    print("n_batches: ", n_batches)
    if n_batches > 0:
        _save_data(
            os.path.join(
                program_dir,
                "data",
                "transformer",
                "supervised",
                f"sl_data_{data_counter}",
            ),
            stones_data[0:n_batches * BATCH_SIZE],
            games_data[0:n_batches * BATCH_SIZE],
            stone_masks_data[0:n_batches * BATCH_SIZE],
            policy_data[0:n_batches * BATCH_SIZE],
            value_data[0:n_batches * BATCH_SIZE],
            log_counter,
        )
