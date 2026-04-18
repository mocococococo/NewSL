"""
dcl2 の局面から shot16 の policy , value を学習するためのデータ生成のためのコード

1. 15投目まで終了時の dcl2 の局面を読み込む
2. 盤面を特徴平面に変換する
3. 16投目を選択するための探索を行う
"""
import numpy as np
import os
import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.translate_state import convert_team_stoi, scores_to_scorediff_for_team0
from nn.feature import generate_input_planes
from nn.utility import load_network, get_torch_device
from mcts.search import mcts_search, set_root_state
from board.constant import VX_SIZE, VY_SIZE
from learning_param import BATCH_SIZE, DATA_SET_SIZE

N_ACTIONS = VX_SIZE * VY_SIZE * 2


def _flip_policy_target(policy_target):
    policy = np.asarray(policy_target)
    if policy.size != N_ACTIONS:
        raise ValueError(f"policy_target size must be {N_ACTIONS}, got {policy.size}")

    policy_3d = policy.reshape(2, VY_SIZE, VX_SIZE)
    flipped = policy_3d[::-1, :, ::-1]
    return flipped.reshape(N_ACTIONS).copy()

def _save_data(save_file_path: str, input_data: np.ndarray, policy_data: np.ndarray,\
    value_data: np.ndarray, log_counter: int) -> None:
    """学習データをnpzファイルとして出力する。

    Args:
        save_file_path (str): 保存するファイルパス。
        input_data (np.ndarray): 入力データ。
        policy_data (np.ndarray): Policyのデータ。
        value_data (np.ndarray): Valueのデータ
        log_counter (int): データセットにある棋譜データの個数。
    """
    Path(save_file_path).parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "input": np.array(input_data[0:DATA_SET_SIZE]),
        "policy": np.array(policy_data[0:DATA_SET_SIZE]),
        "value": np.array(value_data[0:DATA_SET_SIZE], dtype=np.int32),
        "log_count": np.array(log_counter)
    }
    print(f"Saving data to {save_file_path}")
    np.savez_compressed(save_file_path, **save_data)


def main(
    log_path: str = "path/to/dcl2/records",
    save_path: str = "path/to/save/data",
    data_size: int = 1000,
    target_end: int = 9,
    target_shot: int = 15,
    model: str = "path/to/shot16/model",
    use_gpu: bool = True,
) -> None:
    log_size = 0
    log_counter = 1
    data_counter = 0
    input_data = []
    policy_data = []
    value_data = []
    
    device = get_torch_device(use_gpu=use_gpu)
    network = load_network(model, use_gpu=use_gpu)
    network.to(device)

    for one_log in os.listdir(log_path):
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
                if not ((end == target_end) and (shot == target_shot)):
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
            # 特徴平面を生成するコード
            input_planes = generate_input_planes(
                stones=stones,
                end=end,
                shot=shot,
                hammer=hammer,
                score_diff_for_team0=scorediff_for_team0
            )
            # 探索して、action, policy_target, value_target を得る
            _, policy_target, value_target = mcts_search(root_state=root, is_create_data=True)
            # 生成した action, policy_target, value_target を保存するコード
            input_data.append(input_planes)
            policy_data.append(policy_target)
            value_data.append(value_target)

            flipped_input_planes = np.flip(input_planes, axis=2).copy()
            flipped_policy_target = _flip_policy_target(policy_target)
            input_data.append(flipped_input_planes)
            policy_data.append(flipped_policy_target)
            value_data.append(value_target)

            # 生成したデータを保存するコード
            if len(value_data) >= DATA_SET_SIZE:
                print(f"sl_data{data_counter}")
                _save_data(os.path.join
                        (
                            save_path,
                            f"sl_data_{data_counter}"
                        ),
                    input_data,
                    policy_data,
                    value_data,
                    log_counter
                )
                input_data = input_data[DATA_SET_SIZE:]
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
        _save_data(os.path.join(save_path, f"sl_data_{data_counter}"), \
            input_data[0:n_batches*BATCH_SIZE], policy_data[0:n_batches*BATCH_SIZE], \
            value_data[0:n_batches*BATCH_SIZE], log_counter)
    
if __name__ == "__main__":
    main(
        log_path=Path(__file__).resolve().parents[1] / "LearnLog" / "jiritsu-vs-silicon",
        save_path=Path(__file__).resolve().parents[1] / "data",
        data_size=1000,
        target_end=9,
        target_shot=15,
        model=Path(__file__).resolve().parents[1] / "model" / "js20000CP-32-9-LeaRate1000-vx32-vy25-batchsize1024.bin",
        use_gpu=True
    )
