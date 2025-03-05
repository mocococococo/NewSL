"""学習データの生成処理。
"""
import glob
import os
import random
import json
from typing import List, NoReturn
from pathlib import Path
import numpy as np
from nn.feature import generate_input_planes, generate_target_data, generate_value_data
from learning_param import BATCH_SIZE, DATA_SET_SIZE

def create_file_if_not_exist(file_path: str) -> NoReturn:
    """データ保存用のnpzファイルを作成する。
    Args:
        data_dir (str): 保存するファイルパス。
    """
    if not os.path.isfile(file_path + ".npz"):
        print(f"Creating file: {file_path}")
        # 空のファイルを作成する
        print(f"Parent file: {Path(file_path).parent}")
        with open(Path(file_path).parent / f"{file_path}.npz", mode='w') as file:
            # ここでファイルに内容を書くこともできますが、ここでは空のファイルを作成します
            pass
        print(f"File created: {file_path}.npz")
    else:
        print(f"File already exists: {file_path}")
    return

def _save_data(save_file_path: str, input_data: np.ndarray, policy_data: np.ndarray,\
    value_data: np.ndarray, log_counter: int) -> NoReturn:
    """学習データをnpzファイルとして出力する。

    Args:
        save_file_path (str): 保存するファイルパス。
        input_data (np.ndarray): 入力データ。
        policy_data (np.ndarray): Policyのデータ。
        value_data (np.ndarray): Valueのデータ
        log_counter (int): データセットにある棋譜データの個数。
    """
    save_data = {
        "input": np.array(input_data[0:DATA_SET_SIZE]),
        "policy": np.array(policy_data[0:DATA_SET_SIZE]),
        "value": np.array(value_data[0:DATA_SET_SIZE], dtype=np.int32),
        "log_count": np.array(log_counter)
    }
    print(f"Saving data to {save_file_path}")
    create_file_if_not_exist(save_file_path)
    np.savez_compressed(save_file_path, **save_data)

def generate_supervised_learning_data(
        program_dir: str,
        log_dir: str,
        data_size: int
    ):
    """教師データをnpzファイルとして生成する。

    Args:
        program_dir (str): 保存するファイルパス。
        log_dir (str): ログデータのディレクトリパス。
        data_size (int): データにする試合の個数。
    """
    input_data = []
    policy_data = []
    value_data = []

    log_counter = 1
    data_counter = 0
    print(f"start generate {data_size} data!")
    
    game_size = 0

    log_files = os.listdir(log_dir)
    for one_log in random.sample(log_files, len(log_files)):
        if os.path.isdir(os.path.join(log_dir, one_log)):
            if (game_size >= data_size):
                break
            dcl2_path = os.path.join(log_dir, one_log, "game.dcl2")
            if not os.path.exists(dcl2_path):
                continue
            with open(dcl2_path) as dclfile:
                dcl2_data = dclfile.readlines()
            dcl_json_data = json.loads(dcl2_data[-2])
            try:
                if dcl_json_data['log']['state'] :
                    print("one_log", game_size, "=", one_log)
                    game_size += 1
            except KeyError:
                continue
            for log_path in sorted(glob.glob(os.path.join(log_dir, one_log, "*.json"))):
                # print("log_path: ", log_path)
                with open(log_path, 'r') as file:
                    data = json.load(file)
                    try:
                        if data['log']['end'] <= 9:
                            input_data.append(generate_input_planes(stones = data['log']['simulator_storage']['stones'], scores=dcl_json_data['log']['state']['scores'], end=data['log']['end'], shot=data['log']['shot']))
                            policy_data.append(generate_target_data(data['log']['selected_move']))
                            value_data.append(generate_value_data(dcl_json_data, data['log']['end'], data['log']['shot']))
                    except Exception as e:
                            print(f"Error processing log: {log_path}")
                            print(f"Error message: {e}")
                # print("len(value_data): ", len(value_data), ", DATA_SET_SIZE: ", DATA_SET_SIZE)
                if len(value_data) >= DATA_SET_SIZE:
                    print(f"sl_data{data_counter}")
                    _save_data(os.path.join(program_dir, "data", f"sl_data_{data_counter}"), input_data, policy_data, value_data, log_counter)
                    input_data = input_data[DATA_SET_SIZE:]
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
        _save_data(os.path.join(program_dir, "data", f"sl_data_{data_counter}"), \
            input_data[0:n_batches*BATCH_SIZE], policy_data[0:n_batches*BATCH_SIZE], \
            value_data[0:n_batches*BATCH_SIZE], log_counter)
