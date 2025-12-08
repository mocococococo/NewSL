"""学習データの生成処理。
"""
import glob
import os
import random
import copy
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, NoReturn
from pathlib import Path
import numpy as np

from common.translate_state import scores_to_scorediff_for_team0, convert_team_stoi
from nn.learn.feature import generate_input_planes, generate_target_data, generate_value_data
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
    

def visualize_data(box):
    rotations = ['cw', 'ccw']
    
    fig = plt.figure(figsize=(14, 6))
    
    for i in range(2):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        
        # グリッドの各セルの位置を決定
        x_len = box.shape[1]
        y_len = box.shape[2]
        x_range = np.arange(x_len)
        y_range = np.arange(y_len)
        # indexing='ij' とすることで (x, y) 順に生成
        xpos, ypos = np.meshgrid(x_range, y_range, indexing='ij')
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros_like(xpos)
        
        # 各バーの幅（x,y方向）
        dx = 0.8 * np.ones_like(xpos)
        dy = 0.8 * np.ones_like(ypos)
        # 各バーの高さ（Frequency）
        dz = box[i].flatten()
        
        # height_rangeが指定されていれば、バーの高さをクリップする
        
        max_val = np.max(dz) if np.max(dz) > 0 else 1
        ax.set_zlim(0, max_val)
        
        colors = ['yellow' if h >= 1 else 'blue' for h in dz]
        
        # 3Dバーを描画
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, color=colors)
        ax.set_title(f"Initial Speed Distribution 3D ({rotations[i]})")
        ax.set_xlabel("x-index")
        ax.set_ylabel("y-index")
        ax.set_zlabel("Frequency")
        ax.set_box_aspect((x_len, y_len, max_val))
    
    plt.tight_layout()
    plt.show()

def generate_supervised_learning_data(
        program_dir: str,
        log_dir: str,
        data_size: int
    ):
    """教師データをnpzファイルとして生成する。

    Args:
        program_dir (str): 保存するファイルパス。
        log_dir (str): ログデータのディレクトリパス。
    """
    input_data = []
    policy_data = []
    value_data = []

    log_counter = 1
    data_counter = 0
    print(f"start generate {data_size} data from {log_dir} to {program_dir}!")
    
    game_size = 0
    # ibox = np.zeros((2, 32, 56), dtype=int)
    # vbox = np.zeros((2, 32, 56), dtype=int)

    #print("log_dir: ", log_dir)
    """
    log_files = os.listdir(log_dir)
    for one_log in random.sample(log_files, len(log_files)):
    """
    for one_log in os.listdir(log_dir):
        # print("one_log: ", one_log)
        # print("isdir: ", os.path.isdir(os.path.join(log_dir, one_log)))
        if os.path.isdir(os.path.join(log_dir, one_log)):
            if (game_size > data_size):
                break
            # print("is_dcl2_path: ", os.path.exists(os.path.join(log_dir, one_log, "game.dcl2")))
            dcl2_path = os.path.join(log_dir, one_log, "game.dcl2")
            if not os.path.exists(dcl2_path):
                continue
            with open(dcl2_path) as dclfile:
                dcl2_data = dclfile.readlines()
                dcl2_json_data = json.loads(dcl2_data[-2])
            try:
                if json.loads(dcl2_data[-2])['log']['state'] :
                    print("one_log", game_size, "=", one_log)
                    game_size += 1
            except KeyError:
                continue
            for i in range(9, len(dcl2_data)-2, 2):
                dcl2_state = json.loads(dcl2_data[i])['log']['state']
                dcl2_log2 = json.loads(dcl2_data[i+1])['log']
                stones = dcl2_state['stones']['team0'] + dcl2_state['stones']['team1']
                scores = dcl2_json_data['log']['state']['scores']
                scorediff_for_team0 = scores_to_scorediff_for_team0(scores)
                end = dcl2_state['end']
                shot = dcl2_state['shot']
                shot_team = convert_team_stoi(dcl2_log2['team'])
                hammer = convert_team_stoi(dcl2_state['hammer'])
                selected_move = dcl2_log2['move']
                #if end == 0 and shot == 15:
                #    print("stones: ", stones)
                #    print("scores: ", scores)
                #    print("end: ", end)
                #    print("shot: ", shot)
                #    print("selected_move: ", selected_move)
                try:
                    if end < 10:
                        planes = generate_input_planes(stones=stones, end=end, shot=shot, hammer=hammer, score_diff_for_team0=scorediff_for_team0)
                        input_data.append(planes)
                        
                        policy = generate_target_data(selected_move)
                        policy_data.append(policy)
                        
                        value = generate_value_data(scores=scores, end=end, shot_team=shot_team)
                        value_data.append(value)
                        
                        # --- 2. ★追加: 左右反転データの生成 ---
        
                        # (A) 入力平面の反転
                        # planesのshapeは (CHANNELS, HEIGHT, WIDTH) = (53, 56, 32)
                        # axis=2 (Width方向) を反転させる
                        flipped_planes = np.flip(planes, axis=2)
                        input_data.append(flipped_planes)

                        # (B) Policy(正解ラベル)の反転
                        # move情報をコピーして値を書き換える
                        flipped_move = copy.deepcopy(selected_move) # import copy が必要です
                        
                        # vx の符号を反転
                        flipped_move['velocity']['x'] = -1 * flipped_move['velocity']['x']
                        
                        # 回転方向を入れ替え (cw <-> ccw)
                        # カーリングの物理では、左右反転すると曲がる方向も逆になるため回転定義も逆転させる必要がある
                        if flipped_move['rotation'] == 'cw':
                            flipped_move['rotation'] = 'ccw'
                        else:
                            flipped_move['rotation'] = 'cw'
                            
                        flipped_policy = generate_target_data(flipped_move)
                        policy_data.append(flipped_policy)

                        # (C) Valueは盤面を反転しても変わらないので同じ値を使う
                        value_data.append(value)
                        
                        #print(f"shot: {shot}")
                except Exception as e:
                    print(f"Error processing log: {dcl2_data[i]}")
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
                #rint("log counter: ", log_counter)

    # 端数の出力
    n_batches = len(value_data) // BATCH_SIZE
    print("n_batches: ", n_batches)
    if n_batches > 0:
        _save_data(os.path.join(program_dir, "data", f"sl_data_{data_counter}"), \
            input_data[0:n_batches*BATCH_SIZE], policy_data[0:n_batches*BATCH_SIZE], \
            value_data[0:n_batches*BATCH_SIZE], log_counter)
    
    # visualize_data(ibox)
    # visualize_data(vbox)
    


def generate_reinforcement_learning_data(program_dir: str, log_dir: str):
    """自己対戦データをnpzファイルとして生成する。

    Args:
        program_dir (str): 保存するファイルパス。
        log_dir (str): ログデータのディレクトリパス。
    """
    input_data = []
    policy_data = []
    value_data = []

    files = os.listdir(os.path.join(program_dir, "data"))
    log_counter = 1
    data_counter = len(files)
    
    for one_log in os.listdir(log_dir):
        if os.path.isdir(os.path.join(log_dir, one_log)):
            dcl2_path = os.path.join(log_dir, one_log, "game.dcl2")
            if not os.path.exists(dcl2_path):
                continue
            
            with open(dcl2_path) as dclfile:
                dcl2_data = dclfile.readlines()
            dcl_json_data = json.loads(dcl2_data[-2])
            try:
                if dcl_json_data['log']['state'] :
                    print(one_log)
            except KeyError:
                continue
            for log_path in sorted(glob.glob(os.path.join(log_dir, one_log, "*.json"))):
                with open(log_path, 'r') as file:
                    data = json.load(file)
                    try:
                        if data['log']['end'] <= 9 :
                            input_data.append(generate_input_planes(data['log']['simulator_storage']['stones'], data['log']['shot']))
                            policy_data.append(generate_target_data(data['log']['selected_move']))
                            value_data.append(generate_value_data(dcl_json_data, data['log']['end'], data['log']['shot']))
                    except Exception as e:
                            print(f"Error processing log: {log_path}")
                            print(f"Error message: {e}")
                if len(value_data) >= DATA_SET_SIZE:
                    _save_data(os.path.join(program_dir, "data", f"rl_data_{data_counter}"), input_data, policy_data, value_data, log_counter)
                    input_data = input_data[DATA_SET_SIZE:]
                    policy_data = policy_data[DATA_SET_SIZE:]
                    value_data = value_data[DATA_SET_SIZE:]
                    log_counter = 1
                    data_counter += 1
                
                log_counter += 1
    
    # 端数の出力
    n_batches = len(value_data) // BATCH_SIZE
    if n_batches > 0:
        _save_data(os.path.join(program_dir, "data", f"rl_data_{data_counter}"), \
            input_data[0:n_batches*BATCH_SIZE], policy_data[0:n_batches*BATCH_SIZE], \
            value_data[0:n_batches*BATCH_SIZE], log_counter)
