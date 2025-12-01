# -*- coding: utf-8 -*-
"""
モデルの Value ヘッド出力分布を集計するワンショットスクリプト
--------------------------------------------------------------
- checkpoint をロードして forward_for_sl() で推論
- softmax 確率の平均 & argmax クラスの出現回数を収集
- 結果をコンソールに表示し、ヒストグラムを描画（任意）
"""

import glob
import os
import json

import click
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from nn.network.dual_net import DualNet
from nn.feature import generate_input_planes, generate_target_data, generate_value_data
from nn.utility import load_data_set, get_torch_device
from board.constant import PLANES_SIZE, BOARD_SIZE
from learning_param import BATCH_SIZE



def print_histogram(value_data):
    labels = list(range(len(value_data)))  # 0～16 のカテゴリ
    counts = value_data                  # 各カテゴリの出現回数

    plt.figure(figsize=(8,4))
    plt.bar(labels, counts, tick_label=labels)
    plt.xlabel("Value category")
    plt.ylabel("Number of occurrences")
    plt.title("Distribution of Value data")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


@click.command()
@click.option(
    "--data_size",
    type=int,
    default=1000,
    help="試合数"
)
@click.option(
    "--model",
    type=str,
    default="model/cai40000CP-96-15-16shot-5to5.bin",
    help="学習済みモデルファイル (.pt / .bin)"
)
@click.option(
    "--log_dir",
    type=str,
    default="../LearnLog/cai",
    help="評価用データの glob パターン (例: data_dcl2/sl_data_*.npz)"
)
@click.option(
    "--gpu",
    type=bool,
    default=True,
    help="GPU を使う場合は --gpu を指定 (デフォルト: CPU)"
)
def main(data_size: int, model: str, log_dir: str, gpu: bool):
    """モデルの Value ヘッド出力分布を集計する."""
    # デバイス設定
    device = get_torch_device(use_gpu=gpu)
    net = DualNet(device=device).to(device)
    state = torch.load(model, map_location=device)
    net.load_state_dict(state)
    net.eval()

    # デバッグ
    print(f"log_dir = {log_dir}")
    game_size = 0

    value_list = [0 for _ in range(11)]

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
                end = dcl2_state['end']
                shot = dcl2_state['shot']
                team = dcl2_log2['team']
                selected_move = dcl2_log2['move']
                input_planes = generate_input_planes(
                    stones=stones,
                    scores=scores,
                    end=end,
                    shot=shot,
                )
                input_data = torch.tensor(input_planes.reshape(1, PLANES_SIZE, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32).to(device)
                value_data = generate_value_data(dcl2_json_data, end, shot)
                policy, value = net.forward_with_softmax2(input_data)
                
                if shot == 15:
                    pred_classes = value.argmax(dim=1)
                    pred_class = pred_classes[0].item()
                    value_list[pred_class] += 1

    print_histogram(value_list)

if __name__ == "__main__":
    main()
