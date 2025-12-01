import os
import json
import matplotlib.pyplot as plt
from nn.feature import generate_input_planes, generate_target_data, generate_value_data
from board.constant import BOARD_SIZE

DCL2_YPOS_DIFF = 21.0314998626709


def print_histogram(value_data):
    labels = list(range(len(value_data)))  # 0～15 のカテゴリ
    counts = value_data                  # 各カテゴリの出現回数

    plt.figure(figsize=(8,4))
    plt.bar(labels, counts, tick_label=labels)
    plt.xlabel("Value category")
    plt.ylabel("Number of occurrences")
    plt.title("Distribution of Value data")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def generate_supervised_learning_data(
        log_dir: str,
        data_size: int
    ):
    """教師データをnpzファイルとして生成する。

    Args:
        program_dir (str): 保存するファイルパス。
        log_dir (str): ログデータのディレクトリパス。
    """
    input_data = []
    policy_data = [0 for _ in range((BOARD_SIZE ** 2 )* 2)]
    value_data = [0 for _ in range(11)]

    log_counter = 1
    data_counter = 0
    print(f"start generate {data_size} data from {log_dir}!")
    
    game_size = 0

    """
    log_files = os.listdir(log_dir)
    for one_log in random.sample(log_files, len(log_files)):
    """
    for one_log in os.listdir(log_dir):
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
                selected_move = dcl2_log2['move']
                team = dcl2_log2['team']
                for stone in stones:
                    if stone is None:
                        continue
                    stone['position']['y'] -= DCL2_YPOS_DIFF
                try:
                    if end < 10 and shot == 15:
                        planes = generate_input_planes(stones, scores, end, shot)
                        input_data.append(planes)
                        policy = generate_target_data(selected_move)
                        policy_data[policy] += 1
                        value = generate_value_data(dcl2_json_data, end, shot)
                        #if team == "team1":
                        #    value = 10 - value
                        value_data[value] += 1
                        #print(f"shot: {shot}")
                except Exception as e:
                    print(f"Error processing log: {dcl2_data[i]}")
                    print(f"Error message: {e}")
                # print("len(value_data): ", len(value_data), ", DATA_SET_SIZE: ", DATA_SET_SIZE)
    
    print_histogram(value_data)
    
if __name__ == "__main__":
    log_dir = "../LearnLog/cai"
    log_dir = "../LearnLog/jiritsu-vs-silicon"
    data_size = 10000
    generate_supervised_learning_data(log_dir, data_size)
    print("Data generation completed.")