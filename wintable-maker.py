"""各プログラムの対戦データから勝率テーブルを作成するプログラム"""
import os
import json
import shutil
from typing import List, NoReturn
from pathlib import Path

from common.translate_state import convert_team_stoi, scores_to_scorediff_for_team0

def count_subdirectories(path) -> int:
    return len([p for p in Path(path).iterdir() if p.is_dir()])
                
def calc_winrate(log_dir: str, output_path: str = "wintable.py") -> NoReturn:
    game_count = 0
    error_count = 0
    
    # 勝った回数を記録するテーブル
    win_counts = {
        "non-hammer": {i: [0]*17 for i in range(12)},
        "hammer": {i: [0]*17 for i in range(12)}
    }
    
    # その場面が出現した総回数を記録するテーブル（分母用）
    total_counts = {
        "non-hammer": {i: [0]*17 for i in range(12)},
        "hammer": {i: [0]*17 for i in range(12)}
    }
    
    matchs = count_subdirectories(log_dir)
    for one_log in os.listdir(log_dir):
        # if game_count > 1:
        #     break
        if not os.path.isdir(os.path.join(log_dir, one_log)):
            continue
            
        dcl2_path = os.path.join(log_dir, one_log, "game.dcl2")
        if not os.path.exists(dcl2_path):
            print(one_log, ": dcl2 not exist")
            continue
        
        with open(dcl2_path) as dclfile:
            dcl2_data = dclfile.readlines()
            dcl_json_data = json.loads(dcl2_data[-2])
        
        # データの整合性チェック
        if not "state" in dcl_json_data["log"]:
            print(one_log, ": state not exist")
            error_count += 1
            continue
        if dcl_json_data["log"]["state"] is None:
            error_count += 1
            print(one_log, ": state is None")
            # shutil.rmtree(os.path.join(log_dir, one_log))
            continue

        state_data = dcl_json_data["log"]["state"]
        if state_data["game_result"] is None:
            print(one_log, ": game_result is None")
            # shutil.rmtree(os.path.join(log_dir, one_log))
            error_count += 1
            continue
        
        game_count += 1
        winner = state_data["game_result"]["winner"]
        
        for i in range(9, len(dcl2_data)-2, 32):
            dcl2_state = json.loads(dcl2_data[i])['log']['state']
            scores = dcl2_state['scores']
            scorediff_for_team0 = scores_to_scorediff_for_team0(scores)
            if scorediff_for_team0 < -8:
                scorediff_for_team0 = -8
            elif scorediff_for_team0 > 8:
                scorediff_for_team0 = 8
            diff_index_for_0 = scorediff_for_team0 + 8
            diff_index_for_1 = 16 - diff_index_for_0
            end = dcl2_state['end']
            hammer = convert_team_stoi(dcl2_state['hammer'])
            
            if hammer == 0:
                total_counts["hammer"][end][diff_index_for_0] += 1
                total_counts["non-hammer"][end][diff_index_for_1] += 1
            elif hammer == 1:
                total_counts["non-hammer"][end][diff_index_for_0] += 1
                total_counts["hammer"][end][diff_index_for_1] += 1

            if winner == "team0":
                if hammer == 0:
                    # team0 が後攻で勝った時に後攻にとっての勝率テーブルを更新
                    win_counts["hammer"][end][diff_index_for_0] += 1
                elif hammer == 1:
                    # team0 が先攻で勝った時に先攻にとっての勝率テーブルを更新
                    win_counts["non-hammer"][end][diff_index_for_0] += 1
            elif winner == "team1":
                if hammer == 0:
                    # team1 が先攻で勝った時に先攻にとっての勝率テーブルを更新
                    win_counts["non-hammer"][end][diff_index_for_1] += 1
                elif hammer == 1:
                    # team1 が後攻で勝った時に後攻にとっての勝率テーブルを更新
                    win_counts["hammer"][end][diff_index_for_1] += 1
        
        print(f"Processed log: {one_log}", f"(game_count: {game_count}, total: {matchs}, errors: {error_count})")
        
    # 勝率テーブルの計算 (Win / Total)
    final_wintable = {
        "non-hammer": {},
        "hammer": {}
    }

    for category in ["non-hammer", "hammer"]:
        for end in win_counts[category]:
            final_wintable[category][end] = []
            for diff_idx in range(17):
                wins = win_counts[category][end][diff_idx]
                total = total_counts[category][end][diff_idx]
                
                if total > 0:
                    # 小数点第3位などで丸める
                    rate = round(wins / total, 3)
                    final_wintable[category][end].append(rate)
                else:
                    # データがない場合は None とする（wintable.pyの形式に準拠）
                    final_wintable[category][end].append(None)

    # Pythonファイルとして保存
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("WIN_TABLE = {\n")
        
        for category in ["non-hammer", "hammer"]:
            f.write(f'    "{category}": {{\n')
            
            # エンド順にソートして出力
            for end in sorted(final_wintable[category].keys()):
                data_list = final_wintable[category][end]
                
                f.write(f'        {end}: [\n')
                
                # 要素を文字列化して整形 (None は "None", 数値は ".3f" など)
                formatted_elements = []
                for val in data_list:
                    if val is None:
                        formatted_elements.append("None")
                    else:
                        formatted_elements.append(f"{val:.3f}")
                
                # Pythonファイルとして保存
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("WIN_TABLE = {\n")
        
        for category in ["non-hammer", "hammer"]:
            f.write(f'    "{category}": {{\n')
            
            # エンド順にソートして出力
            for end in sorted(final_wintable[category].keys()):
                data_list = final_wintable[category][end]
                
                f.write(f'        {end}: [\n')
                
                # 要素を文字列化して整形 (None は "None", 数値は ".3f" など)
                formatted_elements = []
                for val in data_list:
                    if val is None:
                        formatted_elements.append("None")
                    else:
                        formatted_elements.append(f"{val:.3f}")
                
                # 1行目: 前半8要素 (インデックス 0-7)
                line1 = ", ".join(formatted_elements[:8])
                f.write(f"            {line1},\n")
                
                # 2行目: 中央1要素 (インデックス 8)
                line2 = formatted_elements[8]
                f.write(f"            {line2},\n")
                
                # 3行目: 後半8要素 (インデックス 9-16)
                line3 = ", ".join(formatted_elements[9:])
                f.write(f"            {line3},\n")
                
                f.write('        ],\n')
            f.write('    },\n')
        f.write('}\n')
        
    print(f"Saved wintable to {output_path}")
    

if __name__ == "__main__":
    dir = "./LearnLog/cai"
    calc_winrate(dir)