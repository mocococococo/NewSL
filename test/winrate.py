"""各プログラムの対戦データから勝率を算出するプログラム
"""
import os
import json
import shutil
from pathlib import Path
import matplotlib.pyplot as plt

class team:
    def __init__(self, name: str):
        self.name = name
        self.win_first = 0
        self.lose_first = 0
        self.win_second = 0
        self.lose_second = 0
        self.score_first = 0
        self.score_second = 0

        self.time_first = 0
        self.time_second = 0
        self.match_first = 0
        self.match_second = 0
        
    def add_win_first(self, score: int, time: int):
        self.win_first += 1
        self.score_first += score
        self.time_first += time
        self.match_first += 1
        
    def add_lose_first(self, score: int, time: int):
        self.lose_first += 1
        self.score_first += score
        self.time_first += time
        self.match_first += 1
        
    def add_win_second(self, score: int, time: int):
        self.win_second += 1
        self.score_second += score
        self.time_second += time
        self.match_second += 1
    
    def add_lose_second(self, score: int, time: int):
        self.lose_second += 1
        self.score_second += score
        self.time_second += time
        self.match_second += 1
        
    def get_winrate(self):
        if self.match_first == 0 and self.match_second != 0:
            return 0.0, \
                self.win_second * 100 / self.match_second, \
                self.win_second * 100 / self.match_second
        if self.match_second == 0 and self.match_first != 0:
            return self.win_first * 100 / self.match_first, \
                    0.0, \
                    self.win_first * 100 / self.match_first
        return self.win_first * 100 / self.match_first, \
                self.win_second * 100 / self.match_second, \
                (self.win_first + self.win_second) * 100 / (self.match_first + self.match_second)
                
    def get_score(self):
        if self.match_first == 0 and self.match_second != 0:
            return 0.0, \
                self.score_second / self.match_second, \
                self.score_second / self.match_second
        if self.match_second == 0 and self.match_first != 0:
            return self.score_first / self.match_first, \
                    0.0, \
                    self.score_first / self.match_first
        return self.score_first / self.match_first, \
                self.score_second / self.match_second, \
                (self.score_first + self.score_second) / (self.match_first + self.match_second)
                
    def get_time(self):
        return self.time_first / self.match_first, \
                self.time_second / self.match_second, \
                (self.time_first + self.time_second) / (self.match_first + self.match_second)
        

def count_subdirectories(path) -> int:
    return len([p for p in Path(path).iterdir() if p.is_dir()])
                
def calc_winrate(log_dir: str, teamA:str, teamB: str):
    
    teamA = team(teamA)
    teamB = team(teamB)
    
    history_x = []
    history_wr = []
    
    game_count = 0
    error_count = 0
    
    matchs = count_subdirectories(log_dir)
    for one_log in os.listdir(log_dir):
        # if game_count > 99:
        #     break
        if os.path.isdir(os.path.join(log_dir, one_log)):
            dcl2_path = os.path.join(log_dir, one_log, "game.dcl2")
            if not os.path.exists(dcl2_path):
                # error_count += 1
                print(one_log, ": dcl2 not exist")
                #shutil.rmtree(os.path.join(log_dir, one_log))
                continue
            with open(dcl2_path) as dclfile:
                dcl2_data = dclfile.readlines()
            dcl_json_data = json.loads(dcl2_data[-2])
            team0 = json.loads(dcl2_data[3])["log"]["name"]
            team1 = json.loads(dcl2_data[4])["log"]["name"]
            # print("team0 :", team0, ", team1 :", team1)
            
            if not "state" in dcl_json_data["log"]:
                print(one_log, ": state not exist")
                error_count += 1
                continue
            if dcl_json_data["log"]["state"] is None:
                error_count += 1
                print(one_log, ": state is None")
                #shutil.rmtree(os.path.join(log_dir, one_log))
                continue
            else:
                state_data = dcl_json_data["log"]["state"]
                if state_data["game_result"] is None:
                    print(one_log, ": game_result is None")
                    shutil.rmtree(os.path.join(log_dir, one_log))
                    error_count += 1
                    continue
                else:
                    if (not all(isinstance(i, int) for i in state_data["scores"]["team0"])) or \
                        (not all(isinstance(i, int) for i in state_data["scores"]["team1"])):
                        print(one_log, ": scores is None, time over")
                        shutil.rmtree(os.path.join(log_dir, one_log))
                        error_count += 1
                        continue
                    
                    game_count += 1
                    winner = state_data["game_result"]["winner"]
                    team0_score = sum(state_data["scores"]["team0"])
                    team1_score = sum(state_data["scores"]["team1"])
                    team0_time = state_data["thinking_time_remaining"]["team0"]
                    team1_time = state_data["thinking_time_remaining"]["team1"]
                    if winner == "team0":
                        if team0 == teamA.name and team1 == teamB.name:
                            teamA.add_win_first(team0_score, team0_time)
                            teamB.add_lose_second(team1_score, team1_time)
                            #print(one_log, f": team0 {teamA.name}, team1 {teamB.name}, win {teamA.name}")
                        elif team0 == teamB.name and team1 == teamA.name:
                            teamB.add_win_first(team0_score, team0_time)
                            teamA.add_lose_second(team1_score, team1_time)
                            #print(one_log, f": team0 {teamB.name}, team1 {teamA.name}, win {teamB.name}")
                        else:
                            print(one_log, ": winner is team0, but team info is invalid")
                            print(one_log, f": team0 : {team0}, team1 : {team1}")

                    elif winner == "team1":
                        if team1 == teamA.name and team0 == teamB.name:
                            teamA.add_win_second(team1_score, team1_time)
                            teamB.add_lose_first(team0_score, team0_time)
                            #print(one_log, f": team0 {teamB.name}, team1 {teamA.name}, win {teamA.name}")
                        elif team1 == teamB.name and team0 == teamA.name:
                            teamB.add_win_second(team1_score, team1_time)
                            teamA.add_lose_first(team0_score, team0_time)
                            #print(one_log, f": team0 {teamA.name}, team1 {teamB.name}, win {teamB.name}")
                        else:
                            print(one_log, ": winner is team1, but team info is invalid")
                            print(f": team0 : {team0}, team1 : {team1}")
            # 累積勝率を積む
            _, _, total_wr = teamA.get_winrate()
            history_x.append(game_count)
            history_wr.append(total_wr)
        else:
            # print(one_log, ": not directory")
            continue
    
    print("matchs :", matchs, ", game_count :", game_count, ", error_count :", error_count)
    print(f"game_count: {teamA.match_first + teamA.match_second}")
    print(f"{teamA.name} winrate : {teamA.get_winrate()[0]:.2f}, {teamA.get_winrate()[1]:.2f}, {teamA.get_winrate()[2]:.2f}")
    print(f"{teamA.name} win : {teamA.win_first}, {teamA.win_second}, {teamA.win_first + teamA.win_second}")
    print(f"{teamA.name} lose : {teamA.lose_first}, {teamA.lose_second}, {teamA.lose_first + teamA.lose_second}")
    print(f"{teamA.name} score : {teamA.get_score()[0]:.2f}, {teamA.get_score()[1]:.2f}, {teamA.get_score()[2]:.2f}")
    #print(f"{teamA.name} time : {teamA.get_time()[0]:.2f}, {teamA.get_time()[1]:.2f}, {teamA.get_time()[2]:.2f}")
    #print(f"{teamB.name} winrate : {teamB.get_winrate()[0]:.2f}, {teamB.get_winrate()[1]:.2f}, {teamB.get_winrate()[2]:.2f}")
    #print(f"{teamB.name} win : {teamB.win_first}, {teamB.win_second}, {teamB.win_first + teamB.win_second}")
    #print(f"{teamB.name} lose : {teamB.lose_first}, {teamB.lose_second}, {teamB.lose_first + teamB.lose_second}")
    print(f"{teamB.name} score : {teamB.get_score()[0]:.2f}, {teamB.get_score()[1]:.2f}, {teamB.get_score()[2]:.2f}")
    #print(f"{teamB.name} time : {teamB.get_time()[0]:.2f}, {teamB.get_time()[1]:.2f}, {teamB.get_time()[2]:.2f}")
    
    return history_x, history_wr

def plot_winrate_history(x, y, teamA_name: str, teamB_name: str, out_path="winrate.png", moving_avg_window: int = 0):
    plt.figure()
    plt.plot(x, y, label="winrate (%)")

    if moving_avg_window and moving_avg_window > 1 and len(y) >= moving_avg_window:
        # 単純移動平均（端は描画しない）
        ma = []
        mx = []
        s = 0.0
        for i, v in enumerate(y):
            s += v
            if i >= moving_avg_window:
                s -= y[i - moving_avg_window]
            if i >= moving_avg_window - 1:
                ma.append(s / moving_avg_window)
                mx.append(x[i])
        plt.plot(mx, ma, label=f"moving avg ({moving_avg_window})")

    plt.xlabel("match count")
    plt.ylabel("teamA winrate (%)")
    plt.title(f"{teamA_name} vs {teamB_name} winrate transition")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)

if __name__ == "__main__":
    dir = "./log"
    dir = "./cai-vs-puct-wintable-10000-ver2.1"
    teamA = "PUCT_NewSL"
    teamB = "CAI-chan"
    # teamB = "NewSL"
    x, wr = calc_winrate(dir, teamA, teamB)

    plot_winrate_history(
        x, wr,
        teamA_name=teamA,
        teamB_name=teamB,
        out_path=f"winrate_{Path(dir).name}.png",
        moving_avg_window=50,
    )
    print("saved: winrate.png")