import subprocess
from copy import deepcopy
import json
from dc3client.models import StoneRotation

class CurlingEnv:
    def __init__(self, num_shot=0, end=0, last_end=9, score=None, scores=None, stones=None, team = 'team0'):
        new_stones = []
        if stones is not None:
            for item in stones:
                if item is None:
                    new_stones.append(None)
                else:
                    new_stones.append({
                        "angle": item[0],
                        "position": {
                            "x": item[1],
                            "y": item[2]
                        }

                    })
        else:
            new_stones = [None] * 16

        white_to_move = (num_shot % 2 != 0)

        self.game_state = {
            "num_shot": num_shot,
            "end": end,
            "last_end": last_end,
            "score": score if score is not None else [0] * 10,
            "scores": scores,
            "WhiteToMove": white_to_move,
            "stones": new_stones,
            "team": team
        }



    def step(self, x, y, rotation):
        #実際にショットをシミュレーション
        next_env = deepcopy(self)
        aaaa = next_env.team_check()
        #rint("envdao: ", next_env.game_state["team"])
        rotation_int = 0
        if rotation == StoneRotation.counterclockwise:
            rotation_int = 1
        shot = [self.game_state["num_shot"], rotation_int, x, y]
        converted_stone = []
        for item in self.game_state["stones"]:
            if item is None:
                converted_stone.append(None)
            else:
                if(item["position"]["y"] < 21.0315):
                    print("mainasuaruyo: ", item["position"]["y"])
                converted_stone.append([
                    item["angle"],
                    item["position"]["x"],
                    item["position"]["y"] - 21.0315
                ])
        if(not next_env.team_check()):
            converted_stone = converted_stone[8:16] + converted_stone[0:8]
        stone_str = ' '.join(['null' if item is None else ' '.join(map(str, item)) for item in converted_stone])
        shot_str = ' '.join(map(str, shot))
        #cmd = ['./Simulate_DigitalCurling3/build/Release/simulate_digitalcurling3', stone_str, shot_str]
        cmd = ['./Simulate_DigitalCurling3/build/simulate_digitalcurling3.exe', stone_str, shot_str, "../Starter/kura/config.json"]
        subprocess.run(cmd)
        #jsonファイルから読み取って、各値を更新

        with open('./output.json', 'r') as json_file:
            data = json.load(json_file)
            if data[-1] is not None: #エンド終了時
                #if((data[-1][0] - data[-1][1]) != 3):
                    #rint(data)
                #rint("ooooooooooooo: ",data[-1][0] - data[-1][1])
                    #rint(next_env.game_state["stones"])
                next_env.game_state["stones"] = [None] * 16
                '''if(next_env.team_check()):
                    next_env.game_state["score"][self.game_state["end"]] = data[-1][0] - data[-1][1]
                else:
                    next_env.game_state["score"][self.game_state["end"]] = data[-1][1] - data[-1][0]
                    print(data[-1][1] - data[-1][0])'''
                next_env.game_state["score"][self.game_state["end"]] = data[-1][0] - data[-1][1]
                #この上の行を自分がteam0か1かで条件分岐しなきゃいけなそう。
                if self.game_state["WhiteToMove"] :
                    next_env.game_state["score"][self.game_state["end"]] *= -1
                next_env.game_state["num_shot"] = 0
                next_env.game_state["end"] = self.game_state["end"] + 1
                if next_env.game_state["end"] >= 10 :
                    next_env.game_state["end"] = 0 #とりあえず0にした。もうちょいきれいな方法ありそう。
                next_env.game_state["WhiteToMove"] = False
                #ここでteamを更新するかどうか要検討
            else:
                newstones = []
                for item in data[:-1]:
                    if item is None:
                        newstones.append(None)
                    else:
                        newstones.append({
                            "angle": item[0],
                            "position": {
                                "x": item[1],
                                "y": item[2]#ここも21.0315f補正必要？要検証
                            }
                        })
                if(not next_env.team_check()):
                    newstones = newstones[8:16] + newstones[0:8]
                next_env.game_state["stones"] = newstones
                next_env.game_state["num_shot"] = next_env.game_state["num_shot"] + 1
                next_env.game_state["WhiteToMove"] = not next_env.game_state["WhiteToMove"]
                #ここでteamを更新するかどうか要検討

        return next_env

    def team_check(self):
        if((self.game_state["team"] == 'team0' and (not self.game_state["WhiteToMove"])) or (self.game_state["team"] == 'team1' and (self.game_state["WhiteToMove"]))):
            #rint(11111111)
            return True
        else:
            #rint(22222222)
            return False
            