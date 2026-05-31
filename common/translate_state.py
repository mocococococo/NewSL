from dc3client.models import Stones
from typing import List, Optional, Tuple, Dict

Pos = Tuple[float, float]

def convert_stones_to_list(stones: Stones) -> List[dict]:
    result = [None] * 16  # 16要素のリストを作成し、全てをNoneで初期化

    for i, coordinate in enumerate(stones.team0):
        if coordinate.angle is not None and coordinate.position[0].x is not None and coordinate.position[0].y is not None:
            data = {
                "angle": coordinate.angle,
                "angular_velocity": 0.0,
                "linear_velocity": {"x": 0.0, "y": 0.0},
                "position": {
                    "x": coordinate.position[0].x,
                    "y": coordinate.position[0].y,
                }
            }
            result[i] = data

    for i, coordinate in enumerate(stones.team1):
        if coordinate.angle is not None and coordinate.position[0].x is not None and coordinate.position[0].y is not None:
            data = {
                "angle": coordinate.angle,
                "angular_velocity": 0.0,
                "linear_velocity": {"x": 0.0, "y": 0.0},
                "position": {
                    "x": coordinate.position[0].x,
                    "y": coordinate.position[0].y,
                }
            }
            result[i + 8] = data

    return result

def convert_dc4_stones_to_list(stone_coordinate_data):
    result = [None] * 16

    for i, c in enumerate(stone_coordinate_data["team0"]):
        if not (c.x == 0.0 and c.y == 0.0):
            result[i] = {"position": {"x": c.x, "y": c.y}}

    for i, c in enumerate(stone_coordinate_data["team1"]):
        if not (c.x == 0.0 and c.y == 0.0):
            result[i + 8] = {"position": {"x": c.x, "y": c.y}}

    return result

def convert_scores_to_dict(scores):
    return {"team0": scores.team0, "team1": scores.team1}

def convert_dc4_scores_to_dict(score_obj):
    """
    dc4client.receive_data.ScoreSchema | None
      -> {"team0": list[int], "team1": list[int]}
    """
    if score_obj is None:
        return {"team0": [], "team1": []}

    team0 = score_obj.team0 or []
    team1 = score_obj.team1 or []

    if len(team0) != len(team1):
        raise ValueError(f"score length mismatch: team0={len(team0)} team1={len(team1)}")

    return {
        "team0": [0 if v is None else int(v) for v in team0],
        "team1": [0 if v is None else int(v) for v in team1],
    }

def stones_listdict_to_xy16(stones_list: List[Optional[dict]]) -> List[Optional[Pos]]:
    """convert_stones_to_list() の出力(list[dict|None]) -> list[(x,y)|None]"""
    out: List[Optional[Pos]] = [None] * 16
    for i, s in enumerate(stones_list):
        if s is None:
            continue
        out[i] = (float(s["position"]["x"]), float(s["position"]["y"]))
    return out

def scores_dict_to_list(scores: Dict[str, List[Optional[int]]]) -> List[Optional[Tuple[int, int]]]:
    team0 = scores["team0"]
    team1 = scores["team1"]
    if len(team0) != len(team1):
        raise ValueError(f"score length mismatch: team0={len(team0)} team1={len(team1)}")

    out: List[Optional[Tuple[int, int]]] = []
    for a, b in zip(team0, team1):
        a0 = 0 if a is None else int(a)
        b0 = 0 if b is None else int(b)
        out.append((a0, b0))
    return out

def scores_to_scorediff_for_team0(scores: Dict[str, List[Optional[Tuple[int, int]]]]) -> int:
    score_diff_for_team0 = 0
    for a, b in zip(scores['team0'], scores['team1']):
        a0 = 0 if a is None else int(a)
        b0 = 0 if b is None else int(b)
        score_diff_for_team0 += a0 - b0
    return score_diff_for_team0

def convert_team_stoi(team: str) -> int:
    """チーム名を整数に変換する。team0 -> 0, team1 -> 1"""
    if team == "team0":
        return 0
    elif team == "team1":
        return 1
    else:
        raise ValueError(f"Invalid team name: {team}")

def get_hammer_team(team: str, shot: int) -> int:
    """ハンマーを持っているチームを整数に変換する。team0 -> 0, team1 -> 1"""
    if team == "team0":
        if shot % 2 == 0:
            return 1
        elif shot % 2 == 1:
            return 0
        else:
            raise ValueError(f"Invalid shot number: {shot}")
    elif team == "team1":
        if shot % 2 == 0:
            return 0
        elif shot % 2 == 1:
            return 1
        else:
            raise ValueError(f"Invalid shot number: {shot}")
    else:
        raise ValueError(f"Invalid team name: {team}")