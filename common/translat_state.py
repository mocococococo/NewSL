from dc3client.models import Position, Coordinate, Stones
from typing import List

def convert_stones_to_list(stones: Stones) -> List[dict]:
    result = [None] * 16  # 16要素のリストを作成し、全てをNoneで初期化

    for i, coordinate in enumerate(stones.team0):
        if coordinate.angle is not None and coordinate.position[0].x is not None and coordinate.position[0].y is not None:
            data = {
                "angle": coordinate.angle,
                "angular_velocity": 0.0,
                "linear_velocity": {"x": 0.0, "y": 0.0},
                "position": {"x": coordinate.position[0].x, "y": coordinate.position[0].y}
            }
            result[i] = data

    for i, coordinate in enumerate(stones.team1):
        if coordinate.angle is not None and coordinate.position[0].x is not None and coordinate.position[0].y is not None:
            data = {
                "angle": coordinate.angle,
                "angular_velocity": 0.0,
                "linear_velocity": {"x": 0.0, "y": 0.0},
                "position": {"x": coordinate.position[0].x, "y": coordinate.position[0].y}
            }
            result[i + 8] = data

    return result

