from dataclasses import dataclass

@dataclass
class Shot:
    vx: float # ショットの初速 x ベクトル
    vy: float # ショットの初速 y ベクトル
    spin: int # 回転方向 0 or 1
