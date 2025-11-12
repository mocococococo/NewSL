import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class State:
    stones:         List[Optional[Tuple[float, float]]] # 各石の位置 (x, y) または None (未投)
    hammer:         bool        # True: 後攻, False: 先攻
    throw_index:    int         # 次に投げるショットのインデックス (0-15)
    end:            int         # 現在のエンド (0-9)
    score_diff:     int         # 現在のスコア差 (to-play視点)
    

    def is_end_and_score(self) -> Tuple[bool, int]:
        """
        現局面が終端かどうかと、終端ならエンドスコア（to-play視点でなく客観値）を返す。

        Args:
            state (State): 評価対象の局面。

        Returns:
            Tuple[bool, int]: (終端か, スコア[-8..+8])。
        """
        # ---- 実装者が差し替える想定 ----
        # ダミー：投番==15 で終端、スコア0固定
        terminal = (self.throw_index >= 15)
        score = 0
        return terminal, score