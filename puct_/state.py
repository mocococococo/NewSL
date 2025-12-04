import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

from .params import STATE_POS_SCALE

Team = int  # 0=先攻, 1=後攻
Pos = Tuple[float, float]  # (x, y) 座標
Stones = Tuple[Optional[Pos], ...]  # 各石の位置 (x, y) または None (未投)

@dataclass
class State:
    stones:         Stones # 各石の位置 (x, y) または None (未投)
    hammer:         bool        # True: 後攻, False: 先攻
    shot_index:     int         # 次に投げるショットのインデックス (0-15)
    end:            int         # 現在のエンド (0-9)
    score_diff:     int         # 現在のスコア差 (team0 - team1)
    
    # 以下メソッド群
    
    def __post_init__(self) -> None:
        if len(self.stones) != 16:
            raise ValueError(f"stones must have length 16, got {len(self.stones)}")
        if not (0 <= self.shot_index <= 16):
            raise ValueError(f"shot_index must be in [0,16], got {self.shot_index}")
    
    def is_end_terminal(self) -> bool:
        """エンドが終端か（16投完了）"""
        return self.shot_index >= 16
    
    # --- チーム計算 ---
    def hammer_team(self) -> Team:
        """このエンドでハンマーを持つチーム（0/1）"""
        return 1 if self.hammer else 0

    def to_move(self) -> Team:
        """
        次に投げるチーム（0 or 1）。
        shot_index が偶数: 非ハンマー
        shot_index が奇数: ハンマー
        """
        if self.is_end_terminal():
            return -1  # 終端では未定義
        h = self.hammer_team()
        nh = 1 - h
        return nh if (self.shot_index % 2 == 0) else h
    
    def key(self, pos_scale: int = STATE_POS_SCALE) -> Tuple:
        """
        transposition 用キー。
        - 浮動小数の誤差対策で座標を量子化
        - 同一チーム内の石は同質として扱い、チームごとに座標をソートして正規化
        """
        def q(v: float) -> int:
            return int(round(v * pos_scale))

        team0 = []
        team1 = []
        for i, p in enumerate(self.stones):
            if p is None:
                continue
            xq, yq = q(p[0]), q(p[1])
            (team0 if i < 8 else team1).append((xq, yq))

        team0.sort()
        team1.sort()

        # 状態同一性に影響する情報は全部入れる（end/shot/hammer/score_diff）
        return (self.end, self.shot_index, int(self.hammer), self.score_diff, tuple(team0), tuple(team1))
    
    @staticmethod
    def initial(end: int, hammer: bool, stones: List[Optional[Pos]], score_diff: int) -> 'State':
        """エンド開始状態を作る（shot_index=0）"""
        return State(stones=tuple(stones), hammer=hammer, shot_index=0, end=end, score_diff=score_diff)
    
# puct_search 側の呼び出し形式に合わせた薄い関数（任意）
def is_end_terminal(state: State) -> bool:
    """search.py の is_end_terminal(state) 呼び出しに合わせるためのラッパ。"""
    return state.is_end_terminal()

def score_diff_from_scores(scores: List[Optional[Tuple[int, int]]]) -> int:
    """
    scores から score_diff を計算する補助関数。
    """
    team0_score = sum(s[0] for s in scores if s is not None)
    team1_score = sum(s[1] for s in scores if s is not None)
    return team0_score - team1_score