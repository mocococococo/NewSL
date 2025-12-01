from typing import Tuple

from PUCT.state import State
from PUCT.shot import Shot

import fast_simulator as fs

class Simulator:
    """
    1手のショットを適用する高速シミュレータのインタフェース。

    仕様:
        - ルール違反（フリーガードゾーンなど）は simulator 側で判定することを推奨。
        - もし不可能な場合は「ショット前後の状態一致＋ε内判定」を別途実装する。
    """
    def step(self, state: State, shot: Shot) -> Tuple[State, bool]:
        """
        1手だけ状態遷移を行う。

        Args:
            state (State): 現在の局面。
            shot  (Shot): 実行するショット。

        Returns:
            Tuple[State, bool]: (遷移後の局面, 合法フラグ) を返す。
        """
        # ---- 実装者が差し替える想定 ----
        # ダミー：投番+1 と手番トグルだけを進める（物理遷移は未実装）
        next_state = State(
            stones = 
                fs.simulate(
                    stones = state.stones,
                    index = state.throw_index,
                    shot = shot,
                    freeguard = True if state.throw_index < 5 else False,
                    rink_only = True,
                ),
            hammer = not state.hammer,
            throw_index = state.throw_index + 1,
            end = state.end,
            score_diff = state.score_diff,
        )
        legal = self.is_legal(state.stones, next_state.stones)
        
        return next_state, legal
    
    def is_legal(self, bofore_stones, after_stones) -> bool:
        """
        ショット前後の盤面を比較して合法手かどうかを判定する。

        Args:
            bofore_stones: ショット前の石配置。
            after_stones: ショット後の石配置。
        
        Returns:
            bool: 合法手なら True, 無効投なら False を返す。
        """
        return bofore_stones != after_stones
        