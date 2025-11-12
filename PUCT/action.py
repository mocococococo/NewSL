from typing import List, Tuple

from board.constant import BOARD_SIZE
from PUCT.shot import Shot

class ActionSpace:
    """
    32×32×2 に離散化されたショット空間の変換ユーティリティ。

    - vx ∈ [-0.25, +0.25] を 32 等分
    - vy ∈ [ +2.2,  +3.5] を 32 等分
    - spin ∈ {0(cw), 1(ccw)}
    """

    VX_MIN = -0.25
    VX_MAX = 0.25
    VY_MIN = 2.2
    VY_MAX = 3.5
    VX_BINS = 32
    VY_BINS = 32
    SPIN_BINS = 2
    ACTION_SIZE = VX_BINS * VY_BINS * SPIN_BINS

    @classmethod
    def _linspace(cls, lo: float, hi: float, bins: int) -> List[float]:
        """閉区間 [lo, hi] を bins 等分した中心値リストを返す。"""
        if bins == 1:
            return [(lo + hi) / 2.0]
        step = (hi - lo) / (bins - 1)
        return [lo + i * step for i in range(bins)]

    @classmethod
    def decode(cls, action_id: int) -> Shot:
        """
        action_id から (vx, vy, spin) を復元する。

        Args:
            action_id (int): 0〜2047 のアクションID。

        Returns:
            Shot: 復元されたショット。
        """
        assert 0 <= action_id < cls.ACTION_SIZE
        spin = action_id % 2
        rem = action_id // 2
        iy = rem % cls.VY_BINS
        ix = rem // cls.VY_BINS

        vx_vals = cls._linspace(cls.VX_MIN, cls.VX_MAX, cls.VX_BINS)
        vy_vals = cls._linspace(cls.VY_MIN, cls.VY_MAX, cls.VY_BINS)
        return Shot(vx=vx_vals[ix], vy=vy_vals[iy], spin=spin)

    @classmethod
    def encode(cls, vx: float, vy: float, spin: int) -> int:
        """
        連続値 (vx, vy, spin) を最も近い離散インデックスに丸めて action_id を返す。

        Args:
            vx (float): x方向初速。
            vy (float): y方向初速。
            spin (int): 0 または 1。

        Returns:
            int: 0〜2047 のアクションID。
        """
        def nearest_idx(vals: List[float], x: float) -> int:
            best = 0
            best_diff = float("inf")
            for i, v in enumerate(vals):
                d = abs(v - x)
                if d < best_diff:
                    best = i
                    best_diff = d
            return best

        vx_vals = cls._linspace(cls.VX_MIN, cls.VX_MAX, cls.VX_BINS)
        vy_vals = cls._linspace(cls.VY_MIN, cls.VY_MAX, cls.VY_BINS)

        ix = nearest_idx(vx_vals, vx)
        iy = nearest_idx(vy_vals, vy)
        assert spin in (0, 1)
        return ((ix * cls.VY_BINS) + iy) * 2 + spin