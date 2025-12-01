from dataclasses import dataclass

@dataclass
class PUCTConfig:
    """
    PUCT と Progressive Widening の設定。

    Attributes:
        cpuct (float): 探索バイアスの強さ。
        widen_c0 (int): 最低解放数 c0（M(N)=max(c0, floor(k*N^a))）。
        widen_k (float): k。
        widen_a (float): a。
        root_decision (str): ルートの手の確定規則。"N" or "Q"。
    """
    cpuct: float = 1.5
    widen_c0: int = 4
    widen_k: float = 1.0
    widen_a: float = 0.5
    root_decision: str = "N"  # or "Q"