from dataclasses import dataclass
from typing import Optional

from PUCT.shot import Shot
from PUCT.node import Node

@dataclass
class Edge:
    action_id:  int
    Shot:       Shot
    P:          float        # prior
    N:          int   = 0    # visit count
    W:          float = 0.0  # total value
    Q:          float = 0.0  # mean value
    child:      Optional[Node] = None  # child node
    illegal:    bool  = False  # 無効投なら True, 以後探索しない

    def update_backup(self, incoming_value: float) -> None:
        """バックアップ用：訪問回数と累積価値を更新してQを再計算する。
        
        Args:
            incoming_value: このエッジから見た価値（to-play視点で符号処理済み）
        """
        self.edge_visit_count += 1
        self.edge_value_sum += incoming_value
        self.edge_q_value = self.edge_value_sum / self.edge_visit_count