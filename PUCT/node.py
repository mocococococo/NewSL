from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

from PUCT.state import State
from PUCT.edge import Edge
from PUCT.action import ActionSpace

@dataclass
class Node:
    state:          State
    policy:         Dict[int, float] = field(default_factory=dict)  # action_id -> prior
    childlen:       Dict[int, Edge]  = field(default_factory=dict) # action_id
    unlocked_nodes: List["Node"] = field(default_factory=list) # 展開した子ノード
    N:              int   = 0    # visit count
    
    def ensure_child(self, action_id: int) -> Edge:
        """
        指定アクションの Edge を存在させて返す（なければ生成）。

        Args:
            action_id (int): アクションID。

        Returns:
            Edge: 対応するエッジ。
        """
        if action_id not in self.children:
            shot = ActionSpace.decode(action_id)
            prior = self.P[action_id] if self.P else 0.0
            self.children[action_id] = Edge(action_id=action_id, prior=prior, shot=shot)
        return self.children[action_id]
