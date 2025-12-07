#盤面のデータ定義と操作処理

from typing import List, NoReturn
from collections import deque
import numpy as np

from board.constant import OB_SIZE
from board.stone import Stone

class CurlingBoard:
    #盤面クラス
    def __init__(self, board_size: int):
        #盤面クラスの初期化
        #board_size(int):盤面の大きさ
        self.board_size = board_size
        self.board_size_with_ob = board_size + OB_SIZE * 2

        def pos(x_coord: int, y_coord: int) -> int:
            #座標を1次元配列のインデックスに変換する。
            return x_coord + y_coord * self.board_size_with_ob

        self.board = [Stone.EMPTY] * (self.board_size_with_ob ** 2) #盤面初期化
        