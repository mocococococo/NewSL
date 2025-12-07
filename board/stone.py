from enum import Enum

class Stone(Enum):
    #ストーンの色を表すクラス
    EMPTY = 0
    FIRST = 1
    LAST = 2
    OUT_OF_BOARD = 3

    @classmethod
    def get_opponent_color(cls, color):
        #相手の色を取得する
        if color == Stone.FIRST:
            return Stone.LAST
        
        if color == Stone.LAST:
            return Stone.FIRST
        
        return color

    @classmethod
    def get_char(cls, color) -> str:
        #色に対応する文字を取得する
        if color == Stone.EMPTY:
            return '+'

        if color == Stone.FIRST:
            return '@'

        if color == Stone.LAST:
            return 'O'

        if color == Stone.OUT_OF_BOARD:
            return '#'

        return '!'