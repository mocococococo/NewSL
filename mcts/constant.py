#シミュレータに依存する定数だから、絶対に変える！！！！
import math

param = 0.145

STD_X = param * 0.5
STD_Y = param * 1.5

VAR_X = math.pow((param * 0.5), 2.0)
VAR_Y = math.pow((param * 1.5), 2.0)