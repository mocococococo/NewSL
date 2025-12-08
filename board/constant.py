##ボードに関する定数

#画像の一辺のサイズ
BOARD_SIZE_X = 32
BOARD_SIZE_Y = 56
#vx, vyの離散化数
VX_SIZE = 32
VY_SIZE = 25
#盤外のサイズ
OB_SIZE = 1
#ストーンの半径
STONE_RADIUS = 0.145
#座標変換用定数
DCL2_YPOS_DIFF = 21.0314998626709

#座標の範囲、coordinate.hppより
X_MIN = -2.375
X_MAX = 2.375
Y_MIN = 10.9725
Y_MAX = 19.3475
Y_TEE = 17.3735
R_HOUSE = 1.829

#初速ベクトルの範囲、ホッグライン超える大きさから、テイクアウト意識したものまで
VX_MIN = -0.25
VX_MAX = 0.25
VY_MIN = 2.21
VY_SHEET_MAX = 2.5
VY_MAX = 3.5


#着手履歴の最大数
MAX_RECORDS = (BOARD_SIZE_X ** BOARD_SIZE_Y) * 3

#特徴平面のサイズ
PLANES_SIZE = 53
