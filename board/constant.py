##ボードに関する定数

#画像の一辺のサイズ
BOARD_SIZE = 32
#盤外のサイズ
OB_SIZE = 1

#座標の範囲、coordinate.hppより
X_MIN = -2.375
X_MAX = 2.375
Y_MIN = 10.9725
Y_MAX = 19.2025
Y_TEE = 17.3735
R_HOUSE = 1.829

#初速ベクトルの範囲、ホッグライン超える大きさから、テイクアウト意識したものまで
VX_MIN = -0.25
VX_MAX = 0.25
VY_MIN = 2.2 #2.2にしたほうがよい？
VY_MAX = 3.5


#着手履歴の最大数
MAX_RECORDS = (BOARD_SIZE ** 2) * 3

#特徴平面のサイズ
PLANES_SIZE = 53