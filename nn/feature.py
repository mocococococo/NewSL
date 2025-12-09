"""ニューラルネットワークの入力特徴生成処理
"""
import numpy as np

from typing import List, Optional, Tuple, Dict
from board.constant import BOARD_SIZE_X, BOARD_SIZE_Y, STONE_RADIUS, \
                            X_MIN, X_MAX, Y_MIN, Y_MAX, Y_TEE, \
                            R_HOUSE, VX_MIN, VX_MAX, VY_MIN, VY_MAX, \
                            PLANES_SIZE, VX_SIZE, VY_SIZE, VY_SHEET_MAX

def discretization(x: float, y: float) -> int:
    """
    連続座標 (x,y) を BOARD_SIZE_X × BOARD_SIZE_Y 個のセルに割り当て、セルの1次元indexを返す
    """
    # セル幅
    dx = (X_MAX - X_MIN) / BOARD_SIZE_X
    dy = (Y_MAX - Y_MIN) / BOARD_SIZE_Y

    # clamp（境界ちょうども最後のセルに入れたいので X_MAX/Y_MAX を少し内側扱い）
    x = max(X_MIN, min(x, X_MAX))
    y = max(Y_MIN, min(y, Y_MAX))

    # どのセルか（floor）
    xi = int((x - X_MIN) / dx)
    yi = int((y - Y_MIN) / dy)

    # x==X_MAX 等で xi==BOARD_SIZE_X になり得るので丸める
    xi = max(0, min(xi, BOARD_SIZE_X - 1))
    yi = max(0, min(yi, BOARD_SIZE_Y - 1))

    return yi * BOARD_SIZE_X + xi

def discretization_velocity(vx: float, vy: float) -> int:
    """
    連続速度 (vx,vy) を VX_SIZE × VY_SIZE 個のセルに割り当て、セルの1次元indexを返す
    x 軸は VX_MIN から VX_MAXで均等に分割
    y 軸は VY_MIN から VY_SHEET_MAX まで均等に VY_SIZE - 5 分割し、VY_SHEET_MAX から VY_MAX までは別途均等に 5 分割する
    """
    # セル幅
    dvx = (VX_MAX - VX_MIN) / VX_SIZE
    dvy = (VY_SHEET_MAX - VY_MIN) / (VY_SIZE - 5)
    dvy_extra = (VY_MAX - VY_SHEET_MAX) / 5
    
    # clamp（境界ちょうども最後のセルに入れたいので VX_MAX/VY_MAX を少し内側扱い）
    vx = max(VX_MIN, min(vx, VX_MAX))
    vy = max(VY_MIN, min(vy, VY_MAX))
    
    # どのセルか（floor）
    vxi = int((vx - VX_MIN) / dvx)
    if vy <= VY_SHEET_MAX:
        vyi = int((vy - VY_MIN) / dvy)
    else:
        vyi = (VY_SIZE - 5) + int((vy - VY_SHEET_MAX) / dvy_extra)
        
    # x==VX_MAX 等で vxi==VX_SIZE になり得るので丸める
    vxi = max(0, min(vxi, VX_SIZE - 1))
    vyi = max(0, min(vyi, VY_SIZE - 1))
    
    return vyi * VX_SIZE + vxi

def is_house(x: float, y: float) -> bool:
    """
    現状: ストーンの中心の座標がハウス内にあるかどうかを判定している
    
    改良案: ストーンの半径を考慮してハウス内にあるかどうかを判定する
    """
    distance = x ** 2 + (y - Y_TEE) ** 2
    return distance <= (R_HOUSE + STONE_RADIUS) ** 2
 
def sort_order_by_distance(order):
    """
    ティーからの距離順に並び替える
    x座標は中心が0, y座標は中心は Y_TEE
    dist[1] = order[1]
    dist[2] = order[2]
    """
    sorted_order = sorted(order, key=lambda x: np.sqrt(x[1] ** 2 + (x[2] - Y_TEE) ** 2))
    return sorted_order


"""
特徴平面の作成
planes は PLANES_SIZE の特徴平面を持つ2次元配列で、各平面は 32 * 56 の1次元配列

各平面の説明
planes[0] : 空点
planes[1] : 自分のストーンの座標インデックス
planes[2] : 相手のストーンの座標インデックス
planes[3] : 定数平面
planes[4] : ハウス内にあるストーン
planes[5] : ターン番号1か？
planes[6] : ターン番号2か？
planes[7] : ターン番号3か？
planes[8] : ターン番号4か？
planes[9] : ターン番号5か？
planes[10] : ターン番号6か？
planes[11] : ターン番号7か？
planes[12] : ターン番号8か？
planes[13] : 先攻のショットか？
planes[14] : 後攻のショットか？
planes[15] : エンド1か？
planes[16] : エンド2か？
planes[17] : エンド3か？
planes[18] : エンド4か？
planes[19] : エンド5か？
planes[20] : エンド6か？
planes[21] : エンド7か？
planes[22] : エンド8か？
planes[23] : エンド9か？
planes[24] : エンド10か？
planes[25] : エクストラエンド以降か？
planes[26] : 何点差か？(-5)
planes[27] : 何点差か？(-4)
planes[28] : 何点差か？(-3)
planes[29] : 何点差か？(-2)
planes[30] : 何点差か？(-1)
planes[31] : 何点差か？(0)
planes[32] : 何点差か？(+1)
planes[33] : 何点差か？(+2)
planes[34] : 何点差か？(+3)
planes[35] : 何点差か？(+4)
planes[36] : 何点差か？(+5)
planes[37] : ティーからの距離順に並び替えたストーン1
planes[38] : ティーからの距離順に並び替えたストーン2
planes[39] : ティーからの距離順に並び替えたストーン3
planes[40] : ティーからの距離順に並び替えたストーン4
planes[41] : ティーからの距離順に並び替えたストーン5
planes[42] : ティーからの距離順に並び替えたストーン6
planes[43] : ティーからの距離順に並び替えたストーン7
planes[44] : ティーからの距離順に並び替えたストーン8
planes[45] : ティーからの距離順に並び替えたストーン9
planes[46] : ティーからの距離順に並び替えたストーン10
planes[47] : ティーからの距離順に並び替えたストーン11
planes[48] : ティーからの距離順に並び替えたストーン12
planes[49] : ティーからの距離順に並び替えたストーン13
planes[50] : ティーからの距離順に並び替えたストーン14
planes[51] : ティーからの距離順に並び替えたストーン15
planes[52] : ティーからの距離順に並び替えたストーン16

"""
"""jsonファイルの['log']['simulator_storage']['stones']と['log']['shot']を入力し、PLANES_SIZEの特徴平面を出力する。"""

def generate_input_planes(stones: List[Optional[dict]], end: int, shot: int, hammer: int, score_diff_for_team0: int) -> np.ndarray:
    """
    入力特徴の生成を行う
    目線はshotを打つチーム目線
    入力:
        stones: ストーンの情報が格納されたリスト (list[dict|None]): team0 が 0~7番, team1 が 8~15番の順番で格納されている
        end: 現在のエンド数 (int)
        shot: 現在のショット数 (int)
        hammer: そのエンドで後攻のチーム番号 (0 or 1)
        score_diff_for_team0: team0にとってのこれまでの得点差 (int)
    出力:
        planes: 入力特徴平面 (np.ndarray) shape=(PLANES_SIZE, BOARD_SIZE, BOARD_SIZE)        
    """
    num_planes = PLANES_SIZE
    planes = np.zeros(shape=(num_planes, BOARD_SIZE_X * BOARD_SIZE_Y))
    
    # 空点
    planes[0][:] = 1
    
    #定数平面
    planes[3][:] = 1
    
    # shot_team を決定
    shot_team = hammer if (shot % 2) == 1 else 1 - hammer
    
    # ターン番号を特徴平面に反映
    turn_number = (shot // 2) + 5
    planes[turn_number][:] = 1
    
    # 自分が現在先攻か、後攻かを特徴平面に反映
    planes[13 if (shot % 2) == 0 else 14][:] = 1
    
    if end <= 9: #エクストラエンド以前か
        planes[end + 15][:] = 1 #エンド番号
    else:
        planes[25][:] = 1 #エクストラエンド以降か
        
    # 得点差は最大5点
    if score_diff_for_team0 > 5:
        score_diff_for_team0 = 5
    elif score_diff_for_team0 < -5:
        score_diff_for_team0 = -5

    # 何点差かを特徴平面に反映
    # ショットするチームがteam0の場合、スコア差をそのまま反映
    if shot_team == 0:
        planes[31 + score_diff_for_team0][:] = 1
    # 自分がteam1の場合、スコア差を反転して反映
    else:
        planes[31 - score_diff_for_team0][:] = 1
    
    order_list: List[Tuple[int, float, float]] = []
    """
    16個のストーンについて
    1次元の位置と2次元座標を記録
    """
    #print("shot", shot)
    for i in range(16): # 16個のストーンの情報を特徴平面に反映
        if stones[i] is None:
            continue
        
        x = float(stones[i]['position']['x'])
        y = float(stones[i]['position']['y'])
        
        # 1次元の位置を計算
        index = discretization(x, y)
        
        # ストーンが存在する位置は空点ではないので、空点平面を 0 にする
        planes[0][index] = 0

        # どのストーンが自分のチーム、どのストーンが相手のチームかを判定して特徴平面に反映
        # shot_team が自分のチームとする
        # team0 の過去 8 投のショット情報が 0~7 にある
        # team1 の過去 8 投のショット情報が 8~15 にある
        # 自分のチームのストーンなら planes[1]、相手のチームのストーンなら planes[2] に 1 を立てる
        stone_team = 0 if i < 8 else 1
        planes[1 if stone_team == shot_team else 2][index] = 1

        if is_house(x, y):
            planes[4][index] = 1 #ハウス内にあるストーン

        order_list.append( (index, x, y) ) #ストーンのインデックスと2次元座標を記録
    
    sorted_order = sort_order_by_distance(order_list)
    j = 37 # ここまでの面数 + 1
    for index, x, y in sorted_order[:16]: # ティーからの距離順に並び替えたストーン情報を特徴平面に反映
        planes[j][index] = 1
        j += 1
    
    # 最後に2次元の特徴平面を3次元に変換して返す
    return planes.reshape(num_planes, BOARD_SIZE_Y, BOARD_SIZE_X).astype(np.float32)


# Policy の正解データを作成する
def generate_target_data(selected_move: dict) ->np.ndarray:
    policy_plane = np.zeros(shape=(2, VX_SIZE * VY_SIZE))
    vx = selected_move['velocity']['x']
    vy = selected_move['velocity']['y']
    vindex = discretization_velocity(vx, vy)
    if selected_move['rotation'] == "cw":
        policy_plane[0][vindex] = 1
    else:
        policy_plane[1][vindex] = 1
    
    return np.argmax(policy_plane.reshape((VX_SIZE * VY_SIZE) * 2).astype(np.int64))


def generate_value_data(scores: Dict[str, List[Optional[int]]], end: int, shot_team: int) -> int:
    """
    ニューラルネットワークのValue出力の正解データを作成する
    入力:
        scores: 各エンドの得点が格納された辞書型データ
        end: その局面のエンド数
        shot_team: ショットを打つチームの番号 (0 or 1)
    出力:
        diff: ショットを打つチームにとっての得点差 (0 ~ 16, 8が引き分け)
    """

    # team0 にとっての実際の得点を計算
    diff = 8 + scores['team0'][end] - scores['team1'][end]

    # ショットを打つチームが team1 の場合、得点差を反転
    if shot_team == 1:
        diff = 16 - diff
        
    return diff
