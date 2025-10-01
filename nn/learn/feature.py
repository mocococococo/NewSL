"""ニューラルネットワークの入力特徴生成処理
"""
import numpy as np
import math

from board.constant import BOARD_SIZE, X_MIN, X_MAX, Y_MIN, Y_MAX, Y_TEE, R_HOUSE, VX_MIN, VX_MAX, VY_MIN, VY_MAX, PLANES_SIZE

def discretization(x: float, y: float, 
                   xmin: float, xmax: float, 
                   ymin: float, ymax: float) -> int:
    """
    ストーンの２次元座標の位置を１次元のインデックスに変換する
    """
    x_clamped = max(xmin, min(x, xmax))
    y_clamped = max(ymin, min(y, ymax))

    x_interval = (xmax - xmin) / (BOARD_SIZE - 1)
    y_interval = (ymax - ymin) / (BOARD_SIZE - 1)

    x_index = int((x_clamped - xmin) / x_interval)
    y_index = int((y_clamped - ymin) / y_interval)

    x_index = max(0, min(x_index, BOARD_SIZE - 1))
    y_index = max(0, min(y_index, BOARD_SIZE - 1))

    index = y_index * BOARD_SIZE + x_index
    return index

def is_house(x: float, y: float):
    """
    現状: ストーンの中心の座標がハウス内にあるかどうかを判定している
    
    改良案: ストーンの半径を考慮してハウス内にあるかどうかを判定する
    """
    distance = math.sqrt(x ** 2 + (y - Y_TEE) ** 2)
    return distance <= R_HOUSE
 
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
def generate_input_planes(stones: list, scores: list, end: int, shot: int) -> np.ndarray:
    num_planes = PLANES_SIZE
    planes = np.zeros(shape=(num_planes, BOARD_SIZE * BOARD_SIZE))
    planes[0][:] = 1 #空点
    planes[3][:] = 1 #定数平面
    turn_number = (shot // 2) + 5
    planes[turn_number][:] = 1 #ターン番号
    me = 1
    if shot % 2 == 1: #自分が先攻か後攻か
        me = 2
    
    
    planes[me + 12][:] = 1 #自分のストーンの順番
    
    if end <= 9: #エクストラエンド以前か
        planes[end + 15][:] = 1 #エンド番号
    else:
        planes[25][:] = 1 #エクストラエンド以降か
        
    # どちらのチームが先攻かを判定 + 何点差かを計算
    team0_score = scores['team0']
    team1_score = scores['team1']
    
    score_diff = 0
    team0_is_first = True
    for i in range(end):
        score_diff += team0_score[i] - team1_score[i]
        if team0_score[i] < team1_score[i]:
            team0_is_first = False
        elif team0_score[i] > team1_score[i]:
            team0_is_first = True
        else:
            continue
    
    # 得点差は最大5点
    if score_diff > 5:
        score_diff = 5
    elif score_diff < -5:
        score_diff = -5
    # 何点差かを特徴平面に反映
    if team0_is_first != (shot % 2 == 0):
        planes[31 + score_diff][:] = 1
    else:
        planes[31 - score_diff][:] = 1
    
    order = np.full(shape=(16, 3), fill_value=-1)
    """
    16個のストーンについて
    1次元の位置と2次元座標を記録
    """
    #print("shot", shot)
    for i in range(16): # 16個のストーンの情報を特徴平面に反映
        if stones[i]:
            x = stones[i]['position']['x']
            y = np.abs(stones[i]['position']['y'])
            index = discretization(x, y, X_MIN, X_MAX, Y_MIN, Y_MAX) # 1次元の位置を計算
            planes[0][index] = 0 #空点の更新

            # ここはストーンの情報を更新するけど、何投目かの情報を与えているわけではない。
            # 先攻の過去8投のショット情報がjsonにある
            if i < 8:
                planes[me][index] = 1 # 先攻のストーンの更新
            # 後攻の過去8投のショット情報がjsonにある            
            else:
                planes[3 - me][index] = 1 # 後攻のストーンの更新

            if is_house(x, y):
                planes[4][index] = 1 #ハウス内にあるストーン

            order[i] = [index, x, y] # ストーンの位置と2次元座標を記録
    
    sorted_order = sort_order_by_distance(order)
    j = 37 # ここまでの面数 + 1
    for item in sorted_order: #ティーからの順番
        if item[0] == -1:
            break
        planes[j][item[0]] = 1
        j += 1
    
    # 最後に2次元の特徴平面を3次元に変換して返す
    return planes.reshape(num_planes, BOARD_SIZE, BOARD_SIZE).astype(np.float32)#, box


# Policy の正解データを作成する
def generate_target_data(selected_move: dict) ->np.ndarray:
    policy_plane = np.zeros(shape=(2, BOARD_SIZE * BOARD_SIZE))
    vx = selected_move['velocity']['x']
    vy = selected_move['velocity']['y']

    vindex = discretization(vx, vy, VX_MIN, VX_MAX, VY_MIN, VY_MAX)
    if selected_move['rotation'] == "cw":
        policy_plane[0][vindex] = 1
    else:
        policy_plane[1][vindex] = 1
    
    return np.argmax(policy_plane.reshape((BOARD_SIZE * BOARD_SIZE) * 2).astype(np.int64))


def generate_value_data(dcl_data, end, shot) ->np.ndarray:
    """
    end: その局面のエンド数
    shot: その局面のショットが何投目か？
    """
    team0 = dcl_data['log']['state']['scores']['team0']
    team1 = dcl_data['log']['state']['scores']['team1']
    
    team0_is_first = True
    for i in range(end):
        if team0[i] < team1[i]:
            team0_is_first = False
        elif team0[i] > team1[i]:
            team0_is_first = True
        else:
            continue

    # team0 にとっての得点を計算
    diff = 8 + team0[end] - team1[end]
    
    # 現在のショット数が偶数なら先攻のバリュー、奇数なら後攻のバリューを出力
    if team0_is_first != (shot % 2 == 0):
        diff = 16 - diff
        
    return diff
