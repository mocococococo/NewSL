import numpy as np
from mcts.config import N_SCORE

"""
def idx_to_score(idx):
    assert(0 <= idx and idx <= N_SCORE-1)
    return idx - 8
"""

def score_to_idx(score):
    assert(-8 <= score and score <= 8)
    return int(score + 8)

def dist_v_to_exp_v(dist_v):
    score_array = np.zeros(N_SCORE)
    for i in range(N_SCORE):
        score_array[i] = idx_to_score(i)
        #score_array[i] = i
    exp_v = np.sum(score_array * dist_v)
    return exp_v


def idx_to_score(idx): #勝率テーブル適応（仮）後攻に探索することしか考慮してないし、勝率テーブルも+1と0を入れ替えただけ
    assert(0 <= idx and idx <= N_SCORE-1)
    if idx == 8: #0点の場合
        return idx - 7
    elif idx == 9: #1点の場合
        return idx - 9
    return idx - 8
