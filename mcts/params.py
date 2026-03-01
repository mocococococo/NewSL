# params.py
from __future__ import annotations

# 探索
DEFAULT_MAX_SIMULATIONS = 20000
DEFAULT_CPUCT = 1.0
DEFAULT_TIME_LIMIT_SEC = 2.6
DEFAULT_TIME_LIMIT_SEC_LIST = {
    0: 2.0,
    1: 2.0,
    2: 2.0,
    3: 2.0,
    4: 4.5,
    5: 4.5,
    6: 4.5,
    7: 4.5,
    8: 3.0,
    9: 3.0,
    10: 2.5,
    11: 2.5,
    12: 1.5,
    13: 1.5,
    14: 1.0,
    15: 1.0,
}

# rollout
ROLLOUT_USE_GREEDY_POLICY = True  # Trueならargmax(policy)、Falseならサンプルなどに拡張

# State.key 量子化
STATE_POS_SCALE = 1000            # 座標をround(v*scale)して整数化

PUCT_DEBUG_SCORE_FLAG = False  # スコアデバッグ有効化フラグ
PUCT_DEBUG_SCORE_LIMIT = 30  # スコアデバッグ出力上限

PUCT_DEBUG_SIM = 1
PUCT_DEBUG_SIM_LIMIT = 20

DEFAULT_MAX_DEPTH = 3