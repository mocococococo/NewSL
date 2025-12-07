# params.py
from __future__ import annotations

# 探索
DEFAULT_MAX_SIMULATIONS = 800
DEFAULT_CPUCT = 1.0
DEFAULT_TIME_LIMIT_SEC = 2.5  # float秒 or None

# rollout
ROLLOUT_USE_GREEDY_POLICY = True  # Trueならargmax(policy)、Falseならサンプルなどに拡張

# State.key 量子化
STATE_POS_SCALE = 1000            # 座標をround(v*scale)して整数化

PUCT_DEBUG_SCORE = 1  # スコアデバッグ有効化フラグ
PUCT_DEBUG_SCORE_LIMIT = 30  # スコアデバッグ出力上限

PUCT_DEBUG_SIM = 1
PUCT_DEBUG_SIM_LIMIT = 20