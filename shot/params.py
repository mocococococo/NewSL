from __future__ import annotations

# 探索
DEFAULT_SHOT_MAX_SIMULATIONS = 1020
DEFAULT_SHOT_TIME_LIMIT_SEC = None
DEFAULT_SHOT_TIME_LIMIT_SEC_LIST = {
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

DEFAULT_SHOT_MAX_DEPTH = 1

# policy-guided SHOT
DEFAULT_SHOT_INITIAL_CANDIDATES = 256
SHOT_KEEP_RATIO = 0.5
SHOT_MIN_VISITS_PER_ACTION = 1
