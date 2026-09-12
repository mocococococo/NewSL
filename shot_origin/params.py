from __future__ import annotations
from search_config import get_search_settings

_SETTINGS = get_search_settings("shot_origin")
DEFAULT_SHOT_ORIGIN_MAX_SIMULATIONS = _SETTINGS.max_simulations
DEFAULT_SHOT_ORIGIN_INITIAL_CANDIDATES = _SETTINGS.initial_candidates
DEFAULT_SHOT_ORIGIN_MAX_DEPTH = 1
DEFAULT_SHOT_ORIGIN_INFERENCE_BATCH_SIZE = 64
SHOT_ORIGIN_KEEP_RATIO = 0.5
DEFAULT_SHOT_ORIGIN_TIE_BREAK_SEED = 0
