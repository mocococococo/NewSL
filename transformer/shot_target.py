import math
from typing import Dict, List

import numpy as np

ShotStat = Dict[str, object]


def _one_hot(index: int, size: int) -> np.ndarray:
    if not (0 <= index < size):
        raise ValueError(f"index must be in [0, {size}), got {index}")

    target = np.zeros(size, dtype=np.float32)
    target[index] = 1.0
    return target


def _normalize_distribution(target: np.ndarray, name: str) -> np.ndarray:
    if target.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got {target.shape}")
    if not np.all(np.isfinite(target)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(target < 0):
        raise ValueError(f"{name} contains negative values")

    total = float(target.sum())
    if total <= 0.0:
        raise ValueError(f"{name} sum must be positive, got {total}")

    return (target / total).astype(np.float32, copy=False)


def _mix_rate(value: float) -> float:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"lambda_best must be in [0, 1], got {value}")
    return float(value)


def _find_best_stat(root_candidate_stats: List[ShotStat], best_action_id: int) -> ShotStat:
    for stat in root_candidate_stats:
        if int(stat["action_id"]) == int(best_action_id):
            return stat
    raise ValueError(f"best_action_id {best_action_id} is not included in root_candidate_stats")


def _q(stat: ShotStat) -> float:
    q = float(stat["q"])
    if not math.isfinite(q):
        raise ValueError(f"q must be finite, got {q}")
    return q


def _visit_count(stat: ShotStat) -> int:
    visit_count = int(stat["visit_count"])
    if visit_count < 0:
        raise ValueError(f"visit_count must be non-negative, got {visit_count}")
    return visit_count


def _candidate_weight(
    stat: ShotStat,
    q_best: float,
    alpha_visit: float,
    beta_q: float,
) -> float:
    return (_visit_count(stat) ** float(alpha_visit)) * math.exp(float(beta_q) * (_q(stat) - q_best))


def _filtered_stats(
    root_candidate_stats: List[ShotStat],
    q_best: float,
    min_visit: int,
    delta_q: float,
) -> List[ShotStat]:
    return [
        stat
        for stat in root_candidate_stats
        if _visit_count(stat) >= int(min_visit)
        and _q(stat) >= q_best - float(delta_q)
    ]


def _normalized_score_histogram(stat: ShotStat, value_dim: int) -> np.ndarray:
    histogram = np.asarray(stat["score_histogram"], dtype=np.float32)
    if histogram.shape != (value_dim,):
        raise ValueError(f"score_histogram must have shape ({value_dim},), got {histogram.shape}")
    return _normalize_distribution(histogram, "score_histogram")


def build_policy_target_from_shot_stats(
    root_candidate_stats: List[ShotStat],
    best_action_id: int,
    action_dim: int,
    min_visit: int = 3,
    delta_q: float = 1.0,
    alpha_visit: float = 0.2,
    beta_q: float = 0.3,
    lambda_best: float = 0.5,
) -> np.ndarray:
    """
    SHOTのroot候補統計から、Q差ベースのpolicy教師分布を作る。
    """
    if not root_candidate_stats:
        raise ValueError("root_candidate_stats must not be empty")

    best_stat = _find_best_stat(root_candidate_stats, best_action_id)
    q_best = _q(best_stat)
    filtered = _filtered_stats(root_candidate_stats, q_best, min_visit, delta_q)

    if not filtered:
        return _one_hot(best_action_id, action_dim)

    target = np.zeros(action_dim, dtype=np.float32)
    for stat in filtered:
        action_id = int(stat["action_id"])
        if not (0 <= action_id < action_dim):
            raise ValueError(f"action_id must be in [0, {action_dim}), got {action_id}")
        target[action_id] += _candidate_weight(stat, q_best, alpha_visit, beta_q)

    target = _normalize_distribution(target, "policy_target")
    one_hot_best = _one_hot(best_action_id, action_dim)
    lam = _mix_rate(lambda_best)
    return ((1.0 - lam) * target + lam * one_hot_best).astype(np.float32, copy=False)


def build_value_target_from_shot_stats(
    root_candidate_stats: List[ShotStat],
    best_action_id: int,
    value_dim: int,
    min_visit: int = 3,
    delta_q: float = 0.0,
    alpha_visit: float = 0.5,
    beta_q: float = 0.5,
    lambda_best: float = 0.5,
) -> np.ndarray:
    """
    SHOTのroot候補統計から、候補ごとのscore_histogramを合成したvalue教師分布を作る。
    """
    if not root_candidate_stats:
        raise ValueError("root_candidate_stats must not be empty")

    best_stat = _find_best_stat(root_candidate_stats, best_action_id)
    q_best = _q(best_stat)
    filtered = _filtered_stats(root_candidate_stats, q_best, min_visit, delta_q)
    best_distribution = _normalized_score_histogram(best_stat, value_dim)

    if not filtered:
        return best_distribution.copy()

    target = np.zeros(value_dim, dtype=np.float32)
    total_weight = 0.0
    for stat in filtered:
        weight = _candidate_weight(stat, q_best, alpha_visit, beta_q)
        target += weight * _normalized_score_histogram(stat, value_dim)
        total_weight += weight

    if total_weight <= 0.0:
        base_distribution = best_distribution
    else:
        base_distribution = (target / total_weight).astype(np.float32, copy=False)

    lam = _mix_rate(lambda_best)
    value_target = (1.0 - lam) * base_distribution + lam * best_distribution
    return _normalize_distribution(value_target.astype(np.float32, copy=False), "value_target")


def build_win_value_target_from_shot_stats(
    root_candidate_stats: List[ShotStat],
    best_action_id: int,
    min_visit: int = 3,
    delta_q: float = 1.0,
    alpha_visit: float = 0.2,
    beta_q: float = 0.3,
    lambda_best: float = 0.5,
) -> float:
    """
    SHOTのroot候補統計から、Qを重み付き平均したwin_value教師値を作る。
    """
    if not root_candidate_stats:
        raise ValueError("root_candidate_stats must not be empty")

    best_stat = _find_best_stat(root_candidate_stats, best_action_id)
    q_best = _q(best_stat)
    filtered = _filtered_stats(root_candidate_stats, q_best, min_visit, delta_q)

    if not filtered:
        return q_best

    total_weight = 0.0
    weighted_q = 0.0
    for stat in filtered:
        weight = _candidate_weight(stat, q_best, alpha_visit, beta_q)
        weighted_q += weight * _q(stat)
        total_weight += weight

    if total_weight <= 0.0:
        weighted_average_q = q_best
    else:
        weighted_average_q = weighted_q / total_weight

    lam = _mix_rate(lambda_best)
    return float((1.0 - lam) * weighted_average_q + lam * q_best)
