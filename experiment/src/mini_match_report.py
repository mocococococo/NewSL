from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, wilcoxon


RESULT_WIN_IDX = 0
RESULT_DRAW_IDX = 1
RESULT_LOSE_IDX = 2
ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER = ("<=-2", "-1", "0", "+1", ">=+2")
PLAYER_A_START_KEY = "player_a_start"
PLAYER_B_START_KEY = "player_b_start"
POSITION_BOOTSTRAP_RESAMPLES = 100000
POSITION_BOOTSTRAP_SEED = 12345


def _format_p_value(value: float) -> str:
    if value < 1e-6:
        return f"{value:.6e}"
    return f"{value:.6f}"


def save_position_json(save_file_path: Path, position_result: dict[str, Any]) -> None:
    save_file_path.parent.mkdir(parents=True, exist_ok=True)
    save_file_path.write_text(
        json.dumps(position_result, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )


def save_result_plot(
    save_file_path: Path,
    player_a_start_result_means_x: np.ndarray,
    player_b_start_result_means_x: np.ndarray,
    player_a_start_label: str = "Player A-start",
    player_b_start_label: str = "Player B-start",
) -> None:
    import matplotlib.pyplot as plt

    indices = np.arange(len(player_a_start_result_means_x))
    diff_scores = player_b_start_result_means_x - player_a_start_result_means_x
    if len(indices) == 0:
        raise ValueError("result arrays must not be empty")
    x_max = len(indices) - 1

    figure, (ax_score, ax_diff) = plt.subplots(2, 1, figsize=(12, 8))

    x_tick_step = max(1, int(np.ceil((x_max + 1) / 20)))
    x_ticks = np.arange(0, x_max + 1, x_tick_step)
    if x_ticks[-1] != x_max:
        x_ticks = np.append(x_ticks, x_max)

    ax_score.plot(
        indices,
        player_a_start_result_means_x,
        label=player_a_start_label,
        linewidth=1.0,
        alpha=0.5,
    )
    ax_score.plot(
        indices,
        player_b_start_result_means_x,
        label=player_b_start_label,
        linewidth=1.0,
        alpha=0.5,
    )
    ax_score.set_ylabel("Result Mean Over X Runs")
    ax_score.set_title(f"{player_a_start_label} vs {player_b_start_label}")
    ax_score.set_ylim(-1, 1)
    ax_score.grid(True, alpha=0.3)
    ax_score.legend()
    if x_max == 0:
        ax_score.set_xlim(-0.5, 0.5)
        ax_score.set_xticks([0])
    else:
        ax_score.set_xlim(0, x_max)
        ax_score.set_xticks(x_ticks)
    ax_score.set_xlabel("Position Index")

    ax_diff.plot(indices, diff_scores, color="tab:green", linewidth=1.0, alpha=0.6)
    ax_diff.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax_diff.set_xlabel("Position Index")
    ax_diff.set_ylabel("Result Mean Diff")
    ax_diff.set_title(f"{player_b_start_label} - {player_a_start_label}")
    ax_diff.grid(True, alpha=0.3)
    if x_max == 0:
        ax_diff.set_xlim(-0.5, 0.5)
        ax_diff.set_xticks([0])
    else:
        ax_diff.set_xlim(0, x_max)
        ax_diff.set_xticks(x_ticks)

    save_file_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(save_file_path, dpi=150)
    plt.close(figure)


def read_json(json_path: Path) -> dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{json_path} must contain a JSON object")
    return data


def is_position_record(record: dict[str, Any]) -> bool:
    position = record.get("position")
    player_a_start = record.get(PLAYER_A_START_KEY)
    player_b_start = record.get(PLAYER_B_START_KEY)
    return (
        isinstance(position, dict)
        and isinstance(player_a_start, dict)
        and isinstance(player_b_start, dict)
        and "position_index" in position
        and "root_view_score_diff_bucket" in position
        and "result_mean_x" in player_a_start
        and "result_mean_x" in player_b_start
    )


def load_position_records(json_dir: Path) -> list[dict[str, Any]]:
    if not json_dir.is_dir():
        raise FileNotFoundError(f"{json_dir} does not exist or is not a directory")

    json_paths = sorted(json_dir.glob("*.json"))
    if len(json_paths) == 0:
        raise ValueError(f"No position JSON files found in {json_dir}")

    records = [read_json(json_path) for json_path in json_paths]
    for json_path, record in zip(json_paths, records):
        if not is_position_record(record):
            raise ValueError(f"{json_path} is not a mini-match position JSON file")
    return sorted(records, key=lambda record: int(record["position"]["position_index"]))


def counts_from_model(model_data: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            int(model_data["win_count_x"]),
            int(model_data["draw_count_x"]),
            int(model_data["lose_count_x"]),
        ],
        dtype=np.int64,
    )


def build_bucket_trial_summary(
    counts: np.ndarray,
    num_positions: int,
    x_repeats: int,
) -> dict[str, Any]:
    total_trials = int(num_positions * x_repeats)
    win_count = int(counts[RESULT_WIN_IDX])
    draw_count = int(counts[RESULT_DRAW_IDX])
    lose_count = int(counts[RESULT_LOSE_IDX])

    if total_trials == 0:
        return {
            "num_positions": int(num_positions),
            "total_trials": 0,
            "result_mean_over_all_trials": 0.0,
            "win_count": 0,
            "draw_count": 0,
            "lose_count": 0,
            "win_rate_over_all_trials": 0.0,
            "draw_rate_over_all_trials": 0.0,
            "lose_rate_over_all_trials": 0.0,
        }

    win_rate = win_count / total_trials
    draw_rate = draw_count / total_trials
    lose_rate = lose_count / total_trials
    return {
        "num_positions": int(num_positions),
        "total_trials": total_trials,
        "result_mean_over_all_trials": float(win_rate - lose_rate),
        "win_count": win_count,
        "draw_count": draw_count,
        "lose_count": lose_count,
        "win_rate_over_all_trials": float(win_rate),
        "draw_rate_over_all_trials": float(draw_rate),
        "lose_rate_over_all_trials": float(lose_rate),
    }


def build_direct_match_summary(
    player_a_start_counts: np.ndarray,
    player_b_start_counts: np.ndarray,
    player_a_label: str,
    player_b_label: str,
    ab_reverse: bool = False,
) -> dict[str, Any]:
    original_player_a_win_count = int(
        player_a_start_counts[RESULT_WIN_IDX]
        + player_b_start_counts[RESULT_LOSE_IDX]
    )
    original_player_b_win_count = int(
        player_a_start_counts[RESULT_LOSE_IDX]
        + player_b_start_counts[RESULT_WIN_IDX]
    )
    draw_count = int(
        player_a_start_counts[RESULT_DRAW_IDX]
        + player_b_start_counts[RESULT_DRAW_IDX]
    )

    if ab_reverse:
        view_player_label = player_b_label
        opponent_label = player_a_label
        view_player_win_count = original_player_b_win_count
        opponent_win_count = original_player_a_win_count
        view_label = "Player B view"
    else:
        view_player_label = player_a_label
        opponent_label = player_b_label
        view_player_win_count = original_player_a_win_count
        opponent_win_count = original_player_b_win_count
        view_label = "Player A view"

    total_trials = view_player_win_count + opponent_win_count + draw_count
    decisive_trials = view_player_win_count + opponent_win_count

    if decisive_trials == 0:
        decisive_win_rate = 0.0
        decisive_win_rate_ci_low = 0.0
        decisive_win_rate_ci_high = 0.0
        binom_two_sided_p = 1.0
        binom_greater_p = 1.0
    else:
        decisive_win_rate = view_player_win_count / decisive_trials
        binom_two_sided_result = binomtest(
            view_player_win_count,
            decisive_trials,
            p=0.5,
            alternative="two-sided",
        )
        decisive_win_rate_ci = binom_two_sided_result.proportion_ci(
            confidence_level=0.95
        )
        decisive_win_rate_ci_low = decisive_win_rate_ci.low
        decisive_win_rate_ci_high = decisive_win_rate_ci.high
        binom_two_sided_p = binom_two_sided_result.pvalue
        binom_greater_p = binomtest(
            view_player_win_count,
            decisive_trials,
            p=0.5,
            alternative="greater",
        ).pvalue

    return {
        "view_label": view_label,
        "view_player_label": view_player_label,
        "opponent_label": opponent_label,
        "view_player_win_count": int(view_player_win_count),
        "opponent_win_count": int(opponent_win_count),
        "draw_count": draw_count,
        "total_trials": int(total_trials),
        "decisive_trials": int(decisive_trials),
        "decisive_win_rate_view_player": float(decisive_win_rate),
        "decisive_win_rate_ci_low_view_player": float(decisive_win_rate_ci_low),
        "decisive_win_rate_ci_high_view_player": float(decisive_win_rate_ci_high),
        "binomial_two_sided_p": float(binom_two_sided_p),
        "binomial_one_sided_p_view_player_greater": float(binom_greater_p),
        "ab_reverse": bool(ab_reverse),
    }


def _exact_sign_flip_pvalues(centered_scores: np.ndarray) -> tuple[float, float]:
    nonzero_weights = [
        abs(int(score))
        for score in centered_scores.tolist()
        if int(score) != 0
    ]
    if not nonzero_weights:
        return 1.0, 1.0

    distribution = {0: 1}
    for weight in nonzero_weights:
        next_distribution: dict[int, int] = {}
        for signed_sum, count in distribution.items():
            positive_sum = signed_sum + weight
            negative_sum = signed_sum - weight
            next_distribution[positive_sum] = (
                next_distribution.get(positive_sum, 0) + count
            )
            next_distribution[negative_sum] = (
                next_distribution.get(negative_sum, 0) + count
            )
        distribution = next_distribution

    observed_sum = int(np.sum(centered_scores))
    total_assignments = 1 << len(nonzero_weights)
    two_sided_count = sum(
        count
        for signed_sum, count in distribution.items()
        if abs(signed_sum) >= abs(observed_sum)
    )
    greater_count = sum(
        count
        for signed_sum, count in distribution.items()
        if signed_sum >= observed_sum
    )
    return (
        float(two_sided_count / total_assignments),
        float(greater_count / total_assignments),
    )


def _bootstrap_mean_ci(
    values: np.ndarray,
    confidence_level: float = 0.95,
    resamples: int = POSITION_BOOTSTRAP_RESAMPLES,
    seed: int = POSITION_BOOTSTRAP_SEED,
) -> tuple[float, float]:
    if len(values) == 0:
        raise ValueError("values must not be empty")
    if len(values) == 1:
        value = float(values[0])
        return value, value

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(resamples, dtype=np.float64)
    chunk_size = 1000
    for start in range(0, resamples, chunk_size):
        size = min(chunk_size, resamples - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        bootstrap_means[start:start + size] = values[indices].mean(axis=1)

    alpha = 1.0 - confidence_level
    low, high = np.quantile(
        bootstrap_means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )
    return float(low), float(high)


def build_position_unit_summary(
    records: list[dict[str, Any]],
    x_repeats: int,
    player_a_label: str,
    player_b_label: str,
    ab_reverse: bool = False,
) -> dict[str, Any]:
    total_trials_per_position = 2 * x_repeats
    r_values: list[float] = []
    centered_scores: list[int] = []

    for record in records:
        player_a_start_counts = counts_from_model(record[PLAYER_A_START_KEY])
        player_b_start_counts = counts_from_model(record[PLAYER_B_START_KEY])

        original_a_wins = int(
            player_a_start_counts[RESULT_WIN_IDX]
            + player_b_start_counts[RESULT_LOSE_IDX]
        )
        original_b_wins = int(
            player_a_start_counts[RESULT_LOSE_IDX]
            + player_b_start_counts[RESULT_WIN_IDX]
        )
        draws = int(
            player_a_start_counts[RESULT_DRAW_IDX]
            + player_b_start_counts[RESULT_DRAW_IDX]
        )
        actual_total = original_a_wins + original_b_wins + draws
        if actual_total != total_trials_per_position:
            raise ValueError(
                "Each position must contain exactly "
                f"{total_trials_per_position} trials, got {actual_total}"
            )

        view_wins = original_b_wins if ab_reverse else original_a_wins
        r_value = (view_wins + 0.5 * draws) / total_trials_per_position
        r_values.append(float(r_value))
        centered_scores.append(2 * view_wins + draws - total_trials_per_position)

    r_values_np = np.asarray(r_values, dtype=np.float64)
    d_values_np = r_values_np - 0.5
    centered_scores_np = np.asarray(centered_scores, dtype=np.int64)
    nonzero_count = int(np.count_nonzero(centered_scores_np))
    zero_count = int(len(centered_scores_np) - nonzero_count)

    if nonzero_count == 0:
        wilcoxon_statistic = 0.0
        wilcoxon_two_sided_p = 1.0
        wilcoxon_greater_p = 1.0
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            wilcoxon_two_sided = wilcoxon(
                d_values_np,
                zero_method="wilcox",
                correction=False,
                alternative="two-sided",
                method="approx",
            )
            wilcoxon_greater = wilcoxon(
                d_values_np,
                zero_method="wilcox",
                correction=False,
                alternative="greater",
                method="approx",
            )
        wilcoxon_statistic = float(wilcoxon_two_sided.statistic)
        wilcoxon_two_sided_p = float(wilcoxon_two_sided.pvalue)
        wilcoxon_greater_p = float(wilcoxon_greater.pvalue)

    permutation_two_sided_p, permutation_greater_p = _exact_sign_flip_pvalues(
        centered_scores_np
    )
    bootstrap_ci_low, bootstrap_ci_high = _bootstrap_mean_ci(r_values_np)

    view_player_label = player_b_label if ab_reverse else player_a_label
    opponent_label = player_a_label if ab_reverse else player_b_label
    return {
        "view_player_label": view_player_label,
        "opponent_label": opponent_label,
        "num_positions": int(len(records)),
        "total_trials_per_position": int(total_trials_per_position),
        "mean_r": float(np.mean(r_values_np)),
        "mean_d": float(np.mean(d_values_np)),
        "zero_difference_count": zero_count,
        "nonzero_difference_count": nonzero_count,
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_two_sided_p": wilcoxon_two_sided_p,
        "wilcoxon_one_sided_p_view_player_greater": wilcoxon_greater_p,
        "wilcoxon_zero_method": "wilcox",
        "wilcoxon_method": "approx",
        "sign_flip_statistic_mean_d": float(np.mean(d_values_np)),
        "sign_flip_two_sided_p": permutation_two_sided_p,
        "sign_flip_one_sided_p_view_player_greater": permutation_greater_p,
        "sign_flip_method": "exact",
        "bootstrap_mean_r_ci_low": bootstrap_ci_low,
        "bootstrap_mean_r_ci_high": bootstrap_ci_high,
        "bootstrap_confidence_level": 0.95,
        "bootstrap_method": "percentile_position_resampling",
        "bootstrap_resamples": int(POSITION_BOOTSTRAP_RESAMPLES),
        "bootstrap_seed": int(POSITION_BOOTSTRAP_SEED),
        "ab_reverse": bool(ab_reverse),
    }

def _method_label(experiment: dict[str, Any], key: str, default: str) -> str:
    value = experiment.get(key, default)
    return str(value) if value else default


def build_summary(
    records: list[dict[str, Any]],
    ab_reverse: bool = False,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if len(records) == 0:
        raise ValueError("records must not be empty")

    experiment = records[0].get("experiment", {})
    if not isinstance(experiment, dict):
        experiment = {}
    x_repeats = int(experiment["execution_repeats_x"])

    player_a_start_label = _method_label(
        experiment, "player_a_start_method_label", "Player A-start"
    )
    player_a_start_key = _method_label(
        experiment, "player_a_start_method_key", PLAYER_A_START_KEY
    )
    player_b_start_label = _method_label(
        experiment, "player_b_start_method_label", "Player B-start"
    )
    player_b_start_key = _method_label(
        experiment, "player_b_start_method_key", PLAYER_B_START_KEY
    )
    player_a_label = _method_label(experiment, "player_a_label", "Player A")
    player_b_label = _method_label(experiment, "player_b_label", "Player B")

    player_a_start_result_means_x: list[float] = []
    player_b_start_result_means_x: list[float] = []
    player_a_start_total_counts = np.zeros(3, dtype=np.int64)
    player_b_start_total_counts = np.zeros(3, dtype=np.int64)
    player_b_start_better_by_result_mean_x_count = 0
    player_a_start_better_by_result_mean_x_count = 0
    tie_by_result_mean_x_count = 0
    bucket_position_counts = {label: 0 for label in ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER}
    bucket_player_a_start_counts = {
        label: np.zeros(3, dtype=np.int64) for label in ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER
    }
    bucket_player_b_start_counts = {
        label: np.zeros(3, dtype=np.int64) for label in ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER
    }

    for record in records:
        position = record["position"]
        player_a_start = record[PLAYER_A_START_KEY]
        player_b_start = record[PLAYER_B_START_KEY]
        root_view_score_diff_bucket = str(position["root_view_score_diff_bucket"])
        if root_view_score_diff_bucket not in bucket_position_counts:
            raise ValueError(
                f"Unknown root_view_score_diff_bucket: {root_view_score_diff_bucket}"
            )

        player_a_start_counts = counts_from_model(player_a_start)
        player_b_start_counts = counts_from_model(player_b_start)
        player_a_start_result_mean = float(player_a_start["result_mean_x"])
        player_b_start_result_mean = float(player_b_start["result_mean_x"])
        diff_result_mean = player_b_start_result_mean - player_a_start_result_mean

        player_a_start_result_means_x.append(player_a_start_result_mean)
        player_b_start_result_means_x.append(player_b_start_result_mean)
        player_a_start_total_counts += player_a_start_counts
        player_b_start_total_counts += player_b_start_counts
        bucket_position_counts[root_view_score_diff_bucket] += 1
        bucket_player_a_start_counts[root_view_score_diff_bucket] += player_a_start_counts
        bucket_player_b_start_counts[root_view_score_diff_bucket] += player_b_start_counts

        if diff_result_mean > 0.0:
            player_b_start_better_by_result_mean_x_count += 1
        elif diff_result_mean < 0.0:
            player_a_start_better_by_result_mean_x_count += 1
        else:
            tie_by_result_mean_x_count += 1

    player_a_start_result_means_x_np = np.asarray(
        player_a_start_result_means_x, dtype=np.float32
    )
    player_b_start_result_means_x_np = np.asarray(
        player_b_start_result_means_x, dtype=np.float32
    )
    diff_result_means_x_np = (
        player_b_start_result_means_x_np - player_a_start_result_means_x_np
    )
    total_trials_all_positions = len(records) * x_repeats

    root_view_score_diff_bucket_summary = {}
    for bucket_label in ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER:
        root_view_score_diff_bucket_summary[bucket_label] = {
            PLAYER_A_START_KEY: build_bucket_trial_summary(
                bucket_player_a_start_counts[bucket_label],
                bucket_position_counts[bucket_label],
                x_repeats,
            ),
            PLAYER_B_START_KEY: build_bucket_trial_summary(
                bucket_player_b_start_counts[bucket_label],
                bucket_position_counts[bucket_label],
                x_repeats,
            ),
        }

    direct_match_summary = build_direct_match_summary(
        player_a_start_total_counts,
        player_b_start_total_counts,
        player_a_label,
        player_b_label,
        ab_reverse=ab_reverse,
    )
    position_unit_summary = build_position_unit_summary(
        records,
        x_repeats,
        player_a_label,
        player_b_label,
        ab_reverse=ab_reverse,
    )

    summary = {
        "experiment": experiment,
        "num_positions": int(len(records)),
        "target_end": int(experiment["target_end"]),
        "target_shot": int(experiment["target_shot"]),
        "execution_repeats_x": int(x_repeats),
        "player_a_start_method_key": player_a_start_key,
        "player_a_start_method_label": player_a_start_label,
        "player_b_start_method_key": player_b_start_key,
        "player_b_start_method_label": player_b_start_label,
        "player_a_label": player_a_label,
        "player_b_label": player_b_label,
        "log_size_result_mean_x_player_a_start": float(
            np.mean(player_a_start_result_means_x_np)
        ),
        "log_size_result_mean_x_player_b_start": float(
            np.mean(player_b_start_result_means_x_np)
        ),
        "log_size_diff_result_mean_x_player_b_start_minus_player_a_start": float(
            np.mean(diff_result_means_x_np)
        ),
        "log_size_win_rate_x_player_a_start": float(
            player_a_start_total_counts[RESULT_WIN_IDX] / total_trials_all_positions
        ),
        "log_size_draw_rate_x_player_a_start": float(
            player_a_start_total_counts[RESULT_DRAW_IDX] / total_trials_all_positions
        ),
        "log_size_lose_rate_x_player_a_start": float(
            player_a_start_total_counts[RESULT_LOSE_IDX] / total_trials_all_positions
        ),
        "log_size_win_rate_x_player_b_start": float(
            player_b_start_total_counts[RESULT_WIN_IDX] / total_trials_all_positions
        ),
        "log_size_draw_rate_x_player_b_start": float(
            player_b_start_total_counts[RESULT_DRAW_IDX] / total_trials_all_positions
        ),
        "log_size_lose_rate_x_player_b_start": float(
            player_b_start_total_counts[RESULT_LOSE_IDX] / total_trials_all_positions
        ),
        "total_trials_all_positions": int(total_trials_all_positions),
        "all_trials_win_count_player_a_start": int(
            player_a_start_total_counts[RESULT_WIN_IDX]
        ),
        "all_trials_draw_count_player_a_start": int(
            player_a_start_total_counts[RESULT_DRAW_IDX]
        ),
        "all_trials_lose_count_player_a_start": int(
            player_a_start_total_counts[RESULT_LOSE_IDX]
        ),
        "all_trials_win_count_player_b_start": int(
            player_b_start_total_counts[RESULT_WIN_IDX]
        ),
        "all_trials_draw_count_player_b_start": int(
            player_b_start_total_counts[RESULT_DRAW_IDX]
        ),
        "all_trials_lose_count_player_b_start": int(
            player_b_start_total_counts[RESULT_LOSE_IDX]
        ),
        "all_trials_win_rate_player_a_start": float(
            player_a_start_total_counts[RESULT_WIN_IDX] / total_trials_all_positions
        ),
        "all_trials_draw_rate_player_a_start": float(
            player_a_start_total_counts[RESULT_DRAW_IDX] / total_trials_all_positions
        ),
        "all_trials_lose_rate_player_a_start": float(
            player_a_start_total_counts[RESULT_LOSE_IDX] / total_trials_all_positions
        ),
        "all_trials_win_rate_player_b_start": float(
            player_b_start_total_counts[RESULT_WIN_IDX] / total_trials_all_positions
        ),
        "all_trials_draw_rate_player_b_start": float(
            player_b_start_total_counts[RESULT_DRAW_IDX] / total_trials_all_positions
        ),
        "all_trials_lose_rate_player_b_start": float(
            player_b_start_total_counts[RESULT_LOSE_IDX] / total_trials_all_positions
        ),
        "player_b_start_better_by_result_mean_x_count": int(
            player_b_start_better_by_result_mean_x_count
        ),
        "player_a_start_better_by_result_mean_x_count": int(
            player_a_start_better_by_result_mean_x_count
        ),
        "tie_by_result_mean_x_count": int(tie_by_result_mean_x_count),
        "root_view_score_diff_bucket_order": list(ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER),
        "root_view_score_diff_bucket_summary": root_view_score_diff_bucket_summary,
        "direct_match_summary": direct_match_summary,
        "position_unit_summary": position_unit_summary,
    }
    return summary, player_a_start_result_means_x_np, player_b_start_result_means_x_np


def print_summary(
    json_dir: Path,
    summary: dict[str, Any],
    png_path: Path | None = None,
) -> None:
    player_a_start_label = summary["player_a_start_method_label"]
    player_b_start_label = summary["player_b_start_method_label"]

    print("")
    print(f"Saved position json files to {json_dir}")
    if png_path is not None:
        print(f"Saved plot to {png_path}")
    print(f"num_positions                : {summary['num_positions']}")
    print(f"execution_repeats_x         : {summary['execution_repeats_x']}")
    print(
        f"log_size result_mean_x {player_a_start_label}: "
        f"{summary['log_size_result_mean_x_player_a_start']:.6f}"
    )
    print(
        f"log_size result_mean_x {player_b_start_label}: "
        f"{summary['log_size_result_mean_x_player_b_start']:.6f}"
    )
    print(
        f"log_size diff ({player_b_start_label} - {player_a_start_label}): "
        f"{summary['log_size_diff_result_mean_x_player_b_start_minus_player_a_start']:.6f}"
    )
    print(
        f"log_size win/draw/lose {player_a_start_label}: "
        f"{summary['log_size_win_rate_x_player_a_start']:.6f}, "
        f"{summary['log_size_draw_rate_x_player_a_start']:.6f}, "
        f"{summary['log_size_lose_rate_x_player_a_start']:.6f}"
    )
    print(
        f"log_size win/draw/lose {player_b_start_label}: "
        f"{summary['log_size_win_rate_x_player_b_start']:.6f}, "
        f"{summary['log_size_draw_rate_x_player_b_start']:.6f}, "
        f"{summary['log_size_lose_rate_x_player_b_start']:.6f}"
    )
    print(
        f"all_trials win/draw/lose {player_a_start_label}: "
        f"{summary['all_trials_win_count_player_a_start']}, "
        f"{summary['all_trials_draw_count_player_a_start']}, "
        f"{summary['all_trials_lose_count_player_a_start']}"
    )
    print(
        f"all_trials win/draw/lose {player_b_start_label}: "
        f"{summary['all_trials_win_count_player_b_start']}, "
        f"{summary['all_trials_draw_count_player_b_start']}, "
        f"{summary['all_trials_lose_count_player_b_start']}"
    )
    print(
        f"{player_b_start_label} better by result_mean_x count       : "
        f"{summary['player_b_start_better_by_result_mean_x_count']}"
    )
    print(
        f"{player_a_start_label} better by result_mean_x count        : "
        f"{summary['player_a_start_better_by_result_mean_x_count']}"
    )
    print(f"Tie by result_mean_x count                : {summary['tie_by_result_mean_x_count']}")
    print("")
    print(f"direct match summary ({summary['direct_match_summary']['view_label']})")
    direct_match_summary = summary["direct_match_summary"]
    view_player_label = direct_match_summary["view_player_label"]
    opponent_label = direct_match_summary["opponent_label"]
    print(
        f"{view_player_label} wins / {opponent_label} wins / draws / total: "
        f"{direct_match_summary['view_player_win_count']}, "
        f"{direct_match_summary['opponent_win_count']}, "
        f"{direct_match_summary['draw_count']}, "
        f"{direct_match_summary['total_trials']}"
    )
    print(
        f"draw-excluded decisive winrate {view_player_label}: "
        f"{direct_match_summary['decisive_win_rate_view_player']:.6f} "
        f"({direct_match_summary['view_player_win_count']}/"
        f"{direct_match_summary['decisive_trials']})"
    )
    print(
        f"draw-excluded decisive winrate 95% CI {view_player_label}: "
        f"[{direct_match_summary['decisive_win_rate_ci_low_view_player']:.6f}, "
        f"{direct_match_summary['decisive_win_rate_ci_high_view_player']:.6f}]"
    )
    print(
        "binomial test on decisive games two-sided p: " +
        _format_p_value(direct_match_summary["binomial_two_sided_p"])
    )
    print(
        f"binomial test on decisive games one-sided p ({view_player_label} > {opponent_label}): " +
        _format_p_value(
            direct_match_summary["binomial_one_sided_p_view_player_greater"]
        )
    )
    position_unit_summary = summary["position_unit_summary"]
    print("")
    print("position-unit paired summary")
    print(
        "r_i = (view-player wins + 0.5 * draws) / "
        f"{position_unit_summary['total_trials_per_position']} trials per position"
    )
    print(
        f"mean r_i {view_player_label}: "
        f"{position_unit_summary['mean_r']:.6f}"
    )
    print(
        f"mean d_i = mean(r_i - 0.5): "
        f"{position_unit_summary['mean_d']:.6f}"
    )
    print(
        "zero / nonzero d_i positions: "
        f"{position_unit_summary['zero_difference_count']}, "
        f"{position_unit_summary['nonzero_difference_count']}"
    )
    print(
        "Wilcoxon signed-rank statistic: "
        f"{position_unit_summary['wilcoxon_statistic']:.6f}"
    )
    print(
        "Wilcoxon settings: "
        f"zero_method={position_unit_summary['wilcoxon_zero_method']}, "
        f"method={position_unit_summary['wilcoxon_method']}"
    )
    print(
        "Wilcoxon signed-rank test two-sided p: " +
        _format_p_value(position_unit_summary["wilcoxon_two_sided_p"])
    )
    print(
        f"Wilcoxon signed-rank test one-sided p "
        f"({view_player_label} > {opponent_label}): " +
        _format_p_value(
            position_unit_summary["wilcoxon_one_sided_p_view_player_greater"]
        )
    )
    print(
        "position-level sign-flip statistic mean d_i: "
        f"{position_unit_summary['sign_flip_statistic_mean_d']:.6f}"
    )
    print(
        "exact position-level sign-flip permutation test two-sided p: " +
        _format_p_value(position_unit_summary["sign_flip_two_sided_p"])
    )
    print(
        f"exact position-level sign-flip permutation test one-sided p "
        f"({view_player_label} > {opponent_label}): " +
        _format_p_value(
            position_unit_summary["sign_flip_one_sided_p_view_player_greater"]
        )
    )
    print(
        "position bootstrap settings: "
        f"method={position_unit_summary['bootstrap_method']}, "
        f"resamples={position_unit_summary['bootstrap_resamples']}, "
        f"seed={position_unit_summary['bootstrap_seed']}"
    )
    print(
        f"position bootstrap mean r_i 95% CI {view_player_label}: "
        f"[{position_unit_summary['bootstrap_mean_r_ci_low']:.6f}, "
        f"{position_unit_summary['bootstrap_mean_r_ci_high']:.6f}]"
    )
    print("")
    print("root_view_score_diff_before_shot bucket summary (all trials in each bucket)")
    for bucket_label in summary["root_view_score_diff_bucket_order"]:
        bucket_summary = summary["root_view_score_diff_bucket_summary"][bucket_label]
        player_a_start_bucket = bucket_summary[PLAYER_A_START_KEY]
        player_b_start_bucket = bucket_summary[PLAYER_B_START_KEY]
        print(
            f"bucket {bucket_label:>4} positions={player_a_start_bucket['num_positions']:4d} "
            f"trials={player_a_start_bucket['total_trials']:5d} "
            f"| {player_a_start_label} result_mean="
            f"{player_a_start_bucket['result_mean_over_all_trials']:.4f} "
            f"win/draw/lose={player_a_start_bucket['win_rate_over_all_trials']:.4f},"
            f"{player_a_start_bucket['draw_rate_over_all_trials']:.4f},"
            f"{player_a_start_bucket['lose_rate_over_all_trials']:.4f} "
            f"| {player_b_start_label} result_mean="
            f"{player_b_start_bucket['result_mean_over_all_trials']:.4f} "
            f"win/draw/lose={player_b_start_bucket['win_rate_over_all_trials']:.4f},"
            f"{player_b_start_bucket['draw_rate_over_all_trials']:.4f},"
            f"{player_b_start_bucket['lose_rate_over_all_trials']:.4f}"
        )


def render_report_from_records(
    json_dir: Path,
    png_path: Path,
    records: list[dict[str, Any]],
    ab_reverse: bool = False,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    summary, player_a_start_result_means_x, player_b_start_result_means_x = build_summary(
        records,
        ab_reverse=ab_reverse,
    )
    save_result_plot(
        png_path,
        player_a_start_result_means_x,
        player_b_start_result_means_x,
        player_a_start_label=summary["player_a_start_method_label"],
        player_b_start_label=summary["player_b_start_method_label"],
    )
    print_summary(json_dir, summary, png_path=png_path)
    return summary, player_a_start_result_means_x, player_b_start_result_means_x


def print_report_from_records(
    json_dir: Path,
    records: list[dict[str, Any]],
    ab_reverse: bool = False,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    summary, player_a_start_result_means_x, player_b_start_result_means_x = build_summary(
        records,
        ab_reverse=ab_reverse,
    )
    print_summary(json_dir, summary)
    return summary, player_a_start_result_means_x, player_b_start_result_means_x


def render_report_from_json_dir(
    json_dir: Path,
    png_path: Path | None = None,
    ab_reverse: bool = False,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if png_path is None:
        png_path = json_dir.parent / f"{json_dir.name}.png"
    records = load_position_records(json_dir)
    return render_report_from_records(json_dir, png_path, records, ab_reverse=ab_reverse)


def print_report_from_json_dir(
    json_dir: Path,
    ab_reverse: bool = False,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    records = load_position_records(json_dir)
    return print_report_from_records(json_dir, records, ab_reverse=ab_reverse)
