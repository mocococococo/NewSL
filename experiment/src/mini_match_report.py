from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, t as student_t, ttest_1samp


RESULT_WIN_IDX = 0
RESULT_DRAW_IDX = 1
RESULT_LOSE_IDX = 2
ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER = ("<=-2", "-1", "0", "+1", ">=+2")
PLAYER_A_START_KEY = "player_a_start"
PLAYER_B_START_KEY = "player_b_start"
DRAW_SCORE_FOR_PLAYER_A = 0.2


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
    draw_score_for_player_a: float = DRAW_SCORE_FOR_PLAYER_A,
) -> dict[str, Any]:
    player_a_win_count = int(
        player_a_start_counts[RESULT_WIN_IDX]
        + player_b_start_counts[RESULT_LOSE_IDX]
    )
    player_b_win_count = int(
        player_a_start_counts[RESULT_LOSE_IDX]
        + player_b_start_counts[RESULT_WIN_IDX]
    )
    draw_count = int(
        player_a_start_counts[RESULT_DRAW_IDX]
        + player_b_start_counts[RESULT_DRAW_IDX]
    )
    total_trials = player_a_win_count + player_b_win_count + draw_count
    decisive_trials = player_a_win_count + player_b_win_count

    if total_trials == 0:
        draw_weighted_score_rate = 0.0
    else:
        draw_weighted_score_rate = (
            player_a_win_count + draw_score_for_player_a * draw_count
        ) / total_trials

    if decisive_trials == 0:
        decisive_win_rate = 0.0
        binom_two_sided_p = 1.0
        binom_greater_p = 1.0
    else:
        decisive_win_rate = player_a_win_count / decisive_trials
        binom_two_sided_p = binomtest(
            player_a_win_count,
            decisive_trials,
            p=0.5,
            alternative="two-sided",
        ).pvalue
        binom_greater_p = binomtest(
            player_a_win_count,
            decisive_trials,
            p=0.5,
            alternative="greater",
        ).pvalue

    return {
        "player_a_label": player_a_label,
        "player_b_label": player_b_label,
        "player_a_win_count": player_a_win_count,
        "player_b_win_count": player_b_win_count,
        "draw_count": draw_count,
        "total_trials": int(total_trials),
        "draw_score_for_player_a": float(draw_score_for_player_a),
        "draw_weighted_score_rate_player_a": float(draw_weighted_score_rate),
        "decisive_trials": int(decisive_trials),
        "decisive_win_rate_player_a": float(decisive_win_rate),
        "binomial_two_sided_p": float(binom_two_sided_p),
        "binomial_one_sided_p_player_a_greater": float(binom_greater_p),
    }


def build_draw_weighted_score_stats(
    player_a_win_count: int,
    player_b_win_count: int,
    draw_count: int,
    draw_score_for_player_a: float,
    baseline_score_rate: float = 0.5,
    alpha: float = 0.05,
) -> dict[str, Any]:
    scores = np.concatenate(
        [
            np.full(player_a_win_count, 1.0, dtype=np.float64),
            np.full(draw_count, draw_score_for_player_a, dtype=np.float64),
            np.zeros(player_b_win_count, dtype=np.float64),
        ]
    )
    total_trials = int(scores.size)

    if total_trials == 0:
        return {
            "baseline_score_rate": float(baseline_score_rate),
            "alpha": float(alpha),
            "mean": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "t_test_two_sided_p": 1.0,
            "t_test_greater_p": 1.0,
            "t_test_less_p": 1.0,
        }

    mean = float(np.mean(scores))
    if total_trials <= 1:
        ci_low = mean
        ci_high = mean
    else:
        std = float(np.std(scores, ddof=1))
        sem = std / float(np.sqrt(total_trials))
        ci_margin = float(student_t.ppf(1.0 - alpha / 2.0, total_trials - 1) * sem)
        ci_low = mean - ci_margin
        ci_high = mean + ci_margin

    two_sided_p = _ttest_pvalue(scores, baseline_score_rate, "two-sided", mean)
    greater_p = _ttest_pvalue(scores, baseline_score_rate, "greater", mean)
    less_p = _ttest_pvalue(scores, baseline_score_rate, "less", mean)

    return {
        "baseline_score_rate": float(baseline_score_rate),
        "alpha": float(alpha),
        "mean": mean,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "t_test_two_sided_p": float(two_sided_p),
        "t_test_greater_p": float(greater_p),
        "t_test_less_p": float(less_p),
    }


def _ttest_pvalue(
    scores: np.ndarray,
    baseline_score_rate: float,
    alternative: str,
    mean: float,
) -> float:
    if scores.size <= 1:
        return 1.0

    result = ttest_1samp(scores, baseline_score_rate, alternative=alternative)
    pvalue = float(result.pvalue)
    if np.isnan(pvalue):
        return 1.0 if mean == baseline_score_rate else 0.0
    return pvalue


def _method_label(experiment: dict[str, Any], key: str, default: str) -> str:
    value = experiment.get(key, default)
    return str(value) if value else default


def build_summary(
    records: list[dict[str, Any]],
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
    )
    draw_weighted_score_stats = build_draw_weighted_score_stats(
        direct_match_summary["player_a_win_count"],
        direct_match_summary["player_b_win_count"],
        direct_match_summary["draw_count"],
        direct_match_summary["draw_score_for_player_a"],
    )
    direct_match_summary["draw_weighted_score_stats_player_a"] = (
        draw_weighted_score_stats
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
    print("direct match summary (Player A view)")
    direct_match_summary = summary["direct_match_summary"]
    player_a_label = direct_match_summary["player_a_label"]
    player_b_label = direct_match_summary["player_b_label"]
    draw_score_for_player_a = direct_match_summary["draw_score_for_player_a"]
    print(
        f"{player_a_label} wins / {player_b_label} wins / draws / total: "
        f"{direct_match_summary['player_a_win_count']}, "
        f"{direct_match_summary['player_b_win_count']}, "
        f"{direct_match_summary['draw_count']}, "
        f"{direct_match_summary['total_trials']}"
    )
    print(
        f"draw-weighted score rate {player_a_label} "
        f"(draw={draw_score_for_player_a:.1f}): "
        f"{direct_match_summary['draw_weighted_score_rate_player_a']:.6f}"
    )
    draw_weighted_score_stats = direct_match_summary[
        "draw_weighted_score_stats_player_a"
    ]
    print(
        f"draw-weighted score 95% CI {player_a_label}: "
        f"[{draw_weighted_score_stats['ci_low']:.6f}, "
        f"{draw_weighted_score_stats['ci_high']:.6f}]"
    )
    print(
        f"t-test draw-weighted score two-sided p "
        f"({player_a_label} != {draw_weighted_score_stats['baseline_score_rate']:.1f}): "
        f"{draw_weighted_score_stats['t_test_two_sided_p']:.6f}"
    )
    print(
        f"t-test draw-weighted score one-sided p "
        f"({player_a_label} > {draw_weighted_score_stats['baseline_score_rate']:.1f}): "
        f"{draw_weighted_score_stats['t_test_greater_p']:.6f}"
    )
    print(
        f"t-test draw-weighted score one-sided p "
        f"({player_a_label} < {draw_weighted_score_stats['baseline_score_rate']:.1f}): "
        f"{draw_weighted_score_stats['t_test_less_p']:.6f}"
    )
    print(
        f"draw-excluded decisive winrate {player_a_label}: "
        f"{direct_match_summary['decisive_win_rate_player_a']:.6f} "
        f"({direct_match_summary['player_a_win_count']}/"
        f"{direct_match_summary['decisive_trials']})"
    )
    print(
        "binomial test on decisive games two-sided p: "
        f"{direct_match_summary['binomial_two_sided_p']:.6f}"
    )
    print(
        f"binomial test on decisive games one-sided p ({player_a_label} > {player_b_label}): "
        f"{direct_match_summary['binomial_one_sided_p_player_a_greater']:.6f}"
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
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    summary, player_a_start_result_means_x, player_b_start_result_means_x = build_summary(
        records
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
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    summary, player_a_start_result_means_x, player_b_start_result_means_x = build_summary(
        records
    )
    print_summary(json_dir, summary)
    return summary, player_a_start_result_means_x, player_b_start_result_means_x


def render_report_from_json_dir(
    json_dir: Path,
    png_path: Path | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if png_path is None:
        png_path = json_dir.parent / f"{json_dir.name}.png"
    records = load_position_records(json_dir)
    return render_report_from_records(json_dir, png_path, records)


def print_report_from_json_dir(
    json_dir: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    records = load_position_records(json_dir)
    return print_report_from_records(json_dir, records)
