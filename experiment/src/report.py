from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


RESULT_WIN_IDX = 0
RESULT_DRAW_IDX = 1
RESULT_LOSE_IDX = 2
ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER = ("<=-2", "-1", "0", "+1", ">=+2")


def save_position_json(save_file_path: Path, position_result: dict[str, Any]) -> None:
    save_file_path.parent.mkdir(parents=True, exist_ok=True)
    save_file_path.write_text(
        json.dumps(position_result, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )


def save_result_plot(
    save_file_path: Path,
    baseline_result_means_x: np.ndarray,
    newsl_result_means_x: np.ndarray,
    baseline_label: str = "Baseline",
    newsl_label: str = "NewSL",
) -> None:
    import matplotlib.pyplot as plt

    indices = np.arange(len(baseline_result_means_x))
    diff_scores = newsl_result_means_x - baseline_result_means_x
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
        baseline_result_means_x,
        label=baseline_label,
        linewidth=1.0,
        alpha=0.5,
    )
    ax_score.plot(
        indices,
        newsl_result_means_x,
        label=newsl_label,
        linewidth=1.0,
        alpha=0.5,
    )
    ax_score.set_ylabel("Result Mean Over X Runs")
    ax_score.set_title(f"{baseline_label} vs {newsl_label}")
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
    ax_diff.set_title(f"{newsl_label} - {baseline_label}")
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
    baseline = record.get("baseline")
    newsl = record.get("newsl")
    return (
        isinstance(position, dict)
        and isinstance(baseline, dict)
        and isinstance(newsl, dict)
        and "position_index" in position
        and "root_view_score_diff_bucket" in position
        and "result_mean_x" in baseline
        and "result_mean_x" in newsl
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
            raise ValueError(f"{json_path} is not a baseline/newsl position JSON file")
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

    baseline_label = _method_label(experiment, "baseline_method_label", "Baseline")
    baseline_key = _method_label(experiment, "baseline_method_key", "baseline")
    newsl_label = _method_label(experiment, "newsl_method_label", "NewSL")
    newsl_key = _method_label(experiment, "newsl_method_key", "transformer")

    baseline_result_means_x: list[float] = []
    newsl_result_means_x: list[float] = []
    baseline_total_counts = np.zeros(3, dtype=np.int64)
    newsl_total_counts = np.zeros(3, dtype=np.int64)
    newsl_better_by_result_mean_x_count = 0
    baseline_better_by_result_mean_x_count = 0
    tie_by_result_mean_x_count = 0
    bucket_position_counts = {label: 0 for label in ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER}
    bucket_baseline_counts = {
        label: np.zeros(3, dtype=np.int64) for label in ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER
    }
    bucket_newsl_counts = {
        label: np.zeros(3, dtype=np.int64) for label in ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER
    }

    for record in records:
        position = record["position"]
        baseline = record["baseline"]
        newsl = record["newsl"]
        root_view_score_diff_bucket = str(position["root_view_score_diff_bucket"])
        if root_view_score_diff_bucket not in bucket_position_counts:
            raise ValueError(
                f"Unknown root_view_score_diff_bucket: {root_view_score_diff_bucket}"
            )

        baseline_counts = counts_from_model(baseline)
        newsl_counts = counts_from_model(newsl)
        baseline_result_mean = float(baseline["result_mean_x"])
        newsl_result_mean = float(newsl["result_mean_x"])
        diff_result_mean = newsl_result_mean - baseline_result_mean

        baseline_result_means_x.append(baseline_result_mean)
        newsl_result_means_x.append(newsl_result_mean)
        baseline_total_counts += baseline_counts
        newsl_total_counts += newsl_counts
        bucket_position_counts[root_view_score_diff_bucket] += 1
        bucket_baseline_counts[root_view_score_diff_bucket] += baseline_counts
        bucket_newsl_counts[root_view_score_diff_bucket] += newsl_counts

        if diff_result_mean > 0.0:
            newsl_better_by_result_mean_x_count += 1
        elif diff_result_mean < 0.0:
            baseline_better_by_result_mean_x_count += 1
        else:
            tie_by_result_mean_x_count += 1

    baseline_result_means_x_np = np.asarray(baseline_result_means_x, dtype=np.float32)
    newsl_result_means_x_np = np.asarray(newsl_result_means_x, dtype=np.float32)
    diff_result_means_x_np = newsl_result_means_x_np - baseline_result_means_x_np
    total_trials_all_positions = len(records) * x_repeats

    root_view_score_diff_bucket_summary = {}
    for bucket_label in ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER:
        root_view_score_diff_bucket_summary[bucket_label] = {
            "baseline": build_bucket_trial_summary(
                bucket_baseline_counts[bucket_label],
                bucket_position_counts[bucket_label],
                x_repeats,
            ),
            "newsl": build_bucket_trial_summary(
                bucket_newsl_counts[bucket_label],
                bucket_position_counts[bucket_label],
                x_repeats,
            ),
        }

    summary = {
        "experiment": experiment,
        "num_positions": int(len(records)),
        "target_end": int(experiment["target_end"]),
        "target_shot": int(experiment["target_shot"]),
        "execution_repeats_x": int(x_repeats),
        "baseline_method_key": baseline_key,
        "baseline_method_label": baseline_label,
        "newsl_method_key": newsl_key,
        "newsl_method_label": newsl_label,
        "log_size_result_mean_x_baseline": float(np.mean(baseline_result_means_x_np)),
        "log_size_result_mean_x_newsl": float(np.mean(newsl_result_means_x_np)),
        "log_size_diff_result_mean_x_newsl_minus_baseline": float(
            np.mean(diff_result_means_x_np)
        ),
        "log_size_win_rate_x_baseline": float(
            baseline_total_counts[RESULT_WIN_IDX] / total_trials_all_positions
        ),
        "log_size_draw_rate_x_baseline": float(
            baseline_total_counts[RESULT_DRAW_IDX] / total_trials_all_positions
        ),
        "log_size_lose_rate_x_baseline": float(
            baseline_total_counts[RESULT_LOSE_IDX] / total_trials_all_positions
        ),
        "log_size_win_rate_x_newsl": float(
            newsl_total_counts[RESULT_WIN_IDX] / total_trials_all_positions
        ),
        "log_size_draw_rate_x_newsl": float(
            newsl_total_counts[RESULT_DRAW_IDX] / total_trials_all_positions
        ),
        "log_size_lose_rate_x_newsl": float(
            newsl_total_counts[RESULT_LOSE_IDX] / total_trials_all_positions
        ),
        "total_trials_all_positions": int(total_trials_all_positions),
        "all_trials_win_count_baseline": int(baseline_total_counts[RESULT_WIN_IDX]),
        "all_trials_draw_count_baseline": int(baseline_total_counts[RESULT_DRAW_IDX]),
        "all_trials_lose_count_baseline": int(baseline_total_counts[RESULT_LOSE_IDX]),
        "all_trials_win_count_newsl": int(newsl_total_counts[RESULT_WIN_IDX]),
        "all_trials_draw_count_newsl": int(newsl_total_counts[RESULT_DRAW_IDX]),
        "all_trials_lose_count_newsl": int(newsl_total_counts[RESULT_LOSE_IDX]),
        "all_trials_win_rate_baseline": float(
            baseline_total_counts[RESULT_WIN_IDX] / total_trials_all_positions
        ),
        "all_trials_draw_rate_baseline": float(
            baseline_total_counts[RESULT_DRAW_IDX] / total_trials_all_positions
        ),
        "all_trials_lose_rate_baseline": float(
            baseline_total_counts[RESULT_LOSE_IDX] / total_trials_all_positions
        ),
        "all_trials_win_rate_newsl": float(
            newsl_total_counts[RESULT_WIN_IDX] / total_trials_all_positions
        ),
        "all_trials_draw_rate_newsl": float(
            newsl_total_counts[RESULT_DRAW_IDX] / total_trials_all_positions
        ),
        "all_trials_lose_rate_newsl": float(
            newsl_total_counts[RESULT_LOSE_IDX] / total_trials_all_positions
        ),
        "newsl_better_by_result_mean_x_count": int(
            newsl_better_by_result_mean_x_count
        ),
        "baseline_better_by_result_mean_x_count": int(
            baseline_better_by_result_mean_x_count
        ),
        "tie_by_result_mean_x_count": int(tie_by_result_mean_x_count),
        "root_view_score_diff_bucket_order": list(ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER),
        "root_view_score_diff_bucket_summary": root_view_score_diff_bucket_summary,
    }
    return summary, baseline_result_means_x_np, newsl_result_means_x_np


def print_summary(json_dir: Path, png_path: Path, summary: dict[str, Any]) -> None:
    baseline_label = summary["baseline_method_label"]
    newsl_label = summary["newsl_method_label"]

    print("")
    print(f"Saved position json files to {json_dir}")
    print(f"Saved plot to {png_path}")
    print(f"num_positions                : {summary['num_positions']}")
    print(f"execution_repeats_x         : {summary['execution_repeats_x']}")
    print(
        f"log_size result_mean_x {baseline_label}: "
        f"{summary['log_size_result_mean_x_baseline']:.6f}"
    )
    print(
        f"log_size result_mean_x {newsl_label}: "
        f"{summary['log_size_result_mean_x_newsl']:.6f}"
    )
    print(
        f"log_size diff ({newsl_label} - {baseline_label}): "
        f"{summary['log_size_diff_result_mean_x_newsl_minus_baseline']:.6f}"
    )
    print(
        f"log_size win/draw/lose {baseline_label}: "
        f"{summary['log_size_win_rate_x_baseline']:.6f}, "
        f"{summary['log_size_draw_rate_x_baseline']:.6f}, "
        f"{summary['log_size_lose_rate_x_baseline']:.6f}"
    )
    print(
        f"log_size win/draw/lose {newsl_label}: "
        f"{summary['log_size_win_rate_x_newsl']:.6f}, "
        f"{summary['log_size_draw_rate_x_newsl']:.6f}, "
        f"{summary['log_size_lose_rate_x_newsl']:.6f}"
    )
    print(
        f"all_trials win/draw/lose {baseline_label}: "
        f"{summary['all_trials_win_count_baseline']}, "
        f"{summary['all_trials_draw_count_baseline']}, "
        f"{summary['all_trials_lose_count_baseline']}"
    )
    print(
        f"all_trials win/draw/lose {newsl_label}: "
        f"{summary['all_trials_win_count_newsl']}, "
        f"{summary['all_trials_draw_count_newsl']}, "
        f"{summary['all_trials_lose_count_newsl']}"
    )
    print(
        f"{newsl_label} better by result_mean_x count       : "
        f"{summary['newsl_better_by_result_mean_x_count']}"
    )
    print(
        f"{baseline_label} better by result_mean_x count        : "
        f"{summary['baseline_better_by_result_mean_x_count']}"
    )
    print(f"Tie by result_mean_x count                : {summary['tie_by_result_mean_x_count']}")
    print("")
    print("root_view_score_diff_before_shot bucket summary (all trials in each bucket)")
    for bucket_label in summary["root_view_score_diff_bucket_order"]:
        bucket_summary = summary["root_view_score_diff_bucket_summary"][bucket_label]
        baseline_bucket = bucket_summary["baseline"]
        newsl_bucket = bucket_summary["newsl"]
        print(
            f"bucket {bucket_label:>4} positions={baseline_bucket['num_positions']:4d} "
            f"trials={baseline_bucket['total_trials']:5d} "
            f"| {baseline_label} result_mean={baseline_bucket['result_mean_over_all_trials']:.4f} "
            f"win/draw/lose={baseline_bucket['win_rate_over_all_trials']:.4f},"
            f"{baseline_bucket['draw_rate_over_all_trials']:.4f},"
            f"{baseline_bucket['lose_rate_over_all_trials']:.4f} "
            f"| {newsl_label} result_mean={newsl_bucket['result_mean_over_all_trials']:.4f} "
            f"win/draw/lose={newsl_bucket['win_rate_over_all_trials']:.4f},"
            f"{newsl_bucket['draw_rate_over_all_trials']:.4f},"
            f"{newsl_bucket['lose_rate_over_all_trials']:.4f}"
        )


def render_report_from_records(
    json_dir: Path,
    png_path: Path,
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    summary, baseline_result_means_x, newsl_result_means_x = build_summary(records)
    save_result_plot(
        png_path,
        baseline_result_means_x,
        newsl_result_means_x,
        baseline_label=summary["baseline_method_label"],
        newsl_label=summary["newsl_method_label"],
    )
    print_summary(json_dir, png_path, summary)
    return summary, baseline_result_means_x, newsl_result_means_x


def render_report_from_json_dir(
    json_dir: Path,
    png_path: Path | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if png_path is None:
        png_path = json_dir.parent / f"{json_dir.name}.png"
    records = load_position_records(json_dir)
    return render_report_from_records(json_dir, png_path, records)
