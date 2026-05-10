"""kura_vs_ts.py が保存した局面別 JSON から PNG と集計表示を再生成する。

このスクリプトは Kura vs NewSL 専用。CNN 系や旧形式の単一 JSON は扱わない。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from kura_vs_ts_report import print_kura_vs_ts_summary, save_result_plot


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_PATH = (
    EXPERIMENT_DIR
    / "data"
    / "kura_vs_transformer_end9_shot15_winrate_datasize10_x10"
)
RESULT_WIN_IDX = 0
RESULT_DRAW_IDX = 1
RESULT_LOSE_IDX = 2
ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER = ("<=-2", "-1", "0", "+1", ">=+2")


def read_json(json_path: Path) -> dict[str, Any]:
    """JSON ファイルを読み込み、最上位がオブジェクトであることを確認する。"""

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{json_path} の最上位は JSON オブジェクトである必要があります。")
    return data


def load_position_records(json_dir: Path) -> list[dict[str, Any]]:
    """局面別 JSON ディレクトリを読み込み、position_index 順に並べる。"""

    if not json_dir.is_dir():
        raise FileNotFoundError(f"{json_dir} は存在しないか、ディレクトリではありません。")

    json_paths = sorted(json_dir.glob("*.json"))
    if len(json_paths) == 0:
        raise ValueError(f"{json_dir} に kura_vs_ts の局面別 JSON が見つかりません。")

    records = [read_json(json_path) for json_path in json_paths]
    for json_path, record in zip(json_paths, records):
        if not is_kura_vs_ts_position_record(record):
            raise ValueError(f"{json_path} は kura_vs_ts の局面別 JSON ではありません。")
    return sorted(records, key=lambda record: int(record["position"]["position_index"]))


def is_kura_vs_ts_position_record(record: dict[str, Any]) -> bool:
    """kura_vs_ts.py が保存する局面別 JSON の最小構造を満たすか判定する。"""

    position = record.get("position")
    kura = record.get("kura")
    newsl = record.get("newsl")
    return (
        isinstance(position, dict)
        and isinstance(kura, dict)
        and isinstance(newsl, dict)
        and "position_index" in position
        and "root_view_score_diff_bucket" in position
        and "result_mean_x" in kura
        and "result_mean_x" in newsl
    )


def counts_from_model(model_data: dict[str, Any]) -> np.ndarray:
    """1手法分の win/draw/lose count を numpy 配列に変換する。"""

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
    """score_diff bucket ごとの試行数・勝率・result_mean を集計する。"""

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


def build_summary(records: list[dict[str, Any]]) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """局面別 JSON 群から kura_vs_ts.py と同じ summary 形式を再構築する。"""

    if len(records) == 0:
        raise ValueError("records は空にできません。")

    experiment = records[0].get("experiment", {})
    if not isinstance(experiment, dict):
        experiment = {}
    x_repeats = int(experiment["execution_repeats_x"])

    kura_result_means_x: list[float] = []
    newsl_result_means_x: list[float] = []
    kura_total_counts = np.zeros(3, dtype=np.int64)
    newsl_total_counts = np.zeros(3, dtype=np.int64)
    newsl_better_by_result_mean_x_count = 0
    kura_better_by_result_mean_x_count = 0
    tie_by_result_mean_x_count = 0
    bucket_position_counts = {label: 0 for label in ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER}
    bucket_kura_counts = {
        label: np.zeros(3, dtype=np.int64) for label in ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER
    }
    bucket_newsl_counts = {
        label: np.zeros(3, dtype=np.int64) for label in ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER
    }

    # 各局面の Kura / NewSL の結果を、全体集計と bucket 集計へ逐次加算する。
    for record in records:
        position = record["position"]
        kura = record["kura"]
        newsl = record["newsl"]
        root_view_score_diff_bucket = str(position["root_view_score_diff_bucket"])
        if root_view_score_diff_bucket not in bucket_position_counts:
            raise ValueError(
                f"未知の root_view_score_diff_bucket です: {root_view_score_diff_bucket}"
            )

        kura_counts = counts_from_model(kura)
        newsl_counts = counts_from_model(newsl)
        kura_result_mean = float(kura["result_mean_x"])
        newsl_result_mean = float(newsl["result_mean_x"])
        diff_result_mean = newsl_result_mean - kura_result_mean

        kura_result_means_x.append(kura_result_mean)
        newsl_result_means_x.append(newsl_result_mean)
        kura_total_counts += kura_counts
        newsl_total_counts += newsl_counts
        bucket_position_counts[root_view_score_diff_bucket] += 1
        bucket_kura_counts[root_view_score_diff_bucket] += kura_counts
        bucket_newsl_counts[root_view_score_diff_bucket] += newsl_counts

        if diff_result_mean > 0.0:
            newsl_better_by_result_mean_x_count += 1
        elif diff_result_mean < 0.0:
            kura_better_by_result_mean_x_count += 1
        else:
            tie_by_result_mean_x_count += 1

    kura_result_means_x_np = np.asarray(kura_result_means_x, dtype=np.float32)
    newsl_result_means_x_np = np.asarray(newsl_result_means_x, dtype=np.float32)
    diff_result_means_x_np = newsl_result_means_x_np - kura_result_means_x_np
    total_trials_all_positions = len(records) * x_repeats

    # kura_vs_ts.py のコンソール表示関数へ渡せる形に bucket summary を整える。
    root_view_score_diff_bucket_summary = {}
    for bucket_label in ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER:
        root_view_score_diff_bucket_summary[bucket_label] = {
            "kura": build_bucket_trial_summary(
                bucket_kura_counts[bucket_label],
                bucket_position_counts[bucket_label],
                x_repeats,
            ),
            "transformer": build_bucket_trial_summary(
                bucket_newsl_counts[bucket_label],
                bucket_position_counts[bucket_label],
                x_repeats,
            ),
        }

    summary = {
        "num_positions": int(len(records)),
        "target_end": int(experiment["target_end"]),
        "target_shot": int(experiment["target_shot"]),
        "execution_repeats_x": int(x_repeats),
        "kura_policy_model": str(experiment.get("kura_policy_model", "")),
        "kura_top_k": int(experiment.get("kura_top_k", 0)),
        "kura_num_trials": int(experiment.get("kura_num_trials", 0)),
        "kura_rel_keep": float(experiment.get("kura_rel_keep", 0.0)),
        "transformer_model": str(experiment.get("newsl_transformer_model", "")),
        "log_size_result_mean_x_kura": float(np.mean(kura_result_means_x_np)),
        "log_size_result_mean_x_transformer": float(np.mean(newsl_result_means_x_np)),
        "log_size_diff_result_mean_x_transformer_minus_kura": float(
            np.mean(diff_result_means_x_np)
        ),
        "log_size_win_rate_x_kura": float(
            kura_total_counts[RESULT_WIN_IDX] / total_trials_all_positions
        ),
        "log_size_draw_rate_x_kura": float(
            kura_total_counts[RESULT_DRAW_IDX] / total_trials_all_positions
        ),
        "log_size_lose_rate_x_kura": float(
            kura_total_counts[RESULT_LOSE_IDX] / total_trials_all_positions
        ),
        "log_size_win_rate_x_transformer": float(
            newsl_total_counts[RESULT_WIN_IDX] / total_trials_all_positions
        ),
        "log_size_draw_rate_x_transformer": float(
            newsl_total_counts[RESULT_DRAW_IDX] / total_trials_all_positions
        ),
        "log_size_lose_rate_x_transformer": float(
            newsl_total_counts[RESULT_LOSE_IDX] / total_trials_all_positions
        ),
        "total_trials_all_positions": int(total_trials_all_positions),
        "all_trials_win_count_kura": int(kura_total_counts[RESULT_WIN_IDX]),
        "all_trials_draw_count_kura": int(kura_total_counts[RESULT_DRAW_IDX]),
        "all_trials_lose_count_kura": int(kura_total_counts[RESULT_LOSE_IDX]),
        "all_trials_win_count_transformer": int(newsl_total_counts[RESULT_WIN_IDX]),
        "all_trials_draw_count_transformer": int(newsl_total_counts[RESULT_DRAW_IDX]),
        "all_trials_lose_count_transformer": int(newsl_total_counts[RESULT_LOSE_IDX]),
        "all_trials_win_rate_kura": float(
            kura_total_counts[RESULT_WIN_IDX] / total_trials_all_positions
        ),
        "all_trials_draw_rate_kura": float(
            kura_total_counts[RESULT_DRAW_IDX] / total_trials_all_positions
        ),
        "all_trials_lose_rate_kura": float(
            kura_total_counts[RESULT_LOSE_IDX] / total_trials_all_positions
        ),
        "all_trials_win_rate_transformer": float(
            newsl_total_counts[RESULT_WIN_IDX] / total_trials_all_positions
        ),
        "all_trials_draw_rate_transformer": float(
            newsl_total_counts[RESULT_DRAW_IDX] / total_trials_all_positions
        ),
        "all_trials_lose_rate_transformer": float(
            newsl_total_counts[RESULT_LOSE_IDX] / total_trials_all_positions
        ),
        "transformer_better_by_result_mean_x_count": int(
            newsl_better_by_result_mean_x_count
        ),
        "kura_better_by_result_mean_x_count": int(kura_better_by_result_mean_x_count),
        "tie_by_result_mean_x_count": int(tie_by_result_mean_x_count),
        "root_view_score_diff_bucket_order": list(ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER),
        "root_view_score_diff_bucket_summary": root_view_score_diff_bucket_summary,
    }
    return summary, kura_result_means_x_np, newsl_result_means_x_np


def main(
    target_path: str | Path = DEFAULT_TARGET_PATH,
) -> None:
    """指定ディレクトリの局面別 JSON から PNG とコンソール表示を再生成する。"""

    json_dir = Path(target_path)
    png_path = json_dir.parent / f"{json_dir.name}.png"
    records = load_position_records(json_dir)
    summary, kura_result_means_x, newsl_result_means_x = build_summary(records)

    save_result_plot(png_path, kura_result_means_x, newsl_result_means_x)
    print_kura_vs_ts_summary(json_dir, png_path, summary)


if __name__ == "__main__":
    main(
        target_path=Path(__file__).resolve().parents[1]
        / "data"
        / "kura_vs_transformer_end9_shot15_winrate_datasize1000_x10"
    )
