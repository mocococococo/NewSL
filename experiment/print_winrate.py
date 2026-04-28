"""
Generate png files from experiment_winrate.py output json files.
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_PATH = ROOT_DIR / "experiment" / "data"
REQUIRED_POSITION_KEYS = ("cnn_result_mean_x", "transformer_result_mean_x")
RESULT_WIN_IDX = 0
RESULT_DRAW_IDX = 1
RESULT_LOSE_IDX = 2
ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER = ("<=-2", "-1", "0", "+1", ">=+2")


def save_result_plot(
    save_file_path: Path,
    cnn_result_means_x: np.ndarray,
    transformer_result_means_x: np.ndarray,
) -> None:
    """Save a comparison plot for result_mean_x values."""

    import matplotlib.pyplot as plt

    indices = np.arange(len(cnn_result_means_x))
    diff_result_means_x = transformer_result_means_x - cnn_result_means_x
    if len(indices) == 0:
        raise ValueError("cnn_result_means_x and transformer_result_means_x must not be empty")
    x_max = len(indices) - 1

    figure, (ax_score, ax_diff) = plt.subplots(2, 1, figsize=(12, 8))

    x_tick_step = max(1, int(np.ceil((x_max + 1) / 20)))
    x_ticks = np.arange(0, x_max + 1, x_tick_step)
    if x_ticks[-1] != x_max:
        x_ticks = np.append(x_ticks, x_max)

    ax_score.plot(indices, cnn_result_means_x, label="CNN", linewidth=1.0, alpha=0.5)
    ax_score.plot(indices, transformer_result_means_x, label="Transformer", linewidth=1.0, alpha=0.5)
    ax_score.set_ylabel("Result Mean Over X Runs")
    ax_score.set_title("CNN vs Transformer")
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

    ax_diff.plot(indices, diff_result_means_x, color="tab:green", linewidth=1.0, alpha=0.6)
    ax_diff.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax_diff.set_xlabel("Position Index")
    ax_diff.set_ylabel("Result Mean Diff")
    ax_diff.set_title("Transformer - CNN")
    ax_diff.grid(True, alpha=0.3)
    if x_max == 0:
        ax_diff.set_xlim(-0.5, 0.5)
        ax_diff.set_xticks([0])
    else:
        ax_diff.set_xlim(0, x_max)
        ax_diff.set_xticks(x_ticks)

    figure.tight_layout()
    figure.savefig(save_file_path, dpi=150)
    plt.close(figure)


def score_diff_for_team_view(score_diff_for_team0: int, team: int) -> int:
    return score_diff_for_team0 if team == 0 else -score_diff_for_team0


def to_move_team(shot_index: int, hammer_team: int) -> int:
    return hammer_team if (shot_index % 2 == 1) else 1 - hammer_team


def bucket_root_view_score_diff(score_diff: int) -> str:
    if score_diff <= -2:
        return "<=-2"
    if score_diff == -1:
        return "-1"
    if score_diff == 0:
        return "0"
    if score_diff == 1:
        return "+1"
    return ">=+2"


def summarize_bucket_counts(counts: np.ndarray, num_positions: int) -> dict:
    total_trials = int(np.sum(counts))
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


def infer_root_view_team(row: dict) -> int:
    if "root_view_team" in row:
        return int(row["root_view_team"])

    shot_index = int(row["shot"])
    hammer_team = int(row["hammer"])
    return to_move_team(shot_index, hammer_team)


def extract_model_counts(row: dict, model_prefix: str, execution_repeats_x: int | None) -> np.ndarray:
    count_keys = (
        f"{model_prefix}_win_count_x",
        f"{model_prefix}_draw_count_x",
        f"{model_prefix}_lose_count_x",
    )
    if all(key in row for key in count_keys):
        return np.asarray([int(row[key]) for key in count_keys], dtype=np.int64)

    if execution_repeats_x is None:
        raise ValueError(
            f"Row does not contain {model_prefix} count fields and execution_repeats_x is unavailable"
        )

    rate_keys = (
        f"{model_prefix}_win_rate_x",
        f"{model_prefix}_draw_rate_x",
        f"{model_prefix}_lose_rate_x",
    )
    if not all(key in row for key in rate_keys):
        raise ValueError(
            f"Row does not contain {model_prefix} count fields or rate fields required for fallback"
        )

    counts = np.asarray(
        [int(round(float(row[key]) * execution_repeats_x)) for key in rate_keys],
        dtype=np.int64,
    )
    delta = int(execution_repeats_x - int(np.sum(counts)))
    counts[RESULT_DRAW_IDX] += delta
    return counts


def rebuild_bucket_summary_from_positions(
    positions: list[dict],
    execution_repeats_x: int | None,
) -> dict[str, dict[str, dict]]:
    bucket_position_counts = {label: 0 for label in ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER}
    bucket_counts = {
        "cnn": {
            label: np.zeros(3, dtype=np.int64) for label in ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER
        },
        "transformer": {
            label: np.zeros(3, dtype=np.int64) for label in ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER
        },
    }

    for row in positions:
        root_view_team = infer_root_view_team(row)
        if "root_view_score_diff_before_shot" in row:
            root_view_score_diff_before_shot = int(row["root_view_score_diff_before_shot"])
        else:
            root_view_score_diff_before_shot = score_diff_for_team_view(
                int(row["score_diff_for_team0"]),
                root_view_team,
            )

        bucket_label = row.get("root_view_score_diff_bucket")
        if bucket_label is None:
            bucket_label = bucket_root_view_score_diff(root_view_score_diff_before_shot)
        if bucket_label not in bucket_position_counts:
            continue

        bucket_position_counts[bucket_label] += 1
        bucket_counts["cnn"][bucket_label] += extract_model_counts(
            row,
            "cnn",
            execution_repeats_x,
        )
        bucket_counts["transformer"][bucket_label] += extract_model_counts(
            row,
            "transformer",
            execution_repeats_x,
        )

    bucket_summary = {}
    for bucket_label in ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER:
        bucket_summary[bucket_label] = {
            "cnn": summarize_bucket_counts(
                bucket_counts["cnn"][bucket_label],
                bucket_position_counts[bucket_label],
            ),
            "transformer": summarize_bucket_counts(
                bucket_counts["transformer"][bucket_label],
                bucket_position_counts[bucket_label],
            ),
        }
    return bucket_summary


def resolve_bucket_summary(
    summary: dict,
    positions: list[dict],
) -> tuple[list[str], dict[str, dict[str, dict]]] | None:
    bucket_summary = summary.get("root_view_score_diff_bucket_summary")
    if isinstance(bucket_summary, dict):
        bucket_order = summary.get("root_view_score_diff_bucket_order", list(ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER))
        bucket_order = [str(label) for label in bucket_order]
        return bucket_order, bucket_summary

    execution_repeats_x = summary.get("execution_repeats_x")
    if execution_repeats_x is not None:
        execution_repeats_x = int(execution_repeats_x)

    try:
        rebuilt_summary = rebuild_bucket_summary_from_positions(positions, execution_repeats_x)
    except (KeyError, TypeError, ValueError):
        return None

    return list(ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER), rebuilt_summary


def load_result_means_from_json(json_path: Path) -> tuple[np.ndarray, np.ndarray, dict, list[dict]]:
    """Load result_mean_x arrays and summary from a json file."""

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    positions = data.get("positions")
    if not isinstance(positions, list) or len(positions) == 0:
        raise ValueError(f"{json_path} does not contain a non-empty 'positions' list")

    if all(isinstance(row, dict) and "position_index" in row for row in positions):
        positions = sorted(positions, key=lambda row: row["position_index"])

    try:
        cnn_result_means_x = np.asarray(
            [float(row["cnn_result_mean_x"]) for row in positions],
            dtype=np.float32,
        )
        transformer_result_means_x = np.asarray(
            [float(row["transformer_result_mean_x"]) for row in positions],
            dtype=np.float32,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{json_path} has invalid result_mean_x rows") from error

    if cnn_result_means_x.shape != transformer_result_means_x.shape:
        raise ValueError(
            f"{json_path} has mismatched result_mean_x shapes: "
            f"{tuple(cnn_result_means_x.shape)} != {tuple(transformer_result_means_x.shape)}"
        )

    summary = data.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}

    return cnn_result_means_x, transformer_result_means_x, summary, positions


def json_matches_schema(json_path: Path) -> bool:
    """Return True when the json looks like a winrate experiment output."""

    try:
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False

    positions = data.get("positions")
    if not isinstance(positions, list) or len(positions) == 0:
        return False

    first_row = positions[0]
    if not isinstance(first_row, dict):
        return False

    return all(key in first_row for key in REQUIRED_POSITION_KEYS)


def resolve_json_paths(target_path: Path) -> list[Path]:
    """Resolve a single json file or all json files in a directory."""

    if target_path.is_file():
        if target_path.suffix.lower() != ".json":
            raise ValueError(f"{target_path} is not a json file")
        if not json_matches_schema(target_path):
            raise ValueError(f"{target_path} is not a winrate experiment json file")
        return [target_path]

    if target_path.is_dir():
        json_paths = sorted(
            json_path for json_path in target_path.glob("*.json")
            if json_matches_schema(json_path)
        )
        if len(json_paths) == 0:
            raise ValueError(f"No winrate experiment json files found in {target_path}")
        return json_paths

    raise FileNotFoundError(f"{target_path} does not exist")


def render_png_from_json(json_path: Path) -> tuple[Path, dict, list[dict]]:
    """Render a png file from one experiment json file."""

    cnn_result_means_x, transformer_result_means_x, summary, positions = load_result_means_from_json(json_path)
    png_path = json_path.with_suffix(".png")
    save_result_plot(png_path, cnn_result_means_x, transformer_result_means_x)
    return png_path, summary, positions


def print_summary(summary: dict, positions: list[dict]) -> None:
    """Print a concise summary when the expected keys exist."""

    if not isinstance(summary, dict):
        summary = {}

    keys = [
        "num_positions",
        "execution_repeats_x",
        "log_size_result_mean_x_cnn",
        "log_size_result_mean_x_transformer",
        "log_size_diff_result_mean_x_transformer_minus_cnn",
        "log_size_win_rate_x_cnn",
        "log_size_draw_rate_x_cnn",
        "log_size_lose_rate_x_cnn",
        "log_size_win_rate_x_transformer",
        "log_size_draw_rate_x_transformer",
        "log_size_lose_rate_x_transformer",
        "transformer_better_by_result_mean_x_count",
        "cnn_better_by_result_mean_x_count",
        "tie_by_result_mean_x_count",
    ]
    if all(key in summary for key in keys):
        print(f"  num_positions: {summary['num_positions']}")
        print(f"  execution_repeats_x: {summary['execution_repeats_x']}")
        print(f"  log_size result_mean_x CNN: {summary['log_size_result_mean_x_cnn']:.6f}")
        print(f"  log_size result_mean_x Transformer: {summary['log_size_result_mean_x_transformer']:.6f}")
        print(
            "  log_size diff result_mean_x (T - CNN): "
            f"{summary['log_size_diff_result_mean_x_transformer_minus_cnn']:.6f}"
        )
        print(
            "  log_size win/draw/lose CNN: "
            f"{summary['log_size_win_rate_x_cnn']:.6f}, "
            f"{summary['log_size_draw_rate_x_cnn']:.6f}, "
            f"{summary['log_size_lose_rate_x_cnn']:.6f}"
        )
        print(
            "  log_size win/draw/lose Transformer: "
            f"{summary['log_size_win_rate_x_transformer']:.6f}, "
            f"{summary['log_size_draw_rate_x_transformer']:.6f}, "
            f"{summary['log_size_lose_rate_x_transformer']:.6f}"
        )
        print(
            "  better/tie by result_mean_x (T, CNN, tie): "
            f"{summary['transformer_better_by_result_mean_x_count']}, "
            f"{summary['cnn_better_by_result_mean_x_count']}, "
            f"{summary['tie_by_result_mean_x_count']}"
        )

    bucket_summary_data = resolve_bucket_summary(summary, positions)
    if bucket_summary_data is None:
        return

    bucket_order, bucket_summary = bucket_summary_data
    print("  root_view_score_diff_before_shot bucket summary:")
    for bucket_label in bucket_order:
        bucket_value = bucket_summary.get(bucket_label)
        if not isinstance(bucket_value, dict):
            continue
        cnn_bucket = bucket_value.get("cnn", {})
        transformer_bucket = bucket_value.get("transformer", {})
        print(
            f"    bucket {bucket_label:>4} positions={int(cnn_bucket.get('num_positions', 0)):4d} "
            f"trials={int(cnn_bucket.get('total_trials', 0)):5d} "
            f"| CNN result_mean={float(cnn_bucket.get('result_mean_over_all_trials', 0.0)):.4f} "
            f"win/draw/lose={float(cnn_bucket.get('win_rate_over_all_trials', 0.0)):.4f},"
            f"{float(cnn_bucket.get('draw_rate_over_all_trials', 0.0)):.4f},"
            f"{float(cnn_bucket.get('lose_rate_over_all_trials', 0.0)):.4f} "
            f"| T result_mean={float(transformer_bucket.get('result_mean_over_all_trials', 0.0)):.4f} "
            f"win/draw/lose={float(transformer_bucket.get('win_rate_over_all_trials', 0.0)):.4f},"
            f"{float(transformer_bucket.get('draw_rate_over_all_trials', 0.0)):.4f},"
            f"{float(transformer_bucket.get('lose_rate_over_all_trials', 0.0)):.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate winrate comparison png files from experiment json files."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=str(DEFAULT_TARGET_PATH),
        help="JSON file path or directory containing JSON files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_path = Path(args.target)

    json_paths = resolve_json_paths(target_path)
    for json_path in json_paths:
        png_path, summary, positions = render_png_from_json(json_path)
        print(f"saved: {png_path}")
        print_summary(summary, positions)


if __name__ == "__main__":
    main()
