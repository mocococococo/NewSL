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


def load_result_means_from_json(json_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
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

    return cnn_result_means_x, transformer_result_means_x, summary


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


def render_png_from_json(json_path: Path) -> tuple[Path, dict]:
    """Render a png file from one experiment json file."""

    cnn_result_means_x, transformer_result_means_x, summary = load_result_means_from_json(json_path)
    png_path = json_path.with_suffix(".png")
    save_result_plot(png_path, cnn_result_means_x, transformer_result_means_x)
    return png_path, summary


def print_summary(summary: dict) -> None:
    """Print a concise summary when the expected keys exist."""

    if not summary:
        return

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
    if not all(key in summary for key in keys):
        return

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
        png_path, summary = render_png_from_json(json_path)
        print(f"saved: {png_path}")
        print_summary(summary)


if __name__ == "__main__":
    main()
