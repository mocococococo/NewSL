"""
experiment/data 配下の比較結果 json から png を再生成する。
"""

import argparse
import json
from pathlib import Path

import numpy as np

from experiment_cnn_vs_transformer_9_15 import save_result_plot

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_PATH = ROOT_DIR / "experiment" / "data"
REQUIRED_POSITION_KEYS = ("cnn_mean_score", "transformer_mean_score")


def load_scores_from_json(json_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """json から描画用の平均得点列を読み出す。"""

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    positions = data.get("positions")
    if not isinstance(positions, list) or len(positions) == 0:
        raise ValueError(f"{json_path} does not contain a non-empty 'positions' list")

    if all(isinstance(row, dict) and "position_index" in row for row in positions):
        positions = sorted(positions, key=lambda row: row["position_index"])

    try:
        cnn_scores = np.asarray(
            [float(row["cnn_mean_score"]) for row in positions],
            dtype=np.float32,
        )
        transformer_scores = np.asarray(
            [float(row["transformer_mean_score"]) for row in positions],
            dtype=np.float32,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{json_path} has invalid score rows") from error

    if cnn_scores.shape != transformer_scores.shape:
        raise ValueError(
            f"{json_path} has mismatched score shapes: "
            f"{tuple(cnn_scores.shape)} != {tuple(transformer_scores.shape)}"
        )

    return cnn_scores, transformer_scores


def json_matches_schema(json_path: Path) -> bool:
    """Return True when the json looks like a score experiment output."""

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
    """Resolve score experiment json files from a file or directory."""

    if target_path.is_file():
        if target_path.suffix.lower() != ".json":
            raise ValueError(f"{target_path} is not a json file")
        if not json_matches_schema(target_path):
            raise ValueError(f"{target_path} is not a score experiment json file")
        return [target_path]

    if target_path.is_dir():
        json_paths = sorted(
            json_path for json_path in target_path.glob("*.json")
            if json_matches_schema(json_path)
        )
        if len(json_paths) == 0:
            raise ValueError(f"No score experiment json files found in {target_path}")
        return json_paths

    raise FileNotFoundError(f"{target_path} does not exist")


def render_png_from_json(json_path: Path) -> Path:
    """1 個の json から同名 png を生成する。"""

    cnn_scores, transformer_scores = load_scores_from_json(json_path)
    png_path = json_path.with_suffix(".png")
    save_result_plot(png_path, cnn_scores, transformer_scores)
    return png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate comparison png files from experiment json files."
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
        png_path = render_png_from_json(json_path)
        print(f"saved: {png_path}")


if __name__ == "__main__":
    main()
