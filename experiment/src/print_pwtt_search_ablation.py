from __future__ import annotations

import sys
from pathlib import Path


NEWSL_DIR = Path(__file__).resolve().parents[2]
if str(NEWSL_DIR) not in sys.path:
    sys.path.insert(0, str(NEWSL_DIR))

from experiment.src.pwtt_search_ablation_report import print_report_from_json_dir


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_PATH = (
    EXPERIMENT_DIR
    / "data"
    / "pwtt_search_ablation_non-pw_vs_pwtt_end9_shot15_datasize1000_x1"
)


def main(target_path: str | Path = DEFAULT_TARGET_PATH, max_positions: int | None = None) -> None:
    print_report_from_json_dir(Path(target_path), max_positions=max_positions)


if __name__ == "__main__":
    main(
        target_path=DEFAULT_TARGET_PATH,
        max_positions=None,
    )