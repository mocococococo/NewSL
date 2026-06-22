from __future__ import annotations

import sys
from pathlib import Path


NEWSL_DIR = Path(__file__).resolve().parents[2]
if str(NEWSL_DIR) not in sys.path:
    sys.path.insert(0, str(NEWSL_DIR))

from experiment.src.mini_match_report import print_report_from_json_dir


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_PATH = (
    EXPERIMENT_DIR
    / "data"
    / "value_vs_rollout_shot_transformer_end9_shot14_datasize1000_x1"
)


def main(
    target_path: str | Path = DEFAULT_TARGET_PATH,
    ab_reverse: bool = False,
) -> None:
    print_report_from_json_dir(Path(target_path), ab_reverse=ab_reverse)


if __name__ == "__main__":
    main(
        target_path=DEFAULT_TARGET_PATH,
        ab_reverse=False,
    )
