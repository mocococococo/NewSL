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
    / "mini_match_kura_vs_transformer_end9_shot14_datasize1_x1"
)


def main(target_path: str | Path = DEFAULT_TARGET_PATH, ab_reverse: bool = False) -> None:
    print_report_from_json_dir(Path(target_path), ab_reverse=ab_reverse)


if __name__ == "__main__":
    main(
        target_path=Path(__file__).resolve().parents[1]
        / "data"
        / "mini_match_cnn_vs_transformer_end9_shot9_datasize1000_x1",
        ab_reverse=True,
    )
