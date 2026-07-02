from __future__ import annotations

import sys
from pathlib import Path


NEWSL_DIR = Path(__file__).resolve().parents[2]
if str(NEWSL_DIR) not in sys.path:
    sys.path.insert(0, str(NEWSL_DIR))

from experiment.src.mini_match_report import print_report_from_json_dir, load_position_records, print_report_from_records


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_PATH = (
    EXPERIMENT_DIR
    / "data"
    / "shot_vs_puct_transformer_end9_shot9_datasize1000_x1"
)


def main(
    target_path: str | Path = DEFAULT_TARGET_PATH,
    ab_reverse: bool = False,
    max_positions: int | None = None,
) -> None:
    target_path = Path(target_path)
    records = load_position_records(target_path)

    if max_positions is not None:
        records = records[:max_positions]

    print_report_from_records(
        target_path,
        records,
        ab_reverse=ab_reverse,
    )


if __name__ == "__main__":
    main(
        target_path=DEFAULT_TARGET_PATH,
        ab_reverse=True,
        max_positions=None,
    )
