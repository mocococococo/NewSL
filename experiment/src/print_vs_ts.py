from __future__ import annotations

import sys
from pathlib import Path

NEWSL_DIR = Path(__file__).resolve().parents[2]
if str(NEWSL_DIR) not in sys.path:
    sys.path.insert(0, str(NEWSL_DIR))

from experiment.src.report import render_report_from_json_dir


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_PATH = (
    EXPERIMENT_DIR
    / "data"
    / "kura_vs_transformer_end9_shot15_winrate_datasize1000_x10"
)


def main(target_path: str | Path = DEFAULT_TARGET_PATH) -> None:
    render_report_from_json_dir(Path(target_path))

if __name__ == "__main__":
    main(
        target_path=Path(__file__).resolve().parents[1]
        / "data"
        / "kura_vs_shot_transformer-sl-9-15-model-06-02-AdamW-epoch50-shot_end9_shot15_winrate_datasize1000_x1"
    )
