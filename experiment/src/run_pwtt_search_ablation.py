from __future__ import annotations

import sys
from pathlib import Path


NEWSL_DIR = Path(__file__).resolve().parents[2]
if str(NEWSL_DIR) not in sys.path:
    sys.path.insert(0, str(NEWSL_DIR))

from experiment.src.pwtt_search_ablation import run_experiment
from experiment.src.pwtt_search_ablation_report import print_report_from_json_dir


def main() -> None:
    json_dir = run_experiment(
        log_path=NEWSL_DIR / "LearnLog" / "all",
        save_path=Path(__file__).resolve().parents[1] / "data",
        target_end=9,
        target_shot=15,
        data_size=1000,
        X=1,
        model="js20000CP-32-9-LeaRate1000-vx32-vy25-batchsize1024.bin",
        use_gpu=True,
        use_transformer=False,
        transformer_models_by_shot=None,
        shuffle_seed=12345,
        search_seed=24680,
        condition_a="non-tt",
        condition_b="pwtt",
    )
    print_report_from_json_dir(json_dir)


if __name__ == "__main__":
    main()