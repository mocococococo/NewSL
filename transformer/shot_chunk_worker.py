"""Internal subprocess entry point for one SHOT teacher-data chunk."""
import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from transformer import shot_generator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_file", type=Path)
    args = parser.parse_args()
    config = json.loads(args.job_file.read_text(encoding="utf-8"))
    shot_generator._initialize_chunk_worker(
        config["threads"], config["cudnn_flags"], config["matmul_allow_tf32"],
    )
    shot_generator.BATCH_SIZE = config["batch_size"]
    shot_generator.DATA_SET_SIZE = config["data_set_size"]
    shot_generator._run_chunk(*config["job"])


if __name__ == "__main__":
    main()
