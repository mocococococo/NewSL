"""Run existing SHOT generation and training in reverse end/shot order."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learning_param import BATCH_SIZE, EPOCHS
from shot.params import DEFAULT_SHOT_INFERENCE_BATCH_SIZE
from search_config import ACTIVE_SEARCH_SETTINGS, require_search_mode


def positions(start_end=9, start_shot=15, stop_end=0, stop_shot=0):
    for end, shot in ((start_end, start_shot), (stop_end, stop_shot)):
        if not 0 <= end <= 9 or not 0 <= shot <= 15:
            raise ValueError("end must be 0..9 and shot must be 0..15")
    start, stop = start_end * 16 + start_shot, stop_end * 16 + stop_shot
    if stop > start:
        raise ValueError("The stop position must precede the start in game time")
    return [divmod(index, 16) for index in range(start, stop - 1, -1)]


def model_name(end, shot):
    return f"{ACTIVE_SEARCH_SETTINGS.mode}-end{end}-shot{shot}"


def model_path(program_dir, end, shot):
    return Path(program_dir) / "model" / f"{model_name(end, shot)}.bin"


def data_dir(program_dir, end, shot):
    return Path(program_dir) / "data" / f"end{end}" / f"shot{shot}"


def generation_options(config, end, shot):
    """Keep loop data settings while selecting the complete search profile."""
    require_search_mode(config.get("search_mode", ACTIVE_SEARCH_SETTINGS.mode))
    teacher = model_path(config["program_dir"], end, shot + 1) if shot < 15 else None
    options = dict(
        log_path=config["log_path"], save_path=Path(config["program_dir"]) / "data",
        data_size=70000, target_end=end, target_shot=[shot],
        use_end_augmentation=True, use_score_diff_augmentation=False,
        model=config["base_model"], use_transformer=teacher is not None,
        transformer_model=teacher, transformer_target_end=[end],
        transformer_target_shot=[shot + 1] if teacher is not None else [],
        max_simulations=ACTIVE_SEARCH_SETTINGS.max_simulations, use_gpu=True, shuffle_seed=12345,
        chunk_start=config["chunk_start"], chunk_end=config["chunk_end"], chunk_size=BATCH_SIZE,
        num_workers=config["num_workers"], simulation_seed=config["simulation_seed"],
        inference_batch_size=config["inference_batch_size"],
        policy_min_visit=3, policy_delta_q=1.0, policy_alpha_visit=0.2,
        policy_beta_q=0.3, policy_lambda_best=0.5,
        value_min_visit=3, value_delta_q=0.0, value_alpha_visit=0.5,
        value_beta_q=0.5, value_lambda_best=0.5,
    )
    if ACTIVE_SEARCH_SETTINGS.mode == "shot_origin":
        options["search_mode"] = "shot_origin"
    return options


def run_worker(job):
    import numpy as np
    import torch
    from transformer.learn import train
    from transformer.shot_generator import generate_data
    from transformer.utility import split_train_test_set

    config, end, shot = job["config"], job["end"], job["shot"]
    require_search_mode(config.get("search_mode", ACTIVE_SEARCH_SETTINGS.mode))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GPU pipeline")
    if job["stage"] == "generate":
        options = generation_options(config, end, shot)
        if options["transformer_model"] is not None and not options["transformer_model"].is_file():
            raise FileNotFoundError(f"Required teacher model: {options['transformer_model']}")
        generate_data(**options)
    elif job["stage"] == "train":
        folder = data_dir(config["program_dir"], end, shot)
        files = sorted(str(path) for path in folder.glob("sl_data_*.npz"))
        if not files:
            raise ValueError(f"No generated training data: {folder}")
        train_files, _ = split_train_test_set(files, 0.9)
        # Do not advance to the next position with a model that had no updates.
        has_batch = False
        for path in train_files:
            with np.load(path) as data:
                if len(data["value"]) >= config["batch_size"]:
                    has_batch = True
                    break
        if not has_batch:
            raise ValueError(f"No complete training batch: {folder}")
        train(program_dir=config["program_dir"], data_dir=folder,
              model_name=model_name(end, shot), use_gpu=True,
              batch_size=config["batch_size"], epochs=config["epochs"])
    else:
        raise ValueError(f"Unknown stage: {job['stage']}")


def run_stage(config, end, shot, stage):
    """A fresh process releases generation models before training starts."""
    job = dict(config=config, end=end, shot=shot, stage=stage)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker-job", json.dumps(job)]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if sys.dont_write_bytecode:
        env["PYTHONDONTWRITEBYTECODE"] = "1"
    with subprocess.Popen(
        command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        if process.wait():
            raise RuntimeError(f"end={end} shot={shot}: {stage} failed")


def run_pipeline(config):
    require_search_mode(config.get("search_mode", ACTIVE_SEARCH_SETTINGS.mode))
    config = dict(config, search_mode=ACTIVE_SEARCH_SETTINGS.mode)
    sequence = positions(config["start_end"], config["start_shot"],
                         config["stop_end"], config["stop_shot"])
    # A run learns only its newly generated files, not leftovers from older runs.
    for end, shot in sequence:
        folder = data_dir(config["program_dir"], end, shot)
        if any(folder.glob("sl_data_*.npz")):
            raise FileExistsError(f"Training data already exists: {folder}. Choose another --program-dir.")
        path = model_path(config["program_dir"], end, shot)
        if path.exists():
            raise FileExistsError(f"Model already exists: {path}. Choose another --program-dir.")
    start_end, start_shot = sequence[0]
    if start_shot < 15:
        teacher = model_path(config["program_dir"], start_end, start_shot + 1)
        if not teacher.is_file():
            raise FileNotFoundError(f"Required teacher model: {teacher}")
    for index, (end, shot) in enumerate(sequence, 1):
        print(f"[PIPELINE {index}/{len(sequence)}] end={end} shot={shot}", flush=True)
        run_stage(config, end, shot, "generate")
        run_stage(config, end, shot, "train")
        path = model_path(config["program_dir"], end, shot)
        if not path.is_file():
            raise FileNotFoundError(f"Training did not save its model: {path}")
    print(f"[PIPELINE] completed {len(sequence)} positions", flush=True)


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker-job":
        run_worker(json.loads(sys.argv[2]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk_start", type=int, required=True)
    parser.add_argument("--chunk_end", type=int, required=True)
    parser.add_argument("--num_workers", type=positive, default=4)
    parser.add_argument("--inference_batch_size", type=positive, default=DEFAULT_SHOT_INFERENCE_BATCH_SIZE)
    parser.add_argument("--simulation_seed", type=int, default=0)
    parser.add_argument("--batch-size", type=positive, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=positive, default=EPOCHS)
    parser.add_argument("--program-dir", type=Path, default=ROOT,
                        help="root for data/, model/ and record/ outputs")
    parser.add_argument("--log-path", type=Path, default=ROOT / "LearnLog/all")
    parser.add_argument("--base-model", type=Path,
                        default=ROOT / "model" / ACTIVE_SEARCH_SETTINGS.base_model_name)
    for name, default in (("start-end", 9), ("start-shot", 15), ("stop-end", 0), ("stop-shot", 0)):
        parser.add_argument(f"--{name}", type=int, default=default)
    args = parser.parse_args()
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    positions(args.start_end, args.start_shot, args.stop_end, args.stop_shot)
    if args.chunk_start < 0 or args.chunk_end < args.chunk_start or args.simulation_seed < 0:
        parser.error("Invalid chunk range or simulation seed")
    if not args.log_path.is_dir() or not args.base_model.is_file():
        parser.error("log-path and base-model must exist")
    if args.chunk_end * BATCH_SIZE >= len(os.listdir(args.log_path)):
        parser.error("chunk_end is outside the input records")
    # The legacy CNN loader swallows load errors. Check once before starting the loop.
    import torch
    from nn.network.dual_net import DualNet
    if not torch.cuda.is_available():
        parser.error("CUDA is required for this GPU pipeline")
    if ACTIVE_SEARCH_SETTINGS.mode == "shot":
        network = DualNet(torch.device("cpu"))
    else:
        from transformer.network import TransformerNetwork
        network = TransformerNetwork()
    network.load_state_dict(torch.load(args.base_model, map_location="cpu", weights_only=True))
    del network
    run_pipeline(config)


if __name__ == "__main__":
    main()
