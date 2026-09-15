"""Generate exactly one assigned chunk and record its state on this PC."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from remote_generate.state_store import (ROOT, file_hash, identifier, local_path, now,
                                         object_hash, process_token, transaction, worker)


def source_signature():
    """Generation must use the same committed Python sources on every PC."""
    names = subprocess.check_output(["git", "ls-files", "-z", "--", "*.py"], cwd=ROOT).decode().split("\0")
    names = sorted(name for name in names if name)
    required = {p.relative_to(ROOT).as_posix() for p in Path(__file__).parent.glob("*.py")}
    if not required.issubset(names):
        raise RuntimeError("Commit and distribute remote_generate/*.py before starting")
    for args in (("diff", "--quiet", "--", "*.py"), ("diff", "--cached", "--quiet", "--", "*.py")):
        subprocess.run(["git", *args], cwd=ROOT, check=True)
    return object_hash([(name, hashlib.sha256(local_path(name).read_bytes().replace(b"\r\n", b"\n")).hexdigest())
                        for name in names])


def input_signature(log_path, chunk, chunk_size, shuffle_seed):
    folder = local_path(log_path)
    names = sorted(os.listdir(folder))
    selected = random.Random(shuffle_seed).sample(names, len(names))[chunk * chunk_size:(chunk + 1) * chunk_size]
    if not selected:
        raise ValueError(f"Chunk {chunk} is outside the input records")
    records = []
    for name in selected:
        source = folder / name / "game.dcl2"
        records.append((name, file_hash(source) if source.is_file() else None))
    return dict(names_sha256=object_hash(names), chunk_sha256=object_hash(records))


def output_paths(job):
    folder = ROOT / "data" / f"end{job['end']}" / f"shot{job['shot']}"
    return sorted(folder.glob(f"sl_data_chunk{job['chunk']}_*.npz"))


def generate(job):
    import numpy as np
    import torch
    from search_config import require_search_mode
    from transformer.network import TransformerNetwork
    from transformer.shot_generator import generate_data

    settings = job["generation"]
    require_search_mode(settings["search_mode"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for data generation")
    torch.set_num_threads(job["threads"])
    if source_signature() != job["source_sha256"]:
        raise ValueError("Generation source differs from coordinator")
    if input_signature(job["log_path"], job["chunk"], settings["chunk_size"], settings["shuffle_seed"]) != job["input"]:
        raise ValueError("Input records differ from coordinator")
    if output_paths(job):
        raise FileExistsError("This chunk already has output files; refusing to mix runs")
    base = local_path(settings["base_model"])
    if file_hash(base) != job["base_sha256"]:
        raise ValueError("Base model differs from coordinator")
    # Existing loaders can hide model-loading errors. Validate before generation.
    if settings["search_mode"] == "shot":
        from nn.network.dual_net import DualNet
        network = DualNet(torch.device("cpu"))
    else:
        network = TransformerNetwork()
    network.load_state_dict(torch.load(base, map_location="cpu", weights_only=True))
    del network
    teacher = None
    if job["shot"] < 15:
        expected = f"model/{settings['search_mode']}-end{job['end']}-shot{job['shot'] + 1}.bin"
        if job["teacher"]["path"] != expected:
            raise ValueError("Wrong teacher model path")
        teacher = local_path(expected)
        if file_hash(teacher) != job["teacher"]["sha256"]:
            raise ValueError("Teacher model differs from coordinator")
        network = TransformerNetwork()
        network.load_state_dict(torch.load(teacher, map_location="cpu", weights_only=True))
        del network
    options = {key: value for key, value in settings.items()
               if key not in ("start_end", "start_shot", "stop_end", "stop_shot", "chunk_start", "chunk_end", "base_model")}
    options.update(log_path=str(local_path(job["log_path"])), save_path=ROOT / "data",
                   target_end=job["end"], target_shot=[job["shot"]], model=base,
                   use_transformer=teacher is not None, transformer_model=teacher,
                   transformer_target_end=[job["end"]],
                   transformer_target_shot=[job["shot"] + 1] if teacher else [],
                   chunk_start=job["chunk"], chunk_end=job["chunk"], num_workers=1, use_gpu=True)
    generate_data(**options)
    files = []
    for path in output_paths(job):
        with np.load(path, allow_pickle=False) as data:
            samples = len(data["value"])
            if samples < 1:
                raise ValueError(f"Empty generated file: {path}")
        files.append(dict(path=path.relative_to(ROOT).as_posix(), size=path.stat().st_size,
                          sha256=file_hash(path), samples=samples))
    return dict(success=True, files=files, samples=sum(item["samples"] for item in files))


def execute(job):
    for key in ("pc_id", "worker_id", "job_id"):
        identifier(job[key])
    if (type(job["end"]) is not int or not 0 <= job["end"] <= 9
            or type(job["shot"]) is not int or not 0 <= job["shot"] <= 15
            or type(job["chunk"]) is not int or job["chunk"] < 0):
        raise ValueError("Invalid end, shot or chunk")
    with transaction(job["pc_id"]) as state:
        row = worker(state, job["worker_id"])
        if row.get("job", {}).get("job_id") == job["job_id"]:
            if row["job"] != job:
                raise ValueError("Same job ID has different conditions")
            return  # A repeated SSH launch never regenerates a running/completed job.
        if row.get("last_job_id") == job["job_id"]:
            return  # Delayed launch after collection has already been acknowledged.
        if row["state"] != "idle":
            raise RuntimeError("Worker has an uncollected or running job")
        token = process_token(os.getpid())
        if token is None:
            raise RuntimeError("Cannot identify generator process")
        row.update(state="running", job=job, pid=os.getpid(), process_token=token,
                   started_at=now(), result=None)
    try:
        result = generate(job)
    except BaseException as exc:
        result = dict(success=False, files=[], samples=0, error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    with transaction(job["pc_id"]) as state:
        row = worker(state, job["worker_id"])
        if row.get("job") != job or row.get("pid") != os.getpid():
            raise RuntimeError("Worker ownership changed during generation")
        row.update(state="completed", finished_at=now(), result=result)
    if not result["success"]:
        raise RuntimeError(result["error"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-b64", required=True)
    args = parser.parse_args()
    execute(json.loads(base64.b64decode(args.job_b64, validate=True)))


if __name__ == "__main__":
    main()
