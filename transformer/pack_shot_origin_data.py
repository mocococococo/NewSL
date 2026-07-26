from __future__ import annotations

import json
import re
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List

import click
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from learning_param import BATCH_SIZE, DATA_SET_SIZE
from transformer.params import (
    DEFAULT_TRANSFORMER_CONFIG,
    GAME_FEAT_DIM,
    MAX_STONES,
    STONE_FEAT_DIM,
)

DATA_KEYS = ("stones", "games", "stone_masks", "policy", "value", "win_value")
SPLIT_NAMES = ("train", "valid", "test")
N_ACTIONS = DEFAULT_TRANSFORMER_CONFIG.action_dim
N_VALUE_CLASSES = DEFAULT_TRANSFORMER_CONFIG.value_dim


class NpzBuffer:
    def __init__(self) -> None:
        self.batches: Deque[Dict[str, np.ndarray]] = deque()
        self.size = 0

    def append(self, batch: Dict[str, np.ndarray]) -> None:
        sample_count = _sample_count(batch)
        if sample_count <= 0:
            return
        self.batches.append(batch)
        self.size += sample_count

    def pop(self, sample_count: int) -> Dict[str, np.ndarray]:
        if sample_count > self.size:
            raise ValueError(f"cannot pop {sample_count} samples from buffer size {self.size}")

        remaining = sample_count
        chunks = {key: [] for key in DATA_KEYS}
        while remaining > 0:
            head = self.batches[0]
            head_size = _sample_count(head)
            take_size = min(remaining, head_size)
            for key in DATA_KEYS:
                chunks[key].append(head[key][:take_size])

            if take_size == head_size:
                self.batches.popleft()
            else:
                for key in DATA_KEYS:
                    head[key] = head[key][take_size:]

            self.size -= take_size
            remaining -= take_size

        return {key: np.concatenate(chunks[key], axis=0) for key in DATA_KEYS}


def _sample_count(batch: Dict[str, np.ndarray]) -> int:
    return int(batch["win_value"].shape[0])


def _load_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        missing_keys = [key for key in DATA_KEYS if key not in data]
        if missing_keys:
            raise ValueError(f"{npz_path} is missing keys: {missing_keys}")
        batch = {key: np.asarray(data[key]) for key in DATA_KEYS}
    _validate_batch(batch, npz_path)
    return batch


def _validate_batch(batch: Dict[str, np.ndarray], source_path: Path) -> None:
    sample_count = _sample_count(batch)
    expected_shapes = {
        "stones": (sample_count, MAX_STONES, STONE_FEAT_DIM),
        "games": (sample_count, GAME_FEAT_DIM),
        "stone_masks": (sample_count, MAX_STONES),
        "policy": (sample_count, N_ACTIONS),
        "value": (sample_count, N_VALUE_CLASSES),
        "win_value": (sample_count,),
    }
    for key, expected_shape in expected_shapes.items():
        if batch[key].shape != expected_shape:
            raise ValueError(
                f"{source_path}: {key} must have shape {expected_shape}, got {batch[key].shape}"
            )
    for key in ("stones", "games", "policy", "value", "win_value"):
        if not np.all(np.isfinite(batch[key])):
            raise ValueError(f"{source_path}: {key} contains non-finite values")


def _save_npz(save_file_path: Path, batch: Dict[str, np.ndarray], overwrite: bool) -> None:
    if save_file_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {save_file_path}")
    save_file_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "stones": np.asarray(batch["stones"], dtype=np.float32),
        "games": np.asarray(batch["games"], dtype=np.float32),
        "stone_masks": np.asarray(batch["stone_masks"], dtype=np.bool_),
        "policy": np.asarray(batch["policy"], dtype=np.float32),
        "value": np.asarray(batch["value"], dtype=np.float32),
        "win_value": np.asarray(batch["win_value"], dtype=np.float32),
        "log_count": np.array(_sample_count(batch)),
    }
    print(f"Saving packed data to {save_file_path}")
    np.savez_compressed(save_file_path, **save_data)


def _raw_file_sort_key(raw_file_path: Path) -> tuple[int, int, str]:
    match = re.fullmatch(
        r"sl_data_origin_raw_chunk(\d+)_(?:train|valid|test)_(\d+)\.npz",
        raw_file_path.name,
    )
    if match is None:
        return (10**18, 10**18, raw_file_path.name)
    return (int(match.group(1)), int(match.group(2)), raw_file_path.name)


def _iter_raw_files(raw_split_path: Path) -> Iterable[Path]:
    if not raw_split_path.exists():
        return []
    return sorted(raw_split_path.glob("*.npz"), key=_raw_file_sort_key)


def _pack_split(
    split_name: str,
    raw_path: Path,
    save_path: Path,
    samples_per_file: int,
    batch_size: int,
    save_partial_batches: bool,
    overwrite: bool,
) -> dict:
    raw_split_path = raw_path / split_name
    output_split_path = save_path / split_name
    buffer = NpzBuffer()
    output_counter = 0
    raw_file_count = 0
    loaded_samples = 0
    saved_samples = 0

    for raw_file_path in _iter_raw_files(raw_split_path):
        raw_file_count += 1
        batch = _load_npz(raw_file_path)
        loaded_samples += _sample_count(batch)
        buffer.append(batch)
        while buffer.size >= samples_per_file:
            output_batch = buffer.pop(samples_per_file)
            _save_npz(output_split_path / f"sl_data_origin_{output_counter}.npz", output_batch, overwrite)
            saved_samples += _sample_count(output_batch)
            output_counter += 1

    if save_partial_batches:
        partial_count = (buffer.size // batch_size) * batch_size
        if partial_count > 0:
            output_batch = buffer.pop(partial_count)
            _save_npz(output_split_path / f"sl_data_origin_{output_counter}.npz", output_batch, overwrite)
            saved_samples += _sample_count(output_batch)
            output_counter += 1

    dropped_samples = buffer.size
    if dropped_samples > 0:
        print(f"{split_name}: dropped {dropped_samples} samples (< batch_size or partial disabled)")

    return {
        "split": split_name,
        "raw_files": raw_file_count,
        "loaded_samples": loaded_samples,
        "saved_files": output_counter,
        "saved_samples": saved_samples,
        "dropped_samples": dropped_samples,
    }


def pack_shot_origin_data(
    raw_path: str | Path = ROOT_DIR / "data" / "shot_origin" / "raw",
    save_path: str | Path = ROOT_DIR / "data" / "shot_origin",
    samples_per_file: int = DATA_SET_SIZE,
    batch_size: int = BATCH_SIZE,
    save_partial_batches: bool = True,
    overwrite: bool = False,
) -> None:
    raw_path = Path(raw_path)
    save_path = Path(save_path)
    if samples_per_file <= 0:
        raise ValueError(f"samples_per_file must be positive, got {samples_per_file}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if samples_per_file % batch_size != 0:
        raise ValueError(
            f"samples_per_file must be a multiple of batch_size, got {samples_per_file} and {batch_size}"
        )

    manifest_path = save_path / "pack_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"manifest already exists: {manifest_path}")

    manifest = {
        "raw_path": str(raw_path),
        "save_path": str(save_path),
        "samples_per_file": samples_per_file,
        "batch_size": batch_size,
        "save_partial_batches": save_partial_batches,
        "splits": [],
    }
    for split_name in SPLIT_NAMES:
        manifest["splits"].append(
            _pack_split(
                split_name=split_name,
                raw_path=raw_path,
                save_path=save_path,
                samples_per_file=samples_per_file,
                batch_size=batch_size,
                save_partial_batches=save_partial_batches,
                overwrite=overwrite,
            )
        )

    save_path.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=4)
    print(f"Saving pack manifest to {manifest_path}")


@click.command()
@click.option("--overwrite/--no-overwrite", default=False, help="overwrite generated packed files")
@click.option(
    "--save-partial-batches/--drop-partial-batches",
    default=True,
    help="save the final split remainder if it contains at least one full batch",
)
def main(overwrite: bool, save_partial_batches: bool) -> None:
    pack_shot_origin_data(
        raw_path=ROOT_DIR / "data" / "shot_origin" / "raw",
        save_path=ROOT_DIR / "data" / "shot_origin",
        samples_per_file=DATA_SET_SIZE,
        batch_size=BATCH_SIZE,
        save_partial_batches=save_partial_batches,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    main()
