from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from learning_param import BATCH_SIZE


def _matchup_key(team0_name: str, team1_name: str) -> tuple[str, str]:
    return tuple(sorted((team0_name, team1_name)))


def _read_team_names(dcl2_data: Iterable[str]) -> tuple[str, str]:
    team_names: dict[str, str] = {}

    for line in dcl2_data:
        try:
            log = json.loads(line)["log"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

        if log.get("cmd") != "dc_ok":
            continue

        team = log.get("team")
        name = log.get("name")
        if team in ("team0", "team1") and isinstance(name, str):
            team_names[team] = name
        if len(team_names) == 2:
            break

    if "team0" not in team_names or "team1" not in team_names:
        raise ValueError("team0/team1 names were not found")

    return team_names["team0"], team_names["team1"]


def _has_final_state(dcl2_data: list[str]) -> bool:
    if len(dcl2_data) < 2:
        return False

    try:
        return bool(json.loads(dcl2_data[-2])["log"]["state"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return False


def _print_matchups(title: str, matchups: Counter[tuple[str, str]]) -> None:
    print(title)
    if not matchups:
        print("  (none)")
        return

    for (team_a, team_b), count in sorted(
        matchups.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"  {team_a} vs {team_b}: {count}")


def _aggregate_chunk(
    log_path: Path,
    target_log_files: list[str],
    data_size: int,
) -> dict[str, object]:
    selected_matchups: Counter[tuple[str, str]] = Counter()
    processed_matchups: Counter[tuple[str, str]] = Counter()
    selected_matchup_count = 0
    processed_log_count = 0
    non_directory_count = 0
    missing_dcl2_count = 0
    unreadable_matchup_count = 0
    invalid_final_state_count = 0

    for one_log in target_log_files:
        log_dir = log_path / one_log
        if not log_dir.is_dir():
            non_directory_count += 1
            continue

        dcl2_path = log_dir / "game.dcl2"
        if not dcl2_path.exists():
            missing_dcl2_count += 1
            continue

        with dcl2_path.open(encoding="utf-8") as dclfile:
            dcl2_data = dclfile.readlines()

        try:
            team0_name, team1_name = _read_team_names(dcl2_data)
        except ValueError:
            unreadable_matchup_count += 1
            matchup = None
        else:
            matchup = _matchup_key(team0_name, team1_name)
            selected_matchups[matchup] += 1
            selected_matchup_count += 1

        if processed_log_count >= data_size:
            continue
        if not _has_final_state(dcl2_data):
            invalid_final_state_count += 1
            continue

        processed_log_count += 1
        if matchup is not None:
            processed_matchups[matchup] += 1

    return {
        "selected_matchups": selected_matchups,
        "processed_matchups": processed_matchups,
        "selected_entry_count": len(target_log_files),
        "selected_matchup_count": selected_matchup_count,
        "processed_log_count": processed_log_count,
        "non_directory_count": non_directory_count,
        "missing_dcl2_count": missing_dcl2_count,
        "unreadable_matchup_count": unreadable_matchup_count,
        "invalid_final_state_count": invalid_final_state_count,
    }


def _print_chunk_summary(chunk_index: int, summary: dict[str, object]) -> None:
    print("")
    print(f"chunk {chunk_index}")
    print(f"Selected entries: {summary['selected_entry_count']}")
    print(f"Selected logs with readable matchup: {summary['selected_matchup_count']}")
    print(f"Processing logs: {summary['processed_log_count']}")
    print(f"Non-directory entries: {summary['non_directory_count']}")
    print(f"Missing game.dcl2: {summary['missing_dcl2_count']}")
    print(f"Unreadable matchups: {summary['unreadable_matchup_count']}")
    print(f"Invalid final states: {summary['invalid_final_state_count']}")
    print("")
    _print_matchups("Selected matchups:", summary["selected_matchups"])
    print("")
    _print_matchups("Processed matchups:", summary["processed_matchups"])


def aggregate_shot_generator_matchups(
    log_path: str | Path,
    data_size: int,
    shuffle_seed: int,
    chunk_start: int = 0,
    chunk_end: int | None = None,
    chunk_size: int | None = None,
) -> None:
    log_path = Path(log_path)
    if chunk_end is None:
        chunk_end = chunk_start
    if chunk_start < 0:
        raise ValueError(f"chunk_start must be non-negative, got {chunk_start}")
    if chunk_end < chunk_start:
        raise ValueError(
            f"chunk_end must be greater than or equal to chunk_start, got {chunk_end}"
        )
    if data_size <= 0:
        raise ValueError(f"data_size must be positive, got {data_size}")
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not log_path.is_dir():
        raise FileNotFoundError(f"{log_path} does not exist or is not a directory")

    log_files = os.listdir(log_path)
    rng = random.Random(shuffle_seed)
    shuffled_log_files = rng.sample(log_files, len(log_files))

    total_selected_matchups: Counter[tuple[str, str]] = Counter()
    total_processed_matchups: Counter[tuple[str, str]] = Counter()

    for chunk_index in range(chunk_start, chunk_end + 1):
        if chunk_size is None:
            target_log_files = shuffled_log_files
        else:
            chunk_start_pos = chunk_index * chunk_size
            chunk_end_pos = min(
                chunk_start_pos + chunk_size,
                len(shuffled_log_files),
            )
            target_log_files = shuffled_log_files[chunk_start_pos:chunk_end_pos]

        summary = _aggregate_chunk(log_path, target_log_files, data_size)
        _print_chunk_summary(chunk_index, summary)
        total_selected_matchups.update(summary["selected_matchups"])
        total_processed_matchups.update(summary["processed_matchups"])

    if chunk_start != chunk_end:
        print("")
        print(f"all chunks ({chunk_start}-{chunk_end})")
        print("")
        _print_matchups("Selected matchups:", total_selected_matchups)
        print("")
        _print_matchups("Processed matchups:", total_processed_matchups)


if __name__ == "__main__":
    aggregate_shot_generator_matchups(
        log_path=ROOT_DIR / "LearnLog" / "all",
        data_size=70000,
        shuffle_seed=12345,
        chunk_start=0,
        chunk_end=16,
        chunk_size=BATCH_SIZE,
    )
