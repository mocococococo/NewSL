from __future__ import annotations

import gc
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


NEWSL_DIR = Path(__file__).resolve().parents[2]
if str(NEWSL_DIR) not in sys.path:
    sys.path.insert(0, str(NEWSL_DIR))

from common.translate_state import (
    convert_team_stoi,
    scores_to_scorediff_for_team0,
    stones_listdict_to_xy16,
)
from experiment.src.mini_match_report import (
    load_position_records,
    print_report_from_records,
    save_position_json,
)
from mcts.params import (
    DEFAULT_MAX_SIMULATIONS,
    DEFAULT_TIME_LIMIT_SEC,
    STDDV_ANGLE,
    STDDV_SPEED,
)
from mcts.rollout import _end_score_diff_team0_minus_team1
from mcts.search import mcts_search, set_root_state as set_mcts_root_state
from mcts.simulate import ShotNoise, simulator_step_continuous
from mcts.state import State
from shot.params import (
    DEFAULT_SHOT_MAX_SIMULATIONS,
    DEFAULT_SHOT_MAX_SIMULATIONS_15,
    DEFAULT_SHOT_TIME_LIMIT_SEC,
)
from shot.search import shot_search, set_root_state as set_shot_root_state
from transformer.network import TransformerNetwork
from transformer.utility import load_transformer_network


RESULT_WIN_IDX = 0
RESULT_DRAW_IDX = 1
RESULT_LOSE_IDX = 2


def _resolve_model(model: str | Path) -> Path:
    model_path = Path(model)
    if model_path.is_absolute():
        return model_path
    return NEWSL_DIR / "model" / model_path


def _state_to_raw_stones(state: State) -> list[Optional[dict[str, Any]]]:
    stones: list[Optional[dict[str, Any]]] = []
    for stone in state.stones:
        if stone is None:
            stones.append(None)
        else:
            x, y = stone
            stones.append({"position": {"x": float(x), "y": float(y)}})
    return stones


def _score_diff_for_team_view(score_diff_for_team0: int, team: int) -> int:
    return score_diff_for_team0 if team == 0 else -score_diff_for_team0


def _score_diff_bucket(score_diff: int) -> str:
    if score_diff <= -2:
        return "<=-2"
    if score_diff == -1:
        return "-1"
    if score_diff == 0:
        return "0"
    if score_diff == 1:
        return "+1"
    return ">=+2"


def _release_position_memory(use_gpu: bool) -> None:
    gc.collect()
    if use_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _make_execution_noise(
    seed: int,
    position_index: int,
    x_repeats: int,
    shot_count: int,
) -> list[list[ShotNoise]]:
    noise_by_repeat: list[list[ShotNoise]] = []
    for repeat_index in range(x_repeats):
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, position_index, repeat_index])
        )
        repeat_noise = [
            (
                float(rng.normal(0.0, STDDV_SPEED)),
                float(rng.normal(0.0, STDDV_ANGLE)),
            )
            for _ in range(shot_count)
        ]
        noise_by_repeat.append(repeat_noise)
    return noise_by_repeat


def _normalize_search_method(search_method: str) -> str:
    normalized = search_method.strip().lower()
    if normalized not in ("shot", "puct"):
        raise ValueError(f"Unknown search method: {search_method}")
    return normalized


def _label_to_key(label: str, fallback: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
    return key or fallback


def _resolve_model_group(
    models_by_shot: dict[int, str | Path],
    needed_shots: tuple[int, ...],
    group_name: str,
) -> dict[int, Path]:
    missing_shots = [shot for shot in needed_shots if shot not in models_by_shot]
    if missing_shots:
        raise ValueError(f"Missing transformer models for group {group_name}: {missing_shots}")

    return {
        shot: _resolve_model(models_by_shot[shot])
        for shot in needed_shots
    }


def _load_network_group(
    model_paths_by_shot: dict[int, Path],
    use_gpu: bool,
    network_cache: dict[Path, TransformerNetwork],
) -> dict[int, TransformerNetwork]:
    networks_by_shot: dict[int, TransformerNetwork] = {}
    for shot, model_path in model_paths_by_shot.items():
        resolved_path = model_path.resolve()
        if resolved_path not in network_cache:
            network_cache[resolved_path] = load_transformer_network(
                model_path,
                use_gpu=use_gpu,
            )
        networks_by_shot[shot] = network_cache[resolved_path]
    return networks_by_shot


@dataclass(frozen=True)
class SearchAction:
    vx: float
    vy: float
    spin: int
    meta: dict[str, Any]


class TransformerModelPlayer:
    def __init__(
        self,
        key: str,
        label: str,
        search_method: str,
        networks_by_shot: dict[int, TransformerNetwork],
        model_paths_by_shot: dict[int, Path],
        target_end: int,
    ) -> None:
        self.key = key
        self.label = label
        self.search_method = _normalize_search_method(search_method)
        self.networks_by_shot = networks_by_shot
        self.model_paths_by_shot = model_paths_by_shot
        self.target_end = int(target_end)
        self.target_shots = tuple(sorted(networks_by_shot))

    def select_action(self, state: State) -> SearchAction:
        if state.shot_index not in self.networks_by_shot:
            raise ValueError(
                f"Transformer model for shot {state.shot_index} "
                f"is not configured for {self.label}."
            )

        root_kwargs = {
            "stones": _state_to_raw_stones(state),
            "score_diff": state.score_diff,
            "end": state.end,
            "shot_index": state.shot_index,
            "hammer_team": state.hammer_team,
            "transformer_target_end": (self.target_end,),
            "transformer_target_shot": self.target_shots,
        }

        if self.search_method == "shot":
            root = set_shot_root_state(
                network=None,
                transformer_network=self.networks_by_shot,
                use_transformer=True,
                **root_kwargs,
            )
            vx, vy, spin = shot_search(root_state=root, use_value=True)
        else:
            root = set_mcts_root_state(
                sl_model=None,
                search_based_model=self.networks_by_shot,
                use_search_based_model=True,
                **root_kwargs,
            )
            vx, vy, spin = mcts_search(root_state=root, use_value=True)

        return SearchAction(
            vx=float(vx),
            vy=float(vy),
            spin=1 if int(spin) == 1 else 0,
            meta={
                "search_method": self.search_method,
                "model_group": self.key,
                "model": str(self.model_paths_by_shot[state.shot_index]),
            },
        )


def _print_match_header(
    first_player: TransformerModelPlayer,
    second_player: TransformerModelPlayer,
    search_method: str,
    repeat_index: int,
    x_repeats: int,
) -> None:
    print("")
    print(
        f"{first_player.label} vs {second_player.label} "
        f"({repeat_index + 1}/{x_repeats}, {search_method.upper()})"
    )


def _run_suffix_game(
    root_state: State,
    model_a_player: TransformerModelPlayer,
    model_b_player: TransformerModelPlayer,
    start_with_a: bool,
    repeat_index: int,
    x_repeats: int,
    execution_noise: list[ShotNoise],
) -> tuple[State, list[dict[str, Any]]]:
    state = root_state
    action_log: list[dict[str, Any]] = []
    start_shot = root_state.shot_index
    first_player = model_a_player if start_with_a else model_b_player
    second_player = model_b_player if start_with_a else model_a_player
    _print_match_header(
        first_player,
        second_player,
        model_a_player.search_method,
        repeat_index,
        x_repeats,
    )

    while not state.is_end_terminal():
        offset = state.shot_index - start_shot
        use_a = (offset % 2 == 0) if start_with_a else (offset % 2 == 1)
        player = model_a_player if use_a else model_b_player

        print(
            f"end = {state.end}, shot = {state.shot_index}, "
            f"{player.label} search"
        )
        action = player.select_action(state)
        print(
            f"{player.label} selected "
            f"vx={action.vx:.6f}, vy={action.vy:.6f}, spin={action.spin}"
        )

        noise = execution_noise[offset]
        action_log.append(
            {
                "shot": int(state.shot_index),
                "player_key": player.key,
                "player_label": player.label,
                "vx": float(action.vx),
                "vy": float(action.vy),
                "spin": int(action.spin),
                "execution_noise": {
                    "speed": float(noise[0]),
                    "angle": float(noise[1]),
                },
                "meta": action.meta,
            }
        )
        state = simulator_step_continuous(
            state,
            action.vx,
            action.vy,
            action.spin,
            noise=noise,
        )

    return state, action_log


def _evaluate_start_pattern(
    root_state: State,
    model_a_player: TransformerModelPlayer,
    model_b_player: TransformerModelPlayer,
    start_with_a: bool,
    x_repeats: int,
    root_view_team: int,
    root_view_score_diff_before_shot: int,
    execution_noise_by_repeat: list[list[ShotNoise]],
) -> tuple[np.ndarray, float, float, float, float, list[list[dict[str, Any]]]]:
    counts = np.zeros(3, dtype=np.int32)
    action_logs: list[list[dict[str, Any]]] = []

    for repeat_index in range(x_repeats):
        final_state, action_log = _run_suffix_game(
            root_state,
            model_a_player,
            model_b_player,
            start_with_a=start_with_a,
            repeat_index=repeat_index,
            x_repeats=x_repeats,
            execution_noise=execution_noise_by_repeat[repeat_index],
        )
        action_logs.append(action_log)

        raw_score = _end_score_diff_team0_minus_team1(final_state.stones)
        score_from_root_view = _score_diff_for_team_view(raw_score, root_view_team)
        total_score_from_root_view = (
            root_view_score_diff_before_shot + score_from_root_view
        )

        if total_score_from_root_view > 0:
            counts[RESULT_WIN_IDX] += 1
        elif total_score_from_root_view < 0:
            counts[RESULT_LOSE_IDX] += 1
        else:
            counts[RESULT_DRAW_IDX] += 1

    win_rate = float(counts[RESULT_WIN_IDX] / x_repeats)
    draw_rate = float(counts[RESULT_DRAW_IDX] / x_repeats)
    lose_rate = float(counts[RESULT_LOSE_IDX] / x_repeats)
    result_mean = win_rate - lose_rate
    return counts, result_mean, win_rate, draw_rate, lose_rate, action_logs


def _build_output_stem(
    model_a_key: str,
    model_b_key: str,
    search_method: str,
    target_end: int,
    target_shot: int,
    total_data_size: int,
    x_repeats: int,
) -> str:
    return (
        f"model_vs_model_{model_a_key}_vs_{model_b_key}_{search_method}_transformer"
        f"_end{target_end}_shot{target_shot}"
        f"_datasize{total_data_size}_x{x_repeats}"
    )


def _validate_existing_experiment(
    json_dir: Path,
    experiment_metadata: dict[str, Any],
) -> None:
    existing_json_paths = sorted(json_dir.glob("*.json"))
    if not existing_json_paths:
        return

    with existing_json_paths[0].open(encoding="utf-8") as file:
        existing_record = json.load(file)
    existing_metadata = existing_record.get("experiment")
    if not isinstance(existing_metadata, dict):
        raise ValueError(
            f"Existing JSON has no experiment metadata: {existing_json_paths[0]}"
        )

    range_keys = {
        "position_start",
        "position_count",
        "position_end_exclusive",
    }
    existing_common = {
        key: value
        for key, value in existing_metadata.items()
        if key not in range_keys
    }
    current_common = {
        key: value
        for key, value in experiment_metadata.items()
        if key not in range_keys
    }
    normalized_existing = json.loads(json.dumps(existing_common, sort_keys=True))
    normalized_current = json.loads(json.dumps(current_common, sort_keys=True))
    if normalized_existing != normalized_current:
        raise ValueError(
            "Existing JSON files were created with different experiment settings: "
            f"{existing_json_paths[0]}"
        )

def _search_budget_metadata(search_method: str) -> dict[str, Any]:
    if search_method == "shot":
        return {
            "time_limit_sec": float(DEFAULT_SHOT_TIME_LIMIT_SEC),
            "max_simulations": int(DEFAULT_SHOT_MAX_SIMULATIONS),
            "final_shot_max_simulations": int(DEFAULT_SHOT_MAX_SIMULATIONS_15),
        }
    return {
        "time_limit_sec": float(DEFAULT_TIME_LIMIT_SEC),
        "max_simulations": int(DEFAULT_MAX_SIMULATIONS),
        "final_shot_max_simulations": None,
    }


def main(
    log_path: str | Path = NEWSL_DIR / "LearnLog" / "all",
    save_path: str | Path = Path(__file__).resolve().parents[1] / "data",
    search_method: str = "shot",
    model_a_label: str = "Model A",
    model_b_label: str = "Model B",
    target_end: int = 9,
    target_shot: int = 9,
    total_data_size: int = 1000,
    position_start: int = 0,
    position_count: int = 1000,
    X: int = 1,
    use_gpu: bool = True,
    transformer_models_by_shot_A: Optional[dict[int, str | Path]] = None,
    transformer_models_by_shot_B: Optional[dict[int, str | Path]] = None,
    shuffle_seed: Optional[int] = 12345,
    execution_noise_seed: int = 54321,
) -> None:
    search_method = _normalize_search_method(search_method)
    if not (0 <= target_shot <= 15):
        raise ValueError(f"target_shot must be in [0, 15], got {target_shot}")
    if total_data_size <= 0:
        raise ValueError(
            f"total_data_size must be positive, got {total_data_size}"
        )
    if position_start < 0:
        raise ValueError(f"position_start must be non-negative, got {position_start}")
    if position_count <= 0:
        raise ValueError(f"position_count must be positive, got {position_count}")
    position_end = position_start + position_count
    if position_end > total_data_size:
        raise ValueError(
            "position_start + position_count must not exceed "
            f"total_data_size: {position_start} + {position_count} > "
            f"{total_data_size}"
        )
    if X <= 0:
        raise ValueError(f"X must be positive, got {X}")

    default_models = {
        9: "transformer-sl-9-9-model-06-14-adamw-epoch50-shot.bin",
        10: "transformer-sl-9-10-model-06-11-adamw-epoch50-shot.bin",
        11: "transformer-sl-9-11-model-06-09-adamw-epoch50-shot.bin",
        12: "transformer-sl-9-12-model-06-08-adamw-epoch50-shot.bin",
        13: "transformer-sl-9-13-model-06-06-adamw-epoch50-shot.bin",
        14: "transformer-sl-9-14-model-06-04-adamw-epoch50-shot.bin",
        15: "transformer-sl-9-15-model-06-02-adamw-epoch50-shot.bin",
    }
    if transformer_models_by_shot_A is None:
        transformer_models_by_shot_A = default_models
    if transformer_models_by_shot_B is None:
        transformer_models_by_shot_B = default_models

    needed_shots = tuple(range(target_shot, 16))
    model_paths_by_shot_A = _resolve_model_group(
        transformer_models_by_shot_A,
        needed_shots,
        "A",
    )
    model_paths_by_shot_B = _resolve_model_group(
        transformer_models_by_shot_B,
        needed_shots,
        "B",
    )

    model_a_key = _label_to_key(model_a_label, "model_a")
    model_b_key = _label_to_key(model_b_label, "model_b")
    if model_a_key == model_b_key:
        model_a_key = f"{model_a_key}_a"
        model_b_key = f"{model_b_key}_b"

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_stem = _build_output_stem(
        model_a_key,
        model_b_key,
        search_method,
        target_end,
        target_shot,
        total_data_size,
        X,
    )
    json_dir = save_dir / output_stem
    json_dir.mkdir(parents=True, exist_ok=True)
    position_index_width = max(6, len(str(total_data_size - 1)))
    intended_json_paths = [
        json_dir / f"{position_index:0{position_index_width}d}.json"
        for position_index in range(position_start, position_end)
    ]
    existing_json_paths = [path for path in intended_json_paths if path.exists()]
    if existing_json_paths:
        preview = ", ".join(path.name for path in existing_json_paths[:5])
        if len(existing_json_paths) > 5:
            preview += ", ..."
        raise FileExistsError(
            f"Refusing to overwrite {len(existing_json_paths)} existing JSON files "
            f"in {json_dir}: {preview}"
        )

    experiment_metadata = {
        "experiment_type": "model_vs_model",
        "search_method": search_method,
        "target_end": int(target_end),
        "target_shot": int(target_shot),
        "requested_data_size": int(total_data_size),
        "total_data_size": int(total_data_size),
        "position_start": int(position_start),
        "position_count": int(position_count),
        "position_end_exclusive": int(position_end),
        "execution_repeats_x": int(X),
        "player_a_start_method_key": f"{model_a_key}_start",
        "player_a_start_method_label": f"{model_a_label}-start",
        "player_b_start_method_key": f"{model_b_key}_start",
        "player_b_start_method_label": f"{model_b_label}-start",
        "player_a_kind": "Transformer-Model-A",
        "player_b_kind": "Transformer-Model-B",
        "player_a_label": model_a_label,
        "player_b_label": model_b_label,
        "player_a_search_method": search_method,
        "player_b_search_method": search_method,
        "player_a_use_value": True,
        "player_b_use_value": True,
        "search_budget": _search_budget_metadata(search_method),
        "transformer_models_by_shot_A": {
            int(shot): str(path) for shot, path in model_paths_by_shot_A.items()
        },
        "transformer_models_by_shot_B": {
            int(shot): str(path) for shot, path in model_paths_by_shot_B.items()
        },
        "shuffle_seed": shuffle_seed,
        "execution_noise_seed": int(execution_noise_seed),
        "shared_execution_noise": True,
        "shared_search_randomness": False,
    }

    _validate_existing_experiment(json_dir, experiment_metadata)

    network_cache: dict[Path, TransformerNetwork] = {}
    networks_by_shot_A = _load_network_group(
        model_paths_by_shot_A,
        use_gpu,
        network_cache,
    )
    networks_by_shot_B = _load_network_group(
        model_paths_by_shot_B,
        use_gpu,
        network_cache,
    )
    model_a_player = TransformerModelPlayer(
        model_a_key,
        model_a_label,
        search_method,
        networks_by_shot_A,
        model_paths_by_shot_A,
        target_end,
    )
    model_b_player = TransformerModelPlayer(
        model_b_key,
        model_b_label,
        search_method,
        networks_by_shot_B,
        model_paths_by_shot_B,
        target_end,
    )

    log_files = os.listdir(log_path)
    rng = random.Random(shuffle_seed)
    shuffled_log_files = rng.sample(log_files, len(log_files))

    position_records: list[dict[str, Any]] = []
    matched_position_count = 0
    processed_position_count = 0
    range_complete = False

    for one_log in shuffled_log_files:
        if range_complete or processed_position_count >= position_count:
            break
        log_dir = Path(log_path) / one_log
        if not log_dir.is_dir():
            continue

        dcl2_path = log_dir / "game.dcl2"
        if not dcl2_path.exists():
            continue
        with dcl2_path.open(encoding="utf-8") as dclfile:
            dcl2_data = dclfile.readlines()

        for line_index in range(9, len(dcl2_data) - 2, 2):
            if range_complete or processed_position_count >= position_count:
                break
            try:
                dcl2_log = json.loads(dcl2_data[line_index])["log"]
                dcl2_state = dcl2_log["state"]
                stones = (
                    dcl2_state["stones"]["team0"]
                    + dcl2_state["stones"]["team1"]
                )
                scores = dcl2_state["scores"]
                end = int(dcl2_state["end"])
                shot = int(dcl2_state["shot"])
                hammer = convert_team_stoi(dcl2_state["hammer"])
                if not ((end == target_end) and (shot == target_shot)):
                    continue
            except (KeyError, TypeError, ValueError):
                continue

            global_position_index = matched_position_count
            matched_position_count += 1
            if global_position_index < position_start:
                continue
            if global_position_index >= position_end:
                range_complete = True
                break

            score_diff_for_team0 = scores_to_scorediff_for_team0(scores)
            root_state = State.initial(
                stones=stones_listdict_to_xy16(stones),
                end=end,
                hammer_team=hammer,
                shot_index=shot,
                score_diff=score_diff_for_team0,
            )
            root_view_team = int(root_state.to_move())
            root_view_score_diff_before_shot = _score_diff_for_team_view(
                score_diff_for_team0,
                root_view_team,
            )
            root_view_score_diff_bucket = _score_diff_bucket(
                root_view_score_diff_before_shot
            )
            execution_noise_by_repeat = _make_execution_noise(
                execution_noise_seed,
                global_position_index,
                X,
                16 - target_shot,
            )

            print(
                f"Processing position {global_position_index + 1} "
                f"({processed_position_count + 1}/{position_count} in this run): "
                f"log={one_log}, end={end}, shot={shot}, "
                f"search={search_method.upper()}"
            )
            (
                a_start_counts,
                a_start_result_mean,
                a_start_win_rate,
                a_start_draw_rate,
                a_start_lose_rate,
                a_start_action_logs,
            ) = _evaluate_start_pattern(
                root_state,
                model_a_player,
                model_b_player,
                start_with_a=True,
                x_repeats=X,
                root_view_team=root_view_team,
                root_view_score_diff_before_shot=root_view_score_diff_before_shot,
                execution_noise_by_repeat=execution_noise_by_repeat,
            )
            (
                b_start_counts,
                b_start_result_mean,
                b_start_win_rate,
                b_start_draw_rate,
                b_start_lose_rate,
                b_start_action_logs,
            ) = _evaluate_start_pattern(
                root_state,
                model_a_player,
                model_b_player,
                start_with_a=False,
                x_repeats=X,
                root_view_team=root_view_team,
                root_view_score_diff_before_shot=root_view_score_diff_before_shot,
                execution_noise_by_repeat=execution_noise_by_repeat,
            )

            diff_result_mean_x = float(b_start_result_mean - a_start_result_mean)
            if diff_result_mean_x > 0.0:
                better_by_result_mean_x = "player_b_start"
            elif diff_result_mean_x < 0.0:
                better_by_result_mean_x = "player_a_start"
            else:
                better_by_result_mean_x = "tie"

            position_result = {
                "experiment": experiment_metadata,
                "position": {
                    "position_index": int(global_position_index),
                    "log_name": one_log,
                    "line_index": int(line_index),
                    "end": int(end),
                    "shot": int(shot),
                    "hammer_team": int(hammer),
                    "score_diff_for_team0": int(score_diff_for_team0),
                    "root_view_team": int(root_view_team),
                    "root_view_score_diff_before_shot": int(
                        root_view_score_diff_before_shot
                    ),
                    "root_view_score_diff_bucket": root_view_score_diff_bucket,
                },
                "player_a_start": {
                    "method_key": f"{model_a_key}_start",
                    "method_label": f"{model_a_label}-start",
                    "result_mean_x": float(a_start_result_mean),
                    "win_count_x": int(a_start_counts[RESULT_WIN_IDX]),
                    "draw_count_x": int(a_start_counts[RESULT_DRAW_IDX]),
                    "lose_count_x": int(a_start_counts[RESULT_LOSE_IDX]),
                    "win_rate_x": float(a_start_win_rate),
                    "draw_rate_x": float(a_start_draw_rate),
                    "lose_rate_x": float(a_start_lose_rate),
                    "action_logs_x": a_start_action_logs,
                },
                "player_b_start": {
                    "method_key": f"{model_b_key}_start",
                    "method_label": f"{model_b_label}-start",
                    "result_mean_x": float(b_start_result_mean),
                    "win_count_x": int(b_start_counts[RESULT_WIN_IDX]),
                    "draw_count_x": int(b_start_counts[RESULT_DRAW_IDX]),
                    "lose_count_x": int(b_start_counts[RESULT_LOSE_IDX]),
                    "win_rate_x": float(b_start_win_rate),
                    "draw_rate_x": float(b_start_draw_rate),
                    "lose_rate_x": float(b_start_lose_rate),
                    "action_logs_x": b_start_action_logs,
                },
                "comparison": {
                    "diff_result_mean_x_player_b_start_minus_player_a_start": (
                        diff_result_mean_x
                    ),
                    "better_by_result_mean_x": better_by_result_mean_x,
                },
            }

            position_json_path = (
                json_dir / f"{global_position_index:0{position_index_width}d}.json"
            )
            save_position_json(position_json_path, position_result)
            position_records.append(position_result)
            processed_position_count += 1
            _release_position_memory(use_gpu)

        del dcl2_data
        _release_position_memory(use_gpu)

    if processed_position_count == 0:
        raise RuntimeError(
            f"No positions found in {log_path} "
            f"for end={target_end}, shot={target_shot}, "
            f"positions=[{position_start}, {position_end})."
        )

    all_position_records = load_position_records(json_dir)
    print_report_from_records(json_dir, all_position_records)


if __name__ == "__main__":
    main(
        log_path=NEWSL_DIR / "LearnLog" / "all",
        save_path=Path(__file__).resolve().parents[1] / "data",
        search_method="puct",
        model_a_label="End expanded Model",
        model_b_label="End unexpanded Model",
        target_end=9,
        target_shot=9,
        total_data_size=1000,
        position_start=0,
        position_count=500,
        X=1,
        use_gpu=True,
        transformer_models_by_shot_A={
            9: "transformer-sl-9-9-model-06-14-adamw-epoch50-shot.bin",
            10: "transformer-sl-9-10-model-06-11-adamw-epoch50-shot.bin",
            11: "transformer-sl-9-11-model-06-09-adamw-epoch50-shot.bin",
            12: "transformer-sl-9-12-model-06-08-adamw-epoch50-shot.bin",
            13: "transformer-sl-9-13-model-06-06-adamw-epoch50-shot.bin",
            14: "transformer-sl-9-14-model-06-04-adamw-epoch50-shot.bin",
            15: "transformer-sl-9-15-model-06-02-adamw-epoch50-shot.bin",
        },
        transformer_models_by_shot_B={
            9: "transformer-sl-9-9-model-06-22-adamw-epoch50-shot-non-end-extended.bin",
            10: "transformer-sl-9-10-model-06-22-adamw-epoch50-shot-non-end-extended.bin",
            11: "transformer-sl-9-11-model-06-22-adamw-epoch50-shot-non-end-extended.bin",
            12: "transformer-sl-9-12-model-06-22-adamw-epoch50-shot-non-end-extended.bin",
            13: "transformer-sl-9-13-model-06-22-adamw-epoch50-shot-non-end-extended.bin",
            14: "transformer-sl-9-14-model-06-22-adamw-epoch50-shot-non-end-extended.bin",
            15: "transformer-sl-9-15-model-06-22-adamw-epoch50-shot-non-end-extended.bin",
        },
        shuffle_seed=12345,
        execution_noise_seed=54321,
    )
