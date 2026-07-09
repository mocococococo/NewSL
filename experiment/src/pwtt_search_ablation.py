from __future__ import annotations

import gc
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


NEWSL_DIR = Path(__file__).resolve().parents[2]
if str(NEWSL_DIR) not in sys.path:
    sys.path.insert(0, str(NEWSL_DIR))

from common.translate_state import convert_team_stoi, scores_to_scorediff_for_team0, stones_listdict_to_xy16
from mcts.search import mcts_search, set_root_state as set_mcts_root_state
from mcts.state import State
from nn.utility import get_torch_device, load_network
from transformer.utility import load_transformer_network


DEFAULT_CNN_MODEL = "js20000CP-32-9-LeaRate1000-vx32-vy25-batchsize1024.bin"


def _resolve_model(model: str | Path) -> Path:
    model_path = Path(model)
    if model_path.is_absolute():
        return model_path
    return NEWSL_DIR / "model" / model_path


def _release_memory(use_gpu: bool) -> None:
    gc.collect()
    if use_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _save_json(save_file_path: Path, data: dict[str, Any]) -> None:
    save_file_path.parent.mkdir(parents=True, exist_ok=True)
    save_file_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )


def _build_output_stem(
    target_end: int,
    target_shot: int,
    data_size: int,
    x_repeats: int,
) -> str:
    return (
        "pwtt_search_ablation"
        f"_end{target_end}_shot{target_shot}"
        f"_datasize{data_size}_x{x_repeats}"
    )


def _tt_hit_rate_estimate(stats: dict[str, Any]) -> float:
    simulations = int(stats["simulations"])
    nodes = int(stats["nodes"])
    if simulations <= 0:
        return 0.0
    value = 1.0 - ((nodes - 1) / simulations)
    return float(max(0.0, min(1.0, value)))


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _make_trial_summary(trials: list[dict[str, Any]]) -> dict[str, Any]:
    simulations = [float(trial["simulations"]) for trial in trials]
    elapsed = [float(trial["elapsed"]) for trial in trials]
    nodes = [float(trial["nodes"]) for trial in trials]
    root_visited = [float(trial["root_visited"]) for trial in trials]
    root_expanded = [float(trial["root_expanded"]) for trial in trials]
    root_candidates = [float(trial["root_candidates"]) for trial in trials]
    tt_hit_rates = [_tt_hit_rate_estimate(trial) for trial in trials]
    sims_per_sec = [
        float(trial["simulations"]) / float(trial["elapsed"])
        for trial in trials
        if float(trial["elapsed"]) > 0.0
    ]

    return {
        "mean_simulations": _mean(simulations),
        "mean_elapsed": _mean(elapsed),
        "mean_nodes": _mean(nodes),
        "mean_sims_per_sec": _mean(sims_per_sec),
        "mean_root_visited": _mean(root_visited),
        "mean_root_expanded": _mean(root_expanded),
        "mean_root_candidates": _mean(root_candidates),
        "mean_tt_hit_rate_estimate": _mean(tt_hit_rates),
    }


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_single_search(
    network: Any,
    transformer_network: Any,
    root_state: State,
    use_transformer: bool,
    transformer_target_end: tuple[int, ...],
    transformer_target_shot: tuple[int, ...],
    use_progressive_widening: bool,
    use_transposition_table: bool,
    seed: int,
) -> tuple[tuple[float, float, int], dict[str, Any]]:
    _set_random_seed(seed)
    root = set_mcts_root_state(
        network=network,
        stones=_state_to_raw_stones(root_state),
        score_diff=root_state.score_diff,
        end=root_state.end,
        shot_index=root_state.shot_index,
        hammer_team=root_state.hammer_team,
        transformer_network=transformer_network,
        use_transformer=use_transformer,
        transformer_target_end=transformer_target_end,
        transformer_target_shot=transformer_target_shot,
    )
    action, stats = mcts_search(
        root_state=root,
        use_progressive_widening=use_progressive_widening,
        use_transposition_table=use_transposition_table,
        return_stats=True,
    )
    stats = dict(stats)
    stats["selected_action"] = {
        "vx": float(action[0]),
        "vy": float(action[1]),
        "spin": int(action[2]),
    }
    stats["tt_hit_rate_estimate"] = _tt_hit_rate_estimate(stats)
    return action, stats


def _state_to_raw_stones(state: State) -> list[Optional[dict[str, Any]]]:
    stones: list[Optional[dict[str, Any]]] = []
    for stone in state.stones:
        if stone is None:
            stones.append(None)
        else:
            x, y = stone
            stones.append({"position": {"x": float(x), "y": float(y)}})
    return stones


def _load_transformer_networks(
    transformer_models_by_shot: Optional[dict[int, str | Path]],
    use_gpu: bool,
) -> tuple[dict[int, Any], dict[int, str]]:
    if not transformer_models_by_shot:
        return {}, {}

    networks: dict[int, Any] = {}
    paths: dict[int, str] = {}
    for shot, model in sorted(transformer_models_by_shot.items()):
        model_path = _resolve_model(model)
        networks[int(shot)] = load_transformer_network(model_path, use_gpu=use_gpu)
        paths[int(shot)] = str(model_path)
    return networks, paths


def run_experiment(
    log_path: str | Path = NEWSL_DIR / "LearnLog" / "all",
    save_path: str | Path = Path(__file__).resolve().parents[1] / "data",
    target_end: int = 9,
    target_shot: int = 15,
    data_size: int = 1000,
    X: int = 1,
    model: str | Path = DEFAULT_CNN_MODEL,
    use_gpu: bool = True,
    use_transformer: bool = False,
    transformer_models_by_shot: Optional[dict[int, str | Path]] = None,
    shuffle_seed: Optional[int] = 12345,
    search_seed: int = 24680,
) -> Path:
    if not (0 <= target_shot <= 15):
        raise ValueError(f"target_shot must be in [0, 15], got {target_shot}")
    if data_size <= 0:
        raise ValueError(f"data_size must be positive, got {data_size}")
    if X <= 0:
        raise ValueError(f"X must be positive, got {X}")

    device = get_torch_device(use_gpu=use_gpu)
    model_path = _resolve_model(model)
    network = load_network(model_path, use_gpu=use_gpu)
    network.to(device)

    transformer_networks, transformer_model_paths = _load_transformer_networks(
        transformer_models_by_shot,
        use_gpu=use_gpu,
    )
    transformer_target_shot = tuple(sorted(transformer_networks)) or (target_shot,)
    transformer_target_end = (int(target_end),)

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_stem = _build_output_stem(target_end, target_shot, data_size, X)
    json_dir = save_dir / output_stem
    json_dir.mkdir(parents=True, exist_ok=True)
    position_index_width = max(6, len(str(max(data_size - 1, 0))))

    experiment_metadata = {
        "experiment_type": "pwtt_search_ablation",
        "target_end": int(target_end),
        "target_shot": int(target_shot),
        "requested_data_size": int(data_size),
        "execution_repeats_x": int(X),
        "model": str(model_path),
        "use_transformer": bool(use_transformer),
        "transformer_models_by_shot": transformer_model_paths,
        "shuffle_seed": shuffle_seed,
        "search_seed": int(search_seed),
        "condition_a_key": "non_pwtt",
        "condition_a_label": "non-PWTT",
        "condition_b_key": "pwtt",
        "condition_b_label": "PWTT",
    }
    _save_json(json_dir / "metadata.json", experiment_metadata)

    log_path = Path(log_path)
    log_files = os.listdir(log_path)
    rng = random.Random(shuffle_seed)
    shuffled_log_files = rng.sample(log_files, len(log_files))

    position_count = 0
    for one_log in shuffled_log_files:
        if position_count >= data_size:
            break
        log_dir = log_path / one_log
        if not log_dir.is_dir():
            continue

        dcl2_path = log_dir / "game.dcl2"
        if not dcl2_path.exists():
            continue
        with dcl2_path.open(encoding="utf-8") as dclfile:
            dcl2_data = dclfile.readlines()

        for line_index in range(9, len(dcl2_data) - 2, 2):
            if position_count >= data_size:
                break
            try:
                dcl2_log = json.loads(dcl2_data[line_index])["log"]
                dcl2_state = dcl2_log["state"]
                stones = dcl2_state["stones"]["team0"] + dcl2_state["stones"]["team1"]
                scores = dcl2_state["scores"]
                end = int(dcl2_state["end"])
                shot = int(dcl2_state["shot"])
                hammer = convert_team_stoi(dcl2_state["hammer"])
                if not ((end == target_end) and (shot == target_shot)):
                    continue
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

            score_diff_for_team0 = scores_to_scorediff_for_team0(scores)
            root_state = State.initial(
                stones=stones_listdict_to_xy16(stones),
                end=end,
                hammer_team=hammer,
                shot_index=shot,
                score_diff=score_diff_for_team0,
            )

            print(
                f"Processing position {position_count + 1}: "
                f"log={one_log}, end={end}, shot={shot}"
            )
            non_pwtt_trials: list[dict[str, Any]] = []
            pwtt_trials: list[dict[str, Any]] = []

            for repeat_index in range(X):
                pair_seed = int(search_seed + position_count * 100000 + repeat_index)
                run_pwtt_first = ((position_count + repeat_index) % 2 == 1)

                if run_pwtt_first:
                    _, pwtt_stats = _run_single_search(
                        network,
                        transformer_networks or None,
                        root_state,
                        use_transformer,
                        transformer_target_end,
                        transformer_target_shot,
                        True,
                        True,
                        pair_seed,
                    )
                    _, non_pwtt_stats = _run_single_search(
                        network,
                        transformer_networks or None,
                        root_state,
                        use_transformer,
                        transformer_target_end,
                        transformer_target_shot,
                        False,
                        False,
                        pair_seed,
                    )
                else:
                    _, non_pwtt_stats = _run_single_search(
                        network,
                        transformer_networks or None,
                        root_state,
                        use_transformer,
                        transformer_target_end,
                        transformer_target_shot,
                        False,
                        False,
                        pair_seed,
                    )
                    _, pwtt_stats = _run_single_search(
                        network,
                        transformer_networks or None,
                        root_state,
                        use_transformer,
                        transformer_target_end,
                        transformer_target_shot,
                        True,
                        True,
                        pair_seed,
                    )

                non_pwtt_stats["repeat_index"] = int(repeat_index)
                pwtt_stats["repeat_index"] = int(repeat_index)
                non_pwtt_trials.append(non_pwtt_stats)
                pwtt_trials.append(pwtt_stats)

            non_pwtt_summary = _make_trial_summary(non_pwtt_trials)
            pwtt_summary = _make_trial_summary(pwtt_trials)
            mean_sim_ratio = (
                pwtt_summary["mean_simulations"] / non_pwtt_summary["mean_simulations"]
                if non_pwtt_summary["mean_simulations"] > 0.0
                else 0.0
            )
            mean_log_sim_ratio = float(np.log(mean_sim_ratio)) if mean_sim_ratio > 0.0 else 0.0

            position_result = {
                "experiment": experiment_metadata,
                "position": {
                    "position_index": int(position_count),
                    "log_name": one_log,
                    "line_index": int(line_index),
                    "end": int(end),
                    "shot": int(shot),
                    "hammer_team": int(hammer),
                    "score_diff_for_team0": int(score_diff_for_team0),
                },
                "non_pwtt": {
                    "condition_key": "non_pwtt",
                    "condition_label": "non-PWTT",
                    "trials": non_pwtt_trials,
                    "summary": non_pwtt_summary,
                },
                "pwtt": {
                    "condition_key": "pwtt",
                    "condition_label": "PWTT",
                    "trials": pwtt_trials,
                    "summary": pwtt_summary,
                },
                "comparison": {
                    "mean_sim_diff_pwtt_minus_non_pwtt": float(
                        pwtt_summary["mean_simulations"] - non_pwtt_summary["mean_simulations"]
                    ),
                    "mean_sim_ratio_pwtt_over_non_pwtt": float(mean_sim_ratio),
                    "mean_sim_increase_rate": float(mean_sim_ratio - 1.0),
                    "mean_log_sim_ratio": mean_log_sim_ratio,
                },
            }

            position_json_path = json_dir / f"{position_count:0{position_index_width}d}.json"
            _save_json(position_json_path, position_result)
            position_count += 1
            _release_memory(use_gpu)

        del dcl2_data
        _release_memory(use_gpu)

    if position_count == 0:
        raise RuntimeError(
            f"No positions found in {log_path} "
            f"for end={target_end}, shot={target_shot}."
        )

    return json_dir