from __future__ import annotations

import gc
import json
import os
import random
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
from experiment.src.mini_match_report import print_report_from_records, save_position_json
from mcts.params import STDDV_ANGLE, STDDV_SPEED
from mcts.rollout import _end_score_diff_team0_minus_team1
from mcts.search import mcts_search, set_root_state as set_mcts_root_state
from mcts.simulate import ShotNoise, simulator_step_continuous
from mcts.state import State
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


@dataclass(frozen=True)
class SearchAction:
    vx: float
    vy: float
    spin: int
    meta: dict[str, Any]


class TransformerSearchPlayer:
    def __init__(
        self,
        search_method: str,
        networks_by_shot: dict[int, TransformerNetwork],
        model_paths_by_shot: dict[int, Path],
        target_end: int,
    ) -> None:
        normalized = search_method.strip().lower()
        if normalized not in ("shot", "puct"):
            raise ValueError(f"Unknown search method: {search_method}")

        self.search_method = normalized
        self.key = normalized
        self.label = "SHOT" if normalized == "shot" else "PUCT"
        self.networks_by_shot = networks_by_shot
        self.model_paths_by_shot = model_paths_by_shot
        self.target_end = int(target_end)
        self.target_shots = tuple(sorted(networks_by_shot))

    def select_action(self, state: State) -> SearchAction:
        if state.shot_index not in self.networks_by_shot:
            raise ValueError(
                f"Transformer model for shot {state.shot_index} is not configured."
            )

        root_kwargs = {
            "network": None,
            "stones": _state_to_raw_stones(state),
            "score_diff": state.score_diff,
            "end": state.end,
            "shot_index": state.shot_index,
            "hammer_team": state.hammer_team,
            "transformer_network": self.networks_by_shot,
            "use_transformer": True,
            "transformer_target_end": (self.target_end,),
            "transformer_target_shot": self.target_shots,
        }

        if self.search_method == "shot":
            root = set_shot_root_state(**root_kwargs)
            vx, vy, spin = shot_search(root_state=root)
        else:
            root = set_mcts_root_state(**root_kwargs)
            vx, vy, spin = mcts_search(root_state=root)

        return SearchAction(
            vx=float(vx),
            vy=float(vy),
            spin=1 if int(spin) == 1 else 0,
            meta={
                "search_method": self.search_method,
                "model": str(self.model_paths_by_shot[state.shot_index]),
            },
        )


def _print_match_header(
    first_player: TransformerSearchPlayer,
    second_player: TransformerSearchPlayer,
    repeat_index: int,
    x_repeats: int,
) -> None:
    print("")
    print(
        f"{first_player.label} vs {second_player.label} "
        f"({repeat_index + 1}/{x_repeats})"
    )


def _run_suffix_game(
    root_state: State,
    shot_player: TransformerSearchPlayer,
    puct_player: TransformerSearchPlayer,
    start_with_shot: bool,
    repeat_index: int,
    x_repeats: int,
    execution_noise: list[ShotNoise],
) -> tuple[State, list[dict[str, Any]]]:
    state = root_state
    action_log: list[dict[str, Any]] = []
    start_shot = root_state.shot_index
    first_player = shot_player if start_with_shot else puct_player
    second_player = puct_player if start_with_shot else shot_player
    _print_match_header(first_player, second_player, repeat_index, x_repeats)

    while not state.is_end_terminal():
        offset = state.shot_index - start_shot
        use_shot = (offset % 2 == 0) if start_with_shot else (offset % 2 == 1)
        player = shot_player if use_shot else puct_player

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
    shot_player: TransformerSearchPlayer,
    puct_player: TransformerSearchPlayer,
    start_with_shot: bool,
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
            shot_player,
            puct_player,
            start_with_shot=start_with_shot,
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
    target_end: int,
    target_shot: int,
    data_size: int,
    x_repeats: int,
) -> str:
    return (
        f"shot_vs_puct_transformer"
        f"_end{target_end}_shot{target_shot}"
        f"_datasize{data_size}_x{x_repeats}"
    )


def main(
    log_path: str | Path = NEWSL_DIR / "LearnLog" / "all",
    save_path: str | Path = Path(__file__).resolve().parents[1] / "data",
    target_end: int = 9,
    target_shot: int = 15,
    data_size: int = 1000,
    X: int = 1,
    use_gpu: bool = True,
    transformer_models_by_shot: Optional[dict[int, str | Path]] = None,
    shuffle_seed: Optional[int] = 12345,
    execution_noise_seed: int = 54321,
) -> None:
    if not (0 <= target_shot <= 15):
        raise ValueError(f"target_shot must be in [0, 15], got {target_shot}")
    if data_size <= 0:
        raise ValueError(f"data_size must be positive, got {data_size}")
    if X <= 0:
        raise ValueError(f"X must be positive, got {X}")

    if transformer_models_by_shot is None:
        transformer_models_by_shot = {
            8: "transformer-sl-9-8-model-06-16-adamw-epoch50-shot.bin",
            9: "transformer-sl-9-9-model-06-14-adamw-epoch50-shot.bin",
            10: "transformer-sl-9-10-model-06-11-adamw-epoch50-shot.bin",
            11: "transformer-sl-9-11-model-06-09-adamw-epoch50-shot.bin",
            12: "transformer-sl-9-12-model-06-08-adamw-epoch50-shot.bin",
            13: "transformer-sl-9-13-model-06-06-adamw-epoch50-shot.bin",
            14: "transformer-sl-9-14-model-06-05-adamw-epoch50-shot.bin",
            15: "transformer-sl-9-15-model-06-02-adamw-epoch50-shot.bin",
        }

    needed_shots = tuple(range(target_shot, 16))
    missing_shots = [
        shot for shot in needed_shots if shot not in transformer_models_by_shot
    ]
    if missing_shots:
        raise ValueError(f"Missing transformer models for shots: {missing_shots}")

    model_paths_by_shot = {
        int(shot): _resolve_model(model_path)
        for shot, model_path in transformer_models_by_shot.items()
        if int(shot) in needed_shots
    }
    networks_by_shot = {
        shot: load_transformer_network(model_path, use_gpu=use_gpu)
        for shot, model_path in model_paths_by_shot.items()
    }

    shot_player = TransformerSearchPlayer(
        "shot",
        networks_by_shot,
        model_paths_by_shot,
        target_end,
    )
    puct_player = TransformerSearchPlayer(
        "puct",
        networks_by_shot,
        model_paths_by_shot,
        target_end,
    )

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_stem = _build_output_stem(target_end, target_shot, data_size, X)
    json_dir = save_dir / output_stem
    json_dir.mkdir(parents=True, exist_ok=True)
    position_index_width = max(6, len(str(max(data_size - 1, 0))))

    experiment_metadata = {
        "experiment_type": "shot_vs_puct",
        "target_end": int(target_end),
        "target_shot": int(target_shot),
        "requested_data_size": int(data_size),
        "execution_repeats_x": int(X),
        "player_a_start_method_key": "shot_start",
        "player_a_start_method_label": "SHOT-start",
        "player_b_start_method_key": "puct_start",
        "player_b_start_method_label": "PUCT-start",
        "player_a_kind": "Transformer-SHOT",
        "player_b_kind": "Transformer-PUCT",
        "player_a_label": "SHOT",
        "player_b_label": "PUCT",
        "player_a_search_method": "shot",
        "player_b_search_method": "puct",
        "transformer_models_by_shot": {
            int(shot): str(path) for shot, path in model_paths_by_shot.items()
        },
        "shuffle_seed": shuffle_seed,
        "execution_noise_seed": int(execution_noise_seed),
        "shared_execution_noise": True,
    }

    log_files = os.listdir(log_path)
    rng = random.Random(shuffle_seed)
    shuffled_log_files = rng.sample(log_files, len(log_files))

    position_records: list[dict[str, Any]] = []
    position_count = 0

    for one_log in shuffled_log_files:
        if position_count >= data_size:
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
            if position_count >= data_size:
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
                position_count,
                X,
                16 - target_shot,
            )

            print(
                f"Processing position {position_count + 1}: "
                f"log={one_log}, end={end}, shot={shot}"
            )
            (
                shot_start_counts,
                shot_start_result_mean,
                shot_start_win_rate,
                shot_start_draw_rate,
                shot_start_lose_rate,
                shot_start_action_logs,
            ) = _evaluate_start_pattern(
                root_state,
                shot_player,
                puct_player,
                start_with_shot=True,
                x_repeats=X,
                root_view_team=root_view_team,
                root_view_score_diff_before_shot=root_view_score_diff_before_shot,
                execution_noise_by_repeat=execution_noise_by_repeat,
            )
            (
                puct_start_counts,
                puct_start_result_mean,
                puct_start_win_rate,
                puct_start_draw_rate,
                puct_start_lose_rate,
                puct_start_action_logs,
            ) = _evaluate_start_pattern(
                root_state,
                shot_player,
                puct_player,
                start_with_shot=False,
                x_repeats=X,
                root_view_team=root_view_team,
                root_view_score_diff_before_shot=root_view_score_diff_before_shot,
                execution_noise_by_repeat=execution_noise_by_repeat,
            )

            diff_result_mean_x = float(
                puct_start_result_mean - shot_start_result_mean
            )
            if diff_result_mean_x > 0.0:
                better_by_result_mean_x = "player_b_start"
            elif diff_result_mean_x < 0.0:
                better_by_result_mean_x = "player_a_start"
            else:
                better_by_result_mean_x = "tie"

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
                    "root_view_team": int(root_view_team),
                    "root_view_score_diff_before_shot": int(
                        root_view_score_diff_before_shot
                    ),
                    "root_view_score_diff_bucket": root_view_score_diff_bucket,
                },
                "player_a_start": {
                    "method_key": "shot_start",
                    "method_label": "SHOT-start",
                    "result_mean_x": float(shot_start_result_mean),
                    "win_count_x": int(shot_start_counts[RESULT_WIN_IDX]),
                    "draw_count_x": int(shot_start_counts[RESULT_DRAW_IDX]),
                    "lose_count_x": int(shot_start_counts[RESULT_LOSE_IDX]),
                    "win_rate_x": float(shot_start_win_rate),
                    "draw_rate_x": float(shot_start_draw_rate),
                    "lose_rate_x": float(shot_start_lose_rate),
                    "action_logs_x": shot_start_action_logs,
                },
                "player_b_start": {
                    "method_key": "puct_start",
                    "method_label": "PUCT-start",
                    "result_mean_x": float(puct_start_result_mean),
                    "win_count_x": int(puct_start_counts[RESULT_WIN_IDX]),
                    "draw_count_x": int(puct_start_counts[RESULT_DRAW_IDX]),
                    "lose_count_x": int(puct_start_counts[RESULT_LOSE_IDX]),
                    "win_rate_x": float(puct_start_win_rate),
                    "draw_rate_x": float(puct_start_draw_rate),
                    "lose_rate_x": float(puct_start_lose_rate),
                    "action_logs_x": puct_start_action_logs,
                },
                "comparison": {
                    "diff_result_mean_x_player_b_start_minus_player_a_start": (
                        diff_result_mean_x
                    ),
                    "better_by_result_mean_x": better_by_result_mean_x,
                },
            }

            position_json_path = (
                json_dir / f"{position_count:0{position_index_width}d}.json"
            )
            save_position_json(position_json_path, position_result)
            position_records.append(position_result)
            position_count += 1
            _release_position_memory(use_gpu)

        del dcl2_data
        _release_position_memory(use_gpu)

    if position_count == 0:
        raise RuntimeError(
            f"No positions found in {log_path} "
            f"for end={target_end}, shot={target_shot}."
        )

    print_report_from_records(json_dir, position_records)


if __name__ == "__main__":
    main(
        log_path=NEWSL_DIR / "LearnLog" / "all",
        save_path=Path(__file__).resolve().parents[1] / "data",
        target_end=9,
        target_shot=9,
        data_size=1000,
        X=1,
        use_gpu=True,
        transformer_models_by_shot={
            9: "transformer-sl-9-9-model-06-14-adamw-epoch50-shot.bin",
            10: "transformer-sl-9-10-model-06-11-adamw-epoch50-shot.bin",
            11: "transformer-sl-9-11-model-06-09-adamw-epoch50-shot.bin",
            12: "transformer-sl-9-12-model-06-08-adamw-epoch50-shot.bin",
            13: "transformer-sl-9-13-model-06-06-adamw-epoch50-shot.bin",
            14: "transformer-sl-9-14-model-06-04-adamw-epoch50-shot.bin",
            15: "transformer-sl-9-15-model-06-02-adamw-epoch50-shot.bin",
        },
        shuffle_seed=12345,
        execution_noise_seed=54321,
    )
