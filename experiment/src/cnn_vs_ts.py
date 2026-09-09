from __future__ import annotations

import gc
import json
import os
import random
import sys
from pathlib import Path

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
from experiment.src.report import render_report_from_records, save_position_json
from mcts.rollout import _end_score_diff_team0_minus_team1
from mcts.params import STDDV_ANGLE, STDDV_SPEED
from mcts.search import mcts_search, set_root_state
from mcts.simulate import simulator_step_continuous
from mcts.state import State
from nn.utility import get_torch_device, load_network
from transformer.utility import load_transformer_network


RESULT_WIN_IDX = 0
RESULT_DRAW_IDX = 1
RESULT_LOSE_IDX = 2
ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER = ("<=-2", "-1", "0", "+1", ">=+2")
DEFAULT_CNN_MODEL = "js20000CP-32-9-LeaRate1000-vx32-vy25-batchsize1024.bin"
DEFAULT_TRANSFORMER_MODEL = "transformer-sl16-model-140000data.bin"


def build_output_stem(
    target_end: int,
    target_shot: int,
    data_size: int,
    x_repeats: int,
) -> str:
    return (
        f"cnn_vs_transformer_end{target_end}_shot{target_shot}"
        f"_winrate_datasize{data_size}_x{x_repeats}"
    )


def release_position_memory(use_gpu: bool) -> None:
    gc.collect()
    if use_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_diff_for_team_view(score_diff_for_team0: int, team: int) -> int:
    return score_diff_for_team0 if team == 0 else -score_diff_for_team0


def bucket_root_view_score_diff(score_diff: int) -> str:
    if score_diff <= -2:
        return "<=-2"
    if score_diff == -1:
        return "-1"
    if score_diff == 0:
        return "0"
    if score_diff == 1:
        return "+1"
    return ">=+2"


def evaluate_continuous_action(
    root_state: State,
    vx: float,
    vy: float,
    spin: int,
    x_repeats: int,
    root_view_team: int,
    root_view_score_diff_before_shot: int,
    shot_noises: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, float, float]:
    if shot_noises is not None:
        shot_noises = np.asarray(shot_noises, dtype=np.float64)
        if shot_noises.shape != (x_repeats, 2):
            raise ValueError(
                "shot_noises must have shape "
                f"({x_repeats}, 2), got {shot_noises.shape}"
            )

    counts = np.zeros(3, dtype=np.int32)
    for repeat_index in range(x_repeats):
        noise = None
        if shot_noises is not None:
            noise = (
                float(shot_noises[repeat_index, 0]),
                float(shot_noises[repeat_index, 1]),
            )
        final_state = simulator_step_continuous(root_state, vx, vy, spin, noise=noise)
        raw_score = _end_score_diff_team0_minus_team1(final_state.stones)
        score_from_root_view = score_diff_for_team_view(raw_score, root_view_team)
        total_score_from_root_view = root_view_score_diff_before_shot + score_from_root_view

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
    return counts, result_mean, win_rate, draw_rate, lose_rate


def make_shared_shot_noises(x_repeats: int) -> np.ndarray:
    speed_noises = np.random.normal(0.0, STDDV_SPEED, size=x_repeats)
    angle_noises = np.random.normal(0.0, STDDV_ANGLE, size=x_repeats)
    return np.column_stack((speed_noises, angle_noises))


def main(
    log_path: str | Path = "path/to/dcl2/records",
    save_path: str | Path = "path/to/save/data",
    cnn_model: str = DEFAULT_CNN_MODEL,
    transformer_model: str = DEFAULT_TRANSFORMER_MODEL,
    data_size: int = 1000,
    target_end: int = 9,
    target_shot: int = 15,
    use_gpu: bool = False,
    X: int = 100,
) -> None:
    position_count = 0
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    output_stem = build_output_stem(target_end, target_shot, data_size, X)
    json_dir = save_dir / output_stem
    if json_dir.exists() and any(json_dir.glob("*.json")):
        print(f"Existing position JSON files may be overwritten: {json_dir}")
    json_dir.mkdir(parents=True, exist_ok=True)
    png_path = save_dir / f"{output_stem}.png"
    position_index_width = max(6, len(str(max(data_size - 1, 0))))

    device = get_torch_device(use_gpu=use_gpu)
    cnn_model_path = NEWSL_DIR / "model" / cnn_model
    cnn_network = load_network(cnn_model_path, use_gpu=use_gpu)
    cnn_network.to(device)

    transformer_model_path = NEWSL_DIR / "model" / transformer_model
    transformer_network = load_transformer_network(transformer_model_path, use_gpu=use_gpu)
    transformer_network.to(device)

    experiment_metadata = {
        "target_end": int(target_end),
        "target_shot": int(target_shot),
        "requested_data_size": int(data_size),
        "execution_repeats_x": int(X),
        "evaluation_noise_mode": "shared_per_position",
        "baseline_method_key": "cnn",
        "baseline_method_label": "CNN",
        "newsl_method_key": "transformer",
        "newsl_method_label": "NewSL",
        "cnn_model": str(cnn_model_path),
        "newsl_model_type": "transformer",
        "newsl_transformer_model": str(transformer_model_path),
    }

    position_records: list[dict] = []
    log_files = os.listdir(log_path)
    for one_log in random.sample(log_files, len(log_files)):
        if not os.path.isdir(os.path.join(log_path, one_log)):
            continue
        if position_count >= data_size:
            break

        dcl2_path = os.path.join(log_path, one_log, "game.dcl2")
        if not os.path.exists(dcl2_path):
            continue
        with open(dcl2_path, encoding="utf-8") as dclfile:
            dcl2_data = dclfile.readlines()
        try:
            if json.loads(dcl2_data[-2])["log"]["state"]:
                print(f"Processing log: {one_log}, total processed logs: {position_count + 1}")
        except KeyError:
            del dcl2_data
            continue

        for line_index in range(9, len(dcl2_data) - 2, 2):
            if position_count >= data_size:
                break
            try:
                dcl2_log = json.loads(dcl2_data[line_index])["log"]
                dcl2_state = dcl2_log["state"]
                stones = dcl2_state["stones"]["team0"] + dcl2_state["stones"]["team1"]
                scores_for_scorediff = dcl2_state["scores"]
                end = int(dcl2_state["end"])
                scorediff_for_team0 = scores_to_scorediff_for_team0(scores_for_scorediff)
                shot = int(dcl2_state["shot"])
                hammer = convert_team_stoi(dcl2_state["hammer"])
                if not ((end == target_end) and (shot == target_shot)):
                    continue
            except KeyError:
                continue

            score_index = position_count
            root_state = State.initial(
                stones=stones_listdict_to_xy16(stones),
                end=end,
                hammer_team=hammer,
                shot_index=shot,
                score_diff=scorediff_for_team0,
            )
            root_view_team = int(root_state.to_move())
            root_view_score_diff_before_shot = score_diff_for_team_view(
                scorediff_for_team0,
                root_view_team,
            )
            root_view_score_diff_bucket = bucket_root_view_score_diff(
                root_view_score_diff_before_shot
            )

            cnn_root = set_root_state(
                sl_model=cnn_network,
                stones=stones,
                score_diff=scorediff_for_team0,
                end=end,
                shot_index=shot,
                hammer_team=hammer,
                search_based_model=None,
                debug=False,
                use_search_based_model=False,
            )
            cnn_vx, cnn_vy, cnn_spin = mcts_search(root_state=cnn_root)
            cnn_spin = 1 if int(cnn_spin) == 1 else 0

            transformer_root = set_root_state(
                sl_model=cnn_network,
                stones=stones,
                score_diff=scorediff_for_team0,
                end=end,
                shot_index=shot,
                hammer_team=hammer,
                search_based_model=transformer_network,
                debug=False,
                use_search_based_model=True,
                transformer_target_end=(target_end,),
                transformer_target_shot=(target_shot,),
            )
            transformer_vx, transformer_vy, transformer_spin = mcts_search(
                root_state=transformer_root
            )
            transformer_spin = 1 if int(transformer_spin) == 1 else 0

            shared_shot_noises = make_shared_shot_noises(X)
            (
                cnn_counts,
                cnn_result_mean,
                cnn_win_rate,
                cnn_draw_rate,
                cnn_lose_rate,
            ) = evaluate_continuous_action(
                root_state=root_state,
                vx=cnn_vx,
                vy=cnn_vy,
                spin=cnn_spin,
                x_repeats=X,
                root_view_team=root_view_team,
                root_view_score_diff_before_shot=root_view_score_diff_before_shot,
                shot_noises=shared_shot_noises,
            )
            (
                transformer_counts,
                transformer_result_mean,
                transformer_win_rate,
                transformer_draw_rate,
                transformer_lose_rate,
            ) = evaluate_continuous_action(
                root_state=root_state,
                vx=transformer_vx,
                vy=transformer_vy,
                spin=transformer_spin,
                x_repeats=X,
                root_view_team=root_view_team,
                root_view_score_diff_before_shot=root_view_score_diff_before_shot,
                shot_noises=shared_shot_noises,
            )

            diff_result_mean_x = float(transformer_result_mean - cnn_result_mean)
            if diff_result_mean_x > 0.0:
                better_by_result_mean_x = "newsl"
            elif diff_result_mean_x < 0.0:
                better_by_result_mean_x = "baseline"
            else:
                better_by_result_mean_x = "tie"

            position_result = {
                "experiment": experiment_metadata,
                "position": {
                    "position_index": score_index,
                    "log_name": one_log,
                    "line_index": int(line_index),
                    "end": int(end),
                    "shot": int(shot),
                    "hammer": int(hammer),
                    "root_view_team": root_view_team,
                    "score_diff_for_team0": int(scorediff_for_team0),
                    "root_view_score_diff_before_shot": int(root_view_score_diff_before_shot),
                    "root_view_score_diff_bucket": root_view_score_diff_bucket,
                },
                "baseline": {
                    "method_key": "cnn",
                    "method_label": "CNN",
                    "vx": float(cnn_vx),
                    "vy": float(cnn_vy),
                    "spin": int(cnn_spin),
                    "result_mean_x": float(cnn_result_mean),
                    "win_count_x": int(cnn_counts[RESULT_WIN_IDX]),
                    "draw_count_x": int(cnn_counts[RESULT_DRAW_IDX]),
                    "lose_count_x": int(cnn_counts[RESULT_LOSE_IDX]),
                    "win_rate_x": float(cnn_win_rate),
                    "draw_rate_x": float(cnn_draw_rate),
                    "lose_rate_x": float(cnn_lose_rate),
                },
                "newsl": {
                    "method_key": "transformer",
                    "method_label": "NewSL",
                    "model_type": "transformer",
                    "vx": float(transformer_vx),
                    "vy": float(transformer_vy),
                    "spin": int(transformer_spin),
                    "result_mean_x": float(transformer_result_mean),
                    "win_count_x": int(transformer_counts[RESULT_WIN_IDX]),
                    "draw_count_x": int(transformer_counts[RESULT_DRAW_IDX]),
                    "lose_count_x": int(transformer_counts[RESULT_LOSE_IDX]),
                    "win_rate_x": float(transformer_win_rate),
                    "draw_rate_x": float(transformer_draw_rate),
                    "lose_rate_x": float(transformer_lose_rate),
                },
                "comparison": {
                    "diff_result_mean_x_newsl_minus_baseline": diff_result_mean_x,
                    "better_by_result_mean_x": better_by_result_mean_x,
                },
            }
            position_json_path = json_dir / f"{score_index:0{position_index_width}d}.json"
            save_position_json(position_json_path, position_result)
            position_records.append(position_result)
            position_count += 1

            del (
                position_result,
                position_json_path,
                root_state,
                cnn_root,
                cnn_counts,
                transformer_root,
                transformer_counts,
                shared_shot_noises,
                dcl2_log,
                dcl2_state,
                stones,
                scores_for_scorediff,
            )
            release_position_memory(use_gpu)

        del dcl2_data
        release_position_memory(use_gpu)

    if position_count == 0:
        raise RuntimeError(
            f"No positions found for end={target_end}, shot={target_shot} in {log_path}"
        )

    render_report_from_records(json_dir, png_path, position_records)


if __name__ == "__main__":
    main(
        log_path=NEWSL_DIR / "LearnLog" / "all",
        save_path=Path(__file__).resolve().parents[1] / "data",
        cnn_model=DEFAULT_CNN_MODEL,
        transformer_model="transformer-sl-9-15-model-05-26-AdamW-epoch50.bin",
        data_size=3000,
        target_end=9,
        target_shot=15,
        use_gpu=True,
        X=100,
    )
