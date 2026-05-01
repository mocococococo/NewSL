from __future__ import annotations

import contextlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
NEWSL_DIR = REPO_ROOT / "NewSL"
KURA_DIR = REPO_ROOT / "Kuracurling_SHOT"
KURA_SIM_DIR = KURA_DIR / "mcts" / "FCV1Simulation" / "src"

_SHARED_MODULE_PREFIXES = (
    "board",
    "common",
    "dc3client",
    "learning_param",
    "mcts",
    "nn",
    "other",
    "policy_shot",
)


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)


def _remove_sys_path(path: Path) -> None:
    path_str = str(path)
    while path_str in sys.path:
        sys.path.remove(path_str)


def _purge_shared_modules() -> None:
    for module_name in list(sys.modules):
        if any(
            module_name == prefix or module_name.startswith(prefix + ".")
            for prefix in _SHARED_MODULE_PREFIXES
        ):
            del sys.modules[module_name]


# Load Kura modules first. They use top-level names such as nn, board, and mcts.
_prepend_sys_path(KURA_DIR)
from other.other import index_to_shot as kura_index_to_shot
from other.other import load_checkpoint_to_model as kura_load_checkpoint_to_model
from other.other_2 import decompose_and_collate as kura_decompose_and_collate
from mcts.FCV1Simulation.src.build.Release.simulator import StoneSimulator as KuraStoneSimulator
from mcts.utils import add_noise_to_vector as kura_add_noise_to_vector

_remove_sys_path(KURA_DIR)
_purge_shared_modules()

_prepend_sys_path(NEWSL_DIR)
from common.translate_state import (
    convert_team_stoi,
    scores_to_scorediff_for_team0,
    stones_listdict_to_xy16,
)
from mcts.rollout import _end_score_diff_team0_minus_team1
from mcts.search import mcts_search, set_root_state
from mcts.simulate import simulator_step_continuous
from mcts.state import State
from nn.utility import get_torch_device
from transformer.utility import load_transformer_network


RESULT_WIN_IDX = 0
RESULT_DRAW_IDX = 1
RESULT_LOSE_IDX = 2
ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER = ("<=-2", "-1", "0", "+1", ">=+2")

KURA_ACTION_CLASSES = 32 * 32 * 2
KURA_POLICY_SHOT15_MODEL = "checkpoint_score_visit_17274_epoch_050.pt"
KURA_TOP_K = 36
KURA_NUM_TRIALS = 9
KURA_REL_KEEP = 0.0
KURA_STDDV_SPEED = 0.0076
KURA_STDDV_ANGLE = 0.0018
KURA_TEE_Y = 17.3735
KURA_HACK_Y = 21.0315
KURA_HOUSE_RADIUS = 1.829
KURA_STONE_RADIUS = 0.145


@contextlib.contextmanager
def _pushd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _kura_device(use_gpu: bool) -> torch.device:
    if use_gpu:
        torch.cuda.set_device(0)
        return torch.device("cuda")
    return torch.device("cpu")


def _move_tensors_to_device(
    stones_tensor: torch.Tensor,
    game_feat: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return stones_tensor.to(device), game_feat.to(device), mask.to(device)


def _kura_rotation_to_string(rotation: object) -> str:
    value = getattr(rotation, "value", rotation)
    return "cw" if value == "cw" else "ccw"


def _rotation_string_to_newsl_spin(rotation: str) -> int:
    return 0 if rotation == "cw" else 1


def _shot_team_to_name(shot_team: int) -> str:
    if shot_team == 0:
        return "team0"
    if shot_team == 1:
        return "team1"
    raise ValueError(f"shot_team must be 0 or 1, got {shot_team}")


def _prepare_stones_for_kura(
    stones: list[Optional[dict]],
    shot: int,
    shot_team: int,
) -> tuple[list[Optional[dict]], str]:
    stones_list = list(stones)
    my_team = _shot_team_to_name(shot_team)
    change_team = my_team

    if (my_team == "team0" and shot % 2 != 0) or (
        my_team == "team1" and shot % 2 == 0
    ):
        stones_list = stones_list[8:16] + stones_list[0:8]
        change_team = "team1"
        if my_team == "team1":
            change_team = "team0"

    return stones_list, change_team


def _kura_calculate_score(result: np.ndarray) -> int:
    coordinates = np.asarray(result, dtype=float).reshape(-1, 2)
    target_point = np.array([0.0, KURA_HACK_Y + KURA_TEE_Y])
    max_distance = KURA_HOUSE_RADIUS + KURA_STONE_RADIUS
    filtered_and_sorted_indices = sorted(
        [
            i
            for i in range(len(coordinates))
            if np.linalg.norm(coordinates[i] - target_point) < max_distance
        ],
        key=lambda i: np.linalg.norm(coordinates[i] - target_point),
    )

    score = 0
    if filtered_and_sorted_indices:
        if filtered_and_sorted_indices[0] < 8:
            for i in filtered_and_sorted_indices:
                if i < 8:
                    score += 1
                else:
                    break
        else:
            for i in filtered_and_sorted_indices:
                if i >= 8:
                    score -= 1
                else:
                    break
    return score


@dataclass(frozen=True)
class KuraSearchResult:
    action_index: int
    vx: float
    vy: float
    rotation: str
    spin: int
    mean_expected_score: float


class KuraShot15Searcher:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        use_gpu: bool = False,
        top_k: int = KURA_TOP_K,
        num_trials: int = KURA_NUM_TRIALS,
        rel_keep: float = KURA_REL_KEEP,
    ) -> None:
        self.device = _kura_device(use_gpu)
        self.model_path = model_path or (KURA_DIR / "model" / KURA_POLICY_SHOT15_MODEL)
        self.top_k = int(top_k)
        self.num_trials = int(num_trials)
        self.rel_keep = float(rel_keep)

        self.policy_model = kura_load_checkpoint_to_model(
            checkpoint_path=str(self.model_path),
            device=self.device,
            action_classes=KURA_ACTION_CLASSES,
        )
        self.policy_model.eval()

        with _pushd(KURA_SIM_DIR):
            self.stone_simulator = KuraStoneSimulator()

    def _policy_probs(
        self,
        stones_list: list[Optional[dict]],
        end: int,
        shot: int,
        my_team: str,
    ) -> torch.Tensor:
        stones_tensor, game_feat, mask = kura_decompose_and_collate(
            stones=stones_list,
            scores=None,
            end=int(end) + 1,
            shot=int(shot),
            my_team=my_team,
        )
        stones_tensor, game_feat, mask = _move_tensors_to_device(
            stones_tensor,
            game_feat,
            mask,
            self.device,
        )

        with torch.no_grad():
            policy_logits, _, _ = self.policy_model(stones_tensor, game_feat, mask=mask)
            probs = F.softmax(policy_logits[:, -1, :], dim=-1)[0]
        return probs

    def _decode_action(self, action_index: int) -> tuple[float, float, str, int]:
        vx, vy, rotation = kura_index_to_shot(int(action_index))
        rotation_str = _kura_rotation_to_string(rotation)
        return float(vx), float(vy), rotation_str, _rotation_string_to_newsl_spin(rotation_str)

    def _step_once(
        self,
        stones_list: list[Optional[dict]],
        shot: int,
        vx: float,
        vy: float,
        rotation: str,
    ) -> np.ndarray:
        rand_x, rand_y = kura_add_noise_to_vector(
            vx,
            vy,
            KURA_STDDV_SPEED,
            KURA_STDDV_ANGLE,
        )
        rotation_index = 1.0
        if rotation == "ccw":
            rotation_index = -1.0

        position: list[float] = []
        for stone in stones_list:
            if stone is None:
                position.extend([0.0, 0.0])
            else:
                position.extend(
                    [
                        float(stone["position"]["x"]),
                        float(stone["position"]["y"]),
                    ]
                )

        with _pushd(KURA_SIM_DIR):
            result, _ = self.stone_simulator.simulator(
                np.array(position),
                int(shot),
                np.array([rand_x]),
                np.array([rand_y]),
                np.array([rotation_index]),
            )
        return result

    def select(
        self,
        stones: list[Optional[dict]],
        end: int,
        shot: int,
        shot_team: int,
    ) -> KuraSearchResult:
        if shot != 15:
            raise ValueError("KuraShot15Searcher only implements Kura shot == 15 search.")

        stones_list, my_team = _prepare_stones_for_kura(stones, shot, shot_team)
        probs = self._policy_probs(stones_list, end, shot, my_team)

        keep_threshold = probs.max() * self.rel_keep
        candidate_indices = torch.nonzero(probs >= keep_threshold, as_tuple=False).view(-1)
        if candidate_indices.numel() > self.top_k:
            candidate_probs = probs[candidate_indices]
            selected = torch.topk(candidate_probs, k=self.top_k, dim=-1).indices
            candidate_indices = candidate_indices[selected]

        best_action_index: Optional[int] = None
        best_mean: Optional[float] = None
        best_move: Optional[tuple[float, float, str, int]] = None

        for action_index in candidate_indices.detach().cpu().tolist():
            vx, vy, rotation, spin = self._decode_action(int(action_index))
            trial_scores = []
            for _ in range(self.num_trials):
                result = self._step_once(stones_list, shot, vx, vy, rotation)
                score_raw = _kura_calculate_score(result)
                trial_scores.append(float(-score_raw))

            mean_expected = float(np.mean(trial_scores))
            if best_mean is None or mean_expected > best_mean:
                best_mean = mean_expected
                best_action_index = int(action_index)
                best_move = (vx, vy, rotation, spin)

        if best_action_index is None or best_mean is None or best_move is None:
            raise RuntimeError("Kura search produced no candidate action.")

        vx, vy, rotation, spin = best_move
        return KuraSearchResult(
            action_index=best_action_index,
            vx=vx,
            vy=vy,
            rotation=rotation,
            spin=spin,
            mean_expected_score=best_mean,
        )


def save_result_plot(
    save_file_path: Path,
    kura_result_means_x: np.ndarray,
    transformer_result_means_x: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    indices = np.arange(len(kura_result_means_x))
    diff_scores = transformer_result_means_x - kura_result_means_x
    if len(indices) == 0:
        raise ValueError("kura_result_means_x and transformer_result_means_x must not be empty")
    x_max = len(indices) - 1

    figure, (ax_score, ax_diff) = plt.subplots(2, 1, figsize=(12, 8))

    x_tick_step = max(1, int(np.ceil((x_max + 1) / 20)))
    x_ticks = np.arange(0, x_max + 1, x_tick_step)
    if x_ticks[-1] != x_max:
        x_ticks = np.append(x_ticks, x_max)

    ax_score.plot(indices, kura_result_means_x, label="Kura", linewidth=1.0, alpha=0.5)
    ax_score.plot(indices, transformer_result_means_x, label="Transformer", linewidth=1.0, alpha=0.5)
    ax_score.set_ylabel("Result Mean Over X Runs")
    ax_score.set_title("Kura vs Transformer")
    ax_score.set_ylim(-1, 1)
    ax_score.grid(True, alpha=0.3)
    ax_score.legend()
    if x_max == 0:
        ax_score.set_xlim(-0.5, 0.5)
        ax_score.set_xticks([0])
    else:
        ax_score.set_xlim(0, x_max)
        ax_score.set_xticks(x_ticks)
    ax_score.set_xlabel("Position Index")

    ax_diff.plot(indices, diff_scores, color="tab:green", linewidth=1.0, alpha=0.6)
    ax_diff.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax_diff.set_xlabel("Position Index")
    ax_diff.set_ylabel("Result Mean Diff")
    ax_diff.set_title("Transformer - Kura")
    ax_diff.grid(True, alpha=0.3)
    if x_max == 0:
        ax_diff.set_xlim(-0.5, 0.5)
        ax_diff.set_xticks([0])
    else:
        ax_diff.set_xlim(0, x_max)
        ax_diff.set_xticks(x_ticks)

    figure.tight_layout()
    figure.savefig(save_file_path, dpi=150)
    plt.close(figure)


def save_result_json(
    save_file_path: Path,
    summary: dict,
    result_rows: list[dict],
) -> None:
    save_file_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "positions": result_rows,
            },
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )


def build_output_stem(
    target_end: int,
    target_shot: int,
    data_size: int,
    x_repeats: int,
) -> str:
    return (
        f"kura_vs_transformer_end{target_end}_shot{target_shot}"
        f"_winrate_datasize{data_size}_x{x_repeats}"
    )


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


def build_bucket_trial_summary(counts: np.ndarray, num_positions: int, x_repeats: int) -> dict:
    total_trials = int(num_positions * x_repeats)
    win_count = int(counts[RESULT_WIN_IDX])
    draw_count = int(counts[RESULT_DRAW_IDX])
    lose_count = int(counts[RESULT_LOSE_IDX])

    if total_trials == 0:
        return {
            "num_positions": int(num_positions),
            "total_trials": 0,
            "result_mean_over_all_trials": 0.0,
            "win_count": 0,
            "draw_count": 0,
            "lose_count": 0,
            "win_rate_over_all_trials": 0.0,
            "draw_rate_over_all_trials": 0.0,
            "lose_rate_over_all_trials": 0.0,
        }

    win_rate = win_count / total_trials
    draw_rate = draw_count / total_trials
    lose_rate = lose_count / total_trials

    return {
        "num_positions": int(num_positions),
        "total_trials": total_trials,
        "result_mean_over_all_trials": float(win_rate - lose_rate),
        "win_count": win_count,
        "draw_count": draw_count,
        "lose_count": lose_count,
        "win_rate_over_all_trials": float(win_rate),
        "draw_rate_over_all_trials": float(draw_rate),
        "lose_rate_over_all_trials": float(lose_rate),
    }


def evaluate_continuous_action(
    root_state: State,
    vx: float,
    vy: float,
    spin: int,
    x_repeats: int,
    root_view_team: int,
    root_view_score_diff_before_shot: int,
) -> tuple[np.ndarray, float, float, float, float]:
    counts = np.zeros(3, dtype=np.int32)
    for _ in range(x_repeats):
        final_state = simulator_step_continuous(root_state, vx, vy, spin)
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


def main(
    log_path: str | Path = "path/to/dcl2/records",
    save_path: str | Path = "path/to/save/data",
    transformer_model: str = "transformer-sl16-model-140000data.bin",
    kura_policy_model: str = KURA_POLICY_SHOT15_MODEL,
    data_size: int = 1000,
    target_end: int = 9,
    target_shot: int = 15,
    use_gpu: bool = False,
    X: int = 100,
) -> None:
    if target_shot != 15:
        raise ValueError("This Kura comparison implements only target_shot=15.")

    position_count = 0
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict] = []

    device = get_torch_device(use_gpu=use_gpu)
    transformer_model_path = NEWSL_DIR / "model" / transformer_model
    transformer_network = load_transformer_network(transformer_model_path, use_gpu=use_gpu)
    transformer_network.to(device)

    kura_searcher = KuraShot15Searcher(
        model_path=KURA_DIR / "model" / kura_policy_model,
        use_gpu=use_gpu,
        top_k=KURA_TOP_K,
        num_trials=KURA_NUM_TRIALS,
        rel_keep=KURA_REL_KEEP,
    )

    kura_result_means_x = [0.0 for _ in range(data_size)]
    transformer_result_means_x = [0.0 for _ in range(data_size)]
    kura_counts_x = np.zeros((data_size, 3), dtype=np.int32)
    transformer_counts_x = np.zeros((data_size, 3), dtype=np.int32)
    bucket_position_counts = {label: 0 for label in ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER}
    bucket_kura_counts = {
        label: np.zeros(3, dtype=np.int32) for label in ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER
    }
    bucket_transformer_counts = {
        label: np.zeros(3, dtype=np.int32) for label in ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER
    }

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

            kura_search_result = kura_searcher.select(
                stones=stones,
                end=end,
                shot=shot,
                shot_team=root_view_team,
            )
            (
                kura_counts,
                kura_result_mean,
                kura_win_rate,
                kura_draw_rate,
                kura_lose_rate,
            ) = evaluate_continuous_action(
                root_state=root_state,
                vx=kura_search_result.vx,
                vy=kura_search_result.vy,
                spin=kura_search_result.spin,
                x_repeats=X,
                root_view_team=root_view_team,
                root_view_score_diff_before_shot=root_view_score_diff_before_shot,
            )
            kura_counts_x[score_index] = kura_counts
            kura_result_means_x[score_index] = kura_result_mean

            transformer_root = set_root_state(
                network=None,
                stones=stones,
                score_diff=scorediff_for_team0,
                end=end,
                shot_index=shot,
                hammer_team=hammer,
                transformer_network=transformer_network,
                debug=False,
                use_transformer=True,
                transformer_target_end=(target_end,),
                transformer_target_shot=(target_shot,),
            )
            transformer_vx, transformer_vy, transformer_spin = mcts_search(
                root_state=transformer_root
            )
            transformer_spin = 1 if int(transformer_spin) == 1 else 0
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
            )
            transformer_counts_x[score_index] = transformer_counts
            transformer_result_means_x[score_index] = transformer_result_mean

            bucket_position_counts[root_view_score_diff_bucket] += 1
            bucket_kura_counts[root_view_score_diff_bucket] += kura_counts
            bucket_transformer_counts[root_view_score_diff_bucket] += transformer_counts

            result_rows.append(
                {
                    "position_index": score_index,
                    "log_name": one_log,
                    "end": int(end),
                    "shot": int(shot),
                    "hammer": int(hammer),
                    "root_view_team": root_view_team,
                    "score_diff_for_team0": int(scorediff_for_team0),
                    "root_view_score_diff_before_shot": int(root_view_score_diff_before_shot),
                    "root_view_score_diff_bucket": root_view_score_diff_bucket,
                    "kura_action_index": int(kura_search_result.action_index),
                    "kura_vx": float(kura_search_result.vx),
                    "kura_vy": float(kura_search_result.vy),
                    "kura_rotation": kura_search_result.rotation,
                    "kura_spin": int(kura_search_result.spin),
                    "kura_search_mean_expected_score": float(
                        kura_search_result.mean_expected_score
                    ),
                    "kura_result_mean_x": float(kura_result_mean),
                    "kura_win_count_x": int(kura_counts[RESULT_WIN_IDX]),
                    "kura_draw_count_x": int(kura_counts[RESULT_DRAW_IDX]),
                    "kura_lose_count_x": int(kura_counts[RESULT_LOSE_IDX]),
                    "kura_win_rate_x": kura_win_rate,
                    "kura_draw_rate_x": kura_draw_rate,
                    "kura_lose_rate_x": kura_lose_rate,
                    "transformer_vx": float(transformer_vx),
                    "transformer_vy": float(transformer_vy),
                    "transformer_spin": int(transformer_spin),
                    "transformer_result_mean_x": float(transformer_result_mean),
                    "transformer_win_count_x": int(transformer_counts[RESULT_WIN_IDX]),
                    "transformer_draw_count_x": int(transformer_counts[RESULT_DRAW_IDX]),
                    "transformer_lose_count_x": int(transformer_counts[RESULT_LOSE_IDX]),
                    "transformer_win_rate_x": transformer_win_rate,
                    "transformer_draw_rate_x": transformer_draw_rate,
                    "transformer_lose_rate_x": transformer_lose_rate,
                    "diff_result_mean_x_transformer_minus_kura": float(
                        transformer_result_mean - kura_result_mean
                    ),
                }
            )
            position_count += 1

    if position_count == 0:
        raise RuntimeError(
            f"No positions found for end={target_end}, shot={target_shot} in {log_path}"
        )

    kura_result_means_x_np = np.array(kura_result_means_x[:position_count], dtype=np.float32)
    transformer_result_means_x_np = np.array(
        transformer_result_means_x[:position_count],
        dtype=np.float32,
    )
    diff_result_means_x_np = transformer_result_means_x_np - kura_result_means_x_np
    kura_counts_np = kura_counts_x[:position_count]
    transformer_counts_np = transformer_counts_x[:position_count]
    kura_rates_x_np = kura_counts_np.astype(np.float32) / float(X)
    transformer_rates_x_np = transformer_counts_np.astype(np.float32) / float(X)
    total_trials_all_positions = position_count * X
    root_view_score_diff_bucket_summary = {}
    for bucket_label in ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER:
        root_view_score_diff_bucket_summary[bucket_label] = {
            "kura": build_bucket_trial_summary(
                bucket_kura_counts[bucket_label],
                bucket_position_counts[bucket_label],
                X,
            ),
            "transformer": build_bucket_trial_summary(
                bucket_transformer_counts[bucket_label],
                bucket_position_counts[bucket_label],
                X,
            ),
        }

    summary = {
        "num_positions": int(position_count),
        "target_end": int(target_end),
        "target_shot": int(target_shot),
        "execution_repeats_x": int(X),
        "kura_policy_model": str(kura_searcher.model_path),
        "kura_top_k": int(kura_searcher.top_k),
        "kura_num_trials": int(kura_searcher.num_trials),
        "kura_rel_keep": float(kura_searcher.rel_keep),
        "transformer_model": str(transformer_model_path),
        "log_size_result_mean_x_kura": float(np.mean(kura_result_means_x_np)),
        "log_size_result_mean_x_transformer": float(np.mean(transformer_result_means_x_np)),
        "log_size_diff_result_mean_x_transformer_minus_kura": float(
            np.mean(diff_result_means_x_np)
        ),
        "log_size_win_rate_x_kura": float(np.mean(kura_rates_x_np[:, RESULT_WIN_IDX])),
        "log_size_draw_rate_x_kura": float(np.mean(kura_rates_x_np[:, RESULT_DRAW_IDX])),
        "log_size_lose_rate_x_kura": float(np.mean(kura_rates_x_np[:, RESULT_LOSE_IDX])),
        "log_size_win_rate_x_transformer": float(
            np.mean(transformer_rates_x_np[:, RESULT_WIN_IDX])
        ),
        "log_size_draw_rate_x_transformer": float(
            np.mean(transformer_rates_x_np[:, RESULT_DRAW_IDX])
        ),
        "log_size_lose_rate_x_transformer": float(
            np.mean(transformer_rates_x_np[:, RESULT_LOSE_IDX])
        ),
        "total_trials_all_positions": int(total_trials_all_positions),
        "all_trials_win_count_kura": int(np.sum(kura_counts_np[:, RESULT_WIN_IDX])),
        "all_trials_draw_count_kura": int(np.sum(kura_counts_np[:, RESULT_DRAW_IDX])),
        "all_trials_lose_count_kura": int(np.sum(kura_counts_np[:, RESULT_LOSE_IDX])),
        "all_trials_win_count_transformer": int(
            np.sum(transformer_counts_np[:, RESULT_WIN_IDX])
        ),
        "all_trials_draw_count_transformer": int(
            np.sum(transformer_counts_np[:, RESULT_DRAW_IDX])
        ),
        "all_trials_lose_count_transformer": int(
            np.sum(transformer_counts_np[:, RESULT_LOSE_IDX])
        ),
        "all_trials_win_rate_kura": float(
            np.sum(kura_counts_np[:, RESULT_WIN_IDX]) / total_trials_all_positions
        ),
        "all_trials_draw_rate_kura": float(
            np.sum(kura_counts_np[:, RESULT_DRAW_IDX]) / total_trials_all_positions
        ),
        "all_trials_lose_rate_kura": float(
            np.sum(kura_counts_np[:, RESULT_LOSE_IDX]) / total_trials_all_positions
        ),
        "all_trials_win_rate_transformer": float(
            np.sum(transformer_counts_np[:, RESULT_WIN_IDX]) / total_trials_all_positions
        ),
        "all_trials_draw_rate_transformer": float(
            np.sum(transformer_counts_np[:, RESULT_DRAW_IDX]) / total_trials_all_positions
        ),
        "all_trials_lose_rate_transformer": float(
            np.sum(transformer_counts_np[:, RESULT_LOSE_IDX]) / total_trials_all_positions
        ),
        "transformer_better_by_result_mean_x_count": int(
            np.sum(diff_result_means_x_np > 0.0)
        ),
        "kura_better_by_result_mean_x_count": int(np.sum(diff_result_means_x_np < 0.0)),
        "tie_by_result_mean_x_count": int(np.sum(diff_result_means_x_np == 0.0)),
        "root_view_score_diff_bucket_order": list(ROOT_VIEW_SCORE_DIFF_BUCKET_ORDER),
        "root_view_score_diff_bucket_summary": root_view_score_diff_bucket_summary,
    }

    output_stem = build_output_stem(target_end, target_shot, data_size, X)
    json_path = save_dir / f"{output_stem}.json"
    png_path = save_dir / f"{output_stem}.png"

    save_result_json(json_path, summary, result_rows)
    save_result_plot(png_path, kura_result_means_x_np, transformer_result_means_x_np)

    print("")
    print(f"Saved summary to {json_path}")
    print(f"Saved plot to {png_path}")
    print(f"num_positions                : {summary['num_positions']}")
    print(f"execution_repeats_x         : {summary['execution_repeats_x']}")
    print(f"log_size result_mean_x Kura : {summary['log_size_result_mean_x_kura']:.6f}")
    print(f"log_size result_mean_x T    : {summary['log_size_result_mean_x_transformer']:.6f}")
    print(f"log_size diff (T - Kura)    : {summary['log_size_diff_result_mean_x_transformer_minus_kura']:.6f}")
    print(
        "log_size win/draw/lose Kura: "
        f"{summary['log_size_win_rate_x_kura']:.6f}, "
        f"{summary['log_size_draw_rate_x_kura']:.6f}, "
        f"{summary['log_size_lose_rate_x_kura']:.6f}"
    )
    print(
        "log_size win/draw/lose T   : "
        f"{summary['log_size_win_rate_x_transformer']:.6f}, "
        f"{summary['log_size_draw_rate_x_transformer']:.6f}, "
        f"{summary['log_size_lose_rate_x_transformer']:.6f}"
    )
    print(
        "all_trials win/draw/lose Kura: "
        f"{summary['all_trials_win_count_kura']}, "
        f"{summary['all_trials_draw_count_kura']}, "
        f"{summary['all_trials_lose_count_kura']}"
    )
    print(
        "all_trials win/draw/lose T   : "
        f"{summary['all_trials_win_count_transformer']}, "
        f"{summary['all_trials_draw_count_transformer']}, "
        f"{summary['all_trials_lose_count_transformer']}"
    )
    print(f"Transformer better by result_mean_x count : {summary['transformer_better_by_result_mean_x_count']}")
    print(f"Kura better by result_mean_x count        : {summary['kura_better_by_result_mean_x_count']}")
    print(f"Tie by result_mean_x count                : {summary['tie_by_result_mean_x_count']}")
    print("")
    print("root_view_score_diff_before_shot bucket summary (all trials in each bucket)")
    for bucket_label in summary["root_view_score_diff_bucket_order"]:
        bucket_summary = summary["root_view_score_diff_bucket_summary"][bucket_label]
        kura_bucket = bucket_summary["kura"]
        transformer_bucket = bucket_summary["transformer"]
        print(
            f"bucket {bucket_label:>4} positions={kura_bucket['num_positions']:4d} trials={kura_bucket['total_trials']:5d} "
            f"| Kura result_mean={kura_bucket['result_mean_over_all_trials']:.4f} "
            f"win/draw/lose={kura_bucket['win_rate_over_all_trials']:.4f},"
            f"{kura_bucket['draw_rate_over_all_trials']:.4f},"
            f"{kura_bucket['lose_rate_over_all_trials']:.4f} "
            f"| T result_mean={transformer_bucket['result_mean_over_all_trials']:.4f} "
            f"win/draw/lose={transformer_bucket['win_rate_over_all_trials']:.4f},"
            f"{transformer_bucket['draw_rate_over_all_trials']:.4f},"
            f"{transformer_bucket['lose_rate_over_all_trials']:.4f}"
        )


if __name__ == "__main__":
    main(
        log_path=NEWSL_DIR / "LearnLog" / "all",
        save_path=Path(__file__).resolve().parents[0] / "data",
        transformer_model="transformer-sl16-model-140000data.bin",
        kura_policy_model=KURA_POLICY_SHOT15_MODEL,
        data_size=10,
        target_end=9,
        target_shot=15,
        use_gpu=False,
        X=10,
    )
