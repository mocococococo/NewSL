from __future__ import annotations

import contextlib
import gc
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
NEWSL_DIR = REPO_ROOT / "NewSL"
KURA_DIR = REPO_ROOT / "Kuracurling_SHOT"
KURA_SIM_DIR = KURA_DIR / "mcts" / "FCV1Simulation" / "src"
KURA_SIM_RELEASE_DIR = KURA_SIM_DIR / "build" / "Release"

_SHARED_MODULE_PREFIXES = (
    "board",
    "common",
    "dc3client",
    "learning_param",
    "mcts",
    "nn",
    "other",
    "policy_shot",
    "shot",
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


# Kura modules use top-level names that overlap with NewSL modules.
_prepend_sys_path(KURA_DIR)
from other.other import index_to_shot as kura_index_to_shot
from other.other import load_checkpoint_to_model as kura_load_checkpoint_to_model
from other.other_2 import decompose_and_collate as kura_decompose_and_collate

_prepend_sys_path(KURA_SIM_RELEASE_DIR)
from simulator import StoneSimulator as KuraStoneSimulator

_remove_sys_path(KURA_SIM_RELEASE_DIR)
_remove_sys_path(KURA_DIR)
_purge_shared_modules()

_prepend_sys_path(NEWSL_DIR)
from common.translate_state import (
    convert_team_stoi,
    scores_to_scorediff_for_team0,
    stones_listdict_to_xy16,
)
from mcts.rollout import _end_score_diff_team0_minus_team1
from mcts.simulate import simulator_step_continuous
from mcts.state import State
from nn.utility import get_torch_device, load_network
from shot.search import shot_search, set_root_state as set_shot_root_state
from transformer.utility import load_transformer_network

from experiment.src.mini_match_report import print_report_from_records, save_position_json


RESULT_WIN_IDX = 0
RESULT_DRAW_IDX = 1
RESULT_LOSE_IDX = 2

KURA_ACTION_CLASSES = 32 * 32 * 2
KURA_POLICY_MODELS_BY_SHOT = {
    1: "checkpoint_2_policy_depth1_epoch_050.pt",
    2: "checkpoint_3_policy_depth1_epoch_050.pt",
    3: "checkpoint_4_policy_depth1_epoch_050.pt",
    4: "checkpoint_5_policy_depth1_epoch_050.pt",
    5: "checkpoint_6_policy_depth1_epoch_050.pt",
    6: "checkpoint_7_policy_depth1_epoch_050.pt",
    7: "checkpoint_8_policy_depth1_epoch_050.pt",
    8: "checkpoint_9_policy_depth1_epoch_050.pt",
    9: "checkpoint_10_policy_depth1_epoch_050.pt",
    10: "checkpoint_11_policy_depth1_epoch_034.pt",
    11: "checkpoint_12_policy_depth1_epoch_032.pt",
    12: "checkpoint_13_policy_depth1_epoch_050.pt",
    13: "checkpoint_14_policy_depth1_epoch_048.pt",
    14: "checkpoint_15_score_visit_18955_epoch_033.pt",
    15: "checkpoint_score_visit_17274_epoch_050.pt",
}
KURA_VALUE_MODELS_BY_SHOT = {
    2: "checkpoint_3_value_depth1_epoch_050.pt",
    3: "checkpoint_4_value_depth1_epoch_050.pt",
    4: "checkpoint_5_value_depth1_epoch_050.pt",
    5: "checkpoint_6_value_depth1_epoch_050.pt",
    6: "checkpoint_7_value_depth1_epoch_050.pt",
    7: "checkpoint_8_value_depth1_epoch_050.pt",
    8: "checkpoint_9_value_depth1_epoch_050.pt",
    9: "checkpoint_10_value_depth1_epoch_050.pt",
    10: "checkpoint_11_value_depth1_epoch_050.pt",
    11: "checkpoint_12_value_depth1_epoch_047.pt",
    12: "checkpoint_13_value_depth1_epoch_050.pt",
    13: "checkpoint_14_value_depth1_epoch_050.pt",
    14: "checkpoint_15_value_1229_epoch_050.pt",
    15: "checkpoint_16_value_1228_epoch_046.pt",
}
KURA_NONFINAL_TOP_K = 6
KURA_NONFINAL_NUM_TRIALS = 3
KURA_FINAL_TOP_K = 36
KURA_FINAL_NUM_TRIALS = 9
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


def _resolve_newsl_model(model: str | Path) -> Path:
    model_path = Path(model)
    if model_path.is_absolute():
        return model_path
    return NEWSL_DIR / "model" / model_path


def _resolve_kura_model(model: str | Path) -> Path:
    model_path = Path(model)
    if model_path.is_absolute():
        return model_path
    return KURA_DIR / "model" / model_path


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


def _shot_team_to_name(shot_team: int) -> str:
    if shot_team == 0:
        return "team0"
    if shot_team == 1:
        return "team1"
    raise ValueError(f"shot_team must be 0 or 1, got {shot_team}")


def _kura_rotation_to_string(rotation: object) -> str:
    value = getattr(rotation, "value", rotation)
    return "cw" if value == "cw" else "ccw"


def _rotation_string_to_newsl_spin(rotation: str) -> int:
    return 0 if rotation == "cw" else 1


def _state_to_raw_stones(state: State) -> list[Optional[dict[str, Any]]]:
    stones: list[Optional[dict[str, Any]]] = []
    for stone in state.stones:
        if stone is None:
            stones.append(None)
        else:
            x, y = stone
            stones.append({"position": {"x": float(x), "y": float(y)}})
    return stones


def _prepare_stones_for_kura(
    stones: list[Optional[dict[str, Any]]],
    shot: int,
    shot_team: int,
) -> tuple[list[Optional[dict[str, Any]]], str]:
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


def _kura_result_to_stone_list(
    result: np.ndarray,
    angle_default: float = 0.0,
    tol: float = 1e-8,
) -> list[Optional[dict[str, Any]]]:
    values = np.asarray(result, dtype=float)
    if values.ndim == 2:
        if values.shape[0] != 1:
            raise ValueError(f"unexpected simulator result shape: {values.shape}")
        values = values[0]
    if values.ndim != 1 or values.size % 2 != 0:
        raise ValueError(f"unexpected simulator result size: {values.size}")

    stones: list[Optional[dict[str, Any]]] = []
    for i in range(0, values.size, 2):
        x = float(values[i])
        y = float(values[i + 1])
        if abs(x) < tol and abs(y) < tol:
            stones.append(None)
        else:
            stones.append(
                {
                    "angle": float(angle_default),
                    "angular_velocity": 0.0,
                    "linear_velocity": {"x": 0.0, "y": 0.0},
                    "position": {"x": x, "y": y},
                }
            )
    return stones


def _kura_add_noise_to_vector(
    x: float,
    y: float,
    stddev_speed: float,
    stddev_angle: float,
) -> tuple[float, float]:
    magnitude = np.sqrt(x**2 + y**2)
    angle = np.arctan2(y, x)
    noisy_magnitude = magnitude + np.random.normal(0.0, stddev_speed)
    noisy_angle = angle + np.random.normal(0.0, stddev_angle)
    return (
        float(noisy_magnitude * np.cos(noisy_angle)),
        float(noisy_magnitude * np.sin(noisy_angle)),
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


def release_position_memory(use_gpu: bool) -> None:
    gc.collect()
    if use_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass(frozen=True)
class ShotAction:
    vx: float
    vy: float
    spin: int
    meta: dict[str, Any]


class MiniMatchPlayer:
    key: str
    label: str
    search_method: Optional[str]

    def select_action(self, state: State) -> ShotAction:
        raise NotImplementedError


class KuraPlayer(MiniMatchPlayer):
    def __init__(
        self,
        policy_models_by_shot: Optional[dict[int, str | Path]] = None,
        value_models_by_shot: Optional[dict[int, str | Path]] = None,
        use_gpu: bool = False,
        nonfinal_top_k: int = KURA_NONFINAL_TOP_K,
        nonfinal_num_trials: int = KURA_NONFINAL_NUM_TRIALS,
        final_top_k: int = KURA_FINAL_TOP_K,
        final_num_trials: int = KURA_FINAL_NUM_TRIALS,
        rel_keep: float = KURA_REL_KEEP,
    ) -> None:
        self.key = "kura"
        self.label = "Kura"
        self.search_method = "kura_original"
        self.device = _kura_device(use_gpu)
        policy_models_by_shot = policy_models_by_shot or KURA_POLICY_MODELS_BY_SHOT
        value_models_by_shot = value_models_by_shot or KURA_VALUE_MODELS_BY_SHOT
        self.policy_model_paths_by_shot = {
            int(shot): _resolve_kura_model(model_path)
            for shot, model_path in policy_models_by_shot.items()
        }
        self.value_model_paths_by_shot = {
            int(shot): _resolve_kura_model(model_path)
            for shot, model_path in value_models_by_shot.items()
        }
        self.policy_models_by_shot: dict[int, Any] = {
            shot: self._load_model(model_path)
            for shot, model_path in self.policy_model_paths_by_shot.items()
        }
        self.value_models_by_shot: dict[int, Any] = {
            shot: self._load_model(model_path)
            for shot, model_path in self.value_model_paths_by_shot.items()
        }
        self.nonfinal_top_k = int(nonfinal_top_k)
        self.nonfinal_num_trials = int(nonfinal_num_trials)
        self.final_top_k = int(final_top_k)
        self.final_num_trials = int(final_num_trials)
        self.rel_keep = float(rel_keep)
        self.score_values = torch.arange(
            -8,
            9,
            dtype=torch.float32,
            device=self.device,
        )

        with _pushd(KURA_SIM_DIR):
            self.stone_simulator = KuraStoneSimulator()

    def _load_model(self, model_path: Path) -> Any:
        model = kura_load_checkpoint_to_model(
            checkpoint_path=str(model_path),
            device=self.device,
            action_classes=KURA_ACTION_CLASSES,
        )
        model.eval()
        return model

    def _policy_model_for_shot(self, shot: int) -> Any:
        if shot not in self.policy_models_by_shot:
            raise ValueError(f"Kura policy model for shot {shot} is not configured.")
        return self.policy_models_by_shot[shot]

    def _value_model_for_shot(self, shot: int) -> Any:
        if shot not in self.value_models_by_shot:
            raise ValueError(f"Kura value model for shot {shot} is not configured.")
        return self.value_models_by_shot[shot]

    def _policy_probs(
        self,
        stones_list: list[Optional[dict[str, Any]]],
        end: int,
        shot: int,
        my_team: str,
    ) -> torch.Tensor:
        policy_model = self._policy_model_for_shot(shot)
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
            policy_logits, _, _ = policy_model(stones_tensor, game_feat, mask=mask)
            return F.softmax(policy_logits[:, -1, :], dim=-1)[0]

    def _value_expected_score(
        self,
        stones_list: list[Optional[dict[str, Any]]],
        end: int,
        shot: int,
        my_team: str,
    ) -> float:
        value_model = self._value_model_for_shot(shot)
        stones_tensor, game_feat, mask = kura_decompose_and_collate(
            stones=stones_list,
            scores=None,
            end=int(end),
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
            _, _, value_logits = value_model(stones_tensor, game_feat, mask=mask)
            value = F.softmax(value_logits[:, -1, :], dim=1)
            expected_score = (value * self.score_values).sum(dim=1)
        return float(expected_score.item())

    def _decode_action(self, action_index: int) -> tuple[float, float, str, int]:
        vx, vy, rotation = kura_index_to_shot(int(action_index))
        rotation_str = _kura_rotation_to_string(rotation)
        return float(vx), float(vy), rotation_str, _rotation_string_to_newsl_spin(rotation_str)

    def _step_once(
        self,
        stones_list: list[Optional[dict[str, Any]]],
        shot: int,
        vx: float,
        vy: float,
        rotation: str,
    ) -> np.ndarray:
        rand_x, rand_y = _kura_add_noise_to_vector(
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

    def select_action(self, state: State) -> ShotAction:
        if state.shot_index == 0:
            return ShotAction(
                vx=-0.121424,
                vy=2.30874,
                spin=0,
                meta={"source": "kura_fixed_first_shot"},
            )

        shot_team = state.to_move()
        stones_list, my_team = _prepare_stones_for_kura(
            _state_to_raw_stones(state),
            state.shot_index,
            shot_team,
        )
        probs = self._policy_probs(stones_list, state.end, state.shot_index, my_team)

        top_k = self.final_top_k if state.shot_index == 15 else self.nonfinal_top_k
        num_trials = (
            self.final_num_trials
            if state.shot_index == 15
            else self.nonfinal_num_trials
        )
        keep_threshold = probs.max() * self.rel_keep
        candidate_indices = torch.nonzero(probs >= keep_threshold, as_tuple=False).view(-1)
        if candidate_indices.numel() > top_k:
            candidate_probs = probs[candidate_indices]
            selected = torch.topk(candidate_probs, k=top_k, dim=-1).indices
            candidate_indices = candidate_indices[selected]

        best_action_index: Optional[int] = None
        best_mean: Optional[float] = None
        best_move: Optional[tuple[float, float, str, int]] = None
        enemy_team = "team1" if my_team == "team0" else "team0"

        for action_index in candidate_indices.detach().cpu().tolist():
            vx, vy, rotation, spin = self._decode_action(int(action_index))
            trial_scores = []
            for _ in range(num_trials):
                result = self._step_once(stones_list, state.shot_index, vx, vy, rotation)
                if state.shot_index == 15:
                    score_raw = _kura_calculate_score(result)
                    trial_scores.append(float(-score_raw))
                else:
                    stone_list = _kura_result_to_stone_list(result)
                    expected_score = self._value_expected_score(
                        stone_list,
                        state.end,
                        state.shot_index + 1,
                        enemy_team,
                    )
                    trial_scores.append(float(expected_score))

            if state.shot_index == 15:
                mean_expected = float(np.mean(trial_scores))
            else:
                mean_expected = float(-1.0 * np.mean(trial_scores))
                if state.shot_index == 12:
                    mean_expected = -1.0 * mean_expected
            if best_mean is None or mean_expected > best_mean:
                best_mean = mean_expected
                best_action_index = int(action_index)
                best_move = (vx, vy, rotation, spin)

        if best_action_index is None or best_mean is None or best_move is None:
            raise RuntimeError("Kura search produced no candidate action.")

        vx, vy, rotation, spin = best_move
        return ShotAction(
            vx=vx,
            vy=vy,
            spin=spin,
            meta={
                "action_index": int(best_action_index),
                "rotation": rotation,
                "mean_expected_score": float(best_mean),
                "policy_model": str(self.policy_model_paths_by_shot[state.shot_index]),
                "value_model": (
                    None
                    if state.shot_index == 15
                    else str(self.value_model_paths_by_shot[state.shot_index + 1])
                ),
            },
        )


class NewSLCNNPlayer(MiniMatchPlayer):
    def __init__(
        self,
        model_path: str | Path,
        use_gpu: bool,
        max_simulations: int,
    ) -> None:
        self.key = "cnn"
        self.label = "CNN"
        self.search_method = "shot"
        self.model_path = _resolve_newsl_model(model_path)
        self.max_simulations = int(max_simulations)
        device = get_torch_device(use_gpu=use_gpu)
        self.network = load_network(self.model_path, use_gpu=use_gpu)
        self.network.to(device)

    def select_action(self, state: State) -> ShotAction:
        root = set_shot_root_state(
            network=self.network,
            stones=_state_to_raw_stones(state),
            score_diff=state.score_diff,
            end=state.end,
            shot_index=state.shot_index,
            hammer_team=state.hammer_team,
            use_transformer=False,
        )
        vx, vy, spin = shot_search(
            root_state=root,
            max_simulations=self.max_simulations,
        )
        return ShotAction(
            vx=float(vx),
            vy=float(vy),
            spin=1 if int(spin) == 1 else 0,
            meta={"model": str(self.model_path)},
        )


class NewSLTransformerPlayer(MiniMatchPlayer):
    def __init__(
        self,
        models_by_shot: dict[int, str | Path],
        use_gpu: bool,
        max_simulations: int,
        target_end: int,
    ) -> None:
        self.key = "transformer"
        self.label = "Transformer"
        self.search_method = "shot"
        self.max_simulations = int(max_simulations)
        self.target_end = int(target_end)
        self.model_paths_by_shot = {
            int(shot): _resolve_newsl_model(model_path)
            for shot, model_path in models_by_shot.items()
        }
        self.networks_by_shot = {
            shot: load_transformer_network(model_path, use_gpu=use_gpu)
            for shot, model_path in self.model_paths_by_shot.items()
        }

    def select_action(self, state: State) -> ShotAction:
        if state.shot_index not in self.networks_by_shot:
            raise ValueError(
                f"Transformer model for shot {state.shot_index} is not configured."
            )

        root = set_shot_root_state(
            network=None,
            stones=_state_to_raw_stones(state),
            score_diff=state.score_diff,
            end=state.end,
            shot_index=state.shot_index,
            hammer_team=state.hammer_team,
            transformer_network=self.networks_by_shot,
            use_transformer=True,
            transformer_target_end=(self.target_end,),
            transformer_target_shot=tuple(sorted(self.networks_by_shot)),
        )
        vx, vy, spin = shot_search(
            root_state=root,
            max_simulations=self.max_simulations,
        )
        return ShotAction(
            vx=float(vx),
            vy=float(vy),
            spin=1 if int(spin) == 1 else 0,
            meta={
                "models_by_shot": {
                    int(shot): str(path)
                    for shot, path in self.model_paths_by_shot.items()
                }
            },
        )


def build_player(
    kind: str,
    *,
    use_gpu: bool,
    target_end: int,
    cnn_model: str | Path,
    transformer_models_by_shot: dict[int, str | Path],
    kura_policy_models_by_shot: dict[int, str | Path],
    kura_value_models_by_shot: dict[int, str | Path],
    max_simulations: int,
) -> MiniMatchPlayer:
    normalized = kind.strip().lower()
    if normalized == "kura":
        return KuraPlayer(
            kura_policy_models_by_shot,
            kura_value_models_by_shot,
            use_gpu=use_gpu,
        )
    if normalized == "cnn":
        return NewSLCNNPlayer(cnn_model, use_gpu=use_gpu, max_simulations=max_simulations)
    if normalized == "transformer":
        return NewSLTransformerPlayer(
            transformer_models_by_shot,
            use_gpu=use_gpu,
            max_simulations=max_simulations,
            target_end=target_end,
        )
    raise ValueError(f"Unknown player kind: {kind}")


def print_mini_match_header(
    first_player: MiniMatchPlayer,
    second_player: MiniMatchPlayer,
    repeat_index: int,
    x_repeats: int,
) -> None:
    print("")
    print(f"{first_player.label} vs {second_player.label} ({repeat_index + 1}/{x_repeats})")


def print_search_header(state: State, player: MiniMatchPlayer) -> None:
    print(f"end = {state.end}, shot = {state.shot_index}, {player.label} search")


def print_selected_action(player: MiniMatchPlayer, action: ShotAction) -> None:
    print(
        f"{player.label} selected "
        f"vx={action.vx:.6f}, vy={action.vy:.6f}, spin={action.spin}"
    )


def run_suffix_game(
    root_state: State,
    player_a: MiniMatchPlayer,
    player_b: MiniMatchPlayer,
    start_with_a: bool,
    repeat_index: int,
    x_repeats: int,
) -> tuple[State, list[dict[str, Any]]]:
    state = root_state
    action_log: list[dict[str, Any]] = []
    start_shot = root_state.shot_index
    first_player = player_a if start_with_a else player_b
    second_player = player_b if start_with_a else player_a
    print_mini_match_header(first_player, second_player, repeat_index, x_repeats)

    while not state.is_end_terminal():
        offset = state.shot_index - start_shot
        use_a = (offset % 2 == 0) if start_with_a else (offset % 2 == 1)
        player = player_a if use_a else player_b
        print_search_header(state, player)
        action = player.select_action(state)
        print_selected_action(player, action)
        action_log.append(
            {
                "shot": int(state.shot_index),
                "player_key": player.key,
                "player_label": player.label,
                "vx": float(action.vx),
                "vy": float(action.vy),
                "spin": int(action.spin),
                "meta": action.meta,
            }
        )
        state = simulator_step_continuous(
            state,
            action.vx,
            action.vy,
            action.spin,
        )

    return state, action_log


def evaluate_start_pattern(
    root_state: State,
    player_a: MiniMatchPlayer,
    player_b: MiniMatchPlayer,
    start_with_a: bool,
    x_repeats: int,
    root_view_team: int,
    root_view_score_diff_before_shot: int,
) -> tuple[np.ndarray, float, float, float, float, list[dict[str, Any]]]:
    counts = np.zeros(3, dtype=np.int32)
    sample_action_log: list[dict[str, Any]] = []

    for repeat_index in range(x_repeats):
        final_state, action_log = run_suffix_game(
            root_state,
            player_a,
            player_b,
            start_with_a=start_with_a,
            repeat_index=repeat_index,
            x_repeats=x_repeats,
        )
        if repeat_index == 0:
            sample_action_log = action_log

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
    return counts, result_mean, win_rate, draw_rate, lose_rate, sample_action_log


def build_output_stem(
    player_a: str,
    player_b: str,
    target_end: int,
    target_shot: int,
    data_size: int,
    x_repeats: int,
) -> str:
    return (
        f"mini_match_{player_a.lower()}_vs_{player_b.lower()}"
        f"_end{target_end}_shot{target_shot}"
        f"_datasize{data_size}_x{x_repeats}"
    )


def main(
    log_path: str | Path = NEWSL_DIR / "LearnLog" / "all",
    save_path: str | Path = Path(__file__).resolve().parents[1] / "data",
    player_a_kind: str = "Kura",
    player_b_kind: str = "Transformer",
    target_end: int = 9,
    target_shot: int = 14,
    data_size: int = 1000,
    X: int = 1,
    use_gpu: bool = True,
    max_simulations: int = 1000,
    cnn_model: str | Path = "js20000CP-32-9-LeaRate1000-vx32-vy25-batchsize1024.bin",
    transformer_models_by_shot: Optional[dict[int, str | Path]] = None,
    kura_policy_models_by_shot: Optional[dict[int, str | Path]] = None,
    kura_value_models_by_shot: Optional[dict[int, str | Path]] = None,
    shuffle_seed: Optional[int] = 12345,
) -> None:
    if not (0 <= target_shot <= 15):
        raise ValueError(f"target_shot must be in [0, 15], got {target_shot}")
    if X <= 0:
        raise ValueError(f"X must be positive, got {X}")

    if transformer_models_by_shot is None:
        transformer_models_by_shot = {
            14: "transformer-sl-9-14-model-06-04-adamw-epoch50-shot.bin",
            15: "transformer-sl-9-15-model-06-02-adamw-epoch50-shot.bin",
        }
    if kura_policy_models_by_shot is None:
        kura_policy_models_by_shot = KURA_POLICY_MODELS_BY_SHOT
    if kura_value_models_by_shot is None:
        kura_value_models_by_shot = KURA_VALUE_MODELS_BY_SHOT

    needed_transformer_shots = tuple(range(target_shot, 16))
    if any(
        kind.strip().lower() == "transformer"
        for kind in (player_a_kind, player_b_kind)
    ):
        missing = [
            shot for shot in needed_transformer_shots if shot not in transformer_models_by_shot
        ]
        if missing:
            raise ValueError(f"Missing transformer models for shots: {missing}")

    if any(kind.strip().lower() == "kura" for kind in (player_a_kind, player_b_kind)):
        needed_kura_policy_shots = tuple(
            shot for shot in range(target_shot, 16) if shot != 0
        )
        missing_policy = [
            shot
            for shot in needed_kura_policy_shots
            if shot not in kura_policy_models_by_shot
        ]
        needed_kura_value_shots = tuple(range(max(target_shot + 1, 2), 16))
        missing_value = [
            shot
            for shot in needed_kura_value_shots
            if shot not in kura_value_models_by_shot
        ]
        if missing_policy:
            raise ValueError(f"Missing Kura policy models for shots: {missing_policy}")
        if missing_value:
            raise ValueError(f"Missing Kura value models for shots: {missing_value}")

    player_a = build_player(
        player_a_kind,
        use_gpu=use_gpu,
        target_end=target_end,
        cnn_model=cnn_model,
        transformer_models_by_shot=transformer_models_by_shot,
        kura_policy_models_by_shot=kura_policy_models_by_shot,
        kura_value_models_by_shot=kura_value_models_by_shot,
        max_simulations=max_simulations,
    )
    player_b = build_player(
        player_b_kind,
        use_gpu=use_gpu,
        target_end=target_end,
        cnn_model=cnn_model,
        transformer_models_by_shot=transformer_models_by_shot,
        kura_policy_models_by_shot=kura_policy_models_by_shot,
        kura_value_models_by_shot=kura_value_models_by_shot,
        max_simulations=max_simulations,
    )

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_stem = build_output_stem(
        player_a.key,
        player_b.key,
        target_end,
        target_shot,
        data_size,
        X,
    )
    json_dir = save_dir / output_stem
    json_dir.mkdir(parents=True, exist_ok=True)
    position_index_width = max(6, len(str(max(data_size - 1, 0))))

    experiment_metadata = {
        "experiment_type": "mini_match",
        "target_end": int(target_end),
        "target_shot": int(target_shot),
        "requested_data_size": int(data_size),
        "execution_repeats_x": int(X),
        "player_a_start_method_key": f"{player_a.key}_start",
        "player_a_start_method_label": f"{player_a.label}-start",
        "player_b_start_method_key": f"{player_b.key}_start",
        "player_b_start_method_label": f"{player_b.label}-start",
        "player_a_kind": player_a_kind,
        "player_b_kind": player_b_kind,
        "player_a_label": player_a.label,
        "player_b_label": player_b.label,
        "player_a_search_method": player_a.search_method,
        "player_b_search_method": player_b.search_method,
        "max_simulations": int(max_simulations),
        "cnn_model": str(_resolve_newsl_model(cnn_model)),
        "transformer_models_by_shot": {
            int(shot): str(_resolve_newsl_model(model_path))
            for shot, model_path in transformer_models_by_shot.items()
        },
        "kura_policy_models_by_shot": {
            int(shot): str(_resolve_kura_model(model_path))
            for shot, model_path in kura_policy_models_by_shot.items()
        },
        "kura_value_models_by_shot": {
            int(shot): str(_resolve_kura_model(model_path))
            for shot, model_path in kura_value_models_by_shot.items()
        },
    }

    log_files = os.listdir(log_path)
    rng = random.Random(shuffle_seed)
    shuffled_log_files = rng.sample(log_files, len(log_files))

    position_records: list[dict[str, Any]] = []
    position_count = 0

    for one_log in shuffled_log_files:
        if position_count >= data_size:
            break
        if not os.path.isdir(os.path.join(log_path, one_log)):
            continue

        dcl2_path = os.path.join(log_path, one_log, "game.dcl2")
        if not os.path.exists(dcl2_path):
            continue
        with open(dcl2_path, encoding="utf-8") as dclfile:
            dcl2_data = dclfile.readlines()

        for line_index in range(9, len(dcl2_data) - 2, 2):
            if position_count >= data_size:
                break
            try:
                dcl2_log = json.loads(dcl2_data[line_index])["log"]
                dcl2_state = dcl2_log["state"]
                stones = dcl2_state["stones"]["team0"] + dcl2_state["stones"]["team1"]
                scores_for_scorediff = dcl2_state["scores"]
                end = int(dcl2_state["end"])
                shot = int(dcl2_state["shot"])
                hammer = convert_team_stoi(dcl2_state["hammer"])
                if not ((end == target_end) and (shot == target_shot)):
                    continue
            except KeyError:
                continue

            scorediff_for_team0 = scores_to_scorediff_for_team0(scores_for_scorediff)
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

            print(
                f"Processing position {position_count + 1}: "
                f"log={one_log}, end={end}, shot={shot}"
            )
            (
                a_start_counts,
                a_start_result_mean,
                a_start_win_rate,
                a_start_draw_rate,
                a_start_lose_rate,
                a_start_action_log,
            ) = evaluate_start_pattern(
                root_state,
                player_a,
                player_b,
                start_with_a=True,
                x_repeats=X,
                root_view_team=root_view_team,
                root_view_score_diff_before_shot=root_view_score_diff_before_shot,
            )
            (
                b_start_counts,
                b_start_result_mean,
                b_start_win_rate,
                b_start_draw_rate,
                b_start_lose_rate,
                b_start_action_log,
            ) = evaluate_start_pattern(
                root_state,
                player_a,
                player_b,
                start_with_a=False,
                x_repeats=X,
                root_view_team=root_view_team,
                root_view_score_diff_before_shot=root_view_score_diff_before_shot,
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
                    "position_index": int(position_count),
                    "log_name": one_log,
                    "line_index": int(line_index),
                    "end": int(end),
                    "shot": int(shot),
                    "hammer_team": int(hammer),
                    "score_diff_for_team0": int(scorediff_for_team0),
                    "root_view_team": int(root_view_team),
                    "root_view_score_diff_before_shot": int(
                        root_view_score_diff_before_shot
                    ),
                    "root_view_score_diff_bucket": root_view_score_diff_bucket,
                },
                "player_a_start": {
                    "method_key": f"{player_a.key}_start",
                    "method_label": f"{player_a.label}-start",
                    "result_mean_x": float(a_start_result_mean),
                    "win_count_x": int(a_start_counts[RESULT_WIN_IDX]),
                    "draw_count_x": int(a_start_counts[RESULT_DRAW_IDX]),
                    "lose_count_x": int(a_start_counts[RESULT_LOSE_IDX]),
                    "win_rate_x": float(a_start_win_rate),
                    "draw_rate_x": float(a_start_draw_rate),
                    "lose_rate_x": float(a_start_lose_rate),
                    "sample_action_log": a_start_action_log,
                },
                "player_b_start": {
                    "method_key": f"{player_b.key}_start",
                    "method_label": f"{player_b.label}-start",
                    "result_mean_x": float(b_start_result_mean),
                    "win_count_x": int(b_start_counts[RESULT_WIN_IDX]),
                    "draw_count_x": int(b_start_counts[RESULT_DRAW_IDX]),
                    "lose_count_x": int(b_start_counts[RESULT_LOSE_IDX]),
                    "win_rate_x": float(b_start_win_rate),
                    "draw_rate_x": float(b_start_draw_rate),
                    "lose_rate_x": float(b_start_lose_rate),
                    "sample_action_log": b_start_action_log,
                },
                "comparison": {
                    "diff_result_mean_x_player_b_start_minus_player_a_start": diff_result_mean_x,
                    "better_by_result_mean_x": better_by_result_mean_x,
                },
            }

            position_json_path = json_dir / f"{position_count:0{position_index_width}d}.json"
            save_position_json(position_json_path, position_result)
            position_records.append(position_result)
            position_count += 1
            release_position_memory(use_gpu)

        del dcl2_data
        release_position_memory(use_gpu)

    if position_count == 0:
        raise RuntimeError(
            f"No positions found in {log_path} for end={target_end}, shot={target_shot}."
        )

    print_report_from_records(json_dir, position_records)


if __name__ == "__main__":
    main(
        log_path=NEWSL_DIR / "LearnLog" / "all",
        save_path=Path(__file__).resolve().parents[1] / "data",
        player_a_kind="Kura",
        player_b_kind="Transformer",
        target_end=9,
        target_shot=14,
        data_size=1000,
        X=1,
        use_gpu=True,
        max_simulations=1022,
        cnn_model="js20000CP-32-9-LeaRate1000-vx32-vy25-batchsize1024.bin",
        transformer_models_by_shot={
            14: "transformer-sl-9-14-model-06-05-adamw-epoch50-shot.bin",
            15: "transformer-sl-9-15-model-06-02-adamw-epoch50-shot.bin",
        },
        kura_policy_models_by_shot=KURA_POLICY_MODELS_BY_SHOT,
        kura_value_models_by_shot=KURA_VALUE_MODELS_BY_SHOT,
        shuffle_seed=12345,
    )
