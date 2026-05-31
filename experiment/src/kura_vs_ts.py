from __future__ import annotations

import contextlib
import gc
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
from mcts.search import mcts_search, set_root_state
from mcts.simulate import simulator_step_continuous
from mcts.state import State
from nn.utility import get_torch_device
from transformer.utility import load_transformer_network

# PNG 描画とコンソール表示は、保存済み JSON から再生成するスクリプトと共有する。
from experiment.src.report import render_report_from_records


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
    raise ValueError(f"shot_team は 0 または 1 である必要があります: {shot_team}")


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


def kura_add_noise_to_vector(
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
            raise ValueError("KuraShot15Searcher は shot == 15 の探索だけに対応しています。")

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
            raise RuntimeError("Kura の探索で候補手が生成されませんでした。")

        vx, vy, rotation, spin = best_move
        return KuraSearchResult(
            action_index=best_action_index,
            vx=vx,
            vy=vy,
            rotation=rotation,
            spin=spin,
            mean_expected_score=best_mean,
        )


def save_position_json(
    save_file_path: Path,
    position_result: dict,
) -> None:
    """1局面分の Kura / NewSL の結果を、1つの JSON ファイルとして保存する。"""

    save_file_path.parent.mkdir(parents=True, exist_ok=True)
    save_file_path.write_text(
        json.dumps(
            position_result,
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
    """end/shot/data_size/X から、PNG名とJSONディレクトリ名の共通stemを作る。"""

    return (
        f"kura_vs_transformer_end{target_end}_shot{target_shot}"
        f"_winrate_datasize{data_size}_x{x_repeats}"
    )


def release_position_memory(use_gpu: bool) -> None:
    """1局面の処理後に、不要な Python / CUDA メモリを解放する。"""

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


def build_bucket_trial_summary(counts: np.ndarray, num_positions: int, x_repeats: int) -> dict:
    """score_diff bucket 内の全試行結果を、表示用の summary 形式にまとめる。"""

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
    """1つの連続ショットを X 回評価し、勝ち/引き分け/負けの統計を返す。"""

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
    """Kura と NewSL のショットを比較し、局面別 JSON と集計 PNG を保存する。"""

    if target_shot != 15:
        raise ValueError("この Kura 比較実験は target_shot=15 のみに対応しています。")

    position_count = 0
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # PNG は experiment/data 直下、局面別 JSON は stem 名のディレクトリ配下に保存する。
    output_stem = build_output_stem(target_end, target_shot, data_size, X)
    json_dir = save_dir / output_stem
    if json_dir.exists() and any(json_dir.glob("*.json")):
        print(
            "既存の局面別 JSON が上書きされる可能性があります。"
            f"古いファイルは自動削除しません: {json_dir}"
        )
    json_dir.mkdir(parents=True, exist_ok=True)
    png_path = save_dir / f"{output_stem}.png"
    position_index_width = max(6, len(str(max(data_size - 1, 0))))

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

    # 各 JSON に共通で入れる実験条件。局面ごとの値とは分けて保存する。
    experiment_metadata = {
        "target_end": int(target_end),
        "target_shot": int(target_shot),
        "requested_data_size": int(data_size),
        "execution_repeats_x": int(X),
        "baseline_method_key": "kura",
        "baseline_method_label": "Kura",
        "newsl_method_key": "transformer",
        "newsl_method_label": "NewSL",
        "kura_policy_model": str(kura_searcher.model_path),
        "kura_top_k": int(kura_searcher.top_k),
        "kura_num_trials": int(kura_searcher.num_trials),
        "kura_rel_keep": float(kura_searcher.rel_keep),
        "newsl_model_type": "transformer",
        "newsl_transformer_model": str(transformer_model_path),
    }

    position_records: list[dict] = []

    # ログをランダム順に走査し、指定 end/shot に一致する局面を data_size 件まで集める。
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

            # ここから1局面分の探索と評価を行う。
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

            # NewSL 側は transformer network を使って MCTS の根を作り、1手を選ぶ。
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

            # PNG と summary に必要な軽量な値だけを保持する。
            diff_result_mean_x = float(transformer_result_mean - kura_result_mean)
            if diff_result_mean_x > 0.0:
                better_by_result_mean_x = "newsl"
            elif diff_result_mean_x < 0.0:
                better_by_result_mean_x = "baseline"
            else:
                better_by_result_mean_x = "tie"

            # メモリを圧迫しないよう、1局面の詳細は即座に JSON へ保存する。
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
                    "method_key": "kura",
                    "method_label": "Kura",
                    "action_index": int(kura_search_result.action_index),
                    "vx": float(kura_search_result.vx),
                    "vy": float(kura_search_result.vy),
                    "rotation": kura_search_result.rotation,
                    "spin": int(kura_search_result.spin),
                    "search_mean_expected_score": float(
                        kura_search_result.mean_expected_score
                    ),
                    "result_mean_x": float(kura_result_mean),
                    "win_count_x": int(kura_counts[RESULT_WIN_IDX]),
                    "draw_count_x": int(kura_counts[RESULT_DRAW_IDX]),
                    "lose_count_x": int(kura_counts[RESULT_LOSE_IDX]),
                    "win_rate_x": float(kura_win_rate),
                    "draw_rate_x": float(kura_draw_rate),
                    "lose_rate_x": float(kura_lose_rate),
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

            # 次の局面へ進む前に、大きい一時オブジェクトを明示的に解放する。
            del (
                position_result,
                position_json_path,
                root_state,
                kura_search_result,
                kura_counts,
                transformer_root,
                transformer_counts,
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
            f"{log_path} に end={target_end}, shot={target_shot} の局面が見つかりませんでした。"
        )

    render_report_from_records(json_dir, png_path, position_records)


if __name__ == "__main__":
    main(
        log_path=NEWSL_DIR / "LearnLog" / "jiritsu-vs-silicon",
        save_path=Path(__file__).resolve().parents[1] / "data",
        transformer_model="transformer-sl-9-15-model-05-26-AdamW-epoch50.bin",
        kura_policy_model=KURA_POLICY_SHOT15_MODEL,
        data_size=10000,
        target_end=9,
        target_shot=15,
        use_gpu=True,
        X=100,
    )
