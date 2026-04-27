"""

"""
import numpy as np
import copy
import os
import sys
import json
import random
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.translate_state import convert_team_stoi, scores_to_scorediff_for_team0
from nn.utility import load_network, get_torch_device
from transformer.feature import generate_input_features
from transformer.utility import load_transformer_network
from transformer.params import (
    DEFAULT_TRANSFORMER_CONFIG,
    GAME_FEAT_DIM,
    MAX_STONES,
    STONE_FEAT_DIM,
)
from mcts.search import mcts_search, set_root_state
from mcts.simulate import simulator_step
from mcts.rollout import _end_score_diff_team0_minus_team1
from board.constant import VX_MIN, VX_MAX, VY_MIN, VY_MAX, VY_SHEET_MAX, VX_SIZE, VY_SIZE
from learning_param import BATCH_SIZE, DATA_SET_SIZE

N_ACTIONS = DEFAULT_TRANSFORMER_CONFIG.action_dim
N_VALUE_CLASSES = DEFAULT_TRANSFORMER_CONFIG.value_dim

RESULT_WIN_IDX = 0
RESULT_DRAW_IDX = 1
RESULT_LOSE_IDX = 2


def encode_action(vx: float, vy: float, spin: int) -> int:
    board_len = VX_SIZE * VY_SIZE

    dvx = (VX_MAX - VX_MIN) / VX_SIZE
    vxi = int(round((vx - VX_MIN) / dvx - 0.5))
    vxi = min(max(vxi, 0), VX_SIZE - 1)

    if vy <= VY_SHEET_MAX:
        dvy = (VY_SHEET_MAX - VY_MIN) / (VY_SIZE - 5)
        vyi = int(round((vy - VY_MIN) / dvy - 0.5))
        vyi = min(max(vyi, 0), (VY_SIZE - 5) - 1)
    else:
        dvy_extra = (VY_MAX - VY_SHEET_MAX) / 5
        vyi = (VY_SIZE - 5) + int(round((vy - VY_SHEET_MAX) / dvy_extra - 0.5))
        vyi = min(max(vyi, VY_SIZE - 5), VY_SIZE - 1)

    action = vyi * VX_SIZE + vxi
    if spin == 1:
        action += board_len
    return action


def save_result_plot(
    save_file_path: Path,
    cnn_result_means_x: np.ndarray,
    transformer_result_means_x: np.ndarray,
) -> None:
    """局面ごとの result_mean_x と差分を png で保存する。"""

    import matplotlib.pyplot as plt

    indices = np.arange(len(cnn_result_means_x))
    diff_scores = transformer_result_means_x - cnn_result_means_x
    if len(indices) == 0:
        raise ValueError("cnn_result_means_x and transformer_result_means_x must not be empty")
    x_max = len(indices) - 1

    figure, (ax_score, ax_diff) = plt.subplots(2, 1, figsize=(12, 8))

    x_tick_step = max(1, int(np.ceil((x_max + 1) / 20)))
    x_ticks = np.arange(0, x_max + 1, x_tick_step)
    if x_ticks[-1] != x_max:
        x_ticks = np.append(x_ticks, x_max)

    ax_score.plot(indices, cnn_result_means_x, label="CNN", linewidth=1.0, alpha=0.5)
    ax_score.plot(indices, transformer_result_means_x, label="Transformer", linewidth=1.0, alpha=0.5)
    ax_score.set_ylabel("Result Mean Over X Runs")
    ax_score.set_title("CNN vs Transformer")
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
    ax_diff.set_title("Transformer - CNN")
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
    """集計結果を json で保存する。"""

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
    """Build a filename stem that is unique for end/shot/data_size/X."""

    return (
        f"cnn_vs_transformer_end{target_end}_shot{target_shot}"
        f"_winrate_datasize{data_size}_x{x_repeats}"
    )


def main(
    log_path: str = "path/to/dcl2/records",
    save_path: str = "path/to/save/data",
    cnn_model: str = "path/to/cnn/model",
    transformer_model: str = "path/to/transformer/model",
    data_size: int = 1000,
    target_end: int = 9,
    target_shot: int = 15,
    use_gpu: bool = True,
    X: int = 100,
) -> None:
    position_count = 0
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    result_rows = []
    
    device = get_torch_device(use_gpu=use_gpu)
    cnn_model_path = Path(__file__).resolve().parents[1] / "model" / cnn_model
    cnn_network = load_network(cnn_model_path, use_gpu=use_gpu)
    cnn_network.to(device)
    
    transformer_model_path = Path(__file__).resolve().parents[1] / "model" / transformer_model
    transformer_network = load_transformer_network(transformer_model_path, use_gpu=use_gpu)
    transformer_network.to(device)
    
    cnn_result_means_x = [0.0 for _ in range(data_size)]
    transformer_result_means_x = [0.0 for _ in range(data_size)]
    cnn_counts_x = np.zeros((data_size, 3), dtype=np.int32)
    transformer_counts_x = np.zeros((data_size, 3), dtype=np.int32)

    log_files = os.listdir(log_path)
    for one_log in random.sample(log_files, len(log_files)):
        if not os.path.isdir(os.path.join(log_path, one_log)):
            continue
        if position_count >= data_size:
            break
        
        dcl2_path = os.path.join(log_path, one_log, "game.dcl2")
        if not os.path.exists(dcl2_path):
            continue
        with open(dcl2_path) as dclfile:
            dcl2_data = dclfile.readlines()
        try:
            if json.loads(dcl2_data[-2])['log']['state'] :
                print(f"Processing log: {one_log}, total processed logs: {position_count + 1}")
        except KeyError:
            continue
        
        for i in range(9, len(dcl2_data)-2, 2):
            if position_count >= data_size:
                break
            try:
                dcl2_state = json.loads(dcl2_data[i])['log']['state']
                stones = dcl2_state['stones']['team0'] + dcl2_state['stones']['team1']
                scores_for_scorediff = dcl2_state['scores']
                end = dcl2_state['end']
                scorediff_for_team0 = scores_to_scorediff_for_team0(scores_for_scorediff)
                # print(f"scores: {scores}, end: {end}, scorediff_for_team0: {scorediff_for_team0}")
                shot = dcl2_state['shot']
                hammer = convert_team_stoi(dcl2_state['hammer'])
                if not ((end == target_end) and (shot == target_shot)):
                    continue
            except KeyError:
                continue

            # CNN モデルで検証
            score_index = position_count
            root = set_root_state(
                network=cnn_network,
                stones=stones,
                score_diff=scorediff_for_team0,
                end=end,
                shot_index=shot,
                hammer_team=hammer,
                transformer_network=None,
                debug=False,
                use_transformer=False,
            )
            vx, vy, spin = mcts_search(root_state=root)
            encoded_action = encode_action(vx, vy, spin)
            # X回実行して win / draw / lose の件数を集計する
            for i in range(X):
                final_state = simulator_step(root, encoded_action)
                raw_score = _end_score_diff_team0_minus_team1(final_state.stones)
                score_from_root_view = raw_score if root.to_move() == 0 else -raw_score
                total_score_from_root_view = scorediff_for_team0 if root.to_move() == 0 else -scorediff_for_team0
                total_score_from_root_view += score_from_root_view
                
                # 勝敗判定
                if total_score_from_root_view > 0:
                    cnn_counts_x[score_index][RESULT_WIN_IDX] += 1
                elif total_score_from_root_view < 0:
                    cnn_counts_x[score_index][RESULT_LOSE_IDX] += 1
                else:
                    cnn_counts_x[score_index][RESULT_DRAW_IDX] += 1
            cnn_win_rate_x = float(cnn_counts_x[score_index][RESULT_WIN_IDX] / X)
            cnn_draw_rate_x = float(cnn_counts_x[score_index][RESULT_DRAW_IDX] / X)
            cnn_lose_rate_x = float(cnn_counts_x[score_index][RESULT_LOSE_IDX] / X)
            cnn_result_means_x[score_index] = cnn_win_rate_x - cnn_lose_rate_x
            
            # Transformer モデルで検証
            root = set_root_state(
                network=cnn_network,
                stones=stones,
                score_diff=scorediff_for_team0,
                end=end,
                shot_index=shot,
                hammer_team=hammer,
                transformer_network=transformer_network,
                debug=False,
                use_transformer=True,
            )
            vx, vy, spin = mcts_search(root_state=root)
            encoded_action = encode_action(vx, vy, spin)
            # X回実行して win / draw / lose の件数を集計する
            for i in range(X):
                final_state = simulator_step(root, encoded_action)
                raw_score = _end_score_diff_team0_minus_team1(final_state.stones)
                score_from_root_view = raw_score if root.to_move() == 0 else -raw_score
                total_score_from_root_view = scorediff_for_team0 if root.to_move() == 0 else -scorediff_for_team0
                total_score_from_root_view += score_from_root_view
                
                # 勝敗判定
                if total_score_from_root_view > 0:
                    transformer_counts_x[score_index][RESULT_WIN_IDX] += 1
                elif total_score_from_root_view < 0:
                    transformer_counts_x[score_index][RESULT_LOSE_IDX] += 1
                else:
                    transformer_counts_x[score_index][RESULT_DRAW_IDX] += 1
            transformer_win_rate_x = float(transformer_counts_x[score_index][RESULT_WIN_IDX] / X)
            transformer_draw_rate_x = float(transformer_counts_x[score_index][RESULT_DRAW_IDX] / X)
            transformer_lose_rate_x = float(transformer_counts_x[score_index][RESULT_LOSE_IDX] / X)
            transformer_result_means_x[score_index] = transformer_win_rate_x - transformer_lose_rate_x
            

            # 結果を保存するための行を追加
            result_rows.append(
                {
                    "position_index": score_index,
                    "log_name": one_log,
                    "end": int(end),
                    "shot": int(shot),
                    "hammer": int(hammer),
                    "score_diff_for_team0": int(scorediff_for_team0),
                    "cnn_result_mean_x": float(cnn_result_means_x[score_index]),
                    "cnn_win_count_x": int(cnn_counts_x[score_index][RESULT_WIN_IDX]),
                    "cnn_draw_count_x": int(cnn_counts_x[score_index][RESULT_DRAW_IDX]),
                    "cnn_lose_count_x": int(cnn_counts_x[score_index][RESULT_LOSE_IDX]),
                    "cnn_win_rate_x": cnn_win_rate_x,
                    "cnn_draw_rate_x": cnn_draw_rate_x,
                    "cnn_lose_rate_x": cnn_lose_rate_x,
                    "transformer_result_mean_x": float(transformer_result_means_x[score_index]),
                    "transformer_win_count_x": int(transformer_counts_x[score_index][RESULT_WIN_IDX]),
                    "transformer_draw_count_x": int(transformer_counts_x[score_index][RESULT_DRAW_IDX]),
                    "transformer_lose_count_x": int(transformer_counts_x[score_index][RESULT_LOSE_IDX]),
                    "transformer_win_rate_x": transformer_win_rate_x,
                    "transformer_draw_rate_x": transformer_draw_rate_x,
                    "transformer_lose_rate_x": transformer_lose_rate_x,
                    "diff_result_mean_x": float(transformer_result_means_x[score_index] - cnn_result_means_x[score_index]),
                }
            )
            position_count += 1

    if position_count == 0:
        raise RuntimeError(
            f"No positions found for end={target_end}, shot={target_shot} in {log_path}"
        )

    cnn_result_means_x_np = np.array(cnn_result_means_x[:position_count], dtype=np.float32)
    transformer_result_means_x_np = np.array(transformer_result_means_x[:position_count], dtype=np.float32)
    diff_result_means_x_np = transformer_result_means_x_np - cnn_result_means_x_np
    cnn_counts_np = cnn_counts_x[:position_count]
    transformer_counts_np = transformer_counts_x[:position_count]
    cnn_rates_x_np = cnn_counts_np.astype(np.float32) / float(X)
    transformer_rates_x_np = transformer_counts_np.astype(np.float32) / float(X)
    total_trials_all_positions = position_count * X

    summary = {
        "num_positions": int(position_count),
        "target_end": int(target_end),
        "target_shot": int(target_shot),
        "execution_repeats_x": int(X),
        "cnn_model": str(cnn_model_path),
        "transformer_model": str(transformer_model_path),
        "log_size_result_mean_x_cnn": float(np.mean(cnn_result_means_x_np)),
        "log_size_result_mean_x_transformer": float(np.mean(transformer_result_means_x_np)),
        "log_size_diff_result_mean_x_transformer_minus_cnn": float(np.mean(diff_result_means_x_np)),
        "log_size_win_rate_x_cnn": float(np.mean(cnn_rates_x_np[:, RESULT_WIN_IDX])),
        "log_size_draw_rate_x_cnn": float(np.mean(cnn_rates_x_np[:, RESULT_DRAW_IDX])),
        "log_size_lose_rate_x_cnn": float(np.mean(cnn_rates_x_np[:, RESULT_LOSE_IDX])),
        "log_size_win_rate_x_transformer": float(np.mean(transformer_rates_x_np[:, RESULT_WIN_IDX])),
        "log_size_draw_rate_x_transformer": float(np.mean(transformer_rates_x_np[:, RESULT_DRAW_IDX])),
        "log_size_lose_rate_x_transformer": float(np.mean(transformer_rates_x_np[:, RESULT_LOSE_IDX])),
        "total_trials_all_positions": int(total_trials_all_positions),
        "all_trials_win_count_cnn": int(np.sum(cnn_counts_np[:, RESULT_WIN_IDX])),
        "all_trials_draw_count_cnn": int(np.sum(cnn_counts_np[:, RESULT_DRAW_IDX])),
        "all_trials_lose_count_cnn": int(np.sum(cnn_counts_np[:, RESULT_LOSE_IDX])),
        "all_trials_win_count_transformer": int(np.sum(transformer_counts_np[:, RESULT_WIN_IDX])),
        "all_trials_draw_count_transformer": int(np.sum(transformer_counts_np[:, RESULT_DRAW_IDX])),
        "all_trials_lose_count_transformer": int(np.sum(transformer_counts_np[:, RESULT_LOSE_IDX])),
        "all_trials_win_rate_cnn": float(np.sum(cnn_counts_np[:, RESULT_WIN_IDX]) / total_trials_all_positions),
        "all_trials_draw_rate_cnn": float(np.sum(cnn_counts_np[:, RESULT_DRAW_IDX]) / total_trials_all_positions),
        "all_trials_lose_rate_cnn": float(np.sum(cnn_counts_np[:, RESULT_LOSE_IDX]) / total_trials_all_positions),
        "all_trials_win_rate_transformer": float(np.sum(transformer_counts_np[:, RESULT_WIN_IDX]) / total_trials_all_positions),
        "all_trials_draw_rate_transformer": float(np.sum(transformer_counts_np[:, RESULT_DRAW_IDX]) / total_trials_all_positions),
        "all_trials_lose_rate_transformer": float(np.sum(transformer_counts_np[:, RESULT_LOSE_IDX]) / total_trials_all_positions),
        "transformer_better_by_result_mean_x_count": int(np.sum(diff_result_means_x_np > 0.0)),
        "cnn_better_by_result_mean_x_count": int(np.sum(diff_result_means_x_np < 0.0)),
        "tie_by_result_mean_x_count": int(np.sum(diff_result_means_x_np == 0.0)),
    }

    output_stem = build_output_stem(target_end, target_shot, data_size, X)
    json_path = save_dir / f"{output_stem}.json"
    png_path = save_dir / f"{output_stem}.png"

    save_result_json(json_path, summary, result_rows)
    save_result_plot(png_path, cnn_result_means_x_np, transformer_result_means_x_np)

    print("")
    print(f"Saved summary to {json_path}")
    print(f"Saved plot to {png_path}")
    print(f"num_positions                : {summary['num_positions']}")
    print(f"execution_repeats_x         : {summary['execution_repeats_x']}")
    print(f"log_size result_mean_x CNN  : {summary['log_size_result_mean_x_cnn']:.6f}")
    print(f"log_size result_mean_x T    : {summary['log_size_result_mean_x_transformer']:.6f}")
    print(f"log_size diff (T - CNN)     : {summary['log_size_diff_result_mean_x_transformer_minus_cnn']:.6f}")
    print(
        "log_size win/draw/lose CNN : "
        f"{summary['log_size_win_rate_x_cnn']:.6f}, "
        f"{summary['log_size_draw_rate_x_cnn']:.6f}, "
        f"{summary['log_size_lose_rate_x_cnn']:.6f}"
    )
    print(
        "log_size win/draw/lose T   : "
        f"{summary['log_size_win_rate_x_transformer']:.6f}, "
        f"{summary['log_size_draw_rate_x_transformer']:.6f}, "
        f"{summary['log_size_lose_rate_x_transformer']:.6f}"
    )
    print(
        "all_trials win/draw/lose CNN: "
        f"{summary['all_trials_win_count_cnn']}, "
        f"{summary['all_trials_draw_count_cnn']}, "
        f"{summary['all_trials_lose_count_cnn']}"
    )
    print(
        "all_trials win/draw/lose T  : "
        f"{summary['all_trials_win_count_transformer']}, "
        f"{summary['all_trials_draw_count_transformer']}, "
        f"{summary['all_trials_lose_count_transformer']}"
    )
    print(f"Transformer better by result_mean_x count : {summary['transformer_better_by_result_mean_x_count']}")
    print(f"CNN better by result_mean_x count         : {summary['cnn_better_by_result_mean_x_count']}")
    print(f"Tie by result_mean_x count                : {summary['tie_by_result_mean_x_count']}")


if __name__ == "__main__":
    main(
        log_path=Path(__file__).resolve().parents[1] / "LearnLog" / "all",
        save_path=Path(__file__).resolve().parents[0] / "data",
        cnn_model="js20000CP-32-9-LeaRate1000-vx32-vy25-batchsize1024.bin",
        transformer_model="transformer-sl16-model-140000data.bin",
        data_size=5000,
        target_end=9,
        target_shot=15,
        use_gpu=True,
        X=10,
    )
