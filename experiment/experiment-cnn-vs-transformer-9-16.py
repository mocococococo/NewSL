"""

"""
import numpy as np
import copy
import os
import sys
import json
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
    cnn_scores: np.ndarray,
    transformer_scores: np.ndarray,
) -> None:
    """局面ごとの平均得点と差分を png で保存する。"""

    import matplotlib.pyplot as plt

    indices = np.arange(len(cnn_scores))
    diff_scores = transformer_scores - cnn_scores
    if len(indices) == 0:
        raise ValueError("cnn_scores and transformer_scores must not be empty")
    x_max = len(indices) - 1

    figure, (ax_score, ax_diff) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax_score.plot(indices, cnn_scores, label="CNN", marker="o")
    ax_score.plot(indices, transformer_scores, label="Transformer", marker="o")
    ax_score.set_ylabel("Mean Terminal Score")
    ax_score.set_title("CNN vs Transformer")
    ax_score.set_ylim(-8, 8)
    ax_score.grid(True, alpha=0.3)
    ax_score.legend()

    ax_diff.plot(indices, diff_scores, label="Transformer - CNN", color="tab:green", marker="o")
    ax_diff.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax_diff.set_xlabel("Position Index")
    ax_diff.set_ylabel("Score Diff")
    ax_diff.set_ylim(-8, 8)
    ax_diff.grid(True, alpha=0.3)
    ax_diff.legend()

    if x_max == 0:
        ax_diff.set_xlim(-0.5, 0.5)
        ax_diff.set_xticks([0])
    else:
        ax_diff.set_xlim(0, x_max)
        ax_diff.set_xticks(np.arange(0, x_max + 1, 1))

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
    
    cnn_score_avg = [0 for _ in range(data_size)]
    transformer_score_avg = [0 for _ in range(data_size)]

    for one_log in os.listdir(log_path):
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
            # 探索結果を X 回、実行して、平均得点を得る
            for i in range(X):
                final_state = simulator_step(root, encoded_action)
                raw_score = _end_score_diff_team0_minus_team1(final_state.stones)
                score_from_root_view = raw_score if root.to_move() == 0 else -raw_score
                cnn_score_avg[score_index] += score_from_root_view
            cnn_score_avg[score_index] = cnn_score_avg[score_index] / X
            
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
            # 探索結果を X 回、実行して、平均得点を得る
            for j in range(X):
                final_state = simulator_step(root, encoded_action)
                raw_score = _end_score_diff_team0_minus_team1(final_state.stones)
                score_from_root_view = raw_score if root.to_move() == 0 else -raw_score
                transformer_score_avg[score_index] += score_from_root_view
            transformer_score_avg[score_index] = transformer_score_avg[score_index] / X

            result_rows.append(
                {
                    "position_index": score_index,
                    "log_name": one_log,
                    "end": int(end),
                    "shot": int(shot),
                    "hammer": int(hammer),
                    "score_diff_for_team0": int(scorediff_for_team0),
                    "cnn_mean_score": float(cnn_score_avg[score_index]),
                    "transformer_mean_score": float(transformer_score_avg[score_index]),
                    "diff_score": float(transformer_score_avg[score_index] - cnn_score_avg[score_index]),
                }
            )
            position_count += 1

    if position_count == 0:
        raise RuntimeError(
            f"No positions found for end={target_end}, shot={target_shot} in {log_path}"
        )

    cnn_scores = np.array(cnn_score_avg[:position_count], dtype=np.float32)
    transformer_scores = np.array(transformer_score_avg[:position_count], dtype=np.float32)
    diff_scores = transformer_scores - cnn_scores

    summary = {
        "num_positions": int(position_count),
        "target_end": int(target_end),
        "target_shot": int(target_shot),
        "execution_repeats": int(X),
        "cnn_model": str(cnn_model_path),
        "transformer_model": str(transformer_model_path),
        "cnn_total_mean_score": float(np.sum(cnn_scores)),
        "transformer_total_mean_score": float(np.sum(transformer_scores)),
        "cnn_average_mean_score": float(np.mean(cnn_scores)),
        "transformer_average_mean_score": float(np.mean(transformer_scores)),
        "diff_total_score": float(np.sum(diff_scores)),
        "diff_average_score": float(np.mean(diff_scores)),
        "transformer_better_count": int(np.sum(diff_scores > 0.0)),
        "cnn_better_count": int(np.sum(diff_scores < 0.0)),
        "tie_count": int(np.sum(diff_scores == 0.0)),
    }

    json_path = save_dir / f"cnn_vs_transformer_end{target_end}_shot{target_shot}.json"
    png_path = save_dir / f"cnn_vs_transformer_end{target_end}_shot{target_shot}.png"

    save_result_json(json_path, summary, result_rows)
    save_result_plot(png_path, cnn_scores, transformer_scores)

    print("")
    print(f"Saved summary to {json_path}")
    print(f"Saved plot to {png_path}")
    print(f"num_positions                : {summary['num_positions']}")
    print(f"CNN total mean score         : {summary['cnn_total_mean_score']:.6f}")
    print(f"Transformer total mean score : {summary['transformer_total_mean_score']:.6f}")
    print(f"CNN average mean score       : {summary['cnn_average_mean_score']:.6f}")
    print(f"Transformer average mean score: {summary['transformer_average_mean_score']:.6f}")
    print(f"Average diff (T - CNN)       : {summary['diff_average_score']:.6f}")
    print(f"Transformer better count     : {summary['transformer_better_count']}")
    print(f"CNN better count             : {summary['cnn_better_count']}")
    print(f"Tie count                    : {summary['tie_count']}")


if __name__ == "__main__":
    main(
        log_path=Path(__file__).resolve().parents[1] / "LearnLog" / "cai",
        save_path=Path(__file__).resolve().parents[0] / "data",
        cnn_model="js20000CP-32-9-LeaRate1000-vx32-vy25-batchsize1024.bin",
        transformer_model="transformer-sl-9-15-model.bin",
        data_size=100,
        target_end=9,
        target_shot=15,
        use_gpu=True,
        X=10,
    )
