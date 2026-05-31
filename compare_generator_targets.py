import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from board.constant import VX_SIZE, VY_SIZE
from common.translate_state import convert_team_stoi, scores_to_scorediff_for_team0
from mcts.rollout import score_to_winvalue
from mcts.search import mcts_search, set_root_state as set_mcts_root_state
from mcts.simulate import decode_action
from nn.utility import get_torch_device, load_network
from shot.search import shot_search, set_root_state as set_shot_root_state
from transformer.params import DEFAULT_TRANSFORMER_CONFIG
from transformer.shot_target import (
    build_policy_target_from_shot_stats,
    build_value_target_from_shot_stats,
    build_win_value_target_from_shot_stats,
)

N_ACTIONS = DEFAULT_TRANSFORMER_CONFIG.action_dim
N_VALUE_CLASSES = DEFAULT_TRANSFORMER_CONFIG.value_dim
SCORE_LABELS = [str(i) for i in range(-8, 9)]


def _normalize_distribution(target, expected_size: int, name: str) -> np.ndarray:
    distribution = np.asarray(target, dtype=np.float32)
    if distribution.shape != (expected_size,):
        raise ValueError(f"{name} must have shape ({expected_size},), got {distribution.shape}")
    if not np.all(np.isfinite(distribution)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(distribution < 0):
        raise ValueError(f"{name} contains negative values")

    total = float(distribution.sum())
    if total <= 0.0:
        raise ValueError(f"{name} sum must be positive, got {total}")

    return (distribution / total).astype(np.float32, copy=False)


def _sample_dcl2_path(log_path: Path, shuffle_seed: Optional[int]) -> Path:
    if not log_path.exists():
        raise FileNotFoundError(log_path)
    if not log_path.is_dir():
        raise ValueError(f"log_path must be a directory, got {log_path}")

    candidates = [
        child / "game.dcl2"
        for child in log_path.iterdir()
        if child.is_dir() and (child / "game.dcl2").exists()
    ]
    if not candidates:
        raise ValueError(f"game.dcl2 was not found under {log_path}")

    rng = random.Random(shuffle_seed)
    return rng.choice(candidates)


def _load_first_target_state(
    dcl2_path: Path,
    target_end: List[int],
    target_shot: List[int],
) -> Tuple[Dict, int]:
    with dcl2_path.open("r", encoding="utf-8") as f:
        dcl2_data = f.readlines()

    for line_index in range(9, len(dcl2_data) - 2, 2):
        try:
            dcl2_state = json.loads(dcl2_data[line_index])["log"]["state"]
            end = int(dcl2_state["end"])
            shot = int(dcl2_state["shot"])
        except (KeyError, json.JSONDecodeError, ValueError):
            continue

        if end in target_end and shot in target_shot:
            return dcl2_state, line_index

    raise ValueError(f"target state was not found in {dcl2_path}")


def _state_fields(dcl2_state: Dict) -> Dict:
    stones = dcl2_state["stones"]["team0"] + dcl2_state["stones"]["team1"]
    score_diff_for_team0 = scores_to_scorediff_for_team0(dcl2_state["scores"])
    return {
        "stones": stones,
        "score_diff": score_diff_for_team0,
        "end": int(dcl2_state["end"]),
        "shot_index": int(dcl2_state["shot"]),
        "hammer_team": convert_team_stoi(dcl2_state["hammer"]),
    }


def _top_actions(policy: np.ndarray, top_k: int) -> np.ndarray:
    top_k = min(top_k, policy.size)
    return np.argsort(policy)[-top_k:][::-1]


def _policy_metrics(mcts_policy: np.ndarray, shot_policy: np.ndarray, top_k: int) -> Dict[str, float]:
    eps = 1e-12
    kl_mcts_shot = float(np.sum(mcts_policy * np.log((mcts_policy + eps) / (shot_policy + eps))))
    kl_shot_mcts = float(np.sum(shot_policy * np.log((shot_policy + eps) / (mcts_policy + eps))))
    l1 = float(np.sum(np.abs(mcts_policy - shot_policy)))
    mcts_top = set(int(a) for a in _top_actions(mcts_policy, top_k))
    shot_top = set(int(a) for a in _top_actions(shot_policy, top_k))
    overlap = len(mcts_top & shot_top) / max(1, top_k)
    return {
        "kl_mcts_shot": kl_mcts_shot,
        "kl_shot_mcts": kl_shot_mcts,
        "l1": l1,
        "top_overlap": float(overlap),
    }


def _win_value_from_score_distribution(
    value_distribution: np.ndarray,
    end: int,
    score_diff: int,
    hammer_team: int,
    root_team: int,
) -> float:
    score_diff_root = score_diff if root_team == 0 else -score_diff
    had_hammer_this_end = root_team == hammer_team

    win_value = 0.0
    for cls, prob in enumerate(value_distribution):
        score = cls - 8
        win_value += float(prob) * score_to_winvalue(
            end,
            score,
            score_diff_root,
            had_hammer_this_end,
        )
    return float(win_value)


def _action_label(action_id: int) -> str:
    vx, vy, spin = decode_action(action_id)
    return f"{action_id}\n({vx:.3f},{vy:.3f},{spin})"


def _action_text(action_id: int) -> str:
    vx, vy, spin = decode_action(action_id)
    return f"{action_id} ({vx:.3f},{vy:.3f},{spin})"


def _plot_policy_topk(mcts_policy: np.ndarray, shot_policy: np.ndarray, top_k: int) -> None:
    union_actions = list(
        dict.fromkeys(
            [int(a) for a in _top_actions(mcts_policy, top_k)]
            + [int(a) for a in _top_actions(shot_policy, top_k)]
        )
    )
    union_actions.sort(key=lambda a: max(float(mcts_policy[a]), float(shot_policy[a])), reverse=True)

    x = np.arange(len(union_actions))
    width = 0.42

    fig, ax = plt.subplots(figsize=(max(12, len(union_actions) * 0.65), 5))
    ax.bar(x - width / 2, [mcts_policy[a] for a in union_actions], width, label="MCTS")
    ax.bar(x + width / 2, [shot_policy[a] for a in union_actions], width, label="SHOT")
    ax.set_title(f"Policy Top Actions (union of top {top_k})")
    ax.set_ylabel("probability")
    ax.set_xticks(x)
    ax.set_xticklabels([_action_label(a) for a in union_actions], rotation=70, ha="right")
    ax.legend()
    fig.tight_layout()


def _plot_policy_heatmaps(mcts_policy: np.ndarray, shot_policy: np.ndarray) -> None:
    mcts = mcts_policy.reshape(2, VY_SIZE, VX_SIZE)
    shot = shot_policy.reshape(2, VY_SIZE, VX_SIZE)
    diff = shot - mcts

    vmax = max(float(mcts.max()), float(shot.max()), 1e-12)
    diff_abs = max(float(np.abs(diff).max()), 1e-12)

    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    for spin in range(2):
        im = axes[0, spin].imshow(mcts[spin], origin="lower", aspect="auto", vmin=0.0, vmax=vmax)
        axes[0, spin].set_title(f"MCTS policy spin={spin}")
        fig.colorbar(im, ax=axes[0, spin], fraction=0.046, pad=0.04)

        im = axes[1, spin].imshow(shot[spin], origin="lower", aspect="auto", vmin=0.0, vmax=vmax)
        axes[1, spin].set_title(f"SHOT policy spin={spin}")
        fig.colorbar(im, ax=axes[1, spin], fraction=0.046, pad=0.04)

        im = axes[2, spin].imshow(
            diff[spin],
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
            vmin=-diff_abs,
            vmax=diff_abs,
        )
        axes[2, spin].set_title(f"SHOT - MCTS spin={spin}")
        fig.colorbar(im, ax=axes[2, spin], fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.set_xlabel("vx index")
        ax.set_ylabel("vy index")

    fig.tight_layout()


def _plot_value(mcts_value: np.ndarray, shot_value: np.ndarray) -> None:
    x = np.arange(N_VALUE_CLASSES)
    width = 0.42

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - width / 2, mcts_value, width, label="MCTS")
    ax.bar(x + width / 2, shot_value, width, label="SHOT")
    ax.set_title("Value Target")
    ax.set_ylabel("probability")
    ax.set_xticks(x)
    ax.set_xticklabels(SCORE_LABELS)
    ax.legend()
    fig.tight_layout()


def _print_summary(
    dcl2_path: Path,
    line_index: int,
    state_fields: Dict,
    mcts_policy: np.ndarray,
    shot_policy: np.ndarray,
    mcts_value: np.ndarray,
    shot_value: np.ndarray,
    mcts_win_value: float,
    shot_win_value: float,
    top_k: int,
) -> None:
    metrics = _policy_metrics(mcts_policy, shot_policy, top_k)
    mcts_top1 = int(np.argmax(mcts_policy))
    shot_top1 = int(np.argmax(shot_policy))
    mcts_expected_score = float(np.sum(mcts_value * np.arange(-8, 9)))
    shot_expected_score = float(np.sum(shot_value * np.arange(-8, 9)))

    print("-----------------------------------------------------")
    print(f"dcl2: {dcl2_path}")
    print(f"line_index: {line_index}")
    print(
        "state: "
        f"end={state_fields['end']} shot={state_fields['shot_index']} "
        f"hammer={state_fields['hammer_team']} score_diff={state_fields['score_diff']}"
    )
    print(f"MCTS top1: {mcts_top1} -> {decode_action(mcts_top1)} prob={mcts_policy[mcts_top1]:.6f}")
    print(f"SHOT top1: {shot_top1} -> {decode_action(shot_top1)} prob={shot_policy[shot_top1]:.6f}")
    print(f"policy KL(MCTS||SHOT): {metrics['kl_mcts_shot']:.6f}")
    print(f"policy KL(SHOT||MCTS): {metrics['kl_shot_mcts']:.6f}")
    print(f"policy L1: {metrics['l1']:.6f}")
    print(f"top-{top_k} overlap: {metrics['top_overlap']:.3f}")
    print(f"MCTS expected score: {mcts_expected_score:.6f}")
    print(f"SHOT expected score: {shot_expected_score:.6f}")
    print(f"MCTS win_value target: {mcts_win_value:.6f}")
    print(f"SHOT win_value target: {shot_win_value:.6f}")
    print("-----------------------------------------------------")


def _print_topk_rank_table(mcts_policy: np.ndarray, shot_policy: np.ndarray, top_k: int) -> None:
    mcts_top = _top_actions(mcts_policy, top_k)
    shot_top = _top_actions(shot_policy, top_k)
    rows = []

    for i in range(min(len(mcts_top), len(shot_top))):
        mcts_action = int(mcts_top[i])
        shot_action = int(shot_top[i])
        rows.append(
            (
                i + 1,
                _action_text(mcts_action),
                float(mcts_policy[mcts_action]),
                i + 1,
                _action_text(shot_action),
                float(shot_policy[shot_action]),
                "yes" if mcts_action == shot_action else "no",
            )
        )

    if not rows:
        print(f"Policy top-{top_k} rank table: no actions")
        print("-----------------------------------------------------")
        return

    headers = (
        "MCTS rank",
        "MCTS action",
        "MCTS prob",
        "SHOT rank",
        "SHOT action",
        "SHOT prob",
        "same?",
    )
    widths = [
        len(headers[0]),
        max(len(headers[1]), *(len(row[1]) for row in rows)),
        len(headers[2]),
        len(headers[3]),
        max(len(headers[4]), *(len(row[4]) for row in rows)),
        len(headers[5]),
        len(headers[6]),
    ]

    print(f"Policy top-{top_k} rank table")
    print(
        f"{headers[0]:>{widths[0]}} | "
        f"{headers[1]:<{widths[1]}} | "
        f"{headers[2]:>{widths[2]}} | "
        f"{headers[3]:>{widths[3]}} | "
        f"{headers[4]:<{widths[4]}} | "
        f"{headers[5]:>{widths[5]}} | "
        f"{headers[6]:>{widths[6]}}"
    )
    print("-" * (sum(widths) + 18))
    for row in rows:
        print(
            f"{row[0]:>{widths[0]}} | "
            f"{row[1]:<{widths[1]}} | "
            f"{row[2]:>{widths[2]}.6f} | "
            f"{row[3]:>{widths[3]}} | "
            f"{row[4]:<{widths[4]}} | "
            f"{row[5]:>{widths[5]}.6f} | "
            f"{row[6]:>{widths[6]}}"
        )
    print("-----------------------------------------------------")


def compare_targets(
    log_path: Path,
    model: Path,
    target_end: List[int],
    target_shot: List[int],
    shuffle_seed: Optional[int] = None,
    mcts_simulations: Optional[int] = None,
    shot_simulations: Optional[int] = None,
    top_k: int = 10,
    use_gpu: bool = True,
    policy_min_visit: int = 3,
    policy_delta_q: float = 1.0,
    policy_alpha_visit: float = 0.2,
    policy_beta_q: float = 0.3,
    policy_lambda_best: float = 0.5,
    value_min_visit: int = 3,
    value_delta_q: float = 0.0,
    value_alpha_visit: float = 0.5,
    value_beta_q: float = 0.5,
    value_lambda_best: float = 0.5,
) -> None:
    dcl2_path = _sample_dcl2_path(Path(log_path), shuffle_seed)
    dcl2_state, line_index = _load_first_target_state(dcl2_path, target_end, target_shot)
    state_fields = _state_fields(dcl2_state)

    device = get_torch_device(use_gpu=use_gpu)
    network = load_network(model, use_gpu=use_gpu)
    network.to(device)

    mcts_root = set_mcts_root_state(network=network, **state_fields)
    mcts_kwargs = {"root_state": mcts_root, "is_create_data": True}
    if mcts_simulations is not None:
        mcts_kwargs["max_simulations"] = mcts_simulations
    _, mcts_policy_counts, mcts_value_counts = mcts_search(**mcts_kwargs)
    mcts_policy = _normalize_distribution(mcts_policy_counts, N_ACTIONS, "mcts_policy")
    mcts_value = _normalize_distribution(mcts_value_counts, N_VALUE_CLASSES, "mcts_value")
    mcts_win_value = _win_value_from_score_distribution(
        mcts_value,
        end=state_fields["end"],
        score_diff=state_fields["score_diff"],
        hammer_team=state_fields["hammer_team"],
        root_team=mcts_root.to_move(),
    )

    shot_root = set_shot_root_state(network=network, **state_fields)
    shot_kwargs = {"root_state": shot_root, "is_create_data": True}
    if shot_simulations is not None:
        shot_kwargs["max_simulations"] = shot_simulations
    shot_best_action_id, shot_stats = shot_search(**shot_kwargs)
    shot_policy = build_policy_target_from_shot_stats(
        shot_stats,
        shot_best_action_id,
        action_dim=N_ACTIONS,
        min_visit=policy_min_visit,
        delta_q=policy_delta_q,
        alpha_visit=policy_alpha_visit,
        beta_q=policy_beta_q,
        lambda_best=policy_lambda_best,
    )
    shot_value = build_value_target_from_shot_stats(
        shot_stats,
        shot_best_action_id,
        value_dim=N_VALUE_CLASSES,
        min_visit=value_min_visit,
        delta_q=value_delta_q,
        alpha_visit=value_alpha_visit,
        beta_q=value_beta_q,
        lambda_best=value_lambda_best,
    )
    shot_win_value = build_win_value_target_from_shot_stats(
        shot_stats,
        shot_best_action_id,
        min_visit=policy_min_visit,
        delta_q=policy_delta_q,
        alpha_visit=policy_alpha_visit,
        beta_q=policy_beta_q,
        lambda_best=policy_lambda_best,
    )

    _print_summary(
        dcl2_path,
        line_index,
        state_fields,
        mcts_policy,
        shot_policy,
        mcts_value,
        shot_value,
        mcts_win_value,
        shot_win_value,
        top_k,
    )
    _print_topk_rank_table(mcts_policy, shot_policy, top_k)
    _plot_policy_topk(mcts_policy, shot_policy, top_k)
    _plot_policy_heatmaps(mcts_policy, shot_policy)
    _plot_value(mcts_value, shot_value)
    plt.show()


if __name__ == "__main__":
    compare_targets(
        log_path=Path(__file__).resolve().parent / "LearnLog" / "all",
        model=Path(__file__).resolve().parent / "model" / "js20000CP-32-9-LeaRate1000-vx32-vy25-batchsize1024.bin",
        target_end=[9],
        target_shot=[15],
        shuffle_seed=None,
        mcts_simulations=10000,
        shot_simulations=1020,
        top_k=10,
        use_gpu=True,
        policy_min_visit=3,
        policy_delta_q=1.0,
        policy_alpha_visit=0.2,
        policy_beta_q=0.3,
        policy_lambda_best=0.5,
        value_min_visit=3,
        value_delta_q=0.0,
        value_alpha_visit=0.5,
        value_beta_q=0.5,
        value_lambda_best=0.5,
    )
