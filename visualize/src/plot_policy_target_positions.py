from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Circle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from board.constant import (  # noqa: E402
    R_HOUSE,
    STONE_RADIUS,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
    Y_TEE,
)
from mcts.hybrid_policy import get_policy  # noqa: E402
from mcts.search import set_root_state as set_mcts_root_state  # noqa: E402
from mcts.simulate import decode_action  # noqa: E402
from transformer.params import TRANSFORMER_VY_MODE, TransformerVyMode  # noqa: E402

from plot_mcts_root_candidates import (  # noqa: E402
    _draw_board,
    _load_model,
    _load_target_state,
    _nominal_target_for_action,
    _safe_filename_part,
    _shot_team,
    _state_fields,
)

PolicyPoint = Dict[str, object]


def _model_kind(sl_model_is_cnn: bool) -> str:
    return "cnn" if sl_model_is_cnn else "transformer"


def _auto_output_path(
    model: Path,
    sl_model_is_cnn: bool,
    state_fields: Dict,
    sample_index: int,
    top_policy_mass_percent: float,
) -> Path:
    parts = [
        "policy_targets",
        _model_kind(sl_model_is_cnn),
        _safe_filename_part(model.stem),
        f"mass{top_policy_mass_percent:g}",
        f"end{int(state_fields['end'])}",
        f"shot{int(state_fields['shot_index'])}",
        f"sample{int(sample_index)}",
    ]
    return PROJECT_ROOT / "visualize" / "data" / "policy_targets" / ("_".join(parts) + ".png")


def _policy_points(
    policy: List[float],
    action_type: TransformerVyMode,
    top_policy_mass_percent: float,
) -> List[PolicyPoint]:
    probs = np.asarray(policy, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError(f"policy must be 1-dimensional, got shape={probs.shape}")
    if not (0.0 < top_policy_mass_percent <= 100.0):
        raise ValueError(
            "top_policy_mass_percent must be in (0, 100], "
            f"got {top_policy_mass_percent}"
        )

    target_mass = float(top_policy_mass_percent) / 100.0
    action_ids = []
    cumulative = 0.0
    for action_id in np.argsort(probs)[::-1]:
        probability = float(probs[int(action_id)])
        if probability <= 0.0:
            continue
        action_ids.append(int(action_id))
        cumulative += probability
        if cumulative >= target_mass:
            break

    points: List[PolicyPoint] = []
    for action_id in action_ids:
        target = _nominal_target_for_action(action_id, action_type)
        if target is None:
            continue

        vx, vy, spin = decode_action(action_id, action_type=action_type)
        points.append(
            {
                "action_id": int(action_id),
                "probability": float(probs[action_id]),
                "vx": float(vx),
                "vy": float(vy),
                "spin": int(spin),
                "target_x": float(target[0]),
                "target_y": float(target[1]),
            }
        )

    points.sort(key=lambda point: float(point["probability"]), reverse=True)
    return points


def _scatter_points(ax, points: List[PolicyPoint], norm):
    if not points:
        return None

    xs = [float(point["target_x"]) for point in points]
    ys = [float(point["target_y"]) for point in points]
    probs = np.asarray([float(point["probability"]) for point in points], dtype=np.float64)
    return ax.scatter(
        xs,
        ys,
        c=probs,
        s=24,
        cmap="viridis",
        norm=norm,
        alpha=0.68,
        edgecolors="none",
        zorder=4,
    )


def _axis_limits_for_points(points: List[PolicyPoint]):
    if not points:
        return (X_MIN - 0.25, X_MAX + 0.25), (Y_MIN - 0.25, Y_MAX + 0.25)

    xs = [float(point["target_x"]) for point in points]
    ys = [float(point["target_y"]) for point in points]
    return (
        min(X_MIN - 0.25, min(xs) - 0.25),
        max(X_MAX + 0.25, max(xs) + 0.25),
    ), (
        min(Y_MIN - 0.25, min(ys) - 0.25),
        max(Y_MAX + 0.25, max(ys) + 0.25),
    )


def _draw_board_foreground(ax, stones) -> None:
    for radius in (R_HOUSE, 1.219, 0.610, 0.152):
        ax.add_patch(
            Circle(
                (0.0, Y_TEE),
                radius,
                facecolor="none",
                edgecolor="#202020",
                linewidth=0.9,
                zorder=10,
            )
        )
    ax.axhline(Y_TEE, color="#202020", linewidth=0.8, alpha=0.85, zorder=10)
    ax.axvline(0.0, color="#202020", linewidth=0.8, alpha=0.85, zorder=10)

    for index, stone in enumerate(stones):
        if stone is None:
            continue
        color = "#d94b42" if index < 8 else "#f0c84b"
        ax.add_patch(
            Circle(
                stone,
                STONE_RADIUS,
                facecolor=color,
                edgecolor="#111111",
                linewidth=0.9,
                zorder=11,
            )
        )


def _print_top_policy_points(points: List[PolicyPoint], top_console_count: int) -> None:
    print("Top plotted policy target positions:")
    for rank, point in enumerate(points[:top_console_count], start=1):
        spin_label = "cw" if int(point["spin"]) == 0 else "ccw"
        print(
            f"{rank:2d}: "
            f"a={int(point['action_id'])} "
            f"p={float(point['probability']):.8f} "
            f"vx={float(point['vx']):.4f} "
            f"vy={float(point['vy']):.4f} "
            f"spin={spin_label} "
            f"target=({float(point['target_x']):.3f}, {float(point['target_y']):.3f})"
        )


def plot_policy_target_positions(
    log_path: Path,
    model: Path,
    target_end: List[int],
    target_shot: List[int],
    output_path: Optional[Path] = None,
    shuffle_seed: Optional[int] = None,
    sample_index: int = 0,
    use_gpu: bool = True,
    sl_model_is_cnn: bool = False,
    transformer_target_end: List[int] = [9],
    transformer_target_shot: List[int] = [15],
    action_type: TransformerVyMode = TRANSFORMER_VY_MODE,
    top_policy_mass_percent: float = 90.0,
    top_console_count: int = 20,
    split_by_spin: bool = True,
    skip_unwinnable: bool = True,
) -> Path:
    dcl2_path, line_index, dcl2_state = _load_target_state(
        log_path=log_path,
        target_end=target_end,
        target_shot=target_shot,
        shuffle_seed=shuffle_seed,
        sample_index=sample_index,
        skip_unwinnable=skip_unwinnable,
    )
    fields = _state_fields(dcl2_state)

    network = _load_model(
        model=model,
        use_gpu=use_gpu,
        sl_model_is_cnn=sl_model_is_cnn,
        action_type=action_type,
    )
    root_state = set_mcts_root_state(
        sl_model=network,
        stones=fields["stones"],
        score_diff=int(fields["score_diff"]),
        end=int(fields["end"]),
        shot_index=int(fields["shot_index"]),
        hammer_team=int(fields["hammer_team"]),
        transformer_target_end=tuple(int(end) for end in transformer_target_end),
        transformer_target_shot=tuple(int(shot) for shot in transformer_target_shot),
        sl_model_is_cnn=sl_model_is_cnn,
        use_search_based_model=False,
    )

    policy = get_policy(root_state, action_type=action_type)
    points = _policy_points(policy, action_type, top_policy_mass_percent)
    if not points:
        raise ValueError("no policy target positions to plot")

    max_probability = max(float(point["probability"]) for point in points)
    min_probability = min(float(point["probability"]) for point in points)
    norm = (
        LogNorm(vmin=min_probability, vmax=max_probability)
        if min_probability < max_probability
        else Normalize(vmin=0.0, vmax=max_probability)
    )

    if split_by_spin:
        fig, axes = plt.subplots(1, 3, figsize=(21, 10), constrained_layout=True)
        plot_specs = [(0, "cw", True), (1, "ccw", True), (None, "board", False)]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8.5, 10), constrained_layout=True)
        axes = [ax]
        plot_specs = [(None, "cw + ccw", True)]

    xlim, ylim = _axis_limits_for_points(points)
    mappable = None
    best_point = points[0]
    colorbar_axes = []
    for ax, (spin, title, show_policy) in zip(axes, plot_specs):
        _draw_board(ax, root_state.stones)
        selected = []
        if show_policy:
            selected = [
                point for point in points if spin is None or int(point["spin"]) == int(spin)
            ]
            mappable = _scatter_points(ax, selected, norm) or mappable
            colorbar_axes.append(ax)
        _draw_board_foreground(ax, root_state.stones)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if show_policy and (spin is None or int(best_point["spin"]) == int(spin)):
            ax.scatter(
                [float(best_point["target_x"])],
                [float(best_point["target_y"])],
                marker="*",
                s=300,
                facecolors="#ffd94a",
                edgecolors="#111111",
                linewidths=1.1,
                zorder=12,
                label="top policy",
            )
            ax.legend(loc="upper right")

        if show_policy:
            ax.set_title(f"{title}: {len(selected)} actions")
        else:
            ax.set_title("board only")

    if mappable is not None:
        fig.colorbar(
            mappable,
            ax=colorbar_axes,
            fraction=0.045,
            pad=0.03,
            label="policy probability",
        )

    shot_team = _shot_team(int(fields["shot_index"]), int(fields["hammer_team"]))
    team_name = "team0 (red)" if shot_team == 0 else "team1 (yellow)"
    score_diff_for_next = int(fields["score_diff"]) if shot_team == 0 else -int(fields["score_diff"])
    model_name = _safe_filename_part(model.stem, max_length=80)
    fig.suptitle(
        "Policy target positions: "
        f"end={int(fields['end'])} shot={int(fields['shot_index'])} "
        f"top_mass={top_policy_mass_percent:g}% actions={len(points)}\n"
        f"Next shot: {team_name}    "
        f"Score diff for next shot team: {score_diff_for_next:+d}\n"
        f"model={model_name}",
        fontsize=14,
    )

    if output_path is None:
        output_path = _auto_output_path(
            model=model,
            sl_model_is_cnn=sl_model_is_cnn,
            state_fields=fields,
            sample_index=sample_index,
            top_policy_mass_percent=top_policy_mass_percent,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    plotted_mass = sum(float(point["probability"]) for point in points)
    print(f"Loaded state: {dcl2_path}:{line_index + 1}")
    print(
        f"Policy actions plotted: {len(points)} "
        f"(top_policy_mass_percent={top_policy_mass_percent}, "
        f"actual_mass={plotted_mass:.6f})"
    )
    _print_top_policy_points(points, top_console_count)
    print(f"Saved plot to {output_path}")
    return output_path


if __name__ == "__main__":
    plot_policy_target_positions(
        log_path=PROJECT_ROOT / "LearnLog" / "all",
        model=PROJECT_ROOT / "model" / "transformer-sl-9-15-model-07-28-adamw-epoch50-shot.bin",
        target_end=[9],
        target_shot=[15],
        output_path=None,
        shuffle_seed=None,
        sample_index=2,
        use_gpu=True,
        sl_model_is_cnn=False,
        transformer_target_end=[9],
        transformer_target_shot=[15],
        top_policy_mass_percent=50.0,
        top_console_count=20,
        split_by_spin=True,
        skip_unwinnable=True,
    )

