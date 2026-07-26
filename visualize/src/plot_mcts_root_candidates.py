from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

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
from common.translate_state import (  # noqa: E402
    convert_team_stoi,
    scores_to_scorediff_for_team0,
)
from mcts import fast_simulator  # noqa: E402
from mcts.search import mcts_search, set_root_state as set_mcts_root_state  # noqa: E402
from mcts.simulate import decode_action, simulator_step  # noqa: E402
from shot.search import shot_search, set_root_state as set_shot_root_state  # noqa: E402
from nn.utility import get_torch_device, load_network  # noqa: E402
from transformer.params import TRANSFORMER_VY_MODE  # noqa: E402
from transformer.utility import load_transformer_network  # noqa: E402

CandidateStat = Dict[str, object]

def _safe_filename_part(value: object, max_length: int = 96) -> str:
    text = str(value).strip()
    safe_chars = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    safe = "".join(safe_chars).strip("._")
    if not safe:
        safe = "unnamed"
    return safe[:max_length]


def _simulation_label(search_method: str, mcts_simulations: Optional[int], shot_simulations: Optional[int]) -> str:
    if search_method == "mcts":
        simulations = mcts_simulations
    elif search_method == "shot":
        simulations = shot_simulations
    else:
        raise ValueError(f"unsupported search_method: {search_method}")
    return "simdefault" if simulations is None else f"sim{int(simulations)}"


def _auto_output_path(
    model: Path,
    search_method: str,
    sl_model_is_cnn: bool,
    mcts_simulations: Optional[int],
    shot_simulations: Optional[int],
    state_fields: Dict[str, int],
    sample_index: int,
) -> Path:
    model_kind = "cnn" if sl_model_is_cnn else "transformer"
    parts = [
        "root_candidates",
        _safe_filename_part(search_method),
        model_kind,
        _safe_filename_part(model.stem),
        _simulation_label(search_method, mcts_simulations, shot_simulations),
        f"end{int(state_fields['end'])}",
        f"shot{int(state_fields['shot_index'])}",
        f"sample{int(sample_index)}",
    ]
    return PROJECT_ROOT / "visualize" / "data" / ("_".join(parts) + ".png")


def _dcl2_paths(log_path: Path, shuffle_seed: Optional[int]) -> Iterable[Path]:
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    if log_path.is_file():
        if log_path.suffix != ".dcl2":
            raise ValueError(f"log_path file must be a .dcl2 file, got {log_path}")
        yield log_path
        return

    children = list(log_path.iterdir()) if shuffle_seed is not None else log_path.iterdir()
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        rng.shuffle(children)

    found = False
    for child in children:
        if not child.is_dir():
            continue
        dcl2_path = child / "game.dcl2"
        if not dcl2_path.exists():
            continue
        found = True
        yield dcl2_path

    if not found:
        raise ValueError(f"game.dcl2 was not found under {log_path}")


def _shot_team(shot: int, hammer_team: int) -> int:
    return hammer_team if (shot % 2) == 1 else 1 - hammer_team


def _count_team_stones_on_sheet(stones: List[object], team: int) -> int:
    if len(stones) != 16:
        raise ValueError(f"stones must have length 16, got {len(stones)}")
    if team == 0:
        team_stones = stones[:8]
    elif team == 1:
        team_stones = stones[8:16]
    else:
        raise ValueError(f"team must be 0 or 1, got {team}")
    return sum(stone is not None for stone in team_stones)


def _is_unwinnable_state(dcl2_state: Dict) -> bool:
    stones = dcl2_state["stones"]["team0"] + dcl2_state["stones"]["team1"]
    score_diff_for_team0 = scores_to_scorediff_for_team0(dcl2_state["scores"])
    shot = int(dcl2_state["shot"])
    hammer_team = convert_team_stoi(dcl2_state["hammer"])
    shot_team = _shot_team(shot, hammer_team)
    scorediff_for_shot_team = (
        score_diff_for_team0 if shot_team == 0 else -score_diff_for_team0
    )
    max_possible_end_score = _count_team_stones_on_sheet(stones, shot_team) + 1
    return max_possible_end_score + scorediff_for_shot_team < 0


def _iter_target_states(
    dcl2_paths: Iterable[Path],
    target_end: List[int],
    target_shot: List[int],
    skip_unwinnable: bool,
) -> Iterable[Tuple[Path, int, Dict]]:
    target_end_set = set(int(end) for end in target_end)
    target_shot_set = set(int(shot) for shot in target_shot)

    for dcl2_path in dcl2_paths:
        with dcl2_path.open("r", encoding="utf-8") as f:
            dcl2_data = f.readlines()

        for line_index, line in enumerate(dcl2_data):
            try:
                dcl2_state = json.loads(line)["log"]["state"]
                end = int(dcl2_state["end"])
                shot = int(dcl2_state["shot"])
            except (KeyError, json.JSONDecodeError, ValueError):
                continue

            if end in target_end_set and shot in target_shot_set:
                if skip_unwinnable and _is_unwinnable_state(dcl2_state):
                    continue
                yield dcl2_path, line_index, dcl2_state


def _load_target_state(
    log_path: Path,
    target_end: List[int],
    target_shot: List[int],
    shuffle_seed: Optional[int],
    sample_index: int,
    skip_unwinnable: bool,
) -> Tuple[Path, int, Dict]:
    if sample_index < 0:
        raise ValueError(f"sample_index must be non-negative, got {sample_index}")

    paths = _dcl2_paths(log_path, shuffle_seed)
    for index, item in enumerate(
        _iter_target_states(paths, target_end, target_shot, skip_unwinnable)
    ):
        if index == sample_index:
            return item

    raise ValueError(
        "target state was not found: "
        f"log_path={log_path}, target_end={target_end}, target_shot={target_shot}, "
        f"sample_index={sample_index}, skip_unwinnable={skip_unwinnable}"
    )


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


def _load_model(model: Path, use_gpu: bool, sl_model_is_cnn: bool, action_type: str):
    if sl_model_is_cnn:
        device = get_torch_device(use_gpu=use_gpu)
        network = load_network(model, use_gpu=use_gpu)
        network.to(device)
        return network

    return load_transformer_network(
        model,
        use_gpu=use_gpu,
        action_type=action_type,
    )


def _draw_board(ax, stones) -> None:
    ax.add_patch(
        Rectangle(
            (X_MIN, Y_MIN),
            X_MAX - X_MIN,
            Y_MAX - Y_MIN,
            facecolor="#f7f4ec",
            edgecolor="#333333",
            linewidth=1.0,
            zorder=0,
        )
    )

    rings = [
        (R_HOUSE, "#2f6fb0"),
        (1.219, "#ffffff"),
        (0.610, "#c83b36"),
        (0.152, "#ffffff"),
    ]
    for radius, color in rings:
        ax.add_patch(
            Circle(
                (0.0, Y_TEE),
                radius,
                facecolor=color,
                edgecolor="#333333",
                linewidth=0.8,
                alpha=0.95,
                zorder=1,
            )
        )

    ax.axhline(Y_TEE, color="#333333", linewidth=0.8, alpha=0.65, zorder=2)
    ax.axvline(0.0, color="#333333", linewidth=0.8, alpha=0.65, zorder=2)

    for index, stone in enumerate(stones):
        if stone is None:
            continue
        color = "#d94b42" if index < 8 else "#f0c84b"
        ax.add_patch(
            Circle(
                stone,
                STONE_RADIUS,
                facecolor=color,
                edgecolor="#202020",
                linewidth=0.8,
                zorder=5,
            )
        )

    ax.set_xlim(X_MIN - 0.25, X_MAX + 0.25)
    ax.set_ylim(Y_MIN - 0.25, Y_MAX + 0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(color="#555555", alpha=0.16, linewidth=0.6)


def _candidate_points(
    candidate_stats: List[CandidateStat],
    min_visit: int,
    spin: Optional[int] = None,
) -> List[CandidateStat]:
    points = []
    for stat in candidate_stats:
        if int(stat["visit_count"]) < min_visit:
            continue
        if spin is not None and int(stat["spin"]) != spin:
            continue
        if stat["target_x"] is None or stat["target_y"] is None:
            continue
        points.append(stat)
    return points


def _nominal_target_for_action(
    action_id: int,
    action_type: str,
) -> Optional[Tuple[float, float]]:
    vx, vy, spin = decode_action(action_id, action_type=action_type)
    x, y = fast_simulator.shot2dest((vx, vy, spin))
    if y <= 0:
        return None
    return float(x), float(y)


def _build_mcts_candidate_stats(
    root_visit_counts: List[int],
    best_action_id: int,
    action_type: str,
) -> List[CandidateStat]:
    stats: List[CandidateStat] = []
    for action_id, visit_count in enumerate(root_visit_counts):
        if int(visit_count) <= 0:
            continue

        vx, vy, spin = decode_action(action_id, action_type=action_type)
        target = _nominal_target_for_action(action_id, action_type)
        target_x = None if target is None else float(target[0])
        target_y = None if target is None else float(target[1])
        stats.append(
            {
                "action_id": int(action_id),
                "visit_count": int(visit_count),
                "q": None,
                "prior": None,
                "vx": float(vx),
                "vy": float(vy),
                "spin": int(spin),
                "target_x": target_x,
                "target_y": target_y,
                "is_best": int(action_id) == int(best_action_id),
                "is_active": True,
            }
        )

    stats.sort(key=lambda stat: int(stat["visit_count"]), reverse=True)
    return stats


def _build_shot_candidate_stats(
    shot_stats: List[CandidateStat],
    best_action_id: int,
    action_type: str,
) -> List[CandidateStat]:
    stats: List[CandidateStat] = []
    for stat in shot_stats:
        action_id = int(stat["action_id"])
        vx, vy, spin = decode_action(action_id, action_type=action_type)
        target = _nominal_target_for_action(action_id, action_type)
        target_x = None if target is None else float(target[0])
        target_y = None if target is None else float(target[1])
        stats.append(
            {
                **stat,
                "action_id": action_id,
                "visit_count": int(stat["visit_count"]),
                "q": float(stat["q"]),
                "prior": float(stat["prior"]),
                "vx": float(vx),
                "vy": float(vy),
                "spin": int(spin),
                "target_x": target_x,
                "target_y": target_y,
                "is_best": action_id == int(best_action_id),
                "is_active": bool(stat.get("is_survivor", False)),
            }
        )

    stats.sort(
        key=lambda item: (
            int(item["visit_count"]),
            float(item["q"]),
            float(item["prior"]),
        ),
        reverse=True,
    )
    return stats


def _best_action_id_from_visit_counts(root_visit_counts: List[int]) -> int:
    if not root_visit_counts:
        raise ValueError("root_visit_counts must not be empty")
    best_action_id = max(range(len(root_visit_counts)), key=lambda a: int(root_visit_counts[a]))
    if int(root_visit_counts[best_action_id]) <= 0:
        raise ValueError("root_visit_counts contains no visited action")
    return int(best_action_id)


def _values_for_color(points: List[CandidateStat], color_by: str) -> Tuple[np.ndarray, str]:
    if color_by not in {"q", "visit_count", "prior"}:
        raise ValueError(f"unsupported color_by: {color_by}")

    values = [point[color_by] for point in points]
    if any(value is None for value in values):
        raise ValueError(f"{color_by} is not available for these candidate stats")

    label = {"q": "Q", "visit_count": "visit count", "prior": "prior"}[color_by]
    return np.asarray([float(value) for value in values], dtype=np.float64), label


def _guide_curve_points(point: CandidateStat) -> Tuple[np.ndarray, np.ndarray]:
    target_x = float(point["target_x"])
    target_y = float(point["target_y"])
    spin = int(point["spin"])
    start_x = 0.0
    start_y = Y_MIN
    progress = np.linspace(0.0, 1.0, 80)
    base_x = start_x + (target_x - start_x) * progress
    base_y = start_y + (target_y - start_y) * progress
    curl_sign = 1.0 if spin == 0 else -1.0
    curl_width = 0.22 + 0.10 * min(abs(target_x), 1.5)
    curl = curl_sign * curl_width * np.sin(np.pi * progress) * (0.35 + 0.65 * progress)
    return base_x + curl, base_y


def _draw_candidate_guides(
    ax,
    points: List[CandidateStat],
    guide_count: int,
) -> None:
    if guide_count <= 0 or not points:
        return

    guide_points = points[:guide_count]
    legend_drawn = set()

    for point in reversed(guide_points):
        spin = int(point["spin"])
        xs, ys = _guide_curve_points(point)
        linestyle = "-" if spin == 0 else "--"
        label = None
        if spin not in legend_drawn:
            label = "cw guide" if spin == 0 else "ccw guide"
            legend_drawn.add(spin)
        ax.plot(
            xs,
            ys,
            color="#4a4a4a",
            linestyle=linestyle,
            linewidth=1.0,
            alpha=0.38,
            zorder=4,
            label=label,
        )


def _scatter_candidates(
    ax,
    points: List[CandidateStat],
    color_by: str,
    draw_guides: bool,
    guide_count: int,
) -> None:
    if not points:
        return

    if draw_guides:
        _draw_candidate_guides(ax, points, guide_count)

    xs = np.asarray([float(point["target_x"]) for point in points], dtype=np.float64)
    ys = np.asarray([float(point["target_y"]) for point in points], dtype=np.float64)
    visits = np.asarray([int(point["visit_count"]) for point in points], dtype=np.float64)
    color_values, color_label = _values_for_color(points, color_by)
    max_visit = max(float(visits.max()), 1.0)
    sizes = 24.0 + 220.0 * np.sqrt(visits / max_visit)
    cmap = "viridis" if color_by == "visit_count" else "coolwarm"

    scatter = ax.scatter(
        xs,
        ys,
        s=sizes,
        c=color_values,
        cmap=cmap,
        alpha=0.72,
        edgecolors="#202020",
        linewidths=0.45,
        zorder=6,
    )
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label=color_label)

    best_points = [point for point in points if bool(point["is_best"])]
    if best_points:
        ax.scatter(
            [float(point["target_x"]) for point in best_points],
            [float(point["target_y"]) for point in best_points],
            marker="*",
            s=360,
            c="#ffd447",
            edgecolors="#101010",
            linewidths=1.2,
            zorder=8,
            label="best action",
        )
        ax.legend(loc="upper right")



def _plot_candidates(
    candidate_stats: List[CandidateStat],
    stones,
    result_stones,
    best_stat: CandidateStat,
    output_path: Path,
    title: str,
    shot_team_label: str,
    score_diff_label: str,
    min_visit: int,
    color_by: str,
    split_by_spin: bool,
    draw_guides: bool,
    guide_count: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    title_text = f"{title}\nNext shot: {shot_team_label}\nScore diff for next shot team: {score_diff_label}"
    result_title = (
        "Best action result (with noise)\n"
        f"a={int(best_stat['action_id'])} "
        f"vx={float(best_stat['vx']):.4f} "
        f"vy={float(best_stat['vy']):.4f} "
        f"spin={int(best_stat['spin'])}"
    )

    if split_by_spin:
        fig, axes = plt.subplots(1, 3, figsize=(18, 9), constrained_layout=True)
        for spin, ax in enumerate(axes[:2]):
            _draw_board(ax, stones)
            points = _candidate_points(candidate_stats, min_visit=min_visit, spin=spin)
            _scatter_candidates(ax, points, color_by, draw_guides, guide_count)
            ax.set_title(f"{title_text} spin={spin}")
        _draw_board(axes[2], result_stones)
        axes[2].set_title(result_title)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 10), constrained_layout=True)
        _draw_board(axes[0], stones)
        points = _candidate_points(candidate_stats, min_visit=min_visit)
        _scatter_candidates(axes[0], points, color_by, draw_guides, guide_count)
        axes[0].set_title(title_text)
        _draw_board(axes[1], result_stones)
        axes[1].set_title(result_title)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _best_candidate_stat(candidate_stats: List[CandidateStat]) -> CandidateStat:
    for stat in candidate_stats:
        if bool(stat["is_best"]):
            return stat
    raise ValueError("best candidate was not found in candidate_stats")


def _print_top_candidates(
    candidate_stats: List[CandidateStat],
    console_top_count: int,
) -> None:
    if console_top_count <= 0:
        return

    rows = candidate_stats[:console_top_count]
    if not rows:
        print("Top root candidates: none")
        return

    print("Top root candidates")
    print("rank action_id visits q prior spin vx vy target_x target_y best")
    for rank, stat in enumerate(rows, start=1):
        best = "*" if bool(stat["is_best"]) else ""
        target_x = stat["target_x"]
        target_y = stat["target_y"]
        target_x_text = "None" if target_x is None else f"{float(target_x):.4f}"
        target_y_text = "None" if target_y is None else f"{float(target_y):.4f}"
        q_text = "None" if stat["q"] is None else f"{float(stat['q']): .5f}"
        prior_text = "None" if stat["prior"] is None else f"{float(stat['prior']): .6f}"
        print(
            f"{rank:>4} "
            f"{int(stat['action_id']):>9} "
            f"{int(stat['visit_count']):>6} "
            f"{q_text:>8} "
            f"{prior_text:>9} "
            f"{int(stat['spin']):>4} "
            f"{float(stat['vx']):> .5f} "
            f"{float(stat['vy']):> .5f} "
            f"{target_x_text:>8} "
            f"{target_y_text:>8} "
            f"{best}"
        )


def _save_json(
    save_json_path: Path,
    dcl2_path: Path,
    line_index: int,
    state_fields: Dict,
    best_action,
    candidate_stats: List[CandidateStat],
) -> None:
    save_json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dcl2_path": str(dcl2_path),
        "line_index": int(line_index),
        "state": {
            "end": int(state_fields["end"]),
            "shot_index": int(state_fields["shot_index"]),
            "hammer_team": int(state_fields["hammer_team"]),
            "score_diff": int(state_fields["score_diff"]),
        },
        "best_action": {
            "vx": float(best_action[0]),
            "vy": float(best_action[1]),
            "spin": int(best_action[2]),
        },
        "candidates": candidate_stats,
    }
    with save_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def plot_mcts_root_candidates(
    log_path: Path,
    model: Path,
    target_end: List[int],
    target_shot: List[int],
    output_path: Optional[Path] = None,
    shuffle_seed: Optional[int] = None,
    sample_index: int = 0,
    skip_unwinnable: bool = True,
    search_method: str = "mcts",
    mcts_simulations: Optional[int] = None,
    shot_simulations: Optional[int] = None,
    use_gpu: bool = True,
    sl_model_is_cnn: bool = True,
    use_value: bool = True,
    use_progressive_widening: bool = True,
    use_transposition_table: bool = True,
    measure_tt_stats: bool = False,
    min_visit: int = 1,
    color_by: str = "visit_count",
    split_by_spin: bool = False,
    console_top_count: int = 10,
    draw_guides: bool = True,
    guide_count: int = 10,
    save_json_path: Optional[Path] = None,
) -> None:
    dcl2_path, line_index, dcl2_state = _load_target_state(
        Path(log_path),
        target_end=target_end,
        target_shot=target_shot,
        shuffle_seed=shuffle_seed,
        sample_index=sample_index,
        skip_unwinnable=skip_unwinnable,
    )
    state_fields = _state_fields(dcl2_state)

    search_method = search_method.lower()
    if search_method not in {"mcts", "shot"}:
        raise ValueError(f"search_method must be 'mcts' or 'shot', got {search_method!r}")

    action_type = "default" if sl_model_is_cnn else TRANSFORMER_VY_MODE
    model_path = Path(model)
    if output_path is None:
        output_path = _auto_output_path(
            model_path,
            search_method,
            sl_model_is_cnn,
            mcts_simulations,
            shot_simulations,
            state_fields,
            sample_index,
        )
    else:
        output_path = Path(output_path)

    network = _load_model(model_path, use_gpu, sl_model_is_cnn, action_type)

    if search_method == "mcts":
        root_state = set_mcts_root_state(
            sl_model=network,
            **state_fields,
            sl_model_is_cnn=sl_model_is_cnn,
        )
        search_action_type = action_type
        search_kwargs = {
            "root_state": root_state,
            "is_create_data": True,
            "use_value": use_value,
            "action_type": search_action_type,
            "use_progressive_widening": use_progressive_widening,
            "use_transposition_table": use_transposition_table,
            "measure_tt_stats": measure_tt_stats,
        }
        if mcts_simulations is not None:
            search_kwargs["max_simulations"] = int(mcts_simulations)

        best_action, root_visit_counts, _ = mcts_search(**search_kwargs)
        best_action_id = _best_action_id_from_visit_counts(root_visit_counts)
        candidate_stats = _build_mcts_candidate_stats(
            root_visit_counts,
            best_action_id,
            search_action_type,
        )
    else:
        search_action_type = action_type
        root_state = set_shot_root_state(
            network=network,
            **state_fields,
            sl_model_is_cnn=sl_model_is_cnn,
        )
        search_kwargs = {
            "root_state": root_state,
            "is_create_data": True,
            "use_value": use_value,
            "action_type": search_action_type,
        }
        if shot_simulations is not None:
            search_kwargs["max_simulations"] = int(shot_simulations)

        best_action_id, shot_stats = shot_search(**search_kwargs)
        best_action = decode_action(best_action_id, action_type=search_action_type)
        candidate_stats = _build_shot_candidate_stats(
            shot_stats,
            best_action_id,
            search_action_type,
        )

    best_stat = _best_candidate_stat(candidate_stats)
    result_state = simulator_step(
        root_state,
        int(best_stat["action_id"]),
        action_type=search_action_type,
    )
    shot_team = root_state.to_move()
    shot_color = "red" if shot_team == 0 else "yellow"
    shot_team_label = f"team{shot_team} ({shot_color})"
    score_diff_for_next = int(state_fields["score_diff"])
    if shot_team == 1:
        score_diff_for_next = -score_diff_for_next
    score_diff_label = f"{score_diff_for_next:+d}"
    title = (
        f"{search_method.upper()} root candidates: end={state_fields['end']} "
        f"shot={state_fields['shot_index']} candidates={len(candidate_stats)}"
    )
    _plot_candidates(
        candidate_stats,
        root_state.stones,
        result_state.stones,
        best_stat,
        output_path,
        title=title,
        shot_team_label=shot_team_label,
        score_diff_label=score_diff_label,
        min_visit=min_visit,
        color_by=color_by,
        split_by_spin=split_by_spin,
        draw_guides=draw_guides,
        guide_count=guide_count,
    )

    _print_top_candidates(candidate_stats, console_top_count)

    if save_json_path is not None:
        _save_json(
            Path(save_json_path),
            dcl2_path,
            line_index,
            state_fields,
            best_action,
            candidate_stats,
        )

    print("-----------------------------------------------------")
    print(f"dcl2: {dcl2_path}")
    print(f"line_index: {line_index}")
    print(f"skip_unwinnable: {skip_unwinnable}")
    print(f"search_method: {search_method}")
    print(f"next_shot: {shot_team_label}")
    print(f"score_diff_for_next_shot_team: {score_diff_label}")
    print(f"output: {output_path}")
    if save_json_path is not None:
        print(f"json: {save_json_path}")
    print("-----------------------------------------------------")


if __name__ == "__main__":
    plot_mcts_root_candidates(
        log_path=PROJECT_ROOT / "LearnLog" / "jiritsu-vs-silicon",
        # model=PROJECT_ROOT / "model" / "js20000CP-32-9-LeaRate1000-vx32-vy25-batchsize1024.bin",
        model=PROJECT_ROOT / "model" / "transformer-supervised-model-AdamW-vy56.bin",
        target_end=[8],
        target_shot=[11],
        # output_path=PROJECT_ROOT / "visualize" / "data" / "mcts_root_candidates_CNN.png",
        output_path=None,
        shuffle_seed=None,
        sample_index=5,
        skip_unwinnable=True,
        search_method="shot",
        mcts_simulations=10000,
        shot_simulations=10000,
        use_gpu=True,
        sl_model_is_cnn=False,
        use_value=True,
        use_progressive_widening=True,
        use_transposition_table=True,
        measure_tt_stats=False,
        min_visit=1,
        color_by="visit_count",
        split_by_spin=False,
        console_top_count=10,
        draw_guides=True,
        guide_count=10,
        save_json_path=None,
    )
