from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NEWSL_DIR = Path(__file__).resolve().parents[2]
if str(NEWSL_DIR) not in sys.path:
    sys.path.insert(0, str(NEWSL_DIR))

from experiment.src.pwtt_search_ablation_report import (
    _condition_info,
    _record_section,
    load_position_records,
)


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_PATH = (
    EXPERIMENT_DIR
    / "data"
    / "pwtt_search_ablation_non-pwtt_vs_pwtt_end9_shot15_datasize1000_x1"
)


COLORS = {
    "condition_a": "#7A869A",
    "condition_b": "#2563EB",
    "ratio": "#0F766E",
    "reference": "#3F3F46",
}


def _bootstrap_mean_ci(values: np.ndarray, seed: int = 12345, resamples: int = 50000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    chunk_size = 2000
    means = np.empty(resamples, dtype=float)
    written = 0
    while written < resamples:
        current_size = min(chunk_size, resamples - written)
        samples = rng.choice(values, size=(current_size, len(values)), replace=True)
        means[written : written + current_size] = samples.mean(axis=1)
        written += current_size
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _position_arrays(records: list[dict]) -> dict[str, np.ndarray | str]:
    condition_a_key, condition_a_label, condition_b_key, condition_b_label = _condition_info(records[0])
    condition_a_sims = []
    condition_b_sims = []
    ratios = []

    for record in records:
        condition_a_sim = float(
            _record_section(record, condition_a_key)["summary"]["mean_simulations"]
        )
        condition_b_sim = float(
            _record_section(record, condition_b_key)["summary"]["mean_simulations"]
        )
        if condition_a_sim <= 0.0:
            continue
        condition_a_sims.append(condition_a_sim)
        condition_b_sims.append(condition_b_sim)
        ratios.append(condition_b_sim / condition_a_sim)

    return {
        "condition_a_key": condition_a_key,
        "condition_a_label": condition_a_label,
        "condition_b_key": condition_b_key,
        "condition_b_label": condition_b_label,
        "condition_a_sims": np.asarray(condition_a_sims, dtype=float),
        "condition_b_sims": np.asarray(condition_b_sims, dtype=float),
        "ratios": np.asarray(ratios, dtype=float),
    }


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D4D4D8", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)


def plot_mean_simulations(records: list[dict], save_path: Path) -> None:
    arrays = _position_arrays(records)
    condition_a_sims = arrays["condition_a_sims"]
    condition_b_sims = arrays["condition_b_sims"]
    condition_a_label = str(arrays["condition_a_label"])
    condition_b_label = str(arrays["condition_b_label"])

    means = np.asarray([condition_a_sims.mean(), condition_b_sims.mean()], dtype=float)
    condition_a_ci = _bootstrap_mean_ci(condition_a_sims)
    condition_b_ci = _bootstrap_mean_ci(condition_b_sims)
    ci_low = np.asarray([condition_a_ci[0], condition_b_ci[0]], dtype=float)
    ci_high = np.asarray([condition_a_ci[1], condition_b_ci[1]], dtype=float)
    yerr = np.vstack([means - ci_low, ci_high - means])

    labels = [condition_a_label, condition_b_label]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(5.2, 7.2))
    bars = ax.bar(
        x,
        means,
        yerr=yerr,
        capsize=7,
        width=0.52,
        color=[COLORS["condition_a"], COLORS["condition_b"]],
        edgecolor="#18181B",
        linewidth=0.8,
        error_kw={"elinewidth": 1.2, "ecolor": "#18181B"},
    )

    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
            color="#18181B",
            fontweight="semibold",
        )

    ax.set_xticks(x, labels, fontsize=12)
    ax.set_ylabel("Mean simulations", fontsize=13)
    ax.set_title("Search Simulations by PW/TT Setting", fontsize=15, pad=14)
    upper = max(ci_high.max() * 1.12, means.max() * 1.18)
    ax.set_ylim(0, upper)
    _style_axes(ax)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_simulation_ratio(records: list[dict], save_path: Path) -> None:
    arrays = _position_arrays(records)
    condition_a_label = str(arrays["condition_a_label"])
    condition_b_label = str(arrays["condition_b_label"])
    ratios = arrays["ratios"]
    ratios = ratios[np.isfinite(ratios)]
    if len(ratios) == 0:
        raise ValueError("No valid simulation ratios found.")

    mean_ratio = float(np.mean(ratios))
    median_ratio = float(np.median(ratios))
    ci_low, ci_high = _bootstrap_mean_ci(ratios)

    fig, ax = plt.subplots(figsize=(5.6, 7.2))
    parts = ax.violinplot(
        ratios,
        positions=[0],
        widths=0.55,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in parts["bodies"]:
        body.set_facecolor(COLORS["ratio"])
        body.set_edgecolor("#134E4A")
        body.set_alpha(0.28)
        body.set_linewidth(1.0)

    rng = np.random.default_rng(12345)
    jitter = rng.normal(0.0, 0.045, size=len(ratios))
    ax.scatter(
        jitter,
        ratios,
        s=18,
        alpha=0.38,
        color=COLORS["ratio"],
        edgecolor="none",
        label="position",
    )

    ax.errorbar(
        0,
        mean_ratio,
        yerr=np.array([[mean_ratio - ci_low], [ci_high - mean_ratio]]),
        fmt="o",
        markersize=8,
        color="#18181B",
        ecolor="#18181B",
        elinewidth=1.4,
        capsize=8,
        label="mean with bootstrap 95% CI",
    )
    ax.scatter([0], [median_ratio], marker="D", s=45, color="#F97316", label="median", zorder=4)
    ax.axhline(1.0, color=COLORS["reference"], linewidth=1.2, linestyle="--", label="no change")

    ax.text(
        0.08,
        mean_ratio,
        f"mean {mean_ratio:.3f}",
        va="center",
        fontsize=11,
        color="#18181B",
    )
    ax.text(
        0.08,
        1.0,
        "1.0",
        va="bottom",
        fontsize=10,
        color=COLORS["reference"],
    )

    ax.set_xlim(-0.55, 0.75)
    ax.set_xticks([0], [f"{condition_b_label} / {condition_a_label}"], fontsize=12)
    ax.set_ylabel("Simulation ratio", fontsize=13)
    ax.set_title("Per-Position Simulation Ratio", fontsize=15, pad=14)

    data_min = float(np.min(ratios))
    data_max = float(np.max(ratios))
    lower = min(0.0, data_min * 0.92, 0.95)
    upper = max(data_max * 1.08, ci_high * 1.08, 1.05)
    ax.set_ylim(lower, upper)
    _style_axes(ax)
    ax.legend(frameon=False, loc="best", fontsize=10)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main(
    target_path: str | Path = DEFAULT_TARGET_PATH,
    output_dir: str | Path | None = None,
) -> None:
    target_path = Path(target_path)
    records = load_position_records(target_path)
    output_dir = Path(output_dir) if output_dir is not None else target_path
    condition_a_key, _, condition_b_key, _ = _condition_info(records[0])

    mean_path = output_dir / f"{condition_a_key}_vs_{condition_b_key}_mean_simulations.png"
    ratio_path = output_dir / f"{condition_a_key}_vs_{condition_b_key}_simulation_ratio.png"
    plot_mean_simulations(records, mean_path)
    plot_simulation_ratio(records, ratio_path)

    print(f"saved: {mean_path}")
    print(f"saved: {ratio_path}")


if __name__ == "__main__":
    main(
        target_path=DEFAULT_TARGET_PATH,
        output_dir=DEFAULT_TARGET_PATH.parents[1] / "plots",
    )