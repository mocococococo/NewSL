from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NEWSL_DIR = Path(__file__).resolve().parents[2]
if str(NEWSL_DIR) not in sys.path:
    sys.path.insert(0, str(NEWSL_DIR))

from experiment.src.pwtt_search_ablation_report import _record_section, load_position_records


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = EXPERIMENT_DIR / "data"
DEFAULT_OUTPUT_PATH = EXPERIMENT_DIR / "plots" / "pwtt_condition_mean_simulations.png"


CONDITION_COLORS = {
    "PWTT": "#2563EB",
    "PW": "#0F766E",
    "TT": "#F97316",
    "non-PWTT": "#7A869A",
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


def _condition_values(json_dir: Path, condition_key: str) -> np.ndarray:
    records = load_position_records(json_dir)
    values = []
    for record in records:
        summary = _record_section(record, condition_key)["summary"]
        values.append(float(summary["mean_simulations"]))
    return np.asarray(values, dtype=float)


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D4D4D8", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)


def plot_condition_summary(
    pwtt_vs_non_pwtt_dir: Path,
    pwtt_vs_non_pw_dir: Path,
    pwtt_vs_non_tt_dir: Path,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    items = [
        ("PWTT", pwtt_vs_non_pwtt_dir, "pwtt"),
        ("PW", pwtt_vs_non_tt_dir, "non-tt"),
        ("TT", pwtt_vs_non_pw_dir, "non-pw"),
        ("non-PWTT", pwtt_vs_non_pwtt_dir, "non-pwtt"),
    ]

    labels = []
    means = []
    ci_lows = []
    ci_highs = []
    counts = []

    for label, json_dir, condition_key in items:
        values = _condition_values(json_dir, condition_key)
        ci_low, ci_high = _bootstrap_mean_ci(values)
        labels.append(label)
        means.append(float(np.mean(values)))
        ci_lows.append(ci_low)
        ci_highs.append(ci_high)
        counts.append(len(values))

    means_arr = np.asarray(means, dtype=float)
    ci_low_arr = np.asarray(ci_lows, dtype=float)
    ci_high_arr = np.asarray(ci_highs, dtype=float)
    yerr = np.vstack([means_arr - ci_low_arr, ci_high_arr - means_arr])
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(6.2, 7.6))
    bars = ax.bar(
        x,
        means_arr,
        yerr=yerr,
        capsize=7,
        width=0.56,
        color=[CONDITION_COLORS[label] for label in labels],
        edgecolor="#18181B",
        linewidth=0.8,
        error_kw={"elinewidth": 1.2, "ecolor": "#18181B"},
    )

    for bar, value, count in zip(bars, means_arr, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
            color="#18181B",
            fontweight="semibold",
        )

    ax.set_xticks(x, labels, fontsize=12)
    ax.set_ylabel("Mean simulations", fontsize=13)
    ax.set_title("Mean Search Simulations by PW/TT Condition", fontsize=15, pad=14)
    upper = max(float(ci_high_arr.max()) * 1.15, float(means_arr.max()) * 1.22)
    ax.set_ylim(0, upper)
    _style_axes(ax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"saved: {output_path}")


def main() -> None:
    pwtt_vs_non_pwtt_dir = DATA_DIR / "pwtt_search_ablation_non-pwtt_vs_pwtt_end9_shot15_datasize1000_x1"
    if not pwtt_vs_non_pwtt_dir.exists():
        pwtt_vs_non_pwtt_dir = DATA_DIR / "pwtt_search_ablation_end9_shot15_datasize1000_x1"
    pwtt_vs_non_pw_dir = DATA_DIR / "pwtt_search_ablation_non-pw_vs_pwtt_end9_shot15_datasize1000_x1"
    pwtt_vs_non_tt_dir = DATA_DIR / "pwtt_search_ablation_non-tt_vs_pwtt_end9_shot15_datasize1000_x1"

    plot_condition_summary(
        pwtt_vs_non_pwtt_dir=pwtt_vs_non_pwtt_dir,
        pwtt_vs_non_pw_dir=pwtt_vs_non_pw_dir,
        pwtt_vs_non_tt_dir=pwtt_vs_non_tt_dir,
        output_path=DEFAULT_OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()