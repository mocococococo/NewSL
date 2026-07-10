from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest


NEWSL_DIR = Path(__file__).resolve().parents[2]
if str(NEWSL_DIR) not in sys.path:
    sys.path.insert(0, str(NEWSL_DIR))

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = EXPERIMENT_DIR / "plots" / "baseline_winrate_summary.png"


@dataclass(frozen=True)
class MethodSpec:
    label: str
    log_dir: Path
    team_name: str


@dataclass(frozen=True)
class WinrateSummary:
    label: str
    team_name: str
    baseline_name: str
    log_dir: Path
    wins: int
    losses: int
    draws: int
    skipped: int
    errors: int

    @property
    def total(self) -> int:
        return self.wins + self.losses

    @property
    def winrate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.wins / self.total

    def exact_ci(self, confidence_level: float = 0.95) -> tuple[float, float]:
        if self.total == 0:
            return 0.0, 0.0
        result = binomtest(self.wins, self.total, p=0.5, alternative="two-sided")
        ci = result.proportion_ci(confidence_level=confidence_level, method="exact")
        return float(ci.low), float(ci.high)


def _game_dirs(log_dir: Path) -> list[Path]:
    if not log_dir.exists():
        return []
    return sorted([path for path in log_dir.iterdir() if path.is_dir()])


def _winner_name(winner: str | None, team0: str, team1: str) -> str | None:
    if winner == "team0":
        return team0
    if winner == "team1":
        return team1
    return None


def _parse_game_result(dcl2_path: Path, team_name: str, baseline_name: str) -> str:
    try:
        lines = dcl2_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if len(lines) < 5:
            return "error"

        team0 = json.loads(lines[3])["log"]["name"]
        team1 = json.loads(lines[4])["log"]["name"]
        valid_pair = {
            team0,
            team1,
        } == {
            team_name,
            baseline_name,
        }
        if not valid_pair:
            return "skipped"

        dcl_json_data = json.loads(lines[-2])
        state_data = dcl_json_data.get("log", {}).get("state")
        if state_data is None:
            return "error"

        game_result = state_data.get("game_result")
        if game_result is None:
            return "error"

        winner = _winner_name(game_result.get("winner"), team0, team1)
        if winner is None:
            return "draw"
        if winner == team_name:
            return "win"
        if winner == baseline_name:
            return "loss"
        return "skipped"
    except (OSError, json.JSONDecodeError, KeyError, IndexError, TypeError):
        return "error"


def calculate_winrate_summary(spec: MethodSpec, baseline_name: str) -> WinrateSummary:
    wins = 0
    losses = 0
    draws = 0
    skipped = 0
    errors = 0

    for game_dir in _game_dirs(spec.log_dir):
        result = _parse_game_result(game_dir / "game.dcl2", spec.team_name, baseline_name)
        if result == "win":
            wins += 1
        elif result == "loss":
            losses += 1
        elif result == "draw":
            draws += 1
        elif result == "skipped":
            skipped += 1
        else:
            errors += 1

    return WinrateSummary(
        label=spec.label,
        team_name=spec.team_name,
        baseline_name=baseline_name,
        log_dir=spec.log_dir,
        wins=wins,
        losses=losses,
        draws=draws,
        skipped=skipped,
        errors=errors,
    )


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D4D4D8", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)


def _print_summary(summaries: list[WinrateSummary]) -> None:
    print("baseline winrate summary")
    for summary in summaries:
        ci_low, ci_high = summary.exact_ci()
        print(
            f"{summary.label}: "
            f"{summary.wins}/{summary.total} = {summary.winrate * 100:.2f}% "
            f"95% CI [{ci_low * 100:.2f}, {ci_high * 100:.2f}] "
            f"draws={summary.draws} skipped={summary.skipped} errors={summary.errors} "
            f"team={summary.team_name} baseline={summary.baseline_name}"
        )


def plot_baseline_winrate_summary(
    method_specs: list[MethodSpec],
    baseline_name: str,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    summaries = [calculate_winrate_summary(spec, baseline_name) for spec in method_specs]
    _print_summary(summaries)

    labels = [summary.label for summary in summaries]
    winrates = np.asarray([summary.winrate * 100.0 for summary in summaries], dtype=float)
    ci_lows = []
    ci_highs = []
    for summary in summaries:
        ci_low, ci_high = summary.exact_ci()
        ci_lows.append(ci_low * 100.0)
        ci_highs.append(ci_high * 100.0)

    ci_low_arr = np.asarray(ci_lows, dtype=float)
    ci_high_arr = np.asarray(ci_highs, dtype=float)
    yerr = np.vstack([winrates - ci_low_arr, ci_high_arr - winrates])
    x = np.arange(len(labels))

    colors = ["#2563EB", "#0F766E", "#F97316", "#7A869A"]
    fig, ax = plt.subplots(figsize=(6.4, 7.6))
    bars = ax.bar(
        x,
        winrates,
        yerr=yerr,
        capsize=7,
        width=0.56,
        color=colors[: len(labels)],
        edgecolor="#18181B",
        linewidth=0.8,
        error_kw={"elinewidth": 1.2, "ecolor": "#18181B"},
    )

    for bar, value in zip(bars, winrates):
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

    ax.axhline(50.0, color="#71717A", linewidth=1.1, linestyle="--")
    ax.set_xticks(x, labels, fontsize=12)
    ax.set_ylabel(f"Winrate vs {baseline_name} (%)", fontsize=13)
    ax.set_title("Winrate Against Baseline", fontsize=15, pad=14)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    _style_axes(ax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"saved: {output_path}")


def main() -> None:
    baseline_name = "CAI-chan"
    method_specs = [
        MethodSpec(
            label="Policy-only",
            log_dir=NEWSL_DIR / "test" / "cai-vs-sl-js",
            team_name="Policy-only",
        ),
        MethodSpec(
            label="PUCT",
            log_dir=NEWSL_DIR / "test" / "cai-vs-puct-js",
            team_name="PUCT",
        ),
        MethodSpec(
            label="PUCT+PW+TT",
            log_dir=NEWSL_DIR / "test" / "cai-vs-puct-pwtt-js",
            team_name="PUCT+PW+TT",
        ),
        MethodSpec(
            label="MCTS_NewSL",
            log_dir=NEWSL_DIR / "test" / "cai-vs-mcts-non-tt",
            team_name="MCTS_NewSL",
        ),
    ]

    plot_baseline_winrate_summary(
        method_specs=method_specs,
        baseline_name=baseline_name,
        output_path=DEFAULT_OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()