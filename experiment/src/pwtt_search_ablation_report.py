from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon


NEWSL_DIR = Path(__file__).resolve().parents[2]
if str(NEWSL_DIR) not in sys.path:
    sys.path.insert(0, str(NEWSL_DIR))


BOOTSTRAP_RESAMPLES = 100000
BOOTSTRAP_SEED = 12345


def _format_p_value(value: float) -> str:
    if value < 1e-6:
        return f"{value:.6e}"
    return f"{value:.6f}"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _is_position_record(record: dict[str, Any]) -> bool:
    return (
        isinstance(record.get("position"), dict)
        and isinstance(record.get("non_pwtt"), dict)
        and isinstance(record.get("pwtt"), dict)
        and isinstance(record.get("comparison"), dict)
    )


def load_position_records(json_dir: Path) -> list[dict[str, Any]]:
    if not json_dir.is_dir():
        raise FileNotFoundError(f"{json_dir} does not exist or is not a directory")

    records: list[dict[str, Any]] = []
    for json_path in sorted(json_dir.glob("*.json")):
        if json_path.name == "metadata.json":
            continue
        record = _read_json(json_path)
        if not _is_position_record(record):
            raise ValueError(f"{json_path} is not a PWTT search ablation JSON file")
        records.append(record)

    if not records:
        raise ValueError(f"No position JSON files found in {json_dir}")
    return sorted(records, key=lambda record: int(record["position"]["position_index"]))


def _position_values(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    non_pwtt_sims = []
    pwtt_sims = []
    non_pwtt_nodes = []
    pwtt_nodes = []
    non_pwtt_hit = []
    pwtt_hit = []
    sim_diff = []
    sim_ratio = []
    log_ratio = []

    for record in records:
        non_summary = record["non_pwtt"]["summary"]
        pwtt_summary = record["pwtt"]["summary"]
        non_sim = float(non_summary["mean_simulations"])
        pwtt_sim = float(pwtt_summary["mean_simulations"])
        ratio = pwtt_sim / non_sim if non_sim > 0.0 else np.nan

        non_pwtt_sims.append(non_sim)
        pwtt_sims.append(pwtt_sim)
        non_pwtt_nodes.append(float(non_summary["mean_nodes"]))
        pwtt_nodes.append(float(pwtt_summary["mean_nodes"]))
        non_pwtt_hit.append(float(non_summary["mean_tt_hit_rate_estimate"]))
        pwtt_hit.append(float(pwtt_summary["mean_tt_hit_rate_estimate"]))
        sim_diff.append(pwtt_sim - non_sim)
        sim_ratio.append(ratio)
        log_ratio.append(float(np.log(ratio)) if ratio > 0.0 else np.nan)

    return {
        "non_pwtt_sims": np.asarray(non_pwtt_sims, dtype=float),
        "pwtt_sims": np.asarray(pwtt_sims, dtype=float),
        "non_pwtt_nodes": np.asarray(non_pwtt_nodes, dtype=float),
        "pwtt_nodes": np.asarray(pwtt_nodes, dtype=float),
        "non_pwtt_hit": np.asarray(non_pwtt_hit, dtype=float),
        "pwtt_hit": np.asarray(pwtt_hit, dtype=float),
        "sim_diff": np.asarray(sim_diff, dtype=float),
        "sim_ratio": np.asarray(sim_ratio, dtype=float),
        "log_ratio": np.asarray(log_ratio, dtype=float),
    }


def _bootstrap_mean_ci(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0, 0.0

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    chunk_size = 2000
    means = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    written = 0
    while written < BOOTSTRAP_RESAMPLES:
        current_size = min(chunk_size, BOOTSTRAP_RESAMPLES - written)
        samples = rng.choice(values, size=(current_size, len(values)), replace=True)
        means[written : written + current_size] = samples.mean(axis=1)
        written += current_size

    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _safe_wilcoxon(values: np.ndarray, alternative: str) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0 or np.allclose(values, 0.0):
        return 1.0
    return float(wilcoxon(values, alternative=alternative, zero_method="wilcox").pvalue)


def print_report_from_records(records: list[dict[str, Any]]) -> None:
    values = _position_values(records)
    n = len(records)
    x_repeats = int(records[0]["experiment"].get("execution_repeats_x", 1))

    mean_non_sim = float(np.mean(values["non_pwtt_sims"]))
    mean_pwtt_sim = float(np.mean(values["pwtt_sims"]))
    mean_non_nodes = float(np.mean(values["non_pwtt_nodes"]))
    mean_pwtt_nodes = float(np.mean(values["pwtt_nodes"]))
    mean_non_hit = float(np.mean(values["non_pwtt_hit"]))
    mean_pwtt_hit = float(np.mean(values["pwtt_hit"]))
    mean_diff = float(np.mean(values["sim_diff"]))
    median_diff = float(np.median(values["sim_diff"]))
    mean_ratio = float(np.nanmean(values["sim_ratio"]))
    median_ratio = float(np.nanmedian(values["sim_ratio"]))
    mean_log_ratio = float(np.nanmean(values["log_ratio"]))
    log_ci_low, log_ci_high = _bootstrap_mean_ci(values["log_ratio"])
    ratio_ci_low = float(np.exp(log_ci_low) - 1.0)
    ratio_ci_high = float(np.exp(log_ci_high) - 1.0)

    wilcoxon_two_sided_p = _safe_wilcoxon(values["log_ratio"], "two-sided")
    wilcoxon_greater_p = _safe_wilcoxon(values["log_ratio"], "greater")

    print("-----------------------------------------------------")
    print("PWTT search ablation summary")
    print(f"positions: {n}, repeats per position: {x_repeats}")
    print(f"non-PWTT mean simulations: {mean_non_sim:.3f}")
    print(f"PWTT mean simulations: {mean_pwtt_sim:.3f}")
    print(f"mean simulation diff (PWTT - non-PWTT): {mean_diff:.3f}")
    print(f"median simulation diff (PWTT - non-PWTT): {median_diff:.3f}")
    print(f"mean simulation increase rate: {(mean_ratio - 1.0) * 100:.3f}%")
    print(f"median simulation increase rate: {(median_ratio - 1.0) * 100:.3f}%")
    print(f"mean log simulation ratio: {mean_log_ratio:.6f}")
    print(
        "bootstrap 95% CI of mean simulation increase rate: "
        f"[{ratio_ci_low * 100:.3f}, {ratio_ci_high * 100:.3f}]%"
    )
    print(f"non-PWTT mean nodes: {mean_non_nodes:.3f}")
    print(f"PWTT mean nodes: {mean_pwtt_nodes:.3f}")
    print(f"non-PWTT estimated TT hit rate: {mean_non_hit * 100:.3f}%")
    print(f"PWTT estimated TT hit rate: {mean_pwtt_hit * 100:.3f}%")
    print(
        "Wilcoxon signed-rank test on log(sim_PWTT / sim_non_PWTT) "
        f"two-sided p: {_format_p_value(wilcoxon_two_sided_p)}"
    )
    print(
        "Wilcoxon signed-rank test on log(sim_PWTT / sim_non_PWTT) "
        f"one-sided p (PWTT > non-PWTT): {_format_p_value(wilcoxon_greater_p)}"
    )
    print("-----------------------------------------------------")


def print_report_from_json_dir(json_dir: str | Path, max_positions: int | None = None) -> None:
    records = load_position_records(Path(json_dir))
    if max_positions is not None:
        records = records[:max_positions]
    print_report_from_records(records)