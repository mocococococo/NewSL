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


def _legacy_key(key: str) -> str:
    return str(key).replace("-", "_")


def _record_section(record: dict[str, Any], key: str) -> dict[str, Any]:
    if key in record and isinstance(record[key], dict):
        return record[key]
    legacy_key = _legacy_key(key)
    if legacy_key in record and isinstance(record[legacy_key], dict):
        return record[legacy_key]
    raise KeyError(f"record does not contain condition section: {key}")


def _condition_info(record: dict[str, Any]) -> tuple[str, str, str, str]:
    experiment = record.get("experiment", {})
    if not isinstance(experiment, dict):
        experiment = {}

    condition_a_key = str(experiment.get("condition_a_key", "non-pwtt")).replace("_", "-")
    condition_b_key = str(experiment.get("condition_b_key", "pwtt")).replace("_", "-")
    condition_a_label = str(experiment.get("condition_a_label", condition_a_key))
    condition_b_label = str(experiment.get("condition_b_label", condition_b_key))
    return condition_a_key, condition_a_label, condition_b_key, condition_b_label


def _is_position_record(record: dict[str, Any]) -> bool:
    if not isinstance(record.get("position"), dict):
        return False
    if not isinstance(record.get("comparison"), dict):
        return False
    try:
        condition_a_key, _, condition_b_key, _ = _condition_info(record)
        _record_section(record, condition_a_key)
        _record_section(record, condition_b_key)
    except KeyError:
        return False
    return True


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
    condition_a_key, _, condition_b_key, _ = _condition_info(records[0])
    condition_a_sims = []
    condition_b_sims = []
    condition_a_nodes = []
    condition_b_nodes = []
    condition_a_hit = []
    condition_b_hit = []
    sim_diff = []
    sim_ratio = []
    log_ratio = []

    for record in records:
        condition_a_summary = _record_section(record, condition_a_key)["summary"]
        condition_b_summary = _record_section(record, condition_b_key)["summary"]
        condition_a_sim = float(condition_a_summary["mean_simulations"])
        condition_b_sim = float(condition_b_summary["mean_simulations"])
        ratio = condition_b_sim / condition_a_sim if condition_a_sim > 0.0 else np.nan

        condition_a_sims.append(condition_a_sim)
        condition_b_sims.append(condition_b_sim)
        condition_a_nodes.append(float(condition_a_summary["mean_nodes"]))
        condition_b_nodes.append(float(condition_b_summary["mean_nodes"]))
        condition_a_hit.append(float(condition_a_summary["mean_tt_hit_rate_estimate"]))
        condition_b_hit.append(float(condition_b_summary["mean_tt_hit_rate_estimate"]))
        sim_diff.append(condition_b_sim - condition_a_sim)
        sim_ratio.append(ratio)
        log_ratio.append(float(np.log(ratio)) if ratio > 0.0 else np.nan)

    return {
        "condition_a_sims": np.asarray(condition_a_sims, dtype=float),
        "condition_b_sims": np.asarray(condition_b_sims, dtype=float),
        "condition_a_nodes": np.asarray(condition_a_nodes, dtype=float),
        "condition_b_nodes": np.asarray(condition_b_nodes, dtype=float),
        "condition_a_hit": np.asarray(condition_a_hit, dtype=float),
        "condition_b_hit": np.asarray(condition_b_hit, dtype=float),
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
    condition_a_key, condition_a_label, condition_b_key, condition_b_label = _condition_info(records[0])
    values = _position_values(records)
    n = len(records)
    x_repeats = int(records[0]["experiment"].get("execution_repeats_x", 1))

    mean_a_sim = float(np.mean(values["condition_a_sims"]))
    mean_b_sim = float(np.mean(values["condition_b_sims"]))
    mean_a_nodes = float(np.mean(values["condition_a_nodes"]))
    mean_b_nodes = float(np.mean(values["condition_b_nodes"]))
    mean_a_hit = float(np.mean(values["condition_a_hit"]))
    mean_b_hit = float(np.mean(values["condition_b_hit"]))
    mean_diff = float(np.mean(values["sim_diff"]))
    median_diff = float(np.median(values["sim_diff"]))
    mean_ratio = float(np.nanmean(values["sim_ratio"]))
    median_ratio = float(np.nanmedian(values["sim_ratio"]))
    mean_log_ratio = float(np.nanmean(values["log_ratio"]))
    increase_rates = values["sim_ratio"] - 1.0
    increase_ci_low, increase_ci_high = _bootstrap_mean_ci(increase_rates)

    wilcoxon_two_sided_p = _safe_wilcoxon(values["log_ratio"], "two-sided")
    wilcoxon_greater_p = _safe_wilcoxon(values["log_ratio"], "greater")

    print("-----------------------------------------------------")
    print("PW/TT search ablation summary")
    print(f"condition A: {condition_a_label} ({condition_a_key})")
    print(f"condition B: {condition_b_label} ({condition_b_key})")
    print(f"positions: {n}, repeats per position: {x_repeats}")
    print(f"{condition_a_label} mean simulations: {mean_a_sim:.3f}")
    print(f"{condition_b_label} mean simulations: {mean_b_sim:.3f}")
    print(f"mean simulation diff ({condition_b_label} - {condition_a_label}): {mean_diff:.3f}")
    print(f"median simulation diff ({condition_b_label} - {condition_a_label}): {median_diff:.3f}")
    print(f"mean simulation increase rate ({condition_b_label} / {condition_a_label}): {(mean_ratio - 1.0) * 100:.3f}%")
    print(f"median simulation increase rate ({condition_b_label} / {condition_a_label}): {(median_ratio - 1.0) * 100:.3f}%")
    print(f"mean log simulation ratio: {mean_log_ratio:.6f}")
    print(
        "bootstrap 95% CI of mean simulation increase rate: "
        f"[{increase_ci_low * 100:.3f}, {increase_ci_high * 100:.3f}]%"
    )
    print(f"{condition_a_label} mean nodes: {mean_a_nodes:.3f}")
    print(f"{condition_b_label} mean nodes: {mean_b_nodes:.3f}")
    print(f"{condition_a_label} estimated TT hit rate: {mean_a_hit * 100:.3f}%")
    print(f"{condition_b_label} estimated TT hit rate: {mean_b_hit * 100:.3f}%")
    print(
        f"Wilcoxon signed-rank test on log(sim_{condition_b_key} / sim_{condition_a_key}) "
        f"two-sided p: {_format_p_value(wilcoxon_two_sided_p)}"
    )
    print(
        f"Wilcoxon signed-rank test on log(sim_{condition_b_key} / sim_{condition_a_key}) "
        f"one-sided p ({condition_b_label} > {condition_a_label}): {_format_p_value(wilcoxon_greater_p)}"
    )
    print("-----------------------------------------------------")


def print_report_from_json_dir(json_dir: str | Path, max_positions: int | None = None) -> None:
    records = load_position_records(Path(json_dir))
    if max_positions is not None:
        records = records[:max_positions]
    print_report_from_records(records)