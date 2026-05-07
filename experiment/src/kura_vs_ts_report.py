"""kura_vs_ts.py と print_kura_vs_ts.py で共通利用する出力処理。

このファイルには、実験結果の PNG 描画とコンソール表示だけを置く。
Kura や NewSL の重いモデル読み込みには依存させない。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_result_plot(
    save_file_path: Path,
    kura_result_means_x: np.ndarray,
    newsl_result_means_x: np.ndarray,
) -> None:
    """局面ごとの result_mean_x と、NewSL - Kura の差分を PNG に保存する。"""

    import matplotlib.pyplot as plt

    # 上段に各手法の result_mean_x、下段に差分を描画する。
    indices = np.arange(len(kura_result_means_x))
    diff_scores = newsl_result_means_x - kura_result_means_x
    if len(indices) == 0:
        raise ValueError("kura_result_means_x と newsl_result_means_x は空にできません。")
    x_max = len(indices) - 1

    figure, (ax_score, ax_diff) = plt.subplots(2, 1, figsize=(12, 8))

    # 局面数が多いと x 軸ラベルが詰まるため、最大 20 分割程度に間引く。
    x_tick_step = max(1, int(np.ceil((x_max + 1) / 20)))
    x_ticks = np.arange(0, x_max + 1, x_tick_step)
    if x_ticks[-1] != x_max:
        x_ticks = np.append(x_ticks, x_max)

    ax_score.plot(indices, kura_result_means_x, label="Kura", linewidth=1.0, alpha=0.5)
    ax_score.plot(indices, newsl_result_means_x, label="NewSL", linewidth=1.0, alpha=0.5)
    ax_score.set_ylabel("Result Mean Over X Runs")
    ax_score.set_title("Kura vs NewSL")
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
    ax_diff.set_title("NewSL - Kura")
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


def print_kura_vs_ts_summary(
    json_dir: Path,
    png_path: Path,
    summary: dict,
) -> None:
    """kura_vs_ts.py と同じ形式で集計結果をコンソールに表示する。"""

    print("")
    print(f"Saved position json files to {json_dir}")
    print(f"Saved plot to {png_path}")
    print(f"num_positions                : {summary['num_positions']}")
    print(f"execution_repeats_x         : {summary['execution_repeats_x']}")
    print(f"log_size result_mean_x Kura : {summary['log_size_result_mean_x_kura']:.6f}")
    print(f"log_size result_mean_x NewSL: {summary['log_size_result_mean_x_transformer']:.6f}")
    print(f"log_size diff (NewSL - Kura): {summary['log_size_diff_result_mean_x_transformer_minus_kura']:.6f}")
    print(
        "log_size win/draw/lose Kura : "
        f"{summary['log_size_win_rate_x_kura']:.6f}, "
        f"{summary['log_size_draw_rate_x_kura']:.6f}, "
        f"{summary['log_size_lose_rate_x_kura']:.6f}"
    )
    print(
        "log_size win/draw/lose NewSL: "
        f"{summary['log_size_win_rate_x_transformer']:.6f}, "
        f"{summary['log_size_draw_rate_x_transformer']:.6f}, "
        f"{summary['log_size_lose_rate_x_transformer']:.6f}"
    )
    print(
        "all_trials win/draw/lose Kura : "
        f"{summary['all_trials_win_count_kura']}, "
        f"{summary['all_trials_draw_count_kura']}, "
        f"{summary['all_trials_lose_count_kura']}"
    )
    print(
        "all_trials win/draw/lose NewSL: "
        f"{summary['all_trials_win_count_transformer']}, "
        f"{summary['all_trials_draw_count_transformer']}, "
        f"{summary['all_trials_lose_count_transformer']}"
    )
    print(f"NewSL better by result_mean_x count       : {summary['transformer_better_by_result_mean_x_count']}")
    print(f"Kura better by result_mean_x count        : {summary['kura_better_by_result_mean_x_count']}")
    print(f"Tie by result_mean_x count                : {summary['tie_by_result_mean_x_count']}")
    print("")
    print("root_view_score_diff_before_shot bucket summary (all trials in each bucket)")
    for bucket_label in summary["root_view_score_diff_bucket_order"]:
        bucket_summary = summary["root_view_score_diff_bucket_summary"][bucket_label]
        kura_bucket = bucket_summary["kura"]
        newsl_bucket = bucket_summary["transformer"]
        print(
            f"bucket {bucket_label:>4} positions={kura_bucket['num_positions']:4d} trials={kura_bucket['total_trials']:5d} "
            f"| Kura result_mean={kura_bucket['result_mean_over_all_trials']:.4f} "
            f"win/draw/lose={kura_bucket['win_rate_over_all_trials']:.4f},"
            f"{kura_bucket['draw_rate_over_all_trials']:.4f},"
            f"{kura_bucket['lose_rate_over_all_trials']:.4f} "
            f"| NewSL result_mean={newsl_bucket['result_mean_over_all_trials']:.4f} "
            f"win/draw/lose={newsl_bucket['win_rate_over_all_trials']:.4f},"
            f"{newsl_bucket['draw_rate_over_all_trials']:.4f},"
            f"{newsl_bucket['lose_rate_over_all_trials']:.4f}"
        )
