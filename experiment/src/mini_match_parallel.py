"""experiment/src/mini_match_parallel.py として配置して実行する並列版。

現在の mini_match.py を変更せず、その対戦・集計処理を利用する。
Windows/CUDA に対応するため、num_workers=1 の場合も spawn を使う。
各ワーカーは起動時にモデルを読み込み、複数の局面で再利用する。
設定は末尾の main(...) で変更する。
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


# 直接実行と python -m experiment.src.mini_match_parallel の両方に対応。
_repo_dir = Path(__file__).resolve().parents[2]
if (_repo_dir / "experiment" / "src" / "mini_match.py").is_file():
    sys.path.insert(0, str(_repo_dir))

from experiment.src import mini_match as mm


NEWSL_DIR = mm.NEWSL_DIR
_WORKER: dict[str, Any] = {}


def _prepare_models(
    kinds, target_end, target_shot, final_end, run_name, cnn_model,
    transformer_models_by_shot, kura_policy_models_by_shot,
    kura_value_models_by_shot,
) -> dict[str, Any]:
    """モデルは親で存在確認だけ行い、読み込みはワーカーに任せる。"""
    if run_name is not None and transformer_models_by_shot is not None:
        raise ValueError("run_name と transformer_models_by_shot は同時に指定できません。")
    if transformer_models_by_shot is None:
        if run_name is not None:
            transformer_models_by_shot = mm._build_distributed_transformer_models(
                run_name, target_end, target_shot, final_end=final_end,
            )
        else:
            transformer_models_by_shot = {
                14: "transformer-sl-9-14-model-06-04-adamw-epoch50-shot.bin",
                15: "transformer-sl-9-15-model-06-02-adamw-epoch50-shot.bin",
            }
    transformers = {}
    for key, path in transformer_models_by_shot.items():
        end, shot = key if isinstance(key, tuple) else (target_end, key)
        transformers[(int(end), int(shot))] = mm._resolve_newsl_model(path)

    policy = {
        int(shot): mm._resolve_kura_model(path)
        for shot, path in (
            mm.KURA_POLICY_MODELS_BY_SHOT
            if kura_policy_models_by_shot is None else kura_policy_models_by_shot
        ).items()
    }
    value = {
        int(shot): mm._resolve_kura_model(path)
        for shot, path in (
            mm.KURA_VALUE_MODELS_BY_SHOT
            if kura_value_models_by_shot is None else kura_value_models_by_shot
        ).items()
    }
    cnn = mm._resolve_newsl_model(cnn_model)
    required_files = []
    if "transformer" in kinds:
        missing = [
            (end, shot)
            for end in range(target_end, final_end + 1)
            for shot in range(target_shot if end == target_end else 0, 16)
            if (end, shot) not in transformers
        ]
        if missing:
            raise ValueError(f"Transformerモデルの指定が不足しています: {missing}")
        required_files.extend(transformers.values())
    if "cnn" in kinds:
        required_files.append(cnn)
    if "kura" in kinds:
        first_shot = 0 if final_end > target_end else target_shot
        for name, models, first in (
            ("policy", policy, max(first_shot, 1)),
            ("value", value, max(first_shot + 1, 2)),
        ):
            missing = [shot for shot in range(first, 16) if shot not in models]
            if missing:
                raise ValueError(f"Kura {name}モデルの指定が不足しています: {missing}")
            required_files.extend(models.values())
    missing_files = sorted({str(path) for path in required_files if not path.is_file()})
    if missing_files:
        raise FileNotFoundError("モデルが見つかりません:\n" + "\n".join(missing_files))
    return {
        "cnn_model": cnn,
        "transformer_models_by_shot": transformers,
        "kura_policy_models_by_shot": policy,
        "kura_value_models_by_shot": value,
    }


def _read_positions(log_path, target_end, target_shot, data_size, shuffle_seed):
    # mini_match.py と同じ順序で選び、モデルや State はプロセス間で渡さない。
    log_names = os.listdir(log_path)
    log_names = random.Random(shuffle_seed).sample(log_names, len(log_names))
    jobs = []
    for log_name in log_names:
        path = Path(log_path) / log_name / "game.dcl2"
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_index in range(9, len(lines) - 2, 2):
            try:
                state = json.loads(lines[line_index])["log"]["state"]
                end, shot = int(state["end"]), int(state["shot"])
                if (end, shot) != (target_end, target_shot):
                    continue
                stones = state["stones"]["team0"] + state["stones"]["team1"]
                hammer = mm.convert_team_stoi(state["hammer"])
                score = mm.scores_to_scorediff_for_team0(state["scores"])
            except KeyError:
                continue
            jobs.append({
                "position_index": len(jobs), "log_name": log_name,
                "line_index": line_index, "end": end, "shot": shot,
                "hammer_team": hammer, "score_diff_for_team0": score,
                "stones": mm.stones_listdict_to_xy16(stones),
            })
            if len(jobs) >= data_size:
                return jobs
    return jobs


def _init_worker(config, models, metadata):
    """ProcessPoolExecutor が各プロセスにつき一度だけ呼び出す。"""
    mm.torch.set_num_threads(config["torch_threads_per_worker"])
    start = time.perf_counter()
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            kwargs = dict(
                models, use_gpu=config["use_gpu"], target_end=config["target_end"],
                max_simulations=config["max_simulations"],
                search_time_limit=config["search_time_limit"],
            )
            player_a = mm.build_player(
                config["player_a_kind"],
                search_method=config["player_a_search_method"],
                **kwargs,
            )
            player_b = mm.build_player(
                config["player_b_kind"],
                search_method=config["player_b_search_method"],
                **kwargs,
            )
    except Exception:
        print(captured.getvalue(), file=sys.stderr, flush=True)
        raise
    metadata = dict(metadata)
    for name, player in (("a", player_a), ("b", player_b)):
        metadata.update({
            f"player_{name}_start_method_key": f"{player.key}_start",
            f"player_{name}_start_method_label": f"{player.label}-start",
            f"player_{name}_label": player.label,
            f"player_{name}_search_method": player.search_method,
        })
    _WORKER.update(
        config=config, metadata=metadata, player_a=player_a, player_b=player_b,
        model_load_seconds=time.perf_counter() - start, positions_processed=0,
    )
    print(f"[worker {os.getpid()}] モデル読み込み完了 "
          f"({_WORKER['model_load_seconds']:.1f}秒)", flush=True)


def _evaluate_position(job):
    config = _WORKER["config"]
    position = dict(job)
    stones = position.pop("stones")
    root = mm.State.initial(
        stones=stones, end=position["end"], shot_index=position["shot"],
        hammer_team=position["hammer_team"],
        score_diff=position["score_diff_for_team0"],
    )
    root_team = int(root.to_move())
    root_score = mm.score_diff_for_team_view(root.score_diff, root_team)
    position.update(
        root_view_team=root_team, root_view_score_diff_before_shot=root_score,
        root_view_score_diff_bucket=mm.bucket_root_view_score_diff(root_score),
    )
    seed = config["simulation_seed"]
    if seed is not None:
        # 担当プロセスや実行順が変わっても、局面ごとの乱数種は変えない。
        seed = (seed + position["position_index"]) % (2 ** 32)
        random.seed(seed)
        mm.np.random.seed(seed)
        mm.torch.manual_seed(seed)
    captured = io.StringIO()
    started = time.perf_counter()
    result = {"experiment": _WORKER["metadata"], "position": position}
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            for side, start_with_a in (("a", True), ("b", False)):
                counts, mean, win, draw, lose, actions = mm.evaluate_start_pattern(
                    root, _WORKER["player_a"], _WORKER["player_b"],
                    start_with_a=start_with_a, x_repeats=config["X"],
                    final_end=config["final_end"], root_view_team=root_team,
                    root_view_score_diff_before_shot=root_score,
                )
                player = _WORKER[f"player_{side}"]
                result[f"player_{side}_start"] = {
                    "method_key": f"{player.key}_start",
                    "method_label": f"{player.label}-start",
                    "result_mean_x": float(mean),
                    "win_count_x": int(counts[mm.RESULT_WIN_IDX]),
                    "draw_count_x": int(counts[mm.RESULT_DRAW_IDX]),
                    "lose_count_x": int(counts[mm.RESULT_LOSE_IDX]),
                    "win_rate_x": float(win), "draw_rate_x": float(draw),
                    "lose_rate_x": float(lose), "sample_action_log": actions,
                }
    except Exception as exc:
        raise RuntimeError(
            f"position={position['position_index']}, log={position['log_name']} "
            f"で失敗しました。\n{captured.getvalue()[-4000:]}"
        ) from exc
    finally:
        # モデルへの参照は _WORKER に残り、次の局面でも再利用される。
        mm.release_position_memory(config["use_gpu"])
    diff = result["player_b_start"]["result_mean_x"] - result["player_a_start"]["result_mean_x"]
    result["comparison"] = {
        "diff_result_mean_x_player_b_start_minus_player_a_start": diff,
        "better_by_result_mean_x": (
            "player_b_start" if diff > 0 else "player_a_start" if diff < 0 else "tie"
        ),
    }
    _WORKER["positions_processed"] += 1
    result["execution"] = {
        "worker_pid": os.getpid(),
        "worker_position_number": _WORKER["positions_processed"],
        "model_load_seconds": _WORKER["model_load_seconds"],
        "position_seconds": time.perf_counter() - started,
        "simulation_seed": seed,
    }
    return result, captured.getvalue()


def main(
    log_path: str | Path = NEWSL_DIR / "LearnLog" / "all",
    save_path: str | Path = NEWSL_DIR / "experiment" / "data",
    player_a_kind: str = "Kura",
    player_b_kind: str = "Transformer",
    target_end: int = 9,
    target_shot: int = 14,
    run_name: str | None = None,
    data_size: int = 1000,
    X: int = 1,
    use_gpu: bool = True,
    max_simulations: int = 1000,
    cnn_model: str | Path = "js20000CP-32-9-LeaRate1000-vx32-vy25-batchsize1024.bin",
    transformer_models_by_shot: dict[int | tuple[int, int], str | Path] | None = None,
    kura_policy_models_by_shot: dict[int, str | Path] | None = None,
    kura_value_models_by_shot: dict[int, str | Path] | None = None,
    shuffle_seed: int | None = 12345,
    final_end: int = 9,
    search_time_limit: float | None = mm.DEFAULT_SHOT_TIME_LIMIT_SEC,
    num_workers: int = 2,
    simulation_seed: int | None = 12345,
    torch_threads_per_worker: int = 1,
    player_a_search_method: str = "shot",
    player_b_search_method: str = "shot",
) -> Path:
    """一つの局面の A-start/B-start を同じワーカーで評価し、保存先を返す。"""
    started = time.perf_counter()
    for name, number in (
        ("num_workers", num_workers), ("data_size", data_size), ("X", X),
        ("max_simulations", max_simulations),
        ("torch_threads_per_worker", torch_threads_per_worker),
    ):
        if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
            raise ValueError(f"{name} は正の整数で指定してください。")
    if not (0 <= target_end <= final_end <= 9 and 0 <= target_shot <= 15):
        raise ValueError("0 <= target_end <= final_end <= 9、0 <= target_shot <= 15 が必要です。")
    if search_time_limit is not None:
        search_time_limit = float(search_time_limit)
        if not math.isfinite(search_time_limit) or search_time_limit <= 0:
            raise ValueError("search_time_limit は正の秒数、または None で指定してください。")
    kinds = (player_a_kind.strip().lower(), player_b_kind.strip().lower())
    if any(kind not in {"kura", "cnn", "transformer"} for kind in kinds):
        raise ValueError("player_a_kind / player_b_kind は Kura、cnn、Transformer のいずれかです。")
    models = _prepare_models(
        kinds, target_end, target_shot, final_end, run_name, cnn_model,
        transformer_models_by_shot, kura_policy_models_by_shot, kura_value_models_by_shot,
    )
    jobs = _read_positions(log_path, target_end, target_shot, data_size, shuffle_seed)
    if not jobs:
        raise RuntimeError(f"{log_path} に end={target_end}, shot={target_shot} の局面がありません。")
    actual_workers = min(num_workers, len(jobs))
    config = dict(
        player_a_kind=player_a_kind, player_b_kind=player_b_kind,
        player_a_search_method=player_a_search_method,
        player_b_search_method=player_b_search_method,
        target_end=target_end, final_end=final_end, X=X, use_gpu=use_gpu,
        max_simulations=max_simulations, search_time_limit=search_time_limit,
        simulation_seed=simulation_seed, torch_threads_per_worker=torch_threads_per_worker,
    )
    transformers = models["transformer_models_by_shot"]
    metadata = {
        "experiment_type": "mini_match", "execution_mode": "process_pool",
        "target_end": target_end, "target_shot": target_shot, "final_end": final_end,
        "requested_data_size": data_size, "actual_data_size": len(jobs),
        "execution_repeats_x": X, "player_a_kind": player_a_kind,
        "player_b_kind": player_b_kind, "max_simulations": max_simulations,
        "search_time_limit": search_time_limit, "use_gpu": use_gpu,
        "requested_num_workers": num_workers, "num_workers": actual_workers,
        "shuffle_seed": shuffle_seed, "simulation_seed": simulation_seed,
        "torch_threads_per_worker": torch_threads_per_worker,
        "transformer_run_name": run_name, "cnn_model": str(models["cnn_model"]),
        "transformer_models_by_shot": {
            int(shot): str(path) for (end, shot), path in transformers.items()
            if end == target_end
        },
        "transformer_models_by_end_shot": {
            str(end): {str(shot): str(path) for (e, shot), path in transformers.items() if e == end}
            for end in sorted({e for e, _ in transformers})
        },
        **{name: {int(shot): str(path) for shot, path in models[name].items()}
           for name in ("kura_policy_models_by_shot", "kura_value_models_by_shot")},
    }
    # 並列数を変えた比較や再実行でも、以前の結果を上書きしない。
    # Windows のパス長を抑え、対戦条件は各 JSON の experiment に記録する。
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    json_dir = Path(save_path) / "mini_match_parallel" / f"{stamp}_workers{actual_workers}"
    json_dir.mkdir(parents=True, exist_ok=False)
    print(f"{len(jobs)}局面 × A/B開始 × {X}回を {actual_workers}プロセスで実行します。")
    print(f"保存先: {json_dir}", flush=True)
    records = []
    width = max(6, len(str(len(jobs) - 1)))
    with ProcessPoolExecutor(
        max_workers=actual_workers, mp_context=mp.get_context("spawn"),
        initializer=_init_worker, initargs=(config, models, metadata),
    ) as executor:
        futures = [executor.submit(_evaluate_position, job) for job in jobs]
        try:
            for future in as_completed(futures):
                record, search_log = future.result()
                index = record["position"]["position_index"]
                # ファイル出力を親に集約し、途中で失敗しても完了分は残す。
                mm.save_position_json(json_dir / f"{index:0{width}d}.json", record)
                (json_dir / f"{index:0{width}d}.log").write_text(search_log, encoding="utf-8")
                records.append(record)
                print(f"完了 {len(records)}/{len(jobs)}: position={index}, "
                      f"worker={record['execution']['worker_pid']}, "
                      f"{record['execution']['position_seconds']:.1f}秒", flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            print(f"実行を中断しました。完了済みの結果: {json_dir}", flush=True)
            raise
    records.sort(key=lambda record: record["position"]["position_index"])
    print(f"対戦処理の経過時間（起動・読み込み・保存を含む）: {time.perf_counter() - started:.1f}秒")
    mm.print_report_from_records(json_dir, records)
    return json_dir


if __name__ == "__main__":
    mp.freeze_support()
    main(
        log_path=NEWSL_DIR / "LearnLog" / "all",
        save_path=NEWSL_DIR / "experiment" / "data",
        player_a_kind="Transformer",
        player_b_kind="kura",
        player_a_search_method="shot",
        player_b_search_method="shot",
        run_name="jiritsu-vs-silicon-wintable-fixed",
        target_end=8,
        target_shot=13,
        final_end=9,
        transformer_models_by_shot=None,
        data_size=1000,
        X=1,
        num_workers=4,
        use_gpu=True,
        max_simulations=1022,
        search_time_limit=None,
        kura_policy_models_by_shot=mm.KURA_POLICY_MODELS_BY_SHOT,
        kura_value_models_by_shot=mm.KURA_VALUE_MODELS_BY_SHOT,
        shuffle_seed=12345,
        simulation_seed=12345,
        torch_threads_per_worker=1,
    )
