import argparse
import subprocess
import sys
import time
from pathlib import Path

from remote_config import load_remotes


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def run_script(script_name, *args):
    command = [
        sys.executable,
        str(HERE / script_name),
        *map(str, args),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    return (
        result.returncode,
        result.stdout,
    )


def start_chunk(
    remote_name,
    chunk,
    end,
    shot,
):
    code, output = run_script(
        "remote_start.py",
        remote_name,
        chunk,
        "--end",
        end,
        "--shot",
        shot,
    )

    print(
        output,
        end="",
    )

    if code != 0:
        raise RuntimeError(
            f"{remote_name}: "
            f"failed to start "
            f"chunk {chunk}"
        )


def check_chunk(
    remote_name,
    chunk,
    end,
    shot,
):
    code, output = run_script(
        "remote_check.py",
        remote_name,
        chunk,
        "--end",
        end,
        "--shot",
        shot,
    )

    if code != 0:
        raise RuntimeError(
            f"{remote_name}: "
            f"failed to check "
            f"chunk {chunk}\n"
            f"{output}"
        )

    for line in output.splitlines():

        if line.startswith(
            "STATUS:"
        ):
            status = (
                line
                .split(
                    ":",
                    1,
                )[1]
                .strip()
            )

            return (
                status,
                output,
            )

    raise RuntimeError(
        f"{remote_name}: "
        f"STATUS not found\n"
        f"{output}"
    )


def collect_chunk(
    remote_name,
    chunk,
    end,
    shot,
):
    code, output = run_script(
        "remote_collect.py",
        remote_name,
        chunk,
        "--end",
        end,
        "--shot",
        shot,
    )

    print(
        output,
        end="",
    )

    if code != 0:
        raise RuntimeError(
            f"{remote_name}: "
            f"failed to collect "
            f"chunk {chunk}"
        )


def local_chunk_files(
    chunk,
    end,
    shot,
):
    folder = (
        ROOT
        / "data"
        / f"end{end}"
        / f"shot{shot}"
    )

    return sorted(
        folder.glob(
            f"sl_data_chunk{chunk}_*.npz"
        )
    )


def is_locally_completed(
    chunk,
    end,
    shot,
):
    return bool(
        local_chunk_files(
            chunk,
            end,
            shot,
        )
    )


def fill_slots(
    remote_name,
    workers,
    pending_chunks,
    running,
    end,
    shot,
):
    while (
        len(
            running[
                remote_name
            ]
        )
        < workers
        and pending_chunks
    ):

        chunk = (
            pending_chunks.pop(
                0
            )
        )

        print(
            f"\nASSIGN: "
            f"{remote_name} "
            f"<- "
            f"end{end}/shot{shot} "
            f"chunk {chunk}"
        )

        start_chunk(
            remote_name,
            chunk,
            end,
            shot,
        )

        running[
            remote_name
        ].append(
            chunk
        )


def discover_existing_jobs(
    remotes,
    chunks,
    end,
    shot,
):
    completed = []
    pending = []

    running = {
        name: []
        for name in remotes
    }

    print(
        "\nCHECK EXISTING STATE:"
    )

    for chunk in chunks:

        # --------------------------------
        # 基盤PCにデータが存在
        # --------------------------------
        if is_locally_completed(
            chunk,
            end,
            shot,
        ):

            print(
                f"  chunk {chunk}: "
                f"ALREADY DONE"
            )

            completed.append(
                chunk
            )

            continue

        found_running = []
        found_done = []

        # --------------------------------
        # 各PCを調査
        # --------------------------------
        for remote_name in remotes:

            status, output = (
                check_chunk(
                    remote_name,
                    chunk,
                    end,
                    shot,
                )
            )

            if status == "RUNNING":

                found_running.append(
                    remote_name
                )

            elif status == "DONE":

                found_done.append(
                    remote_name
                )

        # --------------------------------
        # 同じchunkが複数PCで実行
        # --------------------------------
        if len(
            found_running
        ) > 1:

            raise RuntimeError(
                f"chunk {chunk} "
                f"is running on "
                f"multiple nodes: "
                f"{', '.join(found_running)}"
            )

        # --------------------------------
        # 同じ完成データが複数remote
        # --------------------------------
        if len(
            found_done
        ) > 1:

            raise RuntimeError(
                f"chunk {chunk} "
                f"has completed outputs "
                f"on multiple nodes: "
                f"{', '.join(found_done)}"
            )

        # --------------------------------
        # remoteですでに完了
        # --------------------------------
        if found_done:

            remote_name = (
                found_done[0]
            )

            print(
                f"  chunk {chunk}: "
                f"DONE on "
                f"{remote_name}, "
                f"collecting"
            )

            collect_chunk(
                remote_name,
                chunk,
                end,
                shot,
            )

            completed.append(
                chunk
            )

            continue

        # --------------------------------
        # 実行中を引き継ぐ
        # --------------------------------
        if found_running:

            remote_name = (
                found_running[0]
            )

            print(
                f"  chunk {chunk}: "
                f"RESUME RUNNING "
                f"on {remote_name}"
            )

            running[
                remote_name
            ].append(
                chunk
            )

            continue

        # --------------------------------
        # 未実行
        # --------------------------------
        print(
            f"  chunk {chunk}: "
            f"PENDING"
        )

        pending.append(
            chunk
        )

    return (
        completed,
        running,
        pending,
    )


def train_model(
    end,
    shot,
    epochs=None,
):
    print(
        f"\nSTART TRAINING: "
        f"end{end}/shot{shot}"
    )

    train_args = [
        "--end",
        end,
        "--shot",
        shot,
    ]

    if epochs is not None:

        train_args += [
            "--epochs",
            epochs,
        ]

    code, output = run_script(
        "train_model.py",
        *train_args,
    )

    print(
        output,
        end="",
    )

    if code != 0:
        raise RuntimeError(
            f"Training failed for "
            f"end{end}/shot{shot}"
        )

    print(
        f"\nTRAINING DONE: "
        f"end{end}/shot{shot}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "chunk_start",
        type=int,
    )

    parser.add_argument(
        "chunk_end",
        type=int,
    )

    parser.add_argument(
        "--end",
        type=int,
        default=9,
    )

    parser.add_argument(
        "--shot",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--no-train",
        action="store_true",
    )

    args = parser.parse_args()

    if (
        args.chunk_end
        < args.chunk_start
    ):
        raise SystemExit(
            "chunk_end must be "
            ">= chunk_start"
        )

    if not 0 <= args.end <= 9:
        raise SystemExit(
            "end must be 0..9"
        )

    if not 0 <= args.shot <= 15:
        raise SystemExit(
            "shot must be 0..15"
        )

    remotes = load_remotes()

    if not remotes:
        raise SystemExit(
            "No remotes configured."
        )

    chunks = list(
        range(
            args.chunk_start,
            args.chunk_end + 1,
        )
    )

    print(
        "REMOTES:"
    )

    for (
        name,
        node,
    ) in remotes.items():

        workers = int(
            node.get(
                "workers",
                1,
            )
        )

        print(
            f"  {name} "
            f"(workers={workers})"
        )

    print(
        f"CHUNKS: "
        f"{args.chunk_start}-"
        f"{args.chunk_end}"
    )

    print(
        f"TARGET: "
        f"end{args.end}/"
        f"shot{args.shot}"
    )

    # ------------------------------------
    # 既存状態を調査
    # ------------------------------------
    (
        completed,
        running,
        pending_chunks,
    ) = discover_existing_jobs(
        remotes,
        chunks,
        args.end,
        args.shot,
    )

    # ------------------------------------
    # 最初の割当
    # ------------------------------------
    for (
        remote_name,
        node,
    ) in remotes.items():

        workers = int(
            node.get(
                "workers",
                1,
            )
        )

        fill_slots(
            remote_name,
            workers,
            pending_chunks,
            running,
            args.end,
            args.shot,
        )

    # ------------------------------------
    # 監視
    # ------------------------------------
    while any(
        running.values()
    ):

        time.sleep(
            args.poll_seconds
        )

        for (
            remote_name,
            node,
        ) in remotes.items():

            workers = int(
                node.get(
                    "workers",
                    1,
                )
            )

            for chunk in list(
                running[
                    remote_name
                ]
            ):

                status, output = (
                    check_chunk(
                        remote_name,
                        chunk,
                        args.end,
                        args.shot,
                    )
                )

                print(
                    f"STATUS: "
                    f"{remote_name} "
                    f"chunk {chunk} "
                    f"-> {status}"
                )

                if status == "RUNNING":
                    continue

                # ------------------------
                # 完了
                # ------------------------
                if status == "DONE":

                    print(
                        f"COLLECT: "
                        f"{remote_name} "
                        f"chunk {chunk}"
                    )

                    collect_chunk(
                        remote_name,
                        chunk,
                        args.end,
                        args.shot,
                    )

                    completed.append(
                        chunk
                    )

                    running[
                        remote_name
                    ].remove(
                        chunk
                    )

                    continue

                # ------------------------
                # 異常終了
                # → 再キュー
                # ------------------------
                if status in (
                    "STOPPED",
                    "UNKNOWN",
                ):

                    print(
                        f"REQUEUE: "
                        f"{remote_name} "
                        f"chunk {chunk} "
                        f"({status})"
                    )

                    running[
                        remote_name
                    ].remove(
                        chunk
                    )

                    pending_chunks.append(
                        chunk
                    )

                    continue

                raise RuntimeError(
                    f"{remote_name}: "
                    f"chunk {chunk} "
                    f"unexpected status "
                    f"{status}\n"
                    f"{output}"
                )

            # ----------------------------
            # 空きworkerへ次chunk
            # ----------------------------
            fill_slots(
                remote_name,
                workers,
                pending_chunks,
                running,
                args.end,
                args.shot,
            )

    # ------------------------------------
    # 全chunk確認
    # ------------------------------------
    if pending_chunks:
        raise RuntimeError(
            f"Pending chunks remain: "
            f"{pending_chunks}"
        )

    expected_chunks = set(
        chunks
    )

    completed_chunks = set(
        completed
    )

    if (
        completed_chunks
        != expected_chunks
    ):

        missing = sorted(
            expected_chunks
            - completed_chunks
        )

        raise RuntimeError(
            f"Some chunks are "
            f"not completed: "
            f"{missing}"
        )

    print(
        "\nALL DATA GENERATION DONE"
    )

    print(
        "Completed chunks:",
        ", ".join(
            map(
                str,
                sorted(
                    completed_chunks
                ),
            )
        ),
    )

    # ------------------------------------
    # 学習
    # ------------------------------------
    if args.no_train:

        print(
            "\nTRAINING SKIPPED"
        )

        return

    train_model(
        args.end,
        args.shot,
        args.epochs,
    )

    print(
        "\nALL DONE"
    )


if __name__ == "__main__":
    main()