import argparse
import subprocess
import sys
import time
from pathlib import Path

from remote_config import (
    load_remotes,
    get_run_name,
    get_data_root,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


CONNECTION_ERROR_MARKERS = (
    "connection timed out",
    "connection refused",
    "connection reset",
    "connection closed",
    "connection aborted",
    "no route to host",
    "network is unreachable",
    "could not resolve hostname",
    "stdio forwarding failed",
    "operation timed out",
    "open failed: connect failed",
    "broken pipe",
)


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


def is_connection_error(output):
    text = output.lower()

    return any(
        marker in text
        for marker in CONNECTION_ERROR_MARKERS
    )


def start_chunk(
    remote_name,
    chunk,
    end,
    shot,
    run_name,
):
    code, output = run_script(
        "remote_start.py",
        remote_name,
        chunk,
        "--end",
        end,
        "--shot",
        shot,
        "--run-name",
        run_name,
    )

    print(
        output,
        end="",
    )

    if code == 0:
        return True

    if is_connection_error(output):
        print(
            f"UNREACHABLE: "
            f"{remote_name} while starting "
            f"chunk {chunk}"
        )

        # SSHが切れたタイミングによっては、
        # remote側で起動済みの可能性がある。
        return False

    raise RuntimeError(
        f"{remote_name}: "
        f"failed to start "
        f"chunk {chunk}\n"
        f"{output}"
    )


def check_chunk(
    remote_name,
    chunk,
    end,
    shot,
    run_name,
):
    code, output = run_script(
        "remote_check.py",
        remote_name,
        chunk,
        "--end",
        end,
        "--shot",
        shot,
        "--run-name",
        run_name,
    )

    if code != 0:

        if is_connection_error(output):
            return (
                "UNREACHABLE",
                output,
            )

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
    run_name,
):
    code, output = run_script(
        "remote_collect.py",
        remote_name,
        chunk,
        "--end",
        end,
        "--shot",
        shot,
        "--run-name",
        run_name,
    )

    print(
        output,
        end="",
    )

    if code == 0:
        return True

    if is_connection_error(output):
        print(
            f"UNREACHABLE: "
            f"{remote_name} while collecting "
            f"chunk {chunk}"
        )

        return False

    raise RuntimeError(
        f"{remote_name}: "
        f"failed to collect "
        f"chunk {chunk}\n"
        f"{output}"
    )


def local_chunk_files(
    chunk,
    end,
    shot,
    run_name,
):
    folder = (
        get_data_root(
            ROOT,
            run_name,
        )
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
    run_name,
):
    return bool(
        local_chunk_files(
            chunk,
            end,
            shot,
            run_name,
        )
    )


def fill_slots(
    remote_name,
    workers,
    pending_chunks,
    running,
    end,
    shot,
    run_name,
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

        started = start_chunk(
            remote_name,
            chunk,
            end,
            shot,
            run_name,
        )

        if started:

            running[
                remote_name
            ].append(
                chunk
            )

            continue

        # SSH接続エラーの場合、
        # remote側で起動に成功しているかどうか
        # この時点では判断できない。
        #
        # そのため別PCへ再投入せず、
        # 一旦このPCに割り当てられた状態で保持する。
        running[
            remote_name
        ].append(
            chunk
        )

        print(
            f"WAIT: "
            f"chunk {chunk} may have started "
            f"on {remote_name}; "
            f"will check again later."
        )

        # 接続状態が不明なので、
        # このPCへの追加割当も一旦停止する。
        break
    

def fill_pending_round_robin(
    remotes,
    pending_chunks,
    running,
    end,
    shot,
    run_name,
    unreachable_nodes=None,
):
    if unreachable_nodes is None:
        unreachable_nodes = set()

    while pending_chunks:

        assigned_in_round = False

        for (
            remote_name,
            node,
        ) in remotes.items():

            if not pending_chunks:
                break

            if not node.get(
                "enabled",
                True,
            ):
                continue

            if (
                remote_name
                in unreachable_nodes
            ):
                continue

            workers = int(
                node.get(
                    "workers",
                    1,
                )
            )

            if (
                len(
                    running[
                        remote_name
                    ]
                )
                >= workers
            ):
                continue

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

            started = start_chunk(
                remote_name,
                chunk,
                end,
                shot,
                run_name,
            )

            running[
                remote_name
            ].append(
                chunk
            )

            assigned_in_round = True

            if not started:

                unreachable_nodes.add(
                    remote_name
                )

                print(
                    f"WAIT: "
                    f"chunk {chunk} may have started "
                    f"on {remote_name}; "
                    f"will check again later."
                )

        # 1周してもどのPCにも割り当てられなければ、
        # 現在は全workerが使用中
        if not assigned_in_round:
            break


def discover_existing_jobs(
    remotes,
    chunks,
    end,
    shot,
    run_name,
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

        # 基盤PCにすでにデータがあれば完了済み
        if is_locally_completed(
            chunk,
            end,
            shot,
            run_name,
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

        # enabled=false のPCも含め、
        # 全PCの状態を確認する
        for remote_name in remotes:

            status, output = (
                check_chunk(
                    remote_name,
                    chunk,
                    end,
                    shot,
                )
            )

            # 起動時の状態復元では、
            # PCに接続できないままPENDINGと判断すると
            # 二重生成になる可能性がある。
            #
            # そのため接続が戻るまで待つ。
            if status == "UNREACHABLE":

                raise ConnectionError(
                    f"{remote_name} is temporarily "
                    f"unreachable while checking "
                    f"chunk {chunk}"
                )

            if status == "RUNNING":

                found_running.append(
                    remote_name
                )

            elif status == "DONE":

                found_done.append(
                    remote_name
                )

        if len(
            found_running
        ) > 1:

            raise RuntimeError(
                f"chunk {chunk} "
                f"is running on "
                f"multiple nodes: "
                f"{', '.join(found_running)}"
            )

        if len(
            found_done
        ) > 1:

            raise RuntimeError(
                f"chunk {chunk} "
                f"has completed outputs "
                f"on multiple nodes: "
                f"{', '.join(found_done)}"
            )

        # remoteですでに完成していれば回収
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

            collected = collect_chunk(
                remote_name,
                chunk,
                end,
                shot,
                run_name,
            )

            if not collected:

                raise ConnectionError(
                    f"{remote_name} became "
                    f"unreachable while collecting "
                    f"chunk {chunk}"
                )

            completed.append(
                chunk
            )

            continue

        # すでに動いているものは
        # enabled=false でも監視継続
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
    run_name,
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
        "--run-name",
        run_name,
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
        "--run-name",
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

    # ------------------------------------
    # 全PC
    # ------------------------------------
    remotes = load_remotes()
    
    configured_run_name = (
        get_run_name(
            remotes
        )
    )

    if args.run_name is None:
        args.run_name = (
            configured_run_name
        )

    elif (
        args.run_name
        != configured_run_name
    ):
        raise SystemExit(
            "run-name does not match "
            "the log_path configuration: "
            f"--run-name={args.run_name}, "
            f"log_path={configured_run_name}"
        )

    if not remotes:
        raise SystemExit(
            "No remotes configured."
        )

    # ------------------------------------
    # 新規chunk生成に使うPCだけ
    # ------------------------------------
    generation_remotes = {
        name: node
        for name, node in remotes.items()
        if node.get(
            "enabled",
            True,
        )
    }

    if not generation_remotes:
        raise SystemExit(
            "No enabled remotes configured."
        )

    chunks = list(
        range(
            args.chunk_start,
            args.chunk_end + 1,
        )
    )

    # ------------------------------------
    # PC設定表示
    # ------------------------------------
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

        enabled = bool(
            node.get(
                "enabled",
                True,
            )
        )

        print(
            f"  {name} "
            f"(workers={workers}, "
            f"enabled={enabled})"
        )

    print(
        f"CHUNKS: "
        f"{args.chunk_start}-"
        f"{args.chunk_end}"
    )

    print(
        f"RUN: "
        f"{args.run_name}"
    )

    print(
        f"TARGET: "
        f"end{args.end}/"
        f"shot{args.shot}"
    )

    # ------------------------------------
    # 既存状態を調査
    #
    # 起動直後に一時的なSSH障害があれば、
    # 状態を誤判定しないため接続復旧まで待つ。
    # ------------------------------------
    while True:

        try:
            (
                completed,
                running,
                pending_chunks,
            ) = discover_existing_jobs(
                remotes,
                chunks,
                args.end,
                args.shot,
                args.run_name,
            )

            break

        except ConnectionError as exc:

            print(
                f"\nWAITING FOR CONNECTION: "
                f"{exc}"
            )

            print(
                f"Retrying in "
                f"{args.poll_seconds} seconds."
            )

            time.sleep(
                args.poll_seconds
            )

    # ------------------------------------
    # 最初の割当
    #
    # enabled=true のPCだけ
    # ------------------------------------
    for (
        remote_name,
        node,
    ) in generation_remotes.items():

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
            args.run_name,
        )

    # ------------------------------------
    # 監視
    #
    # runningがある間、
    # または再割当待ちのpendingがある間続ける
    # ------------------------------------
    while (
        any(
            running.values()
        )
        or pending_chunks
    ):

        time.sleep(
            args.poll_seconds
        )

        # このpollで接続不能だったPC
        unreachable_nodes = set()

        # --------------------------------
        # まず全PCの状態を確認する
        #
        # ここではまだ新しいchunkを
        # 割り当てない
        # --------------------------------
        for (
            remote_name,
            node,
        ) in remotes.items():

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
                        args.run_name,
                    )
                )

                print(
                    f"STATUS: "
                    f"{remote_name} "
                    f"chunk {chunk} "
                    f"-> {status}"
                )

                # ------------------------
                # 一時的な接続障害
                # ------------------------
                if status == "UNREACHABLE":

                    unreachable_nodes.add(
                        remote_name
                    )

                    print(
                        f"WAIT: "
                        f"{remote_name} is temporarily "
                        f"unreachable. "
                        f"chunk {chunk} remains assigned."
                    )

                    # このpollでは同じPCを
                    # これ以上確認しない
                    break

                # ------------------------
                # 実行中
                # ------------------------
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

                    collected = collect_chunk(
                        remote_name,
                        chunk,
                        args.end,
                        args.shot,
                        args.run_name,
                    )

                    if not collected:

                        unreachable_nodes.add(
                            remote_name
                        )

                        print(
                            f"WAIT: "
                            f"chunk {chunk} is DONE on "
                            f"{remote_name}, but collection "
                            f"will be retried later."
                        )

                        break

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
                # 接続はできたが
                # プロセスが停止している
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

        # --------------------------------
        # 全PCの状態確認が終わってから、
        # 空いているworkerへpendingを
        # 1チャンクずつ順番に割り当てる
        # --------------------------------
        fill_pending_round_robin(
            remotes,
            pending_chunks,
            running,
            args.end,
            args.shot,
            args.run_name,
            unreachable_nodes,
        )

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

    if args.no_train:

        print(
            "\nTRAINING SKIPPED"
        )

        return

    train_model(
        args.end,
        args.shot,
        args.run_name,
        args.epochs,
    )

    print(
        "\nALL DONE"
    )


if __name__ == "__main__":
    main()