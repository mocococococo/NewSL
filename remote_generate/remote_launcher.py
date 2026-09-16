import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def is_process_running(pid: int) -> bool:
    result = subprocess.run(
        [
            "tasklist",
            "/FI",
            f"PID eq {pid}",
            "/FO",
            "CSV",
            "/NH",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout = result.stdout.decode("cp932", errors="replace")
    return str(pid) in stdout


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chunk-start",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--chunk-end",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--log-path",
        required=True,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    log_dir = ROOT / "log" / "remote_generate"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = (
        log_dir
        / f"chunk_{args.chunk_start}_{args.chunk_end}.log"
    )

    pid_path = (
        log_dir
        / f"chunk_{args.chunk_start}_{args.chunk_end}.pid"
    )

    # 二重起動チェック
    if pid_path.exists():
        try:
            old_pid = int(
                pid_path.read_text(
                    encoding="ascii"
                ).strip()
            )

            if is_process_running(old_pid):
                raise SystemExit(
                    f"Already running: "
                    f"chunk {args.chunk_start}-"
                    f"{args.chunk_end} "
                    f"PID {old_pid}"
                )

        except ValueError:
            pass

    command = [
        sys.executable,
        "-u",
        str(
            ROOT
            / "transformer"
            / "shot_generator.py"
        ),
        "--chunk_start",
        str(args.chunk_start),
        "--chunk_end",
        str(args.chunk_end),
        "--num_workers",
        "1",
        "--log_path",
        args.log_path,
    ]

    if args.dry_run:
        print("COMMAND:")
        print(command)

        print("LOG:")
        print(log_path)

        print("PID FILE:")
        print(pid_path)

        return

    flags = (
        subprocess.DETACHED_PROCESS
        | subprocess.CREATE_NEW_PROCESS_GROUP
        | subprocess.CREATE_BREAKAWAY_FROM_JOB
    )

    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=flags,
            close_fds=True,
        )

    pid_path.write_text(
        str(process.pid),
        encoding="ascii",
    )

    print("PID:", process.pid)
    print("LOG:", log_path)
    print("PID FILE:", pid_path)


if __name__ == "__main__":
    main()