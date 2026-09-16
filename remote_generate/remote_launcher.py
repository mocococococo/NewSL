import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-start", type=int, required=True)
    parser.add_argument("--chunk-end", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    log_dir = ROOT / "log" / "remote_generate"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"chunk_{args.chunk_start}_{args.chunk_end}.log"
    pid_path = log_dir / f"chunk_{args.chunk_start}_{args.chunk_end}.pid"

    command = [
        sys.executable,
        "-u",
        str(ROOT / "transformer" / "shot_generator.py"),
        "--chunk_start",
        str(args.chunk_start),
        "--chunk_end",
        str(args.chunk_end),
        "--num_workers",
        "1",
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

    pid_path.write_text(str(process.pid), encoding="ascii")

    print("PID:", process.pid)
    print("LOG:", log_path)
    print("PID FILE:", pid_path)


if __name__ == "__main__":
    main()