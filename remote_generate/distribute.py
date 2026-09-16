import argparse
import subprocess
import sys
import time
from pathlib import Path

from remote_config import load_remotes


HERE = Path(__file__).resolve().parent


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

    return result.returncode, result.stdout


def start_chunk(remote_name, chunk):
    code, output = run_script(
        "remote_start.py",
        remote_name,
        chunk,
    )

    print(output, end="")

    if code != 0:
        raise RuntimeError(
            f"{remote_name}: failed to start chunk {chunk}"
        )


def check_chunk(remote_name, chunk, end, shot):
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
            f"{remote_name}: failed to check chunk {chunk}\n"
            f"{output}"
        )

    for line in output.splitlines():
        if line.startswith("STATUS:"):
            return line.split(":", 1)[1].strip(), output

    raise RuntimeError(
        f"{remote_name}: STATUS not found\n{output}"
    )


def collect_chunk(remote_name, chunk, end, shot):
    code, output = run_script(
        "remote_collect.py",
        remote_name,
        chunk,
        "--end",
        end,
        "--shot",
        shot,
    )

    print(output, end="")

    if code != 0:
        raise RuntimeError(
            f"{remote_name}: failed to collect chunk {chunk}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("chunk_start", type=int)
    parser.add_argument("chunk_end", type=int)
    parser.add_argument("--end", type=int, default=9)
    parser.add_argument("--shot", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    if args.chunk_end < args.chunk_start:
        raise SystemExit(
            "chunk_end must be >= chunk_start"
        )

    remotes = load_remotes()

    if not remotes:
        raise SystemExit("No remotes configured.")

    remote_names = list(remotes.keys())

    pending_chunks = list(
        range(args.chunk_start, args.chunk_end + 1)
    )

    # remote_name -> chunk
    running = {}

    completed = []

    print("REMOTES:")
    for name in remote_names:
        print(f"  {name}")

    print(
        f"CHUNKS: {args.chunk_start}-{args.chunk_end}"
    )

    # 最初のchunkを各PCへ割り当て
    for remote_name in remote_names:
        if not pending_chunks:
            break

        chunk = pending_chunks.pop(0)

        print(
            f"\nASSIGN: {remote_name} <- chunk {chunk}"
        )

        start_chunk(remote_name, chunk)

        running[remote_name] = chunk

    while running:
        time.sleep(args.poll_seconds)

        for remote_name in list(running):
            chunk = running[remote_name]

            status, output = check_chunk(
                remote_name,
                chunk,
                args.end,
                args.shot,
            )

            print(
                f"STATUS: {remote_name} "
                f"chunk {chunk} -> {status}"
            )

            if status == "RUNNING":
                continue

            if status == "DONE":
                print(
                    f"COLLECT: {remote_name} chunk {chunk}"
                )

                collect_chunk(
                    remote_name,
                    chunk,
                    args.end,
                    args.shot,
                )

                completed.append(chunk)

                del running[remote_name]

                # 空いたPCに次のchunkを渡す
                if pending_chunks:
                    next_chunk = pending_chunks.pop(0)

                    print(
                        f"\nASSIGN: {remote_name} "
                        f"<- chunk {next_chunk}"
                    )

                    start_chunk(
                        remote_name,
                        next_chunk,
                    )

                    running[remote_name] = next_chunk

                continue

            raise RuntimeError(
                f"{remote_name}: chunk {chunk} "
                f"ended with status {status}\n"
                f"{output}"
            )

    print("\nALL DONE")
    print(
        "Completed chunks:",
        ", ".join(map(str, sorted(completed))),
    )


if __name__ == "__main__":
    main()