import argparse
import re
import subprocess
import sys
from pathlib import Path

from remote_config import get_remote


def decode_output(data: bytes) -> str:
    if not data:
        return ""

    for encoding in ("utf-8", "cp932"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass

    return data.decode("utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("remote_name")
    parser.add_argument("chunk_start", type=int)
    parser.add_argument("chunk_end", type=int, nargs="?")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    node = get_remote(args.remote_name)

    chunk_end = (
        args.chunk_start
        if args.chunk_end is None
        else args.chunk_end
    )

    # remotes.json からPCごとの学習元データパスを取得
    log_path = node["log_path"]

    launcher_args = [
        "--chunk-start",
        str(args.chunk_start),
        "--chunk-end",
        str(chunk_end),
        "--log-path",
        log_path,
    ]

    if args.dry_run:
        launcher_args.append("--dry-run")

    node_type = node["type"]

    # -------------------------
    # 基盤PC自身で実行
    # -------------------------
    if node_type == "local":
        root = Path(node["root"])

        command = [
            sys.executable,
            str(
                root
                / "remote_generate"
                / "remote_launcher.py"
            ),
            *launcher_args,
        ]

        result = subprocess.run(
            command,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    # -------------------------
    # SSH先で実行
    # -------------------------
    elif node_type == "ssh":
        host = node["host"]
        remote_root = node["root"]

        remote_command = (
            f'cd /d "{remote_root}" && '
            f'python remote_generate\\remote_launcher.py '
            f'--chunk-start {args.chunk_start} '
            f'--chunk-end {chunk_end} '
            f'--log-path "{log_path}"'
        )

        if args.dry_run:
            remote_command += " --dry-run"

        result = subprocess.run(
            ["ssh", host, remote_command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    else:
        raise SystemExit(
            f"Unknown node type: {node_type}"
        )

    stdout = decode_output(result.stdout)
    stderr = decode_output(result.stderr)

    if stdout:
        print(stdout, end="")

    if stderr:
        print(stderr, end="", file=sys.stderr)

    if result.returncode != 0:
        raise SystemExit(result.returncode)

    # dry-runではPIDは出ない
    if args.dry_run:
        return

    match = re.search(r"PID:\s*(\d+)", stdout)

    if match is None:
        raise SystemExit(
            "Process started, but PID could not be read."
        )

    print(
        f"STARTED: {args.remote_name} "
        f"chunk {args.chunk_start}-{chunk_end} "
        f"PID {match.group(1)}"
    )


if __name__ == "__main__":
    main()