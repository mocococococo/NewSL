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

    return data.decode(
        "utf-8",
        errors="replace",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "remote_name",
    )

    parser.add_argument(
        "chunk_start",
        type=int,
    )

    parser.add_argument(
        "chunk_end",
        type=int,
        nargs="?",
    )

    parser.add_argument(
        "--end",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--shot",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    if not 0 <= args.end <= 9:
        raise SystemExit(
            "end must be 0..9"
        )

    if not 0 <= args.shot <= 15:
        raise SystemExit(
            "shot must be 0..15"
        )

    chunk_end = (
        args.chunk_start
        if args.chunk_end is None
        else args.chunk_end
    )

    node = get_remote(
        args.remote_name
    )

    log_path = node["log_path"]

    launcher_args = [
        "--chunk-start",
        str(args.chunk_start),

        "--chunk-end",
        str(chunk_end),

        "--target-end",
        str(args.end),

        "--target-shot",
        str(args.shot),

        "--log-path",
        str(log_path),
    ]

    if args.dry_run:
        launcher_args.append(
            "--dry-run"
        )

    # ------------------------------------
    # local
    # ------------------------------------
    if node["type"] == "local":

        root = Path(
            node["root"]
        )

        launcher = (
            root
            / "remote_generate"
            / "remote_launcher.py"
        )

        command = [
            sys.executable,
            str(launcher),
            *launcher_args,
        ]

        result = subprocess.run(
            command,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    # ------------------------------------
    # SSH
    # ------------------------------------
    elif node["type"] == "ssh":

        host = node["host"]
        remote_root = node["root"]

        launcher_command = (
            subprocess.list2cmdline(
                [
                    "python",
                    r"remote_generate\remote_launcher.py",
                    *launcher_args,
                ]
            )
        )

        remote_command = (
            f'cd /d "{remote_root}" '
            f"&& {launcher_command}"
        )

        result = subprocess.run(
            [
                "ssh",
                host,
                remote_command,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    else:
        raise SystemExit(
            f"Unknown node type: "
            f"{node['type']}"
        )

    output = decode_output(
        result.stdout
    )

    print(
        output,
        end="",
    )

    if result.returncode != 0:
        raise SystemExit(
            result.returncode
        )

    if args.dry_run:
        return

    match = re.search(
        r"PID:\s*(\d+)",
        output,
    )

    if not match:
        raise SystemExit(
            "PID was not returned "
            "by remote_launcher.py"
        )

    pid = int(
        match.group(1)
    )

    print(
        f"STARTED: "
        f"{args.remote_name} "
        f"end{args.end}/shot{args.shot} "
        f"chunk "
        f"{args.chunk_start}-{chunk_end} "
        f"PID {pid}"
    )


if __name__ == "__main__":
    main()