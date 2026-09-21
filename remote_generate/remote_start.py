import argparse
import re
import subprocess
import sys
from pathlib import Path

from remote_config import get_remote


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def decode_output(data):
    if not data:
        return ""

    for encoding in (
        "utf-8",
        "cp932",
    ):
        try:
            return data.decode(
                encoding
            )
        except UnicodeDecodeError:
            pass

    return data.decode(
        "utf-8",
        errors="replace",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "remote_name"
    )

    parser.add_argument(
        "chunk_start",
        type=int,
    )

    parser.add_argument(
        "chunk_end",
        type=int,
        nargs="?",
        default=None,
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
        "--run-name",
        required=True,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    chunk_end = (
        args.chunk_start
        if args.chunk_end is None
        else args.chunk_end
    )

    node = get_remote(
        args.remote_name
    )

    use_gpu = node.get(
        "use_gpu",
        True,
    )

    inference_batch_size = (
        node.get(
            "inference_batch_size"
        )
    )

    launcher_args = [
        str(args.chunk_start),
        str(chunk_end),
        "--target-end",
        str(args.end),
        "--target-shot",
        str(args.shot),
        "--log-path",
        str(node["log_path"]),
        "--run-name",
        str(args.run_name),
    ]

    if use_gpu:
        launcher_args.append(
            "--use-gpu"
        )
    else:
        launcher_args.append(
            "--no-use-gpu"
        )

    if (
        inference_batch_size
        is not None
    ):
        launcher_args += [
            "--inference-batch-size",
            str(
                inference_batch_size
            ),
        ]

    if args.dry_run:
        launcher_args.append(
            "--dry-run"
        )

    if node["type"] == "local":

        command = [
            sys.executable,
            str(
                HERE
                / "remote_launcher.py"
            ),
            *launcher_args,
        ]

    elif node["type"] == "ssh":

        remote_root = str(
            node["root"]
        )

        remote_command = [
            "python",
            r"remote_generate\remote_launcher.py",
            *launcher_args,
        ]

        command_text = (
            f'cd /d "{remote_root}" && '
            + subprocess.list2cmdline(
                remote_command
            )
        )

        command = [
            "ssh",
            node["host"],
            command_text,
        ]

    else:
        raise ValueError(
            f"Unknown node type: "
            f"{node['type']}"
        )

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout = decode_output(
        result.stdout
    )

    stderr = decode_output(
        result.stderr
    )

    if stdout:
        print(
            stdout,
            end="",
        )

    if stderr:
        print(
            stderr,
            end="",
            file=sys.stderr,
        )

    if result.returncode != 0:
        raise SystemExit(
            result.returncode
        )

    if args.dry_run:
        return

    match = re.search(
        r"PID:\s*(\d+)",
        stdout,
    )

    if not match:
        raise RuntimeError(
            "PID not found in "
            "remote_launcher output"
        )

    print(
        f"REMOTE PID: "
        f"{match.group(1)}"
    )


if __name__ == "__main__":
    main()