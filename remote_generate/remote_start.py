import argparse
import re
import subprocess
import sys


REMOTE_ROOT = r"C:\Users\itolab\DigitalCurling\NewSL"


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
    parser.add_argument("host")
    parser.add_argument("chunk_start", type=int)
    parser.add_argument("chunk_end", type=int, nargs="?")
    args = parser.parse_args()

    chunk_end = (
        args.chunk_start
        if args.chunk_end is None
        else args.chunk_end
    )

    command = (
        f'cd /d "{REMOTE_ROOT}" && '
        f"python remote_generate\\remote_launcher.py "
        f"--chunk-start {args.chunk_start} "
        f"--chunk-end {chunk_end}"
    )

    result = subprocess.run(
        ["ssh", args.host, command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout = decode_output(result.stdout)
    stderr = decode_output(result.stderr)

    if stdout:
        print(stdout, end="")

    if stderr:
        print(stderr, end="", file=sys.stderr)

    if result.returncode != 0:
        raise SystemExit(result.returncode)

    match = re.search(r"PID:\s*(\d+)", stdout)

    if match is None:
        raise SystemExit("Remote process started, but PID could not be read.")

    print(
        f"STARTED: {args.host} "
        f"chunk {args.chunk_start}-{chunk_end} "
        f"PID {match.group(1)}"
    )


if __name__ == "__main__":
    main()