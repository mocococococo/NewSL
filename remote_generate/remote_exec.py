import subprocess
import sys


def decode_output(data):
    if not data:
        return ""

    for encoding in ("utf-8", "cp932"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass

    return data.decode("utf-8", errors="replace")


def run_remote(host, command):
    result = subprocess.run(
        ["ssh", host, command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout = decode_output(result.stdout)
    stderr = decode_output(result.stderr)

    print("exit code:", result.returncode)

    if stdout:
        print("stdout:")
        print(stdout)

    if stderr:
        print("stderr:")
        print(stderr)

    return result.returncode


if __name__ == "__main__":
    host = "remote-01"
    command = " ".join(sys.argv[1:])

    if not command:
        raise SystemExit("Usage: python remote_exec.py <command>")

    raise SystemExit(run_remote(host, command))