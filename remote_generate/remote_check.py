import argparse
import base64
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


def check_local(
    root,
    chunk_start,
    chunk_end,
    end,
    shot,
):
    root = Path(root)

    output_dir = (
        root
        / "data"
        / f"end{end}"
        / f"shot{shot}"
    )

    temp_dir = (
        root
        / ".temp"
        / "remote_generate"
    )

    pid_file = (
        temp_dir
        / "pids"
        / f"chunk_{chunk_start}_{chunk_end}.pid"
    )

    log_file = (
        temp_dir
        / "logs"
        / f"chunk_{chunk_start}_{chunk_end}.log"
    )

    all_done = True
    output_files = []

    for chunk in range(
        chunk_start,
        chunk_end + 1,
    ):
        files = sorted(
            output_dir.glob(
                f"sl_data_chunk{chunk}_*.npz"
            )
        )

        if not files:
            all_done = False
        else:
            output_files.extend(files)

    if all_done:
        print("STATUS: DONE")

        for path in output_files:
            print("OUTPUT:", path)

        return

    if not pid_file.exists():
        print("STATUS: UNKNOWN")
        print("PID file does not exist.")
        print("LOG:", log_file)
        return

    try:
        pid = int(
            pid_file.read_text(
                encoding="ascii"
            ).strip()
        )
    except ValueError:
        print("STATUS: UNKNOWN")
        print("Invalid PID file.")
        print("LOG:", log_file)
        return

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

    stdout = result.stdout.decode(
        "cp932",
        errors="replace",
    )

    if str(pid) in stdout:
        print("STATUS: RUNNING")
        print("PID:", pid)
        print("LOG:", log_file)
        return

    print("STATUS: STOPPED")
    print("PID:", pid)
    print(
        "Process is no longer running "
        "and output is incomplete."
    )
    print("LOG:", log_file)


def ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def check_ssh(
    host,
    remote_root,
    chunk_start,
    chunk_end,
    end,
    shot,
):
    root = ps_quote(remote_root)

    script = f"""
$ProgressPreference = 'SilentlyContinue'

$root = {root}

$outputDir = Join-Path $root 'data\\end{end}\\shot{shot}'

$tempDir = Join-Path $root '.temp\\remote_generate'

$pidFile = Join-Path `
    $tempDir `
    'pids\\chunk_{chunk_start}_{chunk_end}.pid'

$logFile = Join-Path `
    $tempDir `
    'logs\\chunk_{chunk_start}_{chunk_end}.log'

$allDone = $true
$outputFiles = @()

for ($c = {chunk_start}; $c -le {chunk_end}; $c++) {{

    $files = @(
        Get-ChildItem `
            -LiteralPath $outputDir `
            -Filter ("sl_data_chunk{{0}}_*.npz" -f $c) `
            -File `
            -ErrorAction SilentlyContinue
    )

    if ($files.Count -eq 0) {{
        $allDone = $false
    }}
    else {{
        foreach ($file in $files) {{
            $outputFiles += $file.FullName
        }}
    }}
}}

if ($allDone) {{
    Write-Output "STATUS: DONE"

    foreach ($file in $outputFiles) {{
        Write-Output ("OUTPUT: " + $file)
    }}

    exit 0
}}

if (-not (Test-Path -LiteralPath $pidFile)) {{
    Write-Output "STATUS: UNKNOWN"
    Write-Output "PID file does not exist."
    Write-Output ("LOG: " + $logFile)
    exit 0
}}

$jobPid = [int]((
    Get-Content `
        -LiteralPath $pidFile `
        -Raw
).Trim())

$process = Get-Process `
    -Id $jobPid `
    -ErrorAction SilentlyContinue

if ($null -ne $process) {{
    Write-Output "STATUS: RUNNING"
    Write-Output ("PID: " + $jobPid)
    Write-Output ("LOG: " + $logFile)
    exit 0
}}

Write-Output "STATUS: STOPPED"
Write-Output ("PID: " + $jobPid)
Write-Output "Process is no longer running and output is incomplete."
Write-Output ("LOG: " + $logFile)
"""

    encoded = base64.b64encode(
        script.encode("utf-16-le")
    ).decode("ascii")

    result = subprocess.run(
        [
            "ssh",
            host,
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-EncodedCommand",
            encoded,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout = decode_output(result.stdout)
    stderr = decode_output(result.stderr)

    if stdout:
        print(stdout, end="")

    if stderr:
        print(
            stderr,
            end="",
            file=sys.stderr,
        )

    raise SystemExit(
        result.returncode
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
    )

    parser.add_argument(
        "--end",
        type=int,
        default=9,
    )

    parser.add_argument(
        "--shot",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    node = get_remote(
        args.remote_name
    )

    chunk_end = (
        args.chunk_start
        if args.chunk_end is None
        else args.chunk_end
    )

    node_type = node["type"]

    if node_type == "local":
        check_local(
            node["root"],
            args.chunk_start,
            chunk_end,
            args.end,
            args.shot,
        )

    elif node_type == "ssh":
        check_ssh(
            node["host"],
            node["root"],
            args.chunk_start,
            chunk_end,
            args.end,
            args.shot,
        )

    else:
        raise SystemExit(
            f"Unknown node type: "
            f"{node_type}"
        )


if __name__ == "__main__":
    main()