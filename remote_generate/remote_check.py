import argparse
import base64
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


def ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("host")
    parser.add_argument("chunk_start", type=int)
    parser.add_argument("chunk_end", type=int, nargs="?")
    parser.add_argument("--end", type=int, default=9)
    parser.add_argument("--shot", type=int, default=2)
    args = parser.parse_args()

    chunk_end = (
        args.chunk_start
        if args.chunk_end is None
        else args.chunk_end
    )

    root = ps_quote(REMOTE_ROOT)

    script = f"""
$ProgressPreference = 'SilentlyContinue'
    
$root = {root}
$outputDir = Join-Path $root 'data\\end{args.end}\\shot{args.shot}'
$pidFile = Join-Path $root 'log\\remote_generate\\chunk_{args.chunk_start}_{chunk_end}.pid'
$logFile = Join-Path $root 'log\\remote_generate\\chunk_{args.chunk_start}_{chunk_end}.log'

$allDone = $true
$outputFiles = @()

for ($c = {args.chunk_start}; $c -le {chunk_end}; $c++) {{
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
    Get-Content -LiteralPath $pidFile -Raw
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
            args.host,
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
        print(stderr, end="", file=sys.stderr)

    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()