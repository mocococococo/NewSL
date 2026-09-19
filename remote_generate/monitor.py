import argparse
import base64
import csv
import io
import json
import subprocess
import time
from pathlib import Path

from remote_config import load_remotes


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


def read_last_log_lines(
    path: Path,
    line_count: int,
):
    if not path.is_file():
        return []

    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()

            read_size = min(
                size,
                65536,
            )

            f.seek(
                size - read_size
            )

            data = f.read()

        text = decode_output(data)

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        return lines[-line_count:]

    except OSError:
        return []


def get_running_pids_local():
    result = subprocess.run(
        [
            "tasklist",
            "/FO",
            "CSV",
            "/NH",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    text = decode_output(
        result.stdout
    )

    running_pids = set()

    reader = csv.reader(
        io.StringIO(text)
    )

    for row in reader:
        if len(row) < 2:
            continue

        running_pids.add(
            row[1]
        )

    return running_pids


def scan_local_node(
    node_name,
    node,
    chunk_start,
    chunk_end,
    end,
    shot,
    log_lines,
):
    root = Path(
        node["root"]
    )

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
        / f"end{end}"
        / f"shot{shot}"
    )

    log_dir = (
        temp_dir
        / "logs"
    )

    pid_dir = (
        temp_dir
        / "pids"
    )

    running_pids = (
        get_running_pids_local()
    )

    result = {}

    for chunk in range(
        chunk_start,
        chunk_end + 1,
    ):
        output_files = list(
            output_dir.glob(
                f"sl_data_chunk{chunk}_*.npz"
            )
        )

        pid_file = (
            pid_dir
            / f"chunk_{chunk}_{chunk}.pid"
        )

        log_file = (
            log_dir
            / f"chunk_{chunk}_{chunk}.log"
        )

        logs = (
            read_last_log_lines(
                log_file,
                log_lines,
            )
        )

        if output_files:
            status = "DONE"

        elif pid_file.is_file():
            try:
                pid = (
                    pid_file
                    .read_text(
                        encoding="ascii"
                    )
                    .strip()
                )

                if pid in running_pids:
                    status = "RUNNING"
                else:
                    status = "STOPPED"

            except (
                OSError,
                ValueError,
            ):
                status = "UNKNOWN"

        else:
            status = "UNKNOWN"

        result[chunk] = {
            "node": node_name,
            "status": status,
            "log": logs,
            "has_log": log_file.is_file(),
        }

    return result


def ps_quote(value: str) -> str:
    return (
        "'"
        + value.replace(
            "'",
            "''",
        )
        + "'"
    )


def scan_ssh_node(
    node_name,
    node,
    chunk_start,
    chunk_end,
    end,
    shot,
    log_lines,
):
    host = node["host"]
    remote_root = node["root"]

    root = ps_quote(
        remote_root
    )

    script = f"""
$ProgressPreference = 'SilentlyContinue'

[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$root = {root}

$outputDir = Join-Path `
    $root `
    'data\\end{end}\\shot{shot}'

$tempDir = Join-Path `
    $root `
    '.temp\\remote_generate\\end{end}\\shot{shot}'

$logDir = Join-Path `
    $tempDir `
    'logs'

$pidDir = Join-Path `
    $tempDir `
    'pids'

$result = @()

for (
    $c = {chunk_start};
    $c -le {chunk_end};
    $c++
) {{

    $outputFiles = @(
        Get-ChildItem `
            -LiteralPath $outputDir `
            -Filter ("sl_data_chunk{{0}}_*.npz" -f $c) `
            -File `
            -ErrorAction SilentlyContinue
    )

    $pidFile = Join-Path `
        $pidDir `
        ("chunk_{{0}}_{{0}}.pid" -f $c)

    $logFile = Join-Path `
        $logDir `
        ("chunk_{{0}}_{{0}}.log" -f $c)

    $hasLog = Test-Path `
        -LiteralPath $logFile

    $lastLog = @()

    if ($hasLog) {{
        $lines = @(
            Get-Content `
                -LiteralPath $logFile `
                -Tail 100 `
                -ErrorAction SilentlyContinue |
            Where-Object {{
                $_.Trim().Length -gt 0
            }}
        )

        if ($lines.Count -gt 0) {{
            $lastLog = @(
                $lines |
                Select-Object -Last {log_lines}
            )
        }}
    }}

    $status = "UNKNOWN"

    if ($outputFiles.Count -gt 0) {{
        $status = "DONE"
    }}
    elseif (
        Test-Path `
            -LiteralPath $pidFile
    ) {{
        try {{
            $jobPid = [int]((
                Get-Content `
                    -LiteralPath $pidFile `
                    -Raw
            ).Trim())

            $process = Get-Process `
                -Id $jobPid `
                -ErrorAction SilentlyContinue

            if ($null -ne $process) {{
                $status = "RUNNING"
            }}
            else {{
                $status = "STOPPED"
            }}
        }}
        catch {{
            $status = "UNKNOWN"
        }}
    }}

    $result += [PSCustomObject]@{{
        chunk   = $c
        status  = $status
        log     = @($lastLog)
        has_log = $hasLog
    }}
}}

ConvertTo-Json `
    -InputObject @($result) `
    -Compress
"""

    encoded = base64.b64encode(
        script.encode(
            "utf-16-le"
        )
    ).decode("ascii")

    process = subprocess.run(
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

    if process.returncode != 0:
        error = decode_output(
            process.stderr
        ).strip()

        result = {}

        for chunk in range(
            chunk_start,
            chunk_end + 1,
        ):
            result[chunk] = {
                "node": node_name,
                "status": "ERROR",
                "log": [error],
                "has_log": False,
            }

        return result

    text = decode_output(
        process.stdout
    ).strip()

    try:
        data = json.loads(
            text
        )
    except json.JSONDecodeError:
        result = {}

        for chunk in range(
            chunk_start,
            chunk_end + 1,
        ):
            result[chunk] = {
                "node": node_name,
                "status": "ERROR",
                "log": [
                    "Failed to parse remote response"
                ],
                "has_log": False,
            }

        return result

    if isinstance(
        data,
        dict,
    ):
        data = [data]

    result = {}

    for item in data:
        chunk = int(
            item["chunk"]
        )

        logs = item.get(
            "log",
            [],
        )

        if logs is None:
            logs = []

        elif isinstance(
            logs,
            str,
        ):
            logs = [logs]

        elif not isinstance(
            logs,
            list,
        ):
            logs = [
                str(logs)
            ]

        result[chunk] = {
            "node": node_name,
            "status": item["status"],
            "log": logs,
            "has_log": bool(
                item.get(
                    "has_log",
                    False,
                )
            ),
        }

    return result


def scan_all_nodes(
    remotes,
    chunk_start,
    chunk_end,
    end,
    shot,
    log_lines,
):
    states = {
        chunk: []
        for chunk in range(
            chunk_start,
            chunk_end + 1,
        )
    }

    for (
        node_name,
        node,
    ) in remotes.items():

        if node["type"] == "local":
            node_result = (
                scan_local_node(
                    node_name,
                    node,
                    chunk_start,
                    chunk_end,
                    end,
                    shot,
                    log_lines,
                )
            )

        elif node["type"] == "ssh":
            node_result = (
                scan_ssh_node(
                    node_name,
                    node,
                    chunk_start,
                    chunk_end,
                    end,
                    shot,
                    log_lines,
                )
            )

        else:
            continue

        for (
            chunk,
            state,
        ) in node_result.items():
            states[
                chunk
            ].append(
                state
            )

    return states


def choose_state(
    node_states,
):
    running = [
        state
        for state in node_states
        if state["status"]
        == "RUNNING"
    ]

    if len(running) > 1:
        nodes = ", ".join(
            state["node"]
            for state in running
        )

        return {
            "node": nodes,
            "status": "MULTIPLE",
            "log": [
                "Same chunk is running on multiple nodes"
            ],
        }

    if len(running) == 1:
        return running[0]

    done_with_log = [
        state
        for state in node_states
        if (
            state["status"] == "DONE"
            and state["has_log"]
        )
    ]

    if done_with_log:
        return done_with_log[0]

    done = [
        state
        for state in node_states
        if state["status"]
        == "DONE"
    ]

    if done:
        return done[0]

    stopped_with_log = [
        state
        for state in node_states
        if (
            state["status"]
            == "STOPPED"
            and state["has_log"]
        )
    ]

    if stopped_with_log:
        return stopped_with_log[0]

    errors = [
        state
        for state in node_states
        if state["status"]
        == "ERROR"
    ]

    if errors:
        return errors[0]

    old_logs = [
        state
        for state in node_states
        if state["has_log"]
    ]

    if old_logs:
        state = old_logs[0]

        return {
            "node": state["node"],
            "status": "UNKNOWN",
            "log": state["log"],
        }

    return {
        "node": "-",
        "status": "PENDING",
        "log": [],
    }


def shorten(
    value,
    width=90,
):
    if value is None:
        text = ""

    elif isinstance(
        value,
        str,
    ):
        text = value

    elif isinstance(
        value,
        (dict, list),
    ):
        text = json.dumps(
            value,
            ensure_ascii=False,
        )

    else:
        text = str(
            value
        )

    text = (
        text
        .replace(
            "\r",
            " ",
        )
        .replace(
            "\n",
            " ",
        )
        .strip()
    )

    if len(text) <= width:
        return text

    return (
        text[: width - 3]
        + "..."
    )


def display(
    states,
    chunk_start,
    chunk_end,
    end,
    shot,
):
    print(
        "\033[2J\033[H",
        end="",
    )

    print(
        "NewSL distributed generation monitor"
    )

    print(
        f"Target : end{end}/shot{shot}"
    )

    print(
        f"Chunks : "
        f"{chunk_start}-{chunk_end}"
    )

    print(
        "Updated:",
        time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    )

    print()

    print(
        f"{'CHUNK':>5}  "
        f"{'NODE':<12}  "
        f"{'STATUS':<9}  "
        f"LOG"
    )

    print(
        "-" * 120
    )

    counts = {}

    for chunk in range(
        chunk_start,
        chunk_end + 1,
    ):
        state = choose_state(
            states[chunk]
        )

        status = state["status"]

        counts[status] = (
            counts.get(
                status,
                0,
            )
            + 1
        )

        logs = state.get(
            "log",
            [],
        )

        if isinstance(
            logs,
            str,
        ):
            logs = [logs]

        elif not isinstance(
            logs,
            list,
        ):
            logs = [
                str(logs)
            ]

        if not logs:
            logs = [""]

        print(
            f"{chunk:>5}  "
            f"{state['node']:<12}  "
            f"{status:<9}  "
            f"{shorten(logs[0])}"
        )

        for log_line in logs[1:]:
            print(
                f"{'':>5}  "
                f"{'':<12}  "
                f"{'':<9}  "
                f"{shorten(log_line)}"
            )

    print()
    print(
        "SUMMARY:",
        end=" ",
    )

    order = [
        "RUNNING",
        "DONE",
        "PENDING",
        "STOPPED",
        "UNKNOWN",
        "ERROR",
        "MULTIPLE",
    ]

    summary = []

    for status in order:
        count = counts.get(
            status,
            0,
        )

        if count:
            summary.append(
                f"{status}={count}"
            )

    print(
        "  ".join(summary)
    )

    print()
    print(
        "Ctrl+C to stop monitoring."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "chunk_start",
        type=int,
    )

    parser.add_argument(
        "chunk_end",
        type=int,
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

    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--lines",
        type=int,
        default=1,
        help=(
            "Number of recent log lines "
            "to display per chunk."
        ),
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help=(
            "Display once and exit."
        ),
    )

    args = parser.parse_args()

    if (
        args.chunk_end
        < args.chunk_start
    ):
        raise SystemExit(
            "chunk_end must be >= chunk_start"
        )

    if args.interval <= 0:
        raise SystemExit(
            "interval must be > 0"
        )

    if args.lines <= 0:
        raise SystemExit(
            "lines must be > 0"
        )

    remotes = load_remotes()

    if not remotes:
        raise SystemExit(
            "No remotes configured."
        )

    try:
        while True:
            states = scan_all_nodes(
                remotes,
                args.chunk_start,
                args.chunk_end,
                args.end,
                args.shot,
                args.lines,
            )

            display(
                states,
                args.chunk_start,
                args.chunk_end,
                args.end,
                args.shot,
            )

            if args.once:
                break

            time.sleep(
                args.interval
            )

    except KeyboardInterrupt:
        print(
            "\nMonitoring stopped."
        )


if __name__ == "__main__":
    main()