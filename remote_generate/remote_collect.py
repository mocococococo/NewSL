import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

from remote_config import (
    get_remote,
    get_data_root,
)


ROOT = Path(__file__).resolve().parents[1]


def decode_output(data: bytes) -> str:
    if not data:
        return ""

    for encoding in ("utf-8", "cp932"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass

    return data.decode("utf-8", errors="replace")


def collect_local(
    node,
    chunk,
    end,
    shot,
    run_name,
):
    data_dir = (
        get_data_root(
            Path(node["root"]),
            run_name,
        )
        / f"end{end}"
        / f"shot{shot}"
    )

    files = sorted(
        data_dir.glob(
            f"sl_data_chunk{chunk}_*.npz"
        )
    )

    if not files:
        raise SystemExit(
            f"No output found for local chunk {chunk}. "
            "Is generation finished?"
        )

    for path in files:
        print(
            f"OK: {path.name} "
            f"({path.stat().st_size:,} bytes)"
        )

    print(
        f"COLLECTED: local "
        f"chunk {chunk} "
        f"({len(files)} file(s), no copy needed)"
    )


def collect_ssh(
    node,
    remote_name,
    chunk,
    end,
    shot,
    run_name,
):
    host = node["host"]
    remote_root = node["root"]

    remote_dir = (
            get_data_root(
            Path(remote_root),
            run_name,
        )
        / f"end{end}"
        / f"shot{shot}"
    ).as_posix()

    python_code = f"""
from pathlib import Path
import json

p = Path(r"{remote_dir}")

files = sorted(
    p.glob("sl_data_chunk{chunk}_*.npz")
)

print(json.dumps([
    {{
        "path": x.as_posix(),
        "name": x.name,
        "size": x.stat().st_size,
    }}
    for x in files
]))
"""

    encoded = base64.b64encode(
        python_code.encode("utf-8")
    ).decode("ascii")

    remote_command = (
        'python -c '
        f'"import base64;'
        f'exec(base64.b64decode(\'{encoded}\'))"'
    )

    result = subprocess.run(
        ["ssh", host, remote_command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stdout = decode_output(result.stdout)
    stderr = decode_output(result.stderr)

    if result.returncode != 0:
        if stderr:
            print(stderr, file=sys.stderr)
        raise SystemExit(result.returncode)

    try:
        files = json.loads(stdout.strip())
    except json.JSONDecodeError:
        print("Failed to read remote file list.")
        print("stdout:")
        print(stdout)

        if stderr:
            print("stderr:")
            print(stderr)

        raise SystemExit(1)

    if not files:
        raise SystemExit(
            f"No output found for chunk {chunk}. "
            "Is generation finished?"
        )

    local_dir = (
        get_data_root(
            ROOT,
            run_name,
        )
        / f"end{end}"
        / f"shot{shot}"
    )
    local_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for file in files:
        remote_path = file["path"]
        local_path = (
            local_dir
            / file["name"]
        )

        print(
            f"COLLECT: "
            f"{remote_name}:{remote_path}"
        )
        print(
            f"      -> {local_path}"
        )

        copy_result = subprocess.run(
            [
                "scp",
                f"{host}:{remote_path}",
                str(local_path),
            ]
        )

        if copy_result.returncode != 0:
            raise SystemExit(
                f"scp failed: {file['name']}"
            )

        local_size = (
            local_path.stat().st_size
        )

        if local_size != file["size"]:
            raise SystemExit(
                f"Size mismatch: "
                f"{file['name']} "
                f"remote={file['size']} "
                f"local={local_size}"
            )

        print(
            f"OK: {file['name']} "
            f"({local_size:,} bytes)"
        )

    print(
        f"COLLECTED: {remote_name} "
        f"chunk {chunk} "
        f"({len(files)} file(s))"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("remote_name")
    parser.add_argument("chunk", type=int)
    parser.add_argument("--end", type=int, default=9)
    parser.add_argument("--shot", type=int, default=2)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    node = get_remote(args.remote_name)

    node_type = node["type"]

    if node_type == "local":
        collect_local(
            node,
            args.chunk,
            args.end,
            args.shot,
            args.run_name,
        )

    elif node_type == "ssh":
        collect_ssh(
            node,
            args.remote_name,
            args.chunk,
            args.end,
            args.shot,
            args.run_name,
        )

    else:
        raise SystemExit(
            f"Unknown node type: {node_type}"
        )


if __name__ == "__main__":
    main()