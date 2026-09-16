import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

from remote_config import get_remote


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("remote_name")
    parser.add_argument("chunk", type=int)
    parser.add_argument("--end", type=int, default=9)
    parser.add_argument("--shot", type=int, default=2)
    args = parser.parse_args()

    # remotes.json から対象PCの設定を取得
    remote = get_remote(args.remote_name)

    host = remote["host"]
    remote_root = remote["root"]

    remote_dir = (
        f"{remote_root}/data/end{args.end}/shot{args.shot}"
    )

    # 遠隔PCで対象chunkのNPZ一覧を取得
    python_code = f"""
from pathlib import Path
import json

p = Path(r"{remote_dir}")

files = sorted(
    p.glob("sl_data_chunk{args.chunk}_*.npz")
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

    # SSH経由のクォート問題を避けるためBase64化
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
            f"No output found for chunk {args.chunk}. "
            "Is generation finished?"
        )

    local_dir = (
        ROOT
        / "data"
        / f"end{args.end}"
        / f"shot{args.shot}"
    )
    local_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        remote_path = file["path"]
        local_path = local_dir / file["name"]

        print(f"COLLECT: {args.remote_name}:{remote_path}")
        print(f"      -> {local_path}")

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

        # 転送後のサイズ確認
        local_size = local_path.stat().st_size

        if local_size != file["size"]:
            raise SystemExit(
                f"Size mismatch: {file['name']} "
                f"remote={file['size']} "
                f"local={local_size}"
            )

        print(
            f"OK: {file['name']} "
            f"({local_size:,} bytes)"
        )

    print(
        f"COLLECTED: {args.remote_name} "
        f"chunk {args.chunk} "
        f"({len(files)} file(s))"
    )


if __name__ == "__main__":
    main()