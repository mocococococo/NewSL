import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REMOTE_ROOT = "C:/Users/itolab/DigitalCurling/NewSL"


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
    parser.add_argument("chunk", type=int)
    parser.add_argument("--end", type=int, default=9)
    parser.add_argument("--shot", type=int, default=2)
    args = parser.parse_args()

    # 遠隔PCで対象chunkの出力ファイル一覧を取得
    remote_dir = (
        f"{REMOTE_ROOT}/data/end{args.end}/shot{args.shot}"
    )

    python_code = (
        "from pathlib import Path; import json; "
        f"p=Path(r'{remote_dir}'); "
        f"files=sorted(p.glob('sl_data_chunk{args.chunk}_*.npz')); "
        "print(json.dumps(["
        "{'path': x.as_posix(), 'name': x.name, 'size': x.stat().st_size} "
        "for x in files]))"
    )

    result = subprocess.run(
        ["ssh", args.host, "python", "-c", python_code],
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
        print("Failed to read remote file list:")
        print(stdout)
        raise SystemExit(1)

    if not files:
        raise SystemExit(
            f"No output found for chunk {args.chunk}. "
            "Is generation finished?"
        )

    local_dir = ROOT / "data" / f"end{args.end}" / f"shot{args.shot}"
    local_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        remote_path = file["path"]
        local_path = local_dir / file["name"]

        print(f"COLLECT: {args.host}:{remote_path}")
        print(f"      -> {local_path}")

        copy_result = subprocess.run(
            [
                "scp",
                f"{args.host}:{remote_path}",
                str(local_path),
            ]
        )

        if copy_result.returncode != 0:
            raise SystemExit(
                f"scp failed: {file['name']}"
            )

        # 最低限の転送確認
        local_size = local_path.stat().st_size

        if local_size != file["size"]:
            raise SystemExit(
                f"Size mismatch: {file['name']} "
                f"remote={file['size']} local={local_size}"
            )

        print(
            f"OK: {file['name']} "
            f"({local_size:,} bytes)"
        )

    print(
        f"COLLECTED: {args.host} "
        f"chunk {args.chunk} "
        f"({len(files)} file(s))"
    )


if __name__ == "__main__":
    main()