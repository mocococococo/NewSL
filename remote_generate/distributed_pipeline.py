import argparse
import base64
import subprocess
import sys
from pathlib import Path

from remote_config import (
    load_remotes,
    get_run_name,
    get_model_root,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def model_name(end, shot):
    return f"shot-end{end}-shot{shot}.bin"


def model_path(
    end,
    shot,
    run_name,
):
    return (
        get_model_root(
            ROOT,
            run_name,
        )
        / f"end_{end}"
        / model_name(end, shot)
    )


def positions(
    start_end,
    start_shot,
    stop_end,
    stop_shot,
):
    if not 0 <= start_end <= 9:
        raise ValueError(
            "start_end must be 0..9"
        )

    if not 0 <= stop_end <= 9:
        raise ValueError(
            "stop_end must be 0..9"
        )

    if not 0 <= start_shot <= 15:
        raise ValueError(
            "start_shot must be 0..15"
        )

    if not 0 <= stop_shot <= 15:
        raise ValueError(
            "stop_shot must be 0..15"
        )

    start_index = (
        start_end * 16
        + start_shot
    )

    stop_index = (
        stop_end * 16
        + stop_shot
    )

    if stop_index > start_index:
        raise ValueError(
            "stop position must be "
            "before start position"
        )

    result = []

    for index in range(
        start_index,
        stop_index - 1,
        -1,
    ):
        end, shot = divmod(
            index,
            16,
        )

        result.append(
            (end, shot)
        )

    return result


def decode_output(data: bytes):
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


def run_distribute(
    chunk_start,
    chunk_end,
    end,
    shot,
    epochs,
    poll_seconds,
    run_name,
):
    command = [
        sys.executable,
        str(
            HERE
            / "distribute.py"
        ),
        str(chunk_start),
        str(chunk_end),
        "--end",
        str(end),
        "--shot",
        str(shot),
        "--poll-seconds",
        str(poll_seconds),
        "--run-name",
        str(run_name),
    ]

    if epochs is not None:
        command += [
            "--epochs",
            str(epochs),
        ]

    print()
    print("RUN:")
    print(
        " ".join(
            map(
                str,
                command,
            )
        )
    )
    print()

    result = subprocess.run(
        command,
        cwd=ROOT,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"distribute.py failed: "
            f"end{end}/shot{shot}"
        )


def remote_model_path(
    node,
    path,
):
    root = (
        str(
            node["root"]
        )
        .rstrip(
            "/\\"
        )
        .replace(
            "\\",
            "/",
        )
    )

    relative_path = (
        path
        .relative_to(
            ROOT / "model"
        )
        .as_posix()
    )

    return (
        f"{root}/model/"
        f"{relative_path}"
    )


def ensure_remote_model_dir(
    host,
    remote_path,
):
    escaped_path = (
        remote_path.replace(
            "'",
            "''",
        )
    )

    script = f"""
$ErrorActionPreference = 'Stop'

$path = '{escaped_path}'

$modelDir = Split-Path `
    -Parent `
    $path

New-Item `
    -ItemType Directory `
    -Force `
    -Path $modelDir `
    | Out-Null
"""

    encoded = base64.b64encode(
        script.encode(
            "utf-16-le"
        )
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

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to create "
            f"remote model directory "
            f"on {host}:\n"
            f"{decode_output(result.stderr)}"
        )


def verify_remote_model(
    host,
    remote_path,
    expected_size,
):
    escaped_path = (
        remote_path.replace(
            "'",
            "''",
        )
    )

    script = f"""
$ErrorActionPreference = 'Stop'

$path = '{escaped_path}'

if (-not (
    Test-Path `
        -LiteralPath $path
)) {{
    Write-Error "Model does not exist: $path"
    exit 2
}}

$file = Get-Item `
    -LiteralPath $path

Write-Output $file.Length
"""

    encoded = base64.b64encode(
        script.encode(
            "utf-16-le"
        )
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

    if result.returncode != 0:
        raise RuntimeError(
            f"Remote model verification "
            f"failed on {host}:\n"
            f"{decode_output(result.stderr)}"
        )

    text = decode_output(
        result.stdout
    ).strip()

    try:
        remote_size = int(
            text.splitlines()[-1]
        )

    except (
        ValueError,
        IndexError,
    ):
        raise RuntimeError(
            f"Could not read remote "
            f"model size on {host}: "
            f"{text}"
        )

    if remote_size != expected_size:
        raise RuntimeError(
            f"Model size mismatch "
            f"on {host}: "
            f"local={expected_size}, "
            f"remote={remote_size}"
        )


def sync_model_to_enabled_remotes(
    path,
):
    if not path.is_file():
        raise FileNotFoundError(
            f"Model not found: "
            f"{path}"
        )

    remotes = load_remotes()

    local_size = (
        path.stat().st_size
    )

    for (
        name,
        node,
    ) in remotes.items():

        if not node.get(
            "enabled",
            True,
        ):
            continue

        if node["type"] == "local":
            continue

        if node["type"] != "ssh":
            raise RuntimeError(
                f"Unknown node type "
                f"for {name}: "
                f"{node['type']}"
            )

        host = node["host"]

        remote_path = (
            remote_model_path(
                node,
                path,
            )
        )

        print(
            f"SYNC MODEL: "
            f"{path.name} "
            f"-> {name}"
        )

        ensure_remote_model_dir(
            host,
            remote_path,
        )

        result = subprocess.run(
            [
                "scp",
                str(path),
                f"{host}:{remote_path}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to copy "
                f"{path.name} "
                f"to {name}:\n"
                f"{decode_output(result.stderr)}"
            )

        verify_remote_model(
            host,
            remote_path,
            local_size,
        )

        print(
            f"SYNC OK: "
            f"{name} "
            f"({local_size:,} bytes)"
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
        "--start-end",
        type=int,
        default=9,
    )

    parser.add_argument(
        "--start-shot",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--stop-end",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--stop-shot",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=300,
    )

    args = parser.parse_args()
    
    remotes = load_remotes()

    run_name = get_run_name(
        remotes
    )

    if (
        args.chunk_end
        < args.chunk_start
    ):
        raise SystemExit(
            "chunk_end must be "
            ">= chunk_start"
        )

    try:
        sequence = positions(
            args.start_end,
            args.start_shot,
            args.stop_end,
            args.stop_shot,
        )

    except ValueError as exc:
        raise SystemExit(
            str(exc)
        )

    print(
        "================================"
    )

    print(
        "NewSL distributed pipeline"
    )

    print(
        f"Chunks: "
        f"{args.chunk_start}-"
        f"{args.chunk_end}"
    )

    print(
        f"Run  : "
        f"{run_name}"
    )

    print(
        f"Start: "
        f"end{args.start_end}/"
        f"shot{args.start_shot}"
    )

    print(
        f"Stop : "
        f"end{args.stop_end}/"
        f"shot{args.stop_shot}"
    )

    print(
        f"Positions: "
        f"{len(sequence)}"
    )

    print(
        "================================"
    )

    # この実行中にすでにremoteへ送ったモデル
    synced_models = set()

    def ensure_synced(path):
        resolved = (
            path.resolve()
        )

        if resolved in synced_models:
            return

        sync_model_to_enabled_remotes(
            path
        )

        synced_models.add(
            resolved
        )

    total = len(
        sequence
    )

    for index, (
        end,
        shot,
    ) in enumerate(
        sequence,
        1,
    ):
        print()
        print(
            "================================"
        )

        print(
            f"[PIPELINE "
            f"{index}/{total}] "
            f"end{end}/shot{shot}"
        )

        print(
            "================================"
        )

        current_model = (
            model_path(
                end,
                shot,
                run_name,
            )
        )

        # --------------------------------
        # shot14以下では
        # shot+1モデルが教師
        # --------------------------------
        if shot < 15:

            teacher = (
                model_path(
                    end,
                    shot + 1,
                    run_name,
                )
            )

            if not teacher.is_file():
                raise FileNotFoundError(
                    "Required teacher "
                    "model does not exist: "
                    f"{teacher}"
                )

            ensure_synced(
                teacher
            )

        # --------------------------------
        # モデルがすでにあれば
        # その局面は完了済み
        # --------------------------------
        if current_model.is_file():

            print(
                f"SKIP: model already "
                f"exists: "
                f"{current_model}"
            )

            # 後続shotで教師として
            # remoteでも使用できるようにする
            ensure_synced(
                current_model
            )

            continue

        # --------------------------------
        # 分散生成 → 回収 → 学習
        # --------------------------------
        run_distribute(
            args.chunk_start,
            args.chunk_end,
            end,
            shot,
            args.epochs,
            args.poll_seconds,
            run_name,
        )

        # --------------------------------
        # 学習済みモデル確認
        # --------------------------------
        if not current_model.is_file():

            raise FileNotFoundError(
                "Training finished but "
                "model was not created: "
                f"{current_model}"
            )

        print(
            f"MODEL CREATED: "
            f"{current_model}"
        )

        # 次のshotの教師になるのでremoteへ配布
        ensure_synced(
            current_model
        )

    print()
    print(
        "================================"
    )

    print(
        f"PIPELINE ALL DONE: "
        f"{len(sequence)} positions"
    )

    print(
        "================================"
    )


if __name__ == "__main__":
    main()