import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

TEMP_ROOT = (
    ROOT
    / ".temp"
    / "remote_generate"
)


def is_pid_running(
    pid,
):
    result = subprocess.run(
        [
            "tasklist",
            "/FI",
            f"PID eq {pid}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    return (
        str(pid)
        in result.stdout
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
        "--target-end",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--target-shot",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--log-path",
        required=True,
    )

    gpu_group = (
        parser
        .add_mutually_exclusive_group()
    )

    gpu_group.add_argument(
        "--use-gpu",
        dest="use_gpu",
        action="store_true",
    )

    gpu_group.add_argument(
        "--no-use-gpu",
        dest="use_gpu",
        action="store_false",
    )

    parser.set_defaults(
        use_gpu=True
    )

    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    job_temp_root = (
        TEMP_ROOT
        / f"end{args.target_end}"
        / f"shot{args.target_shot}"
    )

    log_dir = (
        job_temp_root
        / "logs"
    )

    pid_dir = (
        job_temp_root
        / "pids"
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pid_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunk_name = (
        f"chunk_"
        f"{args.chunk_start}_"
        f"{args.chunk_end}"
    )

    log_file = (
        log_dir
        / f"{chunk_name}.log"
    )

    pid_file = (
        pid_dir
        / f"{chunk_name}.pid"
    )

    if pid_file.is_file():

        try:
            existing_pid = int(
                pid_file
                .read_text(
                    encoding="utf-8"
                )
                .strip()
            )

        except ValueError:
            existing_pid = None

        if (
            existing_pid is not None
            and is_pid_running(
                existing_pid
            )
        ):
            raise RuntimeError(
                f"Chunk is already running: "
                f"PID={existing_pid}"
            )

    command = [
        sys.executable,
        "-u",
        str(
            ROOT
            / "transformer"
            / "shot_generator.py"
        ),
        "--chunk_start",
        str(args.chunk_start),
        "--chunk_end",
        str(args.chunk_end),
        "--target_end",
        str(args.target_end),
        "--target_shot",
        str(args.target_shot),
        "--num_workers",
        "1",
        "--log_path",
        str(args.log_path),
    ]

    if args.use_gpu:
        command.append(
            "--use_gpu"
        )
    else:
        command.append(
            "--no_use_gpu"
        )

    if (
        args.inference_batch_size
        is not None
    ):
        command += [
            "--inference_batch_size",
            str(
                args.inference_batch_size
            ),
        ]

    if args.dry_run:

        print(
            f"TARGET: "
            f"end={args.target_end}, "
            f"shot={args.target_shot}"
        )

        print(
            f"USE GPU: "
            f"{args.use_gpu}"
        )

        print(
            f"INFERENCE BATCH SIZE: "
            f"{args.inference_batch_size}"
        )

        print(
            "COMMAND:"
        )

        print(
            subprocess.list2cmdline(
                command
            )
        )

        print(
            f"LOG: "
            f"{log_file}"
        )

        print(
            f"PID FILE: "
            f"{pid_file}"
        )

        return

    with log_file.open(
        "w",
        encoding="utf-8",
    ) as log:

        creation_flags = (
            subprocess
            .DETACHED_PROCESS
            | subprocess
            .CREATE_NEW_PROCESS_GROUP
            | subprocess
            .CREATE_BREAKAWAY_FROM_JOB
        )

        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=creation_flags,
            close_fds=True,
        )

    pid_file.write_text(
        str(process.pid),
        encoding="utf-8",
    )

    print(
        f"PID: "
        f"{process.pid}"
    )

    print(
        f"LOG: "
        f"{log_file}"
    )


if __name__ == "__main__":
    main()