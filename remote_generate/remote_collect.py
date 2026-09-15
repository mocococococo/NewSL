"""Copy completed chunk files to this PC and verify them; no state changes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from remote_generate.state_store import file_hash, local_path
from remote_generate.config_loader import DEFAULT_CONFIG, read_config
from remote_generate.ssh_connection import SSHConnection, credentials


def manifest(job, result):
    if result.get("success") is not True:
        raise ValueError("Only successful generation results may be collected")
    prefix = f"data/end{job['end']}/shot{job['shot']}/sl_data_chunk{job['chunk']}_"
    seen = set()
    for item in result["files"]:
        path = item["path"]
        if not re.fullmatch(re.escape(prefix) + r"\d+\.npz", path) or path in seen:
            raise ValueError("Unexpected or duplicate result path")
        local_path(path)
        if (type(item["size"]) is not int or item["size"] <= 0
                or type(item["samples"]) is not int or item["samples"] <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
            raise ValueError("Invalid result metadata")
        seen.add(path)
    if result["samples"] != sum(item["samples"] for item in result["files"]):
        raise ValueError("Inconsistent sample count")
    return result["files"]


def matches(path, item):
    return path.is_file() and path.stat().st_size == item["size"] and file_hash(path) == item["sha256"]


def verify_local(job, result):
    for item in manifest(job, result):
        if not matches(local_path(item["path"]), item):
            raise ValueError(f"Collected file is missing or differs: {item['path']}")


def collect(connection, job, result):
    files = manifest(job, result)
    if not files:
        return  # A chunk can validly contain no retained samples.
    with connection.sftp() as sftp:
        for item in files:
            target = local_path(item["path"])
            if target.exists():
                if matches(target, item):
                    continue
                raise FileExistsError(f"Different local file already exists: {target}")
            source = str(PurePosixPath(connection.node["root"].replace("\\", "/")) / item["path"])
            if sftp.stat(source).st_size != item["size"]:
                raise ValueError("Remote size differs from completion manifest")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
            try:
                digest, size = hashlib.sha256(), 0
                with sftp.open(source, "rb") as incoming, temporary.open("xb") as outgoing:
                    for block in iter(lambda: incoming.read(1024 * 1024), b""):
                        size += len(block)
                        if size > item["size"]:
                            raise ValueError("Remote file grew during collection")
                        digest.update(block)
                        outgoing.write(block)
                    outgoing.flush()
                    os.fsync(outgoing.fileno())
                if size != item["size"] or digest.hexdigest() != item["sha256"]:
                    raise ValueError("Downloaded content differs from completion manifest")
                # Publish without ever overwriting an existing final filename.
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    if not matches(target, item):
                        raise
            finally:
                temporary.unlink(missing_ok=True)
    verify_local(job, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--pc-id", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    spec = next(node for node in config["remotes"] if node["pc_id"] == args.pc_id)
    connection = SSHConnection(config["gateway"], spec, config["transport"], credentials(config["env_file"]))
    status = json.loads(connection.run([spec["python"], "-m", "remote_generate.remote_status", "--pc-id", args.pc_id]))
    row = next(item for item in status["workers"] if item["worker_id"] == args.worker_id)
    if row["state"] != "completed" or row["job"]["job_id"] != args.job_id:
        raise ValueError("Specified job is not completed")
    collect(connection, row["job"], row["result"])
    print("Collected and verified. State has not been changed.")


if __name__ == "__main__":
    main()
