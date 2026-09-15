"""Atomic access to the one persistent remote/state.json on this PC."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "remote/state.json"


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", value):
        raise ValueError(f"Invalid identifier: {value!r}")
    return value


def local_path(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or ":" in str(value):
        raise ValueError(f"Expected repository-relative path: {value!r}")
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError("Path escapes repository")
    return resolved


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     separators=(",", ":")).encode()).hexdigest()


@contextmanager
def mutex(name="state"):
    """No persistent lock file. State lock waits; coordinator lock fails fast."""
    if os.name == "nt":
        from ctypes import wintypes
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
        api.CreateMutexW.restype = wintypes.HANDLE
        api.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        api.WaitForSingleObject.restype = wintypes.DWORD
        api.ReleaseMutex.argtypes = (wintypes.HANDLE,)
        api.CloseHandle.argtypes = (wintypes.HANDLE,)
        key = hashlib.sha256(str(STATE).casefold().encode()).hexdigest()
        handle = api.CreateMutexW(None, False, f"Global\\NewSL-{key}-{name}")
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        acquired = False
        try:
            result = api.WaitForSingleObject(handle, 30000 if name == "state" else 0)
            if result not in (0, 0x80):
                raise RuntimeError(f"Could not acquire {name} lock (another process may own it)")
            acquired = True
            yield
        finally:
            if acquired:
                api.ReleaseMutex(handle)
            api.CloseHandle(handle)
    else:
        import fcntl
        fd = os.open(STATE.parent if name == "state" else ROOT, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (fcntl.LOCK_NB if name != "state" else 0))
            yield
        finally:
            os.close(fd)


def read_state():
    value = json.loads(STATE.read_text(encoding="utf-8-sig"))
    if value.get("schema_version") != 1 or not isinstance(value.get("workers"), list):
        raise ValueError("state.json requires schema_version=1 and workers array")
    identifier(value["pc_id"])
    ids = set()
    for row in value["workers"]:
        wid = identifier(row["worker_id"])
        if wid in ids or row["state"] not in ("idle", "running", "completed"):
            raise ValueError("Invalid worker states or duplicate worker ID")
        ids.add(wid)
    return value


@contextmanager
def transaction(pc_id=None):
    with mutex():
        state = read_state()
        if pc_id is not None and state["pc_id"] != pc_id:
            raise ValueError("state.json belongs to another PC")
        yield state
        write_state(state)


def write_state(value):
    """Caller must hold the state mutex."""
    value["updated_at"] = now()
    temporary = STATE.with_name(f".state.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, STATE)
    finally:
        temporary.unlink(missing_ok=True)


def initialize(pc_id, workers):
    identifier(pc_id)
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be positive")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with mutex():
        state = read_state() if STATE.exists() else dict(schema_version=1, pc_id=pc_id, workers=[])
        if state["pc_id"] != pc_id:
            raise ValueError("state.json belongs to another PC")
        wanted = [f"worker-{index}" for index in range(workers)]
        existing = {row["worker_id"]: row for row in state["workers"]}
        if any(row["state"] != "idle" for key, row in existing.items() if key not in wanted):
            raise RuntimeError("Cannot remove occupied workers")
        state["workers"] = [existing.get(key, dict(worker_id=key, state="idle")) for key in wanted]
        write_state(state)
    return state


def worker(state, worker_id):
    return next(row for row in state["workers"] if row["worker_id"] == worker_id)


def process_token(pid):
    """Windows creation time avoids both PID reuse and os.kill on Windows."""
    if type(pid) is not int or pid <= 0:
        return None
    if os.name != "nt":
        try:
            # /proc starttime is stable across process lifetime on Linux.
            return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        except FileNotFoundError:
            return None
    from ctypes import wintypes
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    api.OpenProcess.restype = wintypes.HANDLE
    api.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
    api.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    api.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = api.OpenProcess(0x1000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:
            return None
        raise ctypes.WinError(error)
    try:
        code = wintypes.DWORD()
        if not api.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        if code.value != 259:
            return None
        times = [wintypes.FILETIME() for _ in range(4)]
        if not api.GetProcessTimes(handle, *(ctypes.byref(item) for item in times)):
            raise ctypes.WinError(ctypes.get_last_error())
        return str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime)
    finally:
        api.CloseHandle(handle)


def acknowledge(pc_id, worker_id, job_id):
    """Only the coordinator requests this, after its durable collection receipt."""
    with transaction(pc_id) as state:
        row = worker(state, worker_id)
        if row["state"] == "idle" and row.get("last_job_id") == job_id:
            return
        if (row["state"] != "completed" or row["job"]["job_id"] != job_id
                or not row["result"].get("success")):
            raise RuntimeError("Cannot acknowledge an unfinished, failed, or different job")
        row.clear()
        row.update(worker_id=worker_id, state="idle", last_job_id=job_id)


def reset_failed(pc_id, worker_id, job_id):
    """Explicit recovery only; never clear a live or successfully completed job."""
    with transaction(pc_id) as state:
        row = worker(state, worker_id)
        if row["state"] == "idle" and row.get("last_reset_job_id") == job_id:
            return
        if row.get("job", {}).get("job_id") != job_id:
            raise RuntimeError("Reset targets a different job")
        token = process_token(row.get("pid"))
        if token is not None and token == row.get("process_token"):
            raise RuntimeError("Cannot reset a live process")
        if row["state"] == "completed" and row["result"].get("success"):
            raise RuntimeError("Collect successful results before acknowledging them")
        row.clear()
        row.update(worker_id=worker_id, state="idle", last_reset_job_id=job_id,
                   reset_token=uuid.uuid4().hex)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "ack", "reset"))
    parser.add_argument("--pc-id", required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--worker-id")
    parser.add_argument("--job-id")
    args = parser.parse_args()
    if args.action == "init":
        initialize(args.pc_id, args.workers)
    else:
        operation = acknowledge if args.action == "ack" else reset_failed
        operation(args.pc_id, identifier(args.worker_id), identifier(args.job_id))
    print(json.dumps({"ok": True}))


if __name__ == "__main__":
    main()
