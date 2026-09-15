"""Run this PC's distributed generation, collection, training and publication loop."""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from remote_generate import state_store as store
from remote_generate.config_loader import DEFAULT_CONFIG, read_config
from remote_generate.remote_collect import collect, manifest, verify_local
from remote_generate.ssh_connection import (ConnectionUnavailable, SSHConnection,
                                            credentials)


def encoded(value):
    return base64.b64encode(json.dumps(value, ensure_ascii=True).encode()).decode()


def progress(message):
    """User-facing logs stay separate from machine-readable subprocess output."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=store.ROOT, stderr=subprocess.PIPE).decode("utf-8").strip()


def positions(config):
    g = config["generation"]
    start = g["start_end"] * 16 + g["start_shot"]
    stop = g["stop_end"] * 16 + g["stop_shot"]
    if stop > start:
        raise ValueError("Generation proceeds backwards from start to stop")
    return [list(divmod(index, 16)) for index in range(start, stop - 1, -1)]


def load_config(path):
    config = read_config(path)
    def keys(value, expected, label):
        if not isinstance(value, dict) or set(value) != set(expected.split()):
            raise ValueError(f"{label} requires exactly these keys: {expected}")
    keys(config, "schema_version env_file gateway local remotes control transport generation training git", "config")
    keys(config["gateway"], "host username key_file known_hosts", "gateway")
    keys(config["local"], "pc_id workers threads_per_worker log_path", "local")
    keys(config["control"], "poll_seconds connection_retries retry_seconds start_grace_seconds start_attempts", "control")
    keys(config["transport"], "connect_timeout_seconds io_timeout_seconds command_timeout_seconds", "transport")
    keys(config["training"], "batch_size epochs", "training")
    keys(config["git"], "remote", "git")
    keys(config["generation"], "search_mode base_model start_end start_shot stop_end stop_shot "
         "chunk_start chunk_end chunk_size data_size shuffle_seed simulation_seed inference_batch_size "
         "max_simulations use_end_augmentation use_score_diff_augmentation "
         "policy_min_visit policy_delta_q policy_alpha_visit policy_beta_q policy_lambda_best "
         "value_min_visit value_delta_q value_alpha_visit value_beta_q value_lambda_best", "generation")
    if not isinstance(config["remotes"], list):
        raise ValueError("remotes must be an array")
    for node in config["remotes"]:
        keys(node, "pc_id workers threads_per_worker root python username password_env known_hosts registry_path log_path", "remote")
    if config["schema_version"] != 1:
        raise ValueError("Unsupported config schema")
    g = config["generation"]
    for name, maximum in (("start_end", 9), ("stop_end", 9), ("start_shot", 15), ("stop_shot", 15)):
        if type(g[name]) is not int or not 0 <= g[name] <= maximum:
            raise ValueError(f"Invalid generation.{name}")
    for name in ("chunk_start", "chunk_end", "shuffle_seed", "simulation_seed"):
        if type(g[name]) is not int or g[name] < 0:
            raise ValueError(f"generation.{name} must be a nonnegative integer")
    if g["chunk_end"] < g["chunk_start"]:
        raise ValueError("chunk_end must include or follow chunk_start")
    for section, names in (("generation", ("chunk_size", "data_size", "inference_batch_size", "max_simulations")),
                           ("training", ("batch_size", "epochs")),
                           ("control", ("start_attempts",))):
        for name in names:
            if type(config[section][name]) is not int or config[section][name] < 1:
                raise ValueError(f"{section}.{name} must be a positive integer")
    for section, names in (("control", ("poll_seconds", "retry_seconds", "start_grace_seconds")),
                           ("transport", ("connect_timeout_seconds", "io_timeout_seconds", "command_timeout_seconds"))):
        for name in names:
            value = config[section][name]
            if type(value) not in (int, float) or not 0 < value < float("inf"):
                raise ValueError(f"{section}.{name} must be finite and positive")
    retries = config["control"]["connection_retries"]
    if type(retries) is not int or retries < 0:
        raise ValueError("connection_retries must be nonnegative")
    ids = set()
    for node in [config["local"], *config["remotes"]]:
        key = store.identifier(node["pc_id"])
        if key in ids:
            raise ValueError("Duplicate PC identity")
        ids.add(key)
        for name in ("workers", "threads_per_worker"):
            if type(node[name]) is not int or node[name] < 1:
                raise ValueError(f"{key}.{name} must be positive")
        store.local_path(node["log_path"])
    store.local_path(g["base_model"])
    store.identifier(config["git"]["remote"])
    if g["search_mode"] not in ("shot", "shot_origin"):
        raise ValueError("Invalid search mode")
    for name in ("use_end_augmentation", "use_score_diff_augmentation"):
        if type(g[name]) is not bool:
            raise ValueError(f"generation.{name} must be boolean")
    for prefix in ("policy", "value"):
        value = g[f"{prefix}_min_visit"]
        if type(value) is not int or value < 1:
            raise ValueError(f"{prefix}_min_visit must be positive")
        for suffix in ("delta_q", "alpha_visit", "beta_q", "lambda_best"):
            value = g[f"{prefix}_{suffix}"]
            if type(value) not in (int, float) or not 0 <= value < float("inf"):
                raise ValueError(f"Invalid {prefix}_{suffix}")
    positions(config)
    return config


# Inspection and Git commands are authored by the coordinator and executed via
# the generic transport. The generator does not distribute or learn models.
PROBE = r'''
import base64, json, pathlib, subprocess, sys
from remote_generate.remote_generate import input_signature, source_signature
from remote_generate.state_store import ROOT, file_hash, local_path
from search_config import require_search_mode
import numpy, torch
r = json.loads(base64.b64decode(sys.argv[1]))
g = r["generation"]
require_search_mode(g["search_mode"])
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required on every generating PC")
def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()
inputs = {str(c): input_signature(r["log_path"], c, g["chunk_size"], g["shuffle_seed"])
          for c in range(g["chunk_start"], g["chunk_end"] + 1)}
teacher = None
if g["start_shot"] < 15:
    name = "model/{}-end{}-shot{}.bin".format(g["search_mode"], g["start_end"], g["start_shot"] + 1)
    teacher = file_hash(local_path(name))
if r["fresh"]:
    for end, shot in r["sequence"]:
        folder = ROOT / "data" / ("end%d" % end) / ("shot%d" % shot)
        model = ROOT / "model" / ("%s-end%d-shot%d.bin" % (g["search_mode"], end, shot))
        if any(folder.glob("sl_data_*.npz")) or model.exists():
            raise FileExistsError("Output already exists for end%d/shot%d" % (end, shot))
print(json.dumps(dict(source_sha256=source_signature(), inputs=inputs,
                     base_sha256=file_hash(local_path(g["base_model"])), teacher_sha256=teacher,
                     branch=git("branch", "--show-current"), commit=git("rev-parse", "HEAD"),
                     runtime=[list(sys.version_info[:2]), torch.__version__, numpy.__version__])))
'''

SYNC = r'''
import base64, json, subprocess, sys
from remote_generate.remote_status import snapshot
from remote_generate.remote_generate import source_signature
from remote_generate.state_store import ROOT, file_hash, local_path
r = json.loads(base64.b64decode(sys.argv[1]))
if any(row["state"] != "idle" for row in snapshot(r["pc_id"])["workers"]):
    raise RuntimeError("Cannot update models while a worker is occupied")
def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.PIPE).decode().strip()
if git("branch", "--show-current") != r["branch"]:
    raise RuntimeError("Remote branch differs")
git("diff", "--quiet")
git("diff", "--cached", "--quiet")
git("pull", "--ff-only", r["remote"], r["branch"])
if git("rev-parse", "HEAD") != r["commit"]:
    raise RuntimeError("Remote did not reach the published commit")
if source_signature() != r["source_sha256"] or file_hash(local_path(r["path"])) != r["sha256"]:
    raise RuntimeError("Downloaded model or Python sources differ")
print(json.dumps({"ok": True, "commit": r["commit"]}))
'''


class Node:
    def __init__(self, config, spec, secrets, local=False):
        self.spec, self.local = spec, local
        self.pc_id = spec["pc_id"]
        self.python = sys.executable if local else spec["python"]
        self.connection = None if local else SSHConnection(config["gateway"], spec, config["transport"], secrets)
        self.timeout = config["transport"]["command_timeout_seconds"]

    def command(self, argv, *, detached=False, log_path=None):
        if self.connection:
            return self.connection.run(argv, detached=detached, log_path=log_path)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        if detached:
            path = store.local_path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            kwargs = dict(creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else dict(start_new_session=True)
            with path.open("ab", buffering=0) as stream:
                process = subprocess.Popen(argv, cwd=store.ROOT, env=env, stdin=subprocess.DEVNULL,
                                           stdout=stream, stderr=stream, close_fds=True, **kwargs)
            return json.dumps({"pid": process.pid})
        result = subprocess.run(argv, cwd=store.ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=self.timeout)
        if result.returncode:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-12000:])
        return result.stdout.decode("utf-8-sig")

    def module(self, name, *args, **kwargs):
        return self.command([self.python, "-u", "-m", f"remote_generate.{name}", *args], **kwargs)

    def inspect(self, source, request):
        return json.loads(self.command([self.python, "-u", "-c", source, encoded(request)]))

    def status(self):
        progress(f"[STATUS][{self.pc_id}] 状態を取得中")
        state = json.loads(self.module("remote_status", "--pc-id", self.pc_id))
        summaries = []
        for row in state["workers"]:
            summary = f"{row['worker_id']}={row['state']}"
            job = row.get("job")
            if job:
                summary += f"(end{job['end']}/shot{job['shot']}/chunk{job['chunk']})"
            if row["state"] == "running" and row.get("process_alive") is False:
                summary += "[プロセス消失]"
            if row["state"] == "completed":
                summary += "[成功]" if (row.get("result") or {}).get("success") else "[失敗]"
            summaries.append(summary)
        progress(f"[STATUS][{self.pc_id}] " + " | ".join(summaries))
        return state


class Pipeline:
    def __init__(self, config, resume):
        self.config, self.resume = config, resume
        secrets = credentials(config["env_file"]) if config["remotes"] else {}
        self.nodes = [Node(config, config["local"], secrets, True)]
        self.nodes += [Node(config, spec, secrets) for spec in config["remotes"]]
        self.lock = asyncio.Lock()
        self.stopping = False
        self.state = None

    def save(self):
        with store.transaction(self.nodes[0].pc_id) as state:
            state["pipeline"] = self.state

    async def retry(self, function, *args, **kwargs):
        for attempt in range(self.config["control"]["connection_retries"] + 1):
            try:
                return await asyncio.to_thread(function, *args, **kwargs)
            except ConnectionUnavailable:
                if attempt == self.config["control"]["connection_retries"]:
                    raise
                print(f"[SSH] connection unavailable; retry {attempt + 1}", flush=True)
                await asyncio.sleep(self.config["control"]["retry_seconds"])

    async def prepare(self):
        progress("[CHECK] 事前確認を開始します")
        config_hash = store.object_hash({k: v for k, v in self.config.items() if k not in ("control", "transport")})
        local = await self.retry(self.nodes[0].status) if store.STATE.exists() else {}
        previous = local.get("pipeline")
        if self.resume:
            if not previous or previous["config_sha256"] != config_hash:
                raise ValueError("Resume needs matching generation, training and PC settings")
            self.state = previous
            if self.state["phase"] == "done":
                progress("[CHECK] 保存された実行は全局面完了済みです")
                return
        elif previous and previous["phase"] != "done":
            raise RuntimeError("An unfinished pipeline exists; use --resume")
        async def inspect_node(node):
            started = time.monotonic()
            g = self.config["generation"]
            progress(f"[CHECK][{node.pc_id}] 入力ログ・モデル・コード・実行環境を確認中 "
                     f"(ログ={node.spec['log_path']}, チャンク={g['chunk_start']}〜{g['chunk_end']})")
            try:
                probe = await self.retry(node.inspect, PROBE,
                    dict(generation=g, log_path=node.spec["log_path"],
                         fresh=not self.resume, sequence=positions(self.config)))
            except Exception:
                progress(f"[CHECK][{node.pc_id}] 事前確認に失敗 ({time.monotonic() - started:.1f}秒)")
                raise
            progress(f"[CHECK][{node.pc_id}] 情報取得完了 ({time.monotonic() - started:.1f}秒)")
            return probe

        probes = await asyncio.gather(*(inspect_node(node) for node in self.nodes))
        baseline = self.state["baseline"] if self.resume else probes[0]
        for node, probe in zip(self.nodes, probes):
            for key in ("source_sha256", "inputs", "base_sha256", "teacher_sha256", "branch", "runtime"):
                if probe[key] != baseline[key]:
                    raise ValueError(f"{node.pc_id}: PCs or saved run differ: {key}")
            if not self.resume and probe["commit"] != baseline["commit"]:
                raise ValueError(f"{node.pc_id}: Start with the same Git commit on every PC")
            progress(f"[CHECK][{node.pc_id}] 入力ログ・モデル・コード・実行環境の照合OK")
        if not baseline["branch"]:
            raise ValueError("Use a Git branch, not detached HEAD")
        git("check-ref-format", "--branch", baseline["branch"])
        for node in self.nodes:
            progress(f"[CHECK][{node.pc_id}] ワーカー状態を初期化・確認中")
            await self.retry(node.module, "state_store", "init", "--pc-id", node.pc_id,
                             "--workers", str(node.spec["workers"]))
            status = await self.retry(node.status)
            for row in status["workers"]:
                if row["state"] == "idle":
                    continue
                assigned = self.state["jobs"].values() if self.resume else []
                if not any(entry["status"] != "done" and entry["job"] == row.get("job") for entry in assigned):
                    raise RuntimeError(f"{node.pc_id} has an occupied worker outside this run")
        if not self.resume:
            self.state = dict(run_id=uuid.uuid4().hex, config_sha256=config_hash, baseline=baseline,
                              sequence=positions(self.config), index=0, phase="generate", jobs={},
                              git_head=baseline["commit"])
            self.save()
        progress("[CHECK] 全PCの事前確認が完了しました")

    def make_job(self, node, worker_id, chunk):
        end, shot = self.state["sequence"][self.state["index"]]
        baseline = self.state["baseline"]
        return dict(job_id=f"{self.state['run_id']}-e{end}s{shot}c{chunk}", pc_id=node.pc_id,
                    worker_id=worker_id, end=end, shot=shot, chunk=chunk,
                    generation=self.config["generation"], threads=node.spec["threads_per_worker"],
                    log_path=node.spec["log_path"], source_sha256=baseline["source_sha256"],
                    input=baseline["inputs"][str(chunk)], base_sha256=baseline["base_sha256"],
                    teacher=self.state["teacher"])

    async def assign(self, node, worker_id):
        async with self.lock:
            jobs = self.state["jobs"]
            if any(e["job"]["pc_id"] == node.pc_id and e["job"]["worker_id"] == worker_id
                   and e["status"] != "done" for e in jobs.values()):
                return
            if self.stopping:
                return
            g = self.config["generation"]
            for chunk in range(g["chunk_start"], g["chunk_end"] + 1):
                if str(chunk) not in jobs:
                    job = self.make_job(node, worker_id, chunk)
                    jobs[str(chunk)] = dict(job=job, status="assigned", result=None,
                                            launches=0, last_launch=0)
                    self.save()  # Durable ownership precedes any launch command.
                    print(f"[ASSIGN] {node.pc_id}/{worker_id}: chunk {chunk}", flush=True)
                    return

    async def handle(self, node, entry, row):
        job = entry["job"]
        if entry["status"] == "collected":
            await asyncio.to_thread(verify_local, job, entry["result"])
            await self.retry(node.module, "state_store", "ack", "--pc-id", node.pc_id,
                             "--worker-id", job["worker_id"], "--job-id", job["job_id"])
            async with self.lock:
                entry["status"] = "done"
                self.save()
            print(f"[COLLECTED] {node.pc_id}: chunk {job['chunk']}", flush=True)
            return
        if row["state"] == "idle":
            if row.get("last_job_id") == job["job_id"]:
                raise RuntimeError("Remote acknowledgement exists without a local collection receipt")
            if (row.get("last_reset_job_id") == job["job_id"]
                    and entry.get("reset_seen") != row["reset_token"]):
                async with self.lock:
                    entry.update(launches=0, last_launch=0, reset_seen=row["reset_token"])
                    self.save()
            if time.time() - entry["last_launch"] < self.config["control"]["start_grace_seconds"]:
                return
            if entry["launches"] >= self.config["control"]["start_attempts"]:
                raise RuntimeError(f"Generator did not record its start; inspect log/remote-{job['worker_id']}.log on {node.pc_id}")
            async with self.lock:
                entry["launches"] += 1
                entry["last_launch"] = time.time()
                self.save()
            await self.retry(node.module, "remote_generate", "--job-b64", encoded(job),
                             detached=True, log_path=f"log/remote-{job['worker_id']}.log")
            return
        if row.get("job") != job:
            raise RuntimeError(f"{node.pc_id}: worker belongs to another assignment")
        if row["state"] == "running":
            if not row["process_alive"]:
                raise RuntimeError(f"{job['job_id']}: generator process disappeared; inspect its log")
            return
        result = row["result"]
        if not result["success"]:
            raise RuntimeError(f"{job['job_id']}: {result.get('error', 'generation failed')}")
        manifest(job, result)
        if node.local:
            await asyncio.to_thread(verify_local, job, result)
        else:
            await self.retry(collect, node.connection, job, result)
        async with self.lock:
            entry.update(status="collected", result=result)
            self.save()  # Persist verified collection before asking the remote to clear its job.

    async def node_loop(self, node):
        pending = {}
        try:
            while not self.stopping:
                for key, task in list(pending.items()):
                    if task.done():
                        del pending[key]
                        task.result()
                for index in range(node.spec["workers"]):
                    await self.assign(node, f"worker-{index}")
                active = [entry for entry in self.state["jobs"].values()
                          if entry["job"]["pc_id"] == node.pc_id and entry["status"] != "done"]
                if not active:
                    return
                status = await self.retry(node.status)  # One read per PC, shared by all its workers.
                rows = {row["worker_id"]: row for row in status["workers"]}
                for entry in active:
                    key = entry["job"]["worker_id"]
                    if key not in pending:
                        pending[key] = asyncio.create_task(self.handle(node, entry, rows[key]))
                if not self.stopping:
                    await asyncio.sleep(self.config["control"]["poll_seconds"])
        except BaseException:
            self.stopping = True
            raise
        finally:
            # In-flight transfers retain their reservations and may finish safely.
            results = await asyncio.gather(*pending.values(), return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException) and not self.stopping:
                    self.stopping = True
                    raise result

    async def generate_position(self):
        end, shot = self.state["sequence"][self.state["index"]]
        teacher = None
        if shot < 15:
            path = f"model/{self.config['generation']['search_mode']}-end{end}-shot{shot + 1}.bin"
            teacher = dict(path=path, sha256=await asyncio.to_thread(store.file_hash, store.local_path(path)))
        if "teacher" in self.state and self.state["teacher"] != teacher:
            raise ValueError("Teacher changed since this generation started")
        self.state["teacher"] = teacher
        self.save()
        await asyncio.gather(*(self.node_loop(node) for node in self.nodes))
        g = self.config["generation"]
        if len(self.state["jobs"]) != g["chunk_end"] - g["chunk_start"] + 1 or any(
                entry["status"] != "done" for entry in self.state["jobs"].values()):
            raise RuntimeError("Generation or collection remains unfinished")
        self.state["phase"] = "train"
        self.save()

    def verify_dataset(self):
        expected = set()
        for entry in self.state["jobs"].values():
            if entry["status"] != "done":
                raise RuntimeError("Cannot train before collection completes")
            verify_local(entry["job"], entry["result"])
            expected.update(item["path"] for item in entry["result"]["files"])
        end, shot = self.state["sequence"][self.state["index"]]
        folder = store.ROOT / "data" / f"end{end}" / f"shot{shot}"
        actual = {path.relative_to(store.ROOT).as_posix() for path in folder.glob("sl_data_*.npz")}
        if not expected or expected != actual:
            raise ValueError("Training files are empty, missing, or include files outside this run")

    async def train_position(self):
        from remote_generate.remote_generate import source_signature
        from transformer.shot_pipeline import run_stage
        await asyncio.to_thread(self.verify_dataset)
        if await asyncio.to_thread(source_signature) != self.state["baseline"]["source_sha256"]:
            raise ValueError("Python sources changed during the run")
        if await asyncio.to_thread(git, "rev-parse", "HEAD") != self.state["git_head"]:
            raise ValueError("Git HEAD changed outside this pipeline")
        end, shot = self.state["sequence"][self.state["index"]]
        options = dict(program_dir=str(store.ROOT), search_mode=self.config["generation"]["search_mode"],
                       **self.config["training"])
        await asyncio.to_thread(run_stage, options, end, shot, "train")
        path = f"model/{options['search_mode']}-end{end}-shot{shot}.bin"
        self.state["publication"] = dict(path=path, sha256=await asyncio.to_thread(store.file_hash, store.local_path(path)))
        self.state["phase"] = "publish"
        self.save()

    def publish_git(self):
        from remote_generate.remote_generate import source_signature
        publication, baseline = self.state["publication"], self.state["baseline"]
        path = publication["path"]
        if git("branch", "--show-current") != baseline["branch"] or source_signature() != baseline["source_sha256"]:
            raise ValueError("Branch or sources changed during the run")
        if store.file_hash(store.local_path(path)) != publication["sha256"]:
            raise ValueError("Trained model changed before publication")
        blob = git("hash-object", "--", path)
        head = git("rev-parse", "HEAD")
        if head != self.state["git_head"]:
            # Recover a crash after our model-only commit but before recording it.
            if (git("rev-parse", "HEAD^") != self.state["git_head"]
                    or git("diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD") != path
                    or git("rev-parse", f"HEAD:{path}") != blob):
                raise ValueError("Unexpected commits appeared during this pipeline")
        try:
            previous = git("rev-parse", f"HEAD:{path}")
        except subprocess.CalledProcessError:
            previous = None
        if previous != blob:
            end, shot = self.state["sequence"][self.state["index"]]
            git("add", "--", path)
            git("commit", "--only", "-m", f"モデルend{end}, shot{shot} 追加", "--", path)
        commit = git("rev-parse", "HEAD")
        git("push", self.config["git"]["remote"], f"HEAD:refs/heads/{baseline['branch']}")
        return commit

    async def publish_position(self):
        commit = await asyncio.to_thread(self.publish_git)
        request = dict(self.state["publication"], source_sha256=self.state["baseline"]["source_sha256"],
                       branch=self.state["baseline"]["branch"], commit=commit, remote=self.config["git"]["remote"])
        await asyncio.gather(*(self.retry(node.inspect, SYNC, dict(request, pc_id=node.pc_id)) for node in self.nodes[1:]))
        self.state["index"] += 1
        self.state["git_head"] = commit
        self.state["jobs"] = {}
        self.state.pop("publication", None)
        self.state.pop("teacher", None)
        self.state["phase"] = "done" if self.state["index"] == len(self.state["sequence"]) else "generate"
        self.save()

    async def run(self):
        await self.prepare()
        while self.state["phase"] != "done":
            print(f"[PIPELINE] {self.state['sequence'][self.state['index']]}: {self.state['phase']}", flush=True)
            await {"generate": self.generate_position, "train": self.train_position,
                   "publish": self.publish_position}[self.state["phase"]]()
        print("[PIPELINE] All requested positions completed.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    with store.mutex("coordinator"):
        asyncio.run(Pipeline(config, args.resume).run())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Coordinator stopped. Detached generators may still be running; use --resume.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
