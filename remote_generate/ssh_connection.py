"""SSH transport only: registry lookup, command execution, and SFTP channels."""
from __future__ import annotations

import base64
from contextlib import ExitStack, contextmanager
from datetime import datetime
import ipaddress
import json
import os
import sys
import time

import paramiko
from dotenv import dotenv_values

from remote_generate.config_loader import config_path
from remote_generate.state_store import identifier


class ConnectionUnavailable(RuntimeError):
    """Retryable transport failure; it says nothing about job completion."""


class RemoteCommandError(RuntimeError):
    pass


class ConnectionTrace:
    """Per-connection stage tracking; concurrent connections never share stages."""
    def __init__(self, report):
        self.report = report
        self.stage = "接続準備"
        self.target = ""
        self.started = time.monotonic()

    def start(self, stage, target):
        self.stage, self.target = stage, target
        self.started = time.monotonic()
        self.report(f"[開始] 段階={stage} 接続先={target}")

    def complete(self):
        self.report(f"[完了] 段階={self.stage} 接続先={self.target} "
                    f"所要時間={time.monotonic() - self.started:.1f}秒")

    def fail(self, exc):
        detail = (str(exc) or repr(exc)).replace("\r", "\\r").replace("\n", "\\n")
        message = (f"段階={self.stage} 接続先={self.target} "
                   f"例外={type(exc).__name__} 詳細={detail} "
                   f"経過={time.monotonic() - self.started:.1f}秒")
        self.report(f"[失敗] {message}")
        return message


class DiagnosticTransport(paramiko.Transport):
    """Observe SSHClient's public transport hooks without replacing its key check."""
    HOST_KEY_STAGE = "遠隔PCのホスト鍵確認"

    def __init__(self, sock, *, trace, **kwargs):
        self.trace = trace
        super().__init__(sock, **kwargs)

    def start_client(self, event=None, timeout=None):
        result = super().start_client(event=event, timeout=timeout)
        # SSHClient calls synchronously. start_client can return on timeout;
        # ensure negotiation actually finished before logging success.
        if event is None:
            super().get_remote_server_key()
            self.trace.complete()
        return result

    def get_remote_server_key(self):
        self.trace.start(self.HOST_KEY_STAGE, self.trace.target)
        return super().get_remote_server_key()

    def auth_password(self, username, password, event=None, fallback=True):
        # SSHClient checks the returned host key before calling authentication.
        # In particular, a BadHostKeyException never reaches this point.
        if self.trace.stage == self.HOST_KEY_STAGE:
            self.trace.complete()
        self.trace.start("遠隔PCのユーザー認証", self.trace.target)
        result = super().auth_password(username, password, event=event, fallback=fallback)
        if event is None:
            if not self.is_authenticated():
                raise paramiko.AuthenticationException("Additional authentication is required")
            self.trace.complete()
        return result


def connection_failure(trace, exc):
    message = trace.fail(exc)
    if isinstance(exc, (paramiko.AuthenticationException, paramiko.BadHostKeyException,
                        FileNotFoundError, PermissionError, FileExistsError)):
        raise exc
    if isinstance(exc, (paramiko.SSHException, EOFError, OSError, ConnectionUnavailable)):
        raise ConnectionUnavailable(message) from exc
    raise exc


def credentials(env_file):
    values = dotenv_values(config_path(env_file), encoding="utf-8-sig", interpolate=False)
    return {**values, **os.environ}


def parse_registry(content, pc_id):
    if len(content) > 65536:
        raise ValueError("Registry exceeds 64 KiB")
    row = json.loads(content.decode("utf-8-sig"))
    if type(row.get("schema_version")) is not int or row["schema_version"] != 1 or row.get("pc_id") != pc_id:
        raise ValueError("Wrong registry schema or PC identity")
    ip = ipaddress.IPv4Address(row["vpn_ip"])
    if ip.is_unspecified or ip.is_loopback or ip.is_link_local or ip.is_multicast:
        raise ValueError("Unusable registry IPv4")
    port = row["ssh_port"]
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Invalid SSH port")
    if datetime.fromisoformat(row["updated_at"].replace("Z", "+00:00")).tzinfo is None:
        raise ValueError("Registry timestamp requires a timezone")
    return str(ip), port


# A generic subprocess launcher, not a generator or a state manager. The large
# request travels on stdin, avoiding the Windows SSH command-line length limit.
BOOTSTRAP = r'''
import json, os, pathlib, subprocess, sys
request = json.load(sys.stdin)
root = pathlib.Path(request["root"]).resolve()
argv = request["argv"]
if request.get("detached"):
    path = (root / request["log_path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Log path escapes repository")
    path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    with path.open("ab", buffering=0) as log:
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_BREAKAWAY_FROM_JOB
        child = subprocess.Popen(argv, cwd=root, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=log, env=env, creationflags=flags,
                                 close_fds=True)
    print(json.dumps({"pid": child.pid}))
else:
    result = subprocess.run(argv, cwd=root, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=request["timeout"], env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    sys.stdout.buffer.write(result.stdout)
    sys.stderr.buffer.write(result.stderr)
    sys.exit(result.returncode)
'''


def ps_string(value):
    return "'" + value.replace("'", "''") + "'"


class SSHConnection:
    def __init__(self, gateway, node, transport, secrets):
        self.gateway = gateway
        self.node = node
        self.transport = transport
        self.secrets = secrets
        identifier(node["pc_id"])

    def report(self, message):
        # This runs on the coordinator, outside the remote command's JSON stream.
        print(f"[{time.strftime('%H:%M:%S')}] [SSH][{self.node['pc_id']}] {message}",
              file=sys.stderr, flush=True)

    @contextmanager
    def connect(self, operation="遠隔PCとの通信"):
        seconds = self.transport["connect_timeout_seconds"]
        trace = ConnectionTrace(self.report)
        gateway_target = f"{self.gateway['host']}:22"
        with ExitStack() as stack:
            try:
                trace.start("接続設定・認証情報の読み込み", gateway_target)
                gateway = stack.enter_context(paramiko.SSHClient())
                remote = stack.enter_context(paramiko.SSHClient())
                gateway.load_host_keys(str(config_path(self.gateway["known_hosts"])))
                gateway.set_missing_host_key_policy(paramiko.RejectPolicy())
                remote.load_host_keys(str(config_path(self.node["known_hosts"])))
                remote.set_missing_host_key_policy(paramiko.RejectPolicy())
                if not remote.get_host_keys().lookup(self.node["pc_id"]):
                    raise ValueError("Missing previously verified host key for PC identity")
                password = self.secrets.get(self.node["password_env"])
                if not password:
                    raise ValueError(f"Missing environment variable: {self.node['password_env']}")
                key = config_path(self.gateway["key_file"])
                if not key.is_file():
                    raise FileNotFoundError(key)
                trace.complete()
                trace.start("中継サーバーへのSSH接続・認証", gateway_target)
                gateway.connect(hostname=self.gateway["host"], username=self.gateway["username"],
                                key_filename=str(key), allow_agent=False, look_for_keys=False,
                                timeout=seconds, banner_timeout=seconds, auth_timeout=seconds)
                trace.complete()
                registry_target = f"{gateway_target} {self.node['registry_path']}"
                trace.start("VPN IP登録JSONの取得", registry_target)
                with gateway.open_sftp() as sftp:
                    sftp.get_channel().settimeout(self.transport["io_timeout_seconds"])
                    with sftp.open(self.node["registry_path"], "rb") as stream:
                        content = stream.read(65537)
                trace.complete()
                trace.start("VPN IP登録JSONの解析・検証", registry_target)
                ip, port = parse_registry(content, self.node["pc_id"])
                trace.complete()
                endpoint = f"{ip}:{port}"
                trace.start("中継サーバーから遠隔PCへのTCP接続", endpoint)
                transport = gateway.get_transport()
                if transport is None or not transport.is_active():
                    raise ConnectionError("Gateway transport is inactive")
                tunnel = stack.enter_context(transport.open_channel(
                    "direct-tcpip", (ip, port), ("127.0.0.1", 0), timeout=seconds))
                trace.complete()
                trace.start("遠隔PCとのSSH接続確立（バナー受信・鍵交換）", endpoint)
                # SSHClient retains its original RejectPolicy and known_hosts
                # identity. Transport hooks only identify the current stage.
                remote.connect(hostname=self.node["pc_id"], sock=tunnel,
                               username=self.node["username"], password=password,
                               allow_agent=False, look_for_keys=False, timeout=seconds,
                               banner_timeout=seconds, auth_timeout=seconds,
                               transport_factory=lambda sock, **kwargs: DiagnosticTransport(
                                   sock, trace=trace, **kwargs))
            except Exception as exc:
                connection_failure(trace, exc)
            # Exceptions raised by the caller after yield are operation failures,
            # not failures in the already completed connection/authentication.
            trace.start(operation, endpoint)
            try:
                yield remote
            except Exception as exc:
                connection_failure(trace, exc)
            else:
                trace.complete()

    @contextmanager
    def sftp(self):
        with self.connect(operation="遠隔PCとのSFTP通信") as pc, pc.open_sftp() as sftp:
            sftp.get_channel().settimeout(self.transport["io_timeout_seconds"])
            yield sftp

    def run(self, argv, *, detached=False, log_path=None):
        timeout = self.transport["command_timeout_seconds"]
        request = dict(root=self.node["root"], argv=argv, detached=detached,
                       log_path=log_path, timeout=timeout)
        bootstrap = "import base64;exec(base64.b64decode('" + base64.b64encode(BOOTSTRAP.encode()).decode() + "'))"
        script = ("$ErrorActionPreference='Stop'; "
                  "$OutputEncoding=[Console]::InputEncoding=[Console]::OutputEncoding="
                  "[System.Text.UTF8Encoding]::new($false); "
                  "$payload=[Console]::In.ReadToEnd(); "
                  f"$payload | & {ps_string(self.node['python'])} -u -c {ps_string(bootstrap)}; "
                  "exit $LASTEXITCODE")
        command = "powershell.exe -NoLogo -NoProfile -NonInteractive -EncodedCommand " + base64.b64encode(script.encode("utf-16-le")).decode()
        with self.connect(operation="遠隔コマンドの実行・応答受信") as pc:
            stdin, stdout, stderr = pc.exec_command(command, timeout=timeout)
            channel = stdout.channel
            try:
                stdin.write(json.dumps(request, ensure_ascii=True))
                stdin.flush()
                channel.shutdown_write()
                output, errors = bytearray(), bytearray()
                deadline = time.monotonic() + timeout
                while True:
                    for ready, read, buffer in ((channel.recv_ready, channel.recv, output),
                                                (channel.recv_stderr_ready, channel.recv_stderr, errors)):
                        while ready():
                            buffer.extend(read(65536))
                            if len(buffer) > 16 * 1024 * 1024:
                                raise RemoteCommandError("Remote response exceeds 16 MiB")
                    if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                        break
                    if time.monotonic() >= deadline:
                        raise ConnectionUnavailable("SSH command response timed out; outcome is unknown")
                    time.sleep(0.02)
                status = channel.recv_exit_status()
                if status == -1:
                    raise ConnectionUnavailable("SSH closed without command exit status")
                if status:
                    raise RemoteCommandError(errors.decode("utf-8", errors="replace")[-12000:])
                return output.decode("utf-8-sig")
            finally:
                channel.close()
