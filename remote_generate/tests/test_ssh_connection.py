"""Offline fault injection: no SSH server, subprocess, or credential file access."""
from contextlib import ExitStack, redirect_stderr
import io
import json
import unittest
from unittest.mock import MagicMock, patch

import paramiko

from remote_generate import ssh_connection as ssh


class Key:
    def get_base64(self):
        return "TEST-PUBLIC-KEY"


class ConnectionDiagnosticsTest(unittest.TestCase):
    def exercise(self, failure=None, operation="遠隔PCとの通信"):
        output = io.StringIO()
        gateway, remote = MagicMock(), MagicMock()
        gateway.__enter__.return_value = gateway
        remote.__enter__.return_value = remote
        remote.get_host_keys.return_value.lookup.return_value = {"ssh-ed25519": Key()}
        stream = gateway.open_sftp.return_value.__enter__.return_value.open.return_value.__enter__.return_value
        stream.read.return_value = json.dumps(dict(schema_version=1, pc_id="remote-01",
                                                   vpn_ip="172.21.110.52", ssh_port=22,
                                                   updated_at="2026-09-15T00:00:00Z")).encode()
        channel = gateway.get_transport.return_value.open_channel
        cause = None
        if failure == "tcp":
            cause = TimeoutError()
            channel.side_effect = cause
        elif failure == "registry":
            stream.read.return_value = b"{"
        elif failure == "gateway":
            cause = paramiko.AuthenticationException("Gateway authentication failed")
            gateway.connect.side_effect = cause

        connection = ssh.SSHConnection(
            dict(host="gateway.invalid", username="test", known_hosts="unused", key_file="unused"),
            dict(pc_id="remote-01", username="test", known_hosts="unused", password_env="TEST_PASSWORD",
                 registry_path="/registry.json"),
            dict(connect_timeout_seconds=15, io_timeout_seconds=30),
            {"TEST_PASSWORD": "secret-must-not-be-logged"})

        def remote_connect(**kwargs):
            transport = kwargs["transport_factory"](kwargs["sock"])
            transport.is_authenticated = lambda: True
            transport.start_client(timeout=kwargs["timeout"])
            transport.get_remote_server_key()
            if failure == "host_key":
                raise paramiko.BadHostKeyException("remote-01", Key(), Key())
            transport.auth_password(kwargs["username"], kwargs["password"])

        remote.connect.side_effect = remote_connect
        with ExitStack() as stack:
            stack.enter_context(redirect_stderr(output))
            stack.enter_context(patch.object(ssh.paramiko, "SSHClient", side_effect=[gateway, remote]))
            stack.enter_context(patch.object(ssh, "config_path", return_value=MagicMock()))
            stack.enter_context(patch.object(paramiko.Transport, "__init__", return_value=None))
            start = stack.enter_context(patch.object(paramiko.Transport, "start_client"))
            get_key = stack.enter_context(patch.object(paramiko.Transport, "get_remote_server_key", return_value=Key()))
            auth = stack.enter_context(patch.object(paramiko.Transport, "auth_password", return_value=[]))
            if failure == "handshake":
                cause = paramiko.SSHException("Error reading SSH protocol banner")
                start.side_effect = cause
            if failure == "negotiation_timeout":
                cause = paramiko.SSHException("No existing session")
                get_key.side_effect = cause
            if failure == "auth":
                cause = paramiko.AuthenticationException("Authentication failed")
                auth.side_effect = cause
            error = None
            try:
                with connection.connect(operation=operation) as client:
                    self.assertIs(client, remote)
                    if failure == "transfer":
                        raise TimeoutError("SFTP read timed out")
            except Exception as exc:
                error = exc
            auth_called = auth.called

        text = output.getvalue()
        self.assertNotIn("secret-must-not-be-logged", text)
        self.assertTrue(gateway.__exit__.called)
        self.assertTrue(remote.__exit__.called)
        if failure not in ("tcp", "registry", "gateway"):
            self.assertTrue(channel.return_value.__exit__.called)
        return error, text, cause, auth_called

    def test_tcp_timeout_preserves_stage_endpoint_and_exception(self):
        error, text, cause, authenticated = self.exercise("tcp")
        self.assertIsInstance(error, ssh.ConnectionUnavailable)
        self.assertIs(error.__cause__, cause)
        self.assertIn("段階=中継サーバーから遠隔PCへのTCP接続", str(error))
        self.assertIn("172.21.110.52:22", str(error))
        self.assertIn("TimeoutError()", str(error))
        self.assertNotIn("[完了] 段階=中継サーバーから遠隔PCへのTCP接続", text)
        self.assertFalse(authenticated)

    def test_ssh_negotiation_failures_are_not_labeled_authentication(self):
        for failure in ("handshake", "negotiation_timeout"):
            with self.subTest(failure=failure):
                error, text, cause, authenticated = self.exercise(failure)
                self.assertIsInstance(error, ssh.ConnectionUnavailable)
                self.assertIn("段階=遠隔PCとのSSH接続確立", str(error))
                self.assertIs(error.__cause__, cause)
                self.assertNotIn("[完了] 段階=遠隔PCとのSSH接続確立", text)
                self.assertFalse(authenticated)

    def test_host_key_failure_remains_non_retryable(self):
        error, text, _, authenticated = self.exercise("host_key")
        self.assertIsInstance(error, paramiko.BadHostKeyException)
        self.assertIn("[失敗] 段階=遠隔PCのホスト鍵確認", text)
        self.assertNotIn("[完了] 段階=遠隔PCのホスト鍵確認", text)
        self.assertFalse(authenticated)

    def test_password_failure_remains_non_retryable(self):
        error, text, _, _ = self.exercise("auth")
        self.assertIsInstance(error, paramiko.AuthenticationException)
        self.assertIn("[完了] 段階=遠隔PCのホスト鍵確認", text)
        self.assertIn("[失敗] 段階=遠隔PCのユーザー認証", text)
        self.assertNotIn("[完了] 段階=遠隔PCのユーザー認証", text)

    def test_registry_parse_failure_has_its_own_stage(self):
        error, text, _, _ = self.exercise("registry")
        self.assertIsInstance(error, json.JSONDecodeError)
        self.assertIn("[失敗] 段階=VPN IP登録JSONの解析・検証", text)

    def test_gateway_failure_has_its_own_stage(self):
        error, text, _, _ = self.exercise("gateway")
        self.assertIsInstance(error, paramiko.AuthenticationException)
        self.assertIn("[失敗] 段階=中継サーバーへのSSH接続・認証", text)

    def test_transfer_error_does_not_blame_authentication(self):
        error, text, _, _ = self.exercise("transfer", operation="遠隔PCとのSFTP通信")
        self.assertIsInstance(error, ssh.ConnectionUnavailable)
        self.assertIn("段階=遠隔PCとのSFTP通信", str(error))
        self.assertIn("[完了] 段階=遠隔PCのユーザー認証", text)

    def test_success_completes_each_stage(self):
        error, text, _, authenticated = self.exercise()
        self.assertIsNone(error)
        self.assertTrue(authenticated)
        self.assertNotIn("[失敗]", text)
        for stage in ("中継サーバーから遠隔PCへのTCP接続", "遠隔PCとのSSH接続確立", "遠隔PCのホスト鍵確認", "遠隔PCのユーザー認証"):
            self.assertIn("[完了] 段階=" + stage, text)


if __name__ == "__main__":
    unittest.main()
