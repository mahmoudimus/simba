"""Tests for client-identity resolution on outbound daemon requests."""

from __future__ import annotations

import pytest

import simba.harness.client as client


@pytest.fixture(autouse=True)
def _clear_client_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start each test from a clean slate (no ambient runtime markers)."""
    for var in (
        "SIMBA_CLIENT",
        "CLAUDECODE",
        "CLAUDE_CODE_ENTRYPOINT",
        "CODEX_SANDBOX",
    ):
        monkeypatch.delenv(var, raising=False)


class TestDetectClient:
    def test_header_constant(self) -> None:
        assert client.CLIENT_HEADER == "X-Simba-Client"

    def test_explicit_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIMBA_CLIENT", "from-env")
        monkeypatch.setenv("CLAUDECODE", "1")
        assert client.detect_client("explicit-name") == "explicit-name"

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIMBA_CLIENT", "my-runtime")
        assert client.detect_client() == "my-runtime"

    def test_claude_code_marker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAUDECODE", "1")
        assert client.detect_client() == client.CLAUDE_CODE

    def test_claude_code_entrypoint_marker(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
        assert client.detect_client() == client.CLAUDE_CODE

    def test_codex_sandbox_marker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CODEX_SANDBOX", "seatbelt")
        assert client.detect_client() == client.CODEX

    def test_default_when_nothing_set(self) -> None:
        assert client.detect_client() == client.CLI

    def test_custom_default(self) -> None:
        assert client.detect_client(default=client.CLAUDE_CODE) == client.CLAUDE_CODE

    def test_explicit_beats_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIMBA_CLIENT", "env-name")
        assert client.detect_client("flag-name") == "flag-name"


class TestDetectClientSource:
    """``detect_client_source`` also reports whether resolution was defaulted."""

    def test_explicit_is_not_defaulted(self) -> None:
        name, defaulted = client.detect_client_source("explicit-name")
        assert name == "explicit-name"
        assert defaulted is False

    def test_env_is_not_defaulted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIMBA_CLIENT", "my-runtime")
        name, defaulted = client.detect_client_source()
        assert name == "my-runtime"
        assert defaulted is False

    def test_claude_code_marker_is_not_defaulted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CLAUDECODE", "1")
        name, defaulted = client.detect_client_source()
        assert name == client.CLAUDE_CODE
        assert defaulted is False

    def test_codex_sandbox_marker_is_not_defaulted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CODEX_SANDBOX", "seatbelt")
        name, defaulted = client.detect_client_source()
        assert name == client.CODEX
        assert defaulted is False

    def test_fallthrough_is_defaulted(self) -> None:
        name, defaulted = client.detect_client_source(default=client.CLAUDE_CODE)
        assert name == client.CLAUDE_CODE
        assert defaulted is True

    def test_detect_client_unaffected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # detect_client keeps its old (non-tuple) signature/behavior.
        monkeypatch.setenv("CODEX_SANDBOX", "seatbelt")
        assert client.detect_client() == client.CODEX


class TestClientHeaders:
    def test_returns_header_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAUDECODE", "1")
        assert client.client_headers() == {client.CLIENT_HEADER: client.CLAUDE_CODE}

    def test_explicit_in_headers(self) -> None:
        assert client.client_headers("pi") == {client.CLIENT_HEADER: "pi"}


class TestOriginNesting:
    """Daemon loopback recalls nest as ``<origin>.daemon`` (origin from contextvar)."""

    def test_origin_roundtrip_and_reset(self) -> None:
        assert client.get_origin_client() is None
        token = client.set_origin_client("claude-code")
        assert client.get_origin_client() == "claude-code"
        client.reset_origin_client(token)
        assert client.get_origin_client() is None

    def test_daemon_with_origin_nests(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIMBA_CLIENT", client.DAEMON)
        token = client.set_origin_client("claude-code")
        try:
            assert client.detect_client() == "claude-code.daemon"
        finally:
            client.reset_origin_client(token)

    def test_daemon_without_origin_stays_flat(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SIMBA_CLIENT", client.DAEMON)
        assert client.detect_client() == client.DAEMON

    def test_collapses_already_nested_origin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SIMBA_CLIENT", client.DAEMON)
        token = client.set_origin_client("pi.daemon")
        try:
            assert client.detect_client() == "pi.daemon"
        finally:
            client.reset_origin_client(token)

    def test_non_daemon_base_ignores_origin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Only the daemon hop composes; a non-daemon process never nests.
        monkeypatch.setenv("SIMBA_CLIENT", client.CLAUDE_CODE)
        token = client.set_origin_client("pi")
        try:
            assert client.detect_client() == client.CLAUDE_CODE
        finally:
            client.reset_origin_client(token)

    def test_headers_carry_nested_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIMBA_CLIENT", client.DAEMON)
        token = client.set_origin_client("codex")
        try:
            assert client.client_headers() == {client.CLIENT_HEADER: "codex.daemon"}
        finally:
            client.reset_origin_client(token)
