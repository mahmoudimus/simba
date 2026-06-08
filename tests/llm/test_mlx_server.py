"""Persistent mlx_lm.server launcher (pure parts + fail-open)."""

from __future__ import annotations

import simba.llm.mlx_server as ms


def test_build_serve_cmd():
    cmd = ms.build_serve_cmd("mlx-community/foo-4bit", 8082, "127.0.0.1")
    assert cmd[0] == "mlx_lm.server"
    assert "--model" in cmd and "mlx-community/foo-4bit" in cmd
    assert "8082" in cmd and "127.0.0.1" in cmd


def test_is_up_true_false(monkeypatch):
    import types

    monkeypatch.setattr(
        ms.httpx, "get", lambda *a, **k: types.SimpleNamespace(status_code=200)
    )
    assert ms.is_up("http://x:1") is True

    def _boom(*a, **k):
        raise ConnectionError("down")

    monkeypatch.setattr(ms.httpx, "get", _boom)
    assert ms.is_up("http://x:1") is False


def test_ensure_server_returns_existing(monkeypatch):
    monkeypatch.setattr(ms, "is_up", lambda url, **k: True)
    called = {"popen": False}
    monkeypatch.setattr(
        ms.subprocess, "Popen", lambda *a, **k: called.__setitem__("popen", True)
    )
    url = ms.ensure_server("m", port=8082)
    assert url == "http://127.0.0.1:8082"
    assert called["popen"] is False  # already up -> no launch


def test_ensure_server_launch_failure_failopen(monkeypatch):
    monkeypatch.setattr(ms, "is_up", lambda url, **k: False)

    def _no_binary(*a, **k):
        raise FileNotFoundError("mlx_lm.server")

    monkeypatch.setattr(ms.subprocess, "Popen", _no_binary)
    assert ms.ensure_server("m", port=8082) is None
