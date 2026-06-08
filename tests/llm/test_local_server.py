"""Persistent local OpenAI-compatible server launcher (engine-agnostic)."""

from __future__ import annotations

import types

import simba.llm.local_server as ls


def _cfg(**kw):
    base = {
        "provider": "mlx-server",
        "base_url": "http://127.0.0.1:8082",
        "model": "mlx-community/foo-4bit",
        "model_path": "",
        "serve_cmd": "",
    }
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_build_serve_cmd_mlx_preset():
    cmd = ls.build_serve_cmd(
        ls.SERVE_PRESETS["mlx-server"], "mlx-community/foo-4bit", "127.0.0.1", 8082
    )
    assert cmd[0] == "mlx_lm.server"
    assert "--model" in cmd and "mlx-community/foo-4bit" in cmd
    assert "8082" in cmd and "127.0.0.1" in cmd


def test_build_serve_cmd_llama_preset():
    cmd = ls.build_serve_cmd(
        ls.SERVE_PRESETS["llama-server"], "/models/q.gguf", "0.0.0.0", 8080
    )
    assert cmd[0] == "llama-server"
    assert "/models/q.gguf" in cmd and "8080" in cmd and "0.0.0.0" in cmd


def test_is_up_true_false(monkeypatch):
    monkeypatch.setattr(
        ls.httpx, "get", lambda *a, **k: types.SimpleNamespace(status_code=200)
    )
    assert ls.is_up("http://x:1") is True

    def _boom(*a, **k):
        raise ConnectionError("down")

    monkeypatch.setattr(ls.httpx, "get", _boom)
    assert ls.is_up("http://x:1") is False


def test_ensure_server_returns_existing(monkeypatch):
    monkeypatch.setattr(ls, "is_up", lambda url, **k: True)
    called = {"popen": False}
    monkeypatch.setattr(
        ls.subprocess, "Popen", lambda *a, **k: called.__setitem__("popen", True)
    )
    url = ls.ensure_server(["llama-server"], base_url="http://127.0.0.1:8082")
    assert url == "http://127.0.0.1:8082"
    assert called["popen"] is False  # already up -> no launch


def test_ensure_server_launch_failure_failopen(monkeypatch):
    monkeypatch.setattr(ls, "is_up", lambda url, **k: False)

    def _no_binary(*a, **k):
        raise FileNotFoundError("llama-server")

    monkeypatch.setattr(ls.subprocess, "Popen", _no_binary)
    assert ls.ensure_server(["llama-server"], base_url="http://127.0.0.1:8082") is None


def test_ensure_for_config_mlx_preset_spawns_mlx(monkeypatch):
    monkeypatch.setattr(ls, "is_up", lambda url, **k: False)
    spawned = {}
    monkeypatch.setattr(
        ls, "ensure_server", lambda cmd, **k: spawned.update(cmd=cmd) or "ok"
    )
    out = ls.ensure_for_config(_cfg(provider="mlx-server"))
    assert out == "ok" and spawned["cmd"][0] == "mlx_lm.server"


def test_ensure_for_config_llama_preset_spawns_llama(monkeypatch):
    monkeypatch.setattr(ls, "is_up", lambda url, **k: False)
    spawned = {}
    monkeypatch.setattr(
        ls, "ensure_server", lambda cmd, **k: spawned.update(cmd=cmd) or "ok"
    )
    cfg = _cfg(
        provider="llama-server",
        base_url="http://127.0.0.1:8080",
        model="",
        model_path="/models/q.gguf",
    )
    out = ls.ensure_for_config(cfg)
    assert out == "ok"
    assert spawned["cmd"][0] == "llama-server" and "/models/q.gguf" in spawned["cmd"]


def test_ensure_for_config_serve_cmd_override(monkeypatch):
    monkeypatch.setattr(ls, "is_up", lambda url, **k: False)
    spawned = {}
    monkeypatch.setattr(
        ls, "ensure_server", lambda cmd, **k: spawned.update(cmd=cmd) or "ok"
    )
    cfg = _cfg(provider="llama-server", serve_cmd="vllm serve {model} --port {port}")
    ls.ensure_for_config(cfg)
    assert spawned["cmd"][0] == "vllm" and "serve" in spawned["cmd"]


def test_ensure_for_config_openai_http_is_noop(monkeypatch):
    # Generic remote endpoint: never spawn anything.
    called = {"popen": False}
    monkeypatch.setattr(
        ls.subprocess, "Popen", lambda *a, **k: called.__setitem__("popen", True)
    )
    assert ls.ensure_for_config(_cfg(provider="openai-http")) is None
    assert called["popen"] is False


def test_ensure_for_config_remote_down_does_not_spawn(monkeypatch):
    # A remote (non-local) host we don't manage: if it's down, don't try to
    # Popen it locally — that would bind the wrong machine.
    monkeypatch.setattr(ls, "is_up", lambda url, **k: False)
    called = {"popen": False}
    monkeypatch.setattr(
        ls.subprocess, "Popen", lambda *a, **k: called.__setitem__("popen", True)
    )
    cfg = _cfg(provider="llama-server", base_url="http://192.168.1.50:8080")
    assert ls.ensure_for_config(cfg) is None
    assert called["popen"] is False
