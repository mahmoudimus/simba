"""Tests for the RLM autonomous engine (claude-cli)."""

from __future__ import annotations

import simba.rlm.engine as engine


class _Cfg:
    def __init__(self, **kw):
        self.engine = kw.get("engine", "claude-cli")
        self.engine_model = kw.get("engine_model", "haiku")
        self.engine_base_url = kw.get("engine_base_url", "")
        self.engine_api_key_env = kw.get("engine_api_key_env", "ANTHROPIC_API_KEY")
        self.engine_allowed_tools = kw.get(
            "engine_allowed_tools", "mcp__neuron__rlm_grep,Bash"
        )
        self.engine_max_turns = kw.get("engine_max_turns", 12)


def test_get_engine_claude_is_none():
    assert engine.get_engine(_Cfg(engine="claude")) is None


def test_get_engine_unknown_is_none():
    assert engine.get_engine(_Cfg(engine="api")) is None  # not in phase 1


def test_get_engine_claude_cli():
    assert isinstance(engine.get_engine(_Cfg()), engine.ClaudeCliEngine)


def test_run_spawns_detached_with_prompt(monkeypatch):
    captured = {}

    def fake_popen(argv, **kw):
        captured["argv"] = argv
        captured["kw"] = kw
        return object()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    engine.ClaudeCliEngine(_Cfg()).run("EPISODE-PROMPT", cwd="/proj")

    argv = captured["argv"]
    assert argv[0] == "claude"
    assert "EPISODE-PROMPT" in argv
    assert captured["kw"]["cwd"] == "/proj"
    assert captured["kw"]["start_new_session"] is True


def test_digest_spawns_detached_cheap(monkeypatch):
    captured = {}

    def fake_popen(argv, **kw):
        captured["argv"] = argv
        captured["kw"] = kw
        return object()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    engine.ClaudeCliEngine(_Cfg()).digest("tid-1", "", cwd="/proj")

    argv = captured["argv"]
    assert argv[0] == "claude"
    assert argv[1] == "-p"
    assert argv[argv.index("--model") + 1] == "haiku"
    assert "--max-turns" in argv
    assert argv[argv.index("--max-turns") + 1] == "12"
    assert "--permission-mode" in argv
    assert captured["kw"]["start_new_session"] is True
    assert captured["kw"]["cwd"] == "/proj"
    prompt = argv[2]
    assert "tid-1" in prompt
    assert "simba memory store" in prompt


def test_digest_proxy_env(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda argv, **kw: captured.update(env=kw.get("env")) or object(),
    )
    monkeypatch.setenv("MYKEY", "secret-token")
    cfg = _Cfg(engine_base_url="http://proxy:1234", engine_api_key_env="MYKEY")
    engine.ClaudeCliEngine(cfg).digest("t", "", cwd="/p")
    assert captured["env"]["ANTHROPIC_BASE_URL"] == "http://proxy:1234"
    assert captured["env"]["ANTHROPIC_AUTH_TOKEN"] == "secret-token"


def test_digest_no_proxy_leaves_env_clean(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda argv, **kw: captured.update(env=kw.get("env")) or object(),
    )
    engine.ClaudeCliEngine(_Cfg()).digest("t", "", cwd="/p")
    assert "ANTHROPIC_BASE_URL" not in captured["env"]
