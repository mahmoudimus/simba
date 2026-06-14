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
        self.digest_prompt = kw.get("digest_prompt", "")


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


# --- Configurable digest prompt (claude-cli engine) ------------------------


def test_claude_digest_uses_configured_prompt(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda argv, **kw: captured.update(argv=argv) or object(),
    )
    cfg = _Cfg(digest_prompt="CUSTOM digest for {tid} in {cwd}")
    engine.ClaudeCliEngine(cfg).digest("T1", "", cwd="/proj")
    assert captured["argv"][2] == "CUSTOM digest for T1 in /proj"


def test_claude_digest_default_prompt_when_unset(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda argv, **kw: captured.update(argv=argv) or object(),
    )
    engine.ClaudeCliEngine(_Cfg()).digest("T1", "", cwd="/proj")
    prompt = captured["argv"][2]
    assert "A coding session just ended" in prompt
    assert "T1" in prompt and "simba memory store" in prompt


def test_claude_run_accepts_session_source_kwarg(monkeypatch):
    # Protocol gained session_source; the agentic engine accepts+ignores it.
    monkeypatch.setattr("subprocess.Popen", lambda argv, **kw: object())
    engine.ClaudeCliEngine(_Cfg()).run("P", cwd="/p", session_source="ignored")


# --- llm-cli completion engine (deepseek via `llm -m`) ---------------------


def test_get_engine_llm_cli():
    assert isinstance(engine.get_engine(_Cfg(engine="llm-cli")), engine.LlmCliEngine)


def test_parse_memories_keeps_valid_drops_invalid():
    text = """[
      {"type": "GOTCHA", "content": "watch the cache", "context": "ttl 5m"},
      {"type": "NONSENSE", "content": "x"},
      {"type": "DECISION", "content": ""},
      {"type": "preference", "content": "user prefers deepseek"}
    ]"""
    mems = engine._parse_memories(text)
    assert [m["type"] for m in mems] == ["GOTCHA", "PREFERENCE"]
    assert mems[0]["context"] == "ttl 5m"
    assert mems[1]["content"] == "user prefers deepseek"


def test_parse_memories_handles_fenced_json():
    text = '```json\n[{"type":"PATTERN","content":"c"}]\n```'
    assert len(engine._parse_memories(text)) == 1


def test_parse_memories_non_list_is_empty():
    assert engine._parse_memories("not json at all") == []
    assert engine._parse_memories('{"type":"GOTCHA","content":"x"}') == []


def test_run_completion_worker_stores_and_marks():
    class FakeClient:
        def complete(self, prompt):
            return (
                '[{"type":"GOTCHA","content":"g1","context":"c1"},'
                '{"type":"DECISION","content":"d1"}]'
            )

    stored = []
    marked = {}

    def fake_store(mem, *, cwd, session_source):
        stored.append((mem["type"], mem["content"], cwd, session_source))
        return True

    def fake_complete(tid, cwd, n):
        marked.update(tid=tid, cwd=cwd, n=n)

    n = engine.run_completion_worker(
        "PROMPT",
        cwd="/proj",
        session_source="T1",
        mark_rlm=True,
        client=FakeClient(),
        store_fn=fake_store,
        complete_job=fake_complete,
    )
    assert n == 2
    assert stored[0] == ("GOTCHA", "g1", "/proj", "T1")
    assert marked == {"tid": "T1", "cwd": "/proj", "n": 2}


def test_run_completion_worker_no_mark_when_disabled():
    class FakeClient:
        def complete(self, prompt):
            return '[{"type":"GOTCHA","content":"g1"}]'

    marked = []
    n = engine.run_completion_worker(
        "P",
        cwd="/p",
        session_source="S",
        mark_rlm=False,
        client=FakeClient(),
        store_fn=lambda m, **k: True,
        complete_job=lambda *a: marked.append(a),
    )
    assert n == 1 and marked == []


def test_run_completion_worker_counts_only_successful_stores():
    class FakeClient:
        def complete(self, prompt):
            return '[{"type":"GOTCHA","content":"a"},{"type":"DECISION","content":"b"}]'

    n = engine.run_completion_worker(
        "P",
        cwd="/p",
        client=FakeClient(),
        store_fn=lambda mem, **k: mem["type"] == "GOTCHA",
    )
    assert n == 1


_PROMPT_SRC = (
    "Transcript: the user paid $2,750 for the laptop and is leading the cloud "
    "migration project at the company."
)


class _GroundedAndHallucinatedClient:
    def complete(self, prompt):
        return (
            '[{"type":"DECISION","content":"user is leading the cloud migration '
            'project"},'
            '{"type":"GOTCHA","content":"user paid $9999 for the laptop"}]'
        )


def test_extraction_validation_drops_hallucinated_claim():
    import types

    cfg = types.SimpleNamespace(
        extraction_validation_enabled=True,
        extraction_validation_min_support=0.5,
    )
    stored = []
    n = engine.run_completion_worker(
        _PROMPT_SRC,
        cwd="/p",
        client=_GroundedAndHallucinatedClient(),
        store_fn=lambda mem, **k: stored.append(mem["content"]) or True,
        cfg=cfg,
    )
    # the $9999 claim is ungrounded (number absent from source) -> dropped
    assert n == 1
    assert stored == ["user is leading the cloud migration project"]


def test_extraction_validation_off_by_default_stores_both():
    import types

    cfg = types.SimpleNamespace(extraction_validation_enabled=False)
    stored = []
    n = engine.run_completion_worker(
        _PROMPT_SRC,
        cwd="/p",
        client=_GroundedAndHallucinatedClient(),
        store_fn=lambda mem, **k: stored.append(mem["content"]) or True,
        cfg=cfg,
    )
    assert n == 2 and len(stored) == 2


def test_llm_digest_spawns_detached_worker(monkeypatch):
    monkeypatch.setattr(engine, "_load_transcript_text", lambda tid: "TRANSCRIPT-BODY")
    captured = {}

    def fake_popen(argv, **kw):
        captured["argv"] = argv
        captured["kw"] = kw
        return object()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    engine.LlmCliEngine(_Cfg(engine="llm-cli")).digest("T1", "", cwd="/proj")

    argv = captured["argv"]
    assert argv[1:4] == ["-m", "simba", "rlm"]
    assert "run-llm" in argv
    assert "--mark-rlm-complete" in argv
    assert argv[argv.index("--session-source") + 1] == "T1"
    assert captured["kw"]["start_new_session"] is True
    assert captured["kw"]["cwd"] == "/proj"
    import pathlib

    pf = argv[argv.index("--prompt-file") + 1]
    body = pathlib.Path(pf).read_text()
    assert "TRANSCRIPT-BODY" in body
    pathlib.Path(pf).unlink()


def test_llm_run_spawns_worker_without_mark(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda argv, **kw: captured.update(argv=argv, kw=kw) or object(),
    )
    engine.LlmCliEngine(_Cfg(engine="llm-cli")).run(
        "EPISODE-PROMPT", cwd="/p", session_source="S9"
    )
    argv = captured["argv"]
    assert "run-llm" in argv
    assert "--mark-rlm-complete" not in argv
    assert argv[argv.index("--session-source") + 1] == "S9"
    import pathlib

    pf = argv[argv.index("--prompt-file") + 1]
    assert "EPISODE-PROMPT" in pathlib.Path(pf).read_text()
    pathlib.Path(pf).unlink()


def test_llm_digest_uses_configured_prompt(monkeypatch):
    monkeypatch.setattr(engine, "_load_transcript_text", lambda tid: "BODY")
    captured = {}
    monkeypatch.setattr(
        "subprocess.Popen",
        lambda argv, **kw: captured.update(argv=argv) or object(),
    )
    cfg = _Cfg(engine="llm-cli", digest_prompt="Extract facts from: {transcript}")
    engine.LlmCliEngine(cfg).digest("T1", "", cwd="/proj")
    import pathlib

    pf = captured["argv"][captured["argv"].index("--prompt-file") + 1]
    assert pathlib.Path(pf).read_text() == "Extract facts from: BODY"
    pathlib.Path(pf).unlink()


def test_run_completion_from_file_reads_and_unlinks(monkeypatch, tmp_path):
    pf = tmp_path / "p.txt"
    pf.write_text("THE-PROMPT")
    seen = {}

    def fake_worker(prompt, *, cwd, session_source, mark_rlm):
        seen.update(
            prompt=prompt, cwd=cwd, session_source=session_source, mark_rlm=mark_rlm
        )
        return 3

    monkeypatch.setattr(engine, "run_completion_worker", fake_worker)
    n = engine.run_completion_from_file(
        str(pf), cwd="/c", session_source="S", mark_rlm=True
    )
    assert n == 3
    assert seen == {
        "prompt": "THE-PROMPT",
        "cwd": "/c",
        "session_source": "S",
        "mark_rlm": True,
    }
    assert not pf.exists()  # temp prompt file cleaned up
