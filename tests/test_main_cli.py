"""Tests for top-level CLI helpers in simba.__main__."""

from __future__ import annotations

import io
import json
import os
import pathlib
import sys

import pytest

import simba.__main__ as cli


class TestResolveHookClient:
    """`simba hook <event> [--client X]` splits the flag and resolves a name."""

    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in (
            "SIMBA_CLIENT",
            "CLAUDECODE",
            "CLAUDE_CODE_ENTRYPOINT",
            "CODEX_SANDBOX",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_explicit_flag(self) -> None:
        args, client, defaulted = cli._resolve_hook_client(
            ["UserPromptSubmit", "--client", "codex"]
        )
        assert args == ["UserPromptSubmit"]
        assert client == "codex"
        assert defaulted is False

    def test_equals_form(self) -> None:
        args, client, defaulted = cli._resolve_hook_client(
            ["PreToolUse", "--client=codex"]
        )
        assert args == ["PreToolUse"]
        assert client == "codex"
        assert defaulted is False

    def test_default_is_claude_code(self) -> None:
        args, client, defaulted = cli._resolve_hook_client(["SessionStart"])
        assert args == ["SessionStart"]
        assert client == "claude-code"
        assert defaulted is True

    def test_env_marker_codex_without_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CODEX_SANDBOX", "seatbelt")
        _args, client, defaulted = cli._resolve_hook_client(["SessionStart"])
        assert client == "codex"
        assert defaulted is False

    def test_explicit_claude_code_is_not_defaulted(self) -> None:
        # An explicit flag that happens to equal the default is still a real
        # resolution -- the payload sniff must never second-guess it.
        _args, client, defaulted = cli._resolve_hook_client(
            ["Stop", "--client", "claude-code"]
        )
        assert client == "claude-code"
        assert defaulted is False

    def test_simba_client_env_is_not_defaulted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SIMBA_CLIENT", "claude-code")
        _args, client, defaulted = cli._resolve_hook_client(["Stop"])
        assert client == "claude-code"
        assert defaulted is False


class TestHookConfigClientTag:
    """Generated Codex hooks self-identify as `codex`; Claude stays default."""

    def test_codex_commands_carry_client_flag(self) -> None:
        cfg = cli._build_codex_hooks_config()
        cmds = [
            h["command"]
            for entries in cfg["hooks"].values()
            for entry in entries
            for h in entry["hooks"]
        ]
        assert cmds, "expected at least one codex hook command"
        assert all(c.endswith(" --client codex") for c in cmds)

    def test_claude_commands_have_no_client_flag(self) -> None:
        cfg = cli._build_hooks_config()
        cmds = [
            h["command"]
            for entries in cfg.values()
            for entry in entries
            for h in entry["hooks"]
        ]
        assert cmds
        assert all("--client" not in c for c in cmds)


class TestHookClientSniff:
    """`simba hook` sniffs a defaulted client from the payload's transcript path.

    A legacy ``.codex/hooks.json`` generated before ``--client codex`` existed
    invokes ``simba hook Stop`` flagless. Codex sets no reliable env marker in
    hook subprocesses, so resolution falls through to the claude-code default
    and Codex receives the claude Stop shape (top-level ``hookSpecificOutput``),
    which its ``StopCommandOutputWire`` deserializer rejects. The payload sniff
    only fires when resolution was genuinely defaulted -- an explicit flag or
    env value always wins.
    """

    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in (
            "SIMBA_CLIENT",
            "CLAUDECODE",
            "CLAUDE_CODE_ENTRYPOINT",
            "CODEX_SANDBOX",
        ):
            monkeypatch.delenv(var, raising=False)

    def _run_stop_hook(
        self,
        monkeypatch: pytest.MonkeyPatch,
        payload: dict,
        argv: list[str],
    ) -> str:
        import simba.harness.core as harness_core

        monkeypatch.setattr(
            cli,
            "_dispatch_canonical",
            lambda event, payload: harness_core.CanonicalResult(
                additional_context="hi"
            ),
        )
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        rc = cli._cmd_hook([*argv, "Stop"])
        assert rc == 0

    def test_codex_transcript_path_sniffed_when_defaulted(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        payload = {
            "transcript_path": ("/Users/x/.codex/sessions/2026/07/22/rollout-abc.jsonl")
        }
        self._run_stop_hook(monkeypatch, payload, [])
        out = capsys.readouterr().out
        assert json.loads(out) == {"stopReason": "hi"}
        assert os.environ["SIMBA_CLIENT"] == "codex"

    def test_explicit_flag_beats_sniff(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        payload = {
            "transcript_path": ("/Users/x/.codex/sessions/2026/07/22/rollout-abc.jsonl")
        }
        self._run_stop_hook(monkeypatch, payload, ["--client", "claude-code"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "hookSpecificOutput" in data
        assert os.environ["SIMBA_CLIENT"] == "claude-code"

    def test_claude_transcript_path_stays_default(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        payload = {"transcript_path": "/Users/x/.claude/projects/foo/00000000.jsonl"}
        self._run_stop_hook(monkeypatch, payload, [])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "hookSpecificOutput" in data
        assert os.environ["SIMBA_CLIENT"] == "claude-code"

    def test_simba_client_env_beats_sniff(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("SIMBA_CLIENT", "claude-code")
        payload = {
            "transcript_path": ("/Users/x/.codex/sessions/2026/07/22/rollout-abc.jsonl")
        }
        self._run_stop_hook(monkeypatch, payload, [])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "hookSpecificOutput" in data
        assert os.environ["SIMBA_CLIENT"] == "claude-code"


def _write_codex_session(
    codex_home: pathlib.Path,
    *,
    session_id: str = "abc123",
    cwd: str | None = None,
    text: str | None = None,
) -> pathlib.Path:
    # Default the session's recorded cwd to the live cwd so codex commands that
    # scope extraction to the current project (the cross-wiring fix) still match.
    if cwd is None:
        cwd = str(pathlib.Path.cwd())
    session_dir = codex_home / "sessions" / "2026" / "02" / "20"
    session_dir.mkdir(parents=True, exist_ok=True)
    codex_jsonl = session_dir / f"rollout-2026-02-20T08-33-13-{session_id}.jsonl"
    lines = [
        {
            "type": "session_meta",
            "payload": {"id": session_id, "cwd": cwd},
        }
    ]
    if text:
        lines.append({"message": {"content": [{"type": "text", "text": text}]}})
    codex_jsonl.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return codex_jsonl


def test_install_codex_skills_copies_skill_and_agents(tmp_path: pathlib.Path) -> None:
    skills_dir = tmp_path / "skills"

    count = cli._install_codex_skills(skills_dir)

    assert count >= 1
    assert (skills_dir / "simba-onboard" / "SKILL.md").is_file()
    assert (skills_dir / "simba-onboard" / "agents" / "openai.yaml").is_file()


def test_remove_codex_skills_removes_installed_skill(tmp_path: pathlib.Path) -> None:
    skills_dir = tmp_path / "skills"
    cli._install_codex_skills(skills_dir)

    removed = cli._remove_codex_skills(skills_dir)

    assert removed >= 1
    assert not (skills_dir / "simba-onboard").exists()


def test_cmd_codex_install_uses_codex_home_env(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    codex_home = tmp_path / "my-codex-home"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    called: list[pathlib.Path] = []

    def _fake_install(skills_dir: pathlib.Path) -> int:
        called.append(skills_dir)
        return 1

    monkeypatch.setattr(cli, "_install_codex_skills", _fake_install)

    rc = cli._cmd_codex_install([])

    assert rc == 0
    assert called == [codex_home / "skills"]


def test_cmd_codex_install_remove_uses_codex_home_env(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    codex_home = tmp_path / "my-codex-home"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    called: list[pathlib.Path] = []

    def _fake_remove(skills_dir: pathlib.Path) -> int:
        called.append(skills_dir)
        return 1

    monkeypatch.setattr(cli, "_remove_codex_skills", _fake_remove)

    rc = cli._cmd_codex_install(["--remove"])

    assert rc == 0
    assert called == [codex_home / "skills"]


def test_ensure_codex_feature_flag_creates_file(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    import tomllib

    codex_home = tmp_path / "codex"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    status = cli._ensure_codex_feature_flag()

    assert status == "added"
    cfg = tomllib.loads((codex_home / "config.toml").read_text())
    assert cfg == {"features": {"hooks": True}}


def test_ensure_codex_feature_flag_preserves_existing(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    import tomllib

    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    config_path = codex_home / "config.toml"
    config_path.write_text('model = "gpt-5.5"\n\n[features]\nmulti_agent = true\n')
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    status = cli._ensure_codex_feature_flag()

    assert status == "added"
    cfg = tomllib.loads(config_path.read_text())
    assert cfg["model"] == "gpt-5.5"
    assert cfg["features"] == {"multi_agent": True, "hooks": True}


def test_ensure_codex_feature_flag_idempotent(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text("[features]\nhooks = true\n")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    assert cli._ensure_codex_feature_flag() == "already-set"


def test_ensure_codex_feature_flag_remove(tmp_path: pathlib.Path, monkeypatch) -> None:
    import tomllib

    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    config_path = codex_home / "config.toml"
    config_path.write_text("[features]\nmulti_agent = true\nhooks = true\n")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    status = cli._ensure_codex_feature_flag(remove=True)

    assert status == "removed"
    cfg = tomllib.loads(config_path.read_text())
    assert cfg["features"] == {"multi_agent": True}


def test_ensure_codex_feature_flag_remove_drops_empty_table(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    import tomllib

    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    config_path = codex_home / "config.toml"
    config_path.write_text("[features]\nhooks = true\n")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    cli._ensure_codex_feature_flag(remove=True)

    cfg = tomllib.loads(config_path.read_text())
    assert "features" not in cfg


def test_ensure_codex_feature_flag_migrates_legacy_key(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    import tomllib

    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    config_path = codex_home / "config.toml"
    config_path.write_text("[features]\nmulti_agent = true\ncodex_hooks = true\n")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    status = cli._ensure_codex_feature_flag()

    assert status == "migrated"
    cfg = tomllib.loads(config_path.read_text())
    assert cfg["features"] == {"multi_agent": True, "hooks": True}


def test_ensure_codex_feature_flag_remove_clears_legacy_key(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    import tomllib

    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    config_path = codex_home / "config.toml"
    config_path.write_text("[features]\ncodex_hooks = true\n")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    status = cli._ensure_codex_feature_flag(remove=True)

    assert status == "removed"
    cfg = tomllib.loads(config_path.read_text())
    assert "features" not in cfg


def test_cmd_codex_extract_mark_done_updates_latest_target(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    codex_home = tmp_path / ".codex"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    session_dir = codex_home / "sessions" / "2026" / "02" / "20"
    session_dir.mkdir(parents=True, exist_ok=True)
    codex_jsonl = session_dir / "rollout-2026-02-20T08-33-13-s1.jsonl"
    codex_jsonl.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {"id": "s1", "cwd": str(pathlib.Path.cwd())},
            }
        )
        + "\n"
    )

    rc = cli._cmd_codex_extract(["--mark-done"])
    assert rc == 0

    # Codex session JSONL has no latest.json metadata file to persist mark-done.
    out = codex_jsonl.read_text()
    assert '"id": "s1"' in out


def test_cmd_codex_status_shows_pending_extraction(
    tmp_path: pathlib.Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    codex_home = tmp_path / ".codex"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    session_dir = codex_home / "sessions" / "2026" / "02" / "20"
    session_dir.mkdir(parents=True, exist_ok=True)
    codex_jsonl = session_dir / "rollout-2026-02-20T08-33-13-s1.jsonl"
    codex_jsonl.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {"id": "s1", "cwd": str(pathlib.Path.cwd())},
            }
        )
        + "\n"
    )

    class _Resp:
        status_code = 503

        def json(self):
            return {}

    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())

    rc = cli._cmd_codex_status(["--auto-extract"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "pending_extraction" in out
    assert "simba codex-extract" in out
    assert "transcript source: codex" in out


def test_cmd_codex_status_auto_extracts_and_marks_ledger(
    tmp_path: pathlib.Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    codex_home = tmp_path / ".codex"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    transcript = _write_codex_session(
        codex_home,
        text=(
            "Always use uv run for Simba CLI commands because it fixes lifecycle tests."
        ),
    )

    class _HealthResp:
        status_code = 200

        def json(self):
            return {"memoryCount": 10, "embeddingModel": "test"}

    class _StoreResp:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"status": "stored", "id": "mem_1"}

    import httpx

    import simba.hooks._memory_client

    posts: list[tuple[str, dict]] = []
    monkeypatch.setattr(httpx, "get", lambda *a, **k: _HealthResp())
    monkeypatch.setattr(
        simba.hooks._memory_client, "daemon_url", lambda: "http://daemon"
    )

    def _fake_post(url: str, json=None, timeout: float = 0.0):
        if url.endswith("/sync"):
            return _StoreResp()
        posts.append((url, json))
        return _StoreResp()

    monkeypatch.setattr(httpx, "post", _fake_post)

    rc = cli._cmd_codex_status(["--auto-extract"])
    assert rc == 0
    assert len(posts) == 1
    assert posts[0][0] == "http://daemon/store"
    payload = posts[0][1]
    assert payload["sessionSource"] == "abc123"
    assert payload["projectPath"] == str(pathlib.Path.cwd())
    assert payload["content"].startswith("Always use uv run")

    ledger_path = codex_home / "simba" / "extractions.jsonl"
    record = json.loads(ledger_path.read_text().strip())
    assert record["status"] == "extracted"
    assert record["session_id"] == "abc123"
    assert record["project_path"] == str(pathlib.Path.cwd())
    assert record["transcript_path"] == str(transcript)
    assert record["stored"] == 1
    out = capsys.readouterr().out
    assert "auto-extract: status=stored" in out

    rc = cli._cmd_codex_status([])
    assert rc == 0
    assert len(posts) == 1
    out = capsys.readouterr().out
    assert "extraction status: extracted" in out


def test_cmd_codex_status_prints_rich_health(
    tmp_path: pathlib.Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    codex_home = tmp_path / ".codex"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    _write_codex_session(codex_home, text="health detail")

    class _HealthResp:
        status_code = 200

        def json(self):
            return {
                "status": "degraded",
                "ready": True,
                "degraded": True,
                "memoryCount": 10,
                "embeddingModel": "test-model",
                "embeddingDims": 1024,
                "dbPath": "/tmp/memories.lance",
                "ftsPath": "/tmp/memory_fts.db",
                "components": {
                    "vector": {"table": "memories", "path": "/tmp/memories.lance"},
                    "fts": {"path": "/tmp/memory_fts.db"},
                    "embedder": {"provider": "gguf", "dims": 1024},
                    "reranker": {"mode": "cross-encoder"},
                },
                "lastError": {
                    "type": "RuntimeError",
                    "endpoint": "/recall",
                    "request_id": "req1",
                },
            }

    class _PostResp:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"status": "ok"}

    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _HealthResp())
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _PostResp())

    rc = cli._cmd_codex_status(["--no-auto-extract"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "readiness: status=degraded ready=True degraded=True" in out
    assert "storage: db=/tmp/memories.lance table=memories" in out
    assert "retrieval: embedding_dims=1024 provider=gguf reranker=cross-encoder" in out
    assert "last error: RuntimeError endpoint=/recall request=req1" in out


def test_cmd_codex_extract_run_skips_already_extracted_transcript(
    tmp_path: pathlib.Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    codex_home = tmp_path / ".codex"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    transcript = _write_codex_session(
        codex_home,
        text=(
            "Always use uv run for Simba CLI commands because it fixes lifecycle tests."
        ),
    )

    import httpx

    import simba.codex.ledger as codex_ledger

    fingerprint = codex_ledger.transcript_fingerprint(transcript)
    assert fingerprint is not None
    codex_ledger.append_extracted(
        codex_home=codex_home,
        transcript_path=str(transcript),
        session_id="abc123",
        project_path=str(pathlib.Path.cwd()),
        fingerprint=fingerprint,
        candidates=1,
        stored=1,
        duplicates=0,
    )

    def _unexpected_post(*args, **kwargs):
        raise AssertionError("already-extracted transcript should not be stored")

    monkeypatch.setattr(httpx, "post", _unexpected_post)

    rc = cli._cmd_codex_extract(["--run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "already_extracted" in out


def test_cmd_codex_extract_run_keeps_pending_on_store_error(
    tmp_path: pathlib.Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    codex_home = tmp_path / ".codex"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    _write_codex_session(
        codex_home,
        text=(
            "Always use uv run for Simba CLI commands because it fixes lifecycle tests."
        ),
    )

    class _StoreResp:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"status": "error"}

    import httpx

    import simba.hooks._memory_client

    monkeypatch.setattr(
        simba.hooks._memory_client, "daemon_url", lambda: "http://daemon"
    )
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _StoreResp())

    trace_dir = tmp_path / "trace-errors"
    rc = cli._cmd_codex_extract(["--run", "--trace-dir", str(trace_dir)])
    assert rc == 1
    assert not (codex_home / "simba" / "extractions.jsonl").exists()
    out = capsys.readouterr().out
    assert "store_errors" in out
    assert "errors=1" in out
    trace_file = next(trace_dir.glob("*.jsonl"))
    events = [
        json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines()
    ]
    negative = next(event for event in events if event["event"] == "negative_lesson")
    assert negative["payload"] == {
        "index": 0,
        "reason": "store_status_unaccepted",
        "status": "error",
    }


def test_cmd_codex_extract_run_writes_trace_artifact(
    tmp_path: pathlib.Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    codex_home = tmp_path / ".codex"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    _write_codex_session(
        codex_home,
        text=(
            "Always use uv run for Simba CLI commands because it fixes lifecycle tests."
        ),
    )

    class _StoreResp:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"status": "stored", "id": "mem_1"}

    import httpx

    import simba.hooks._memory_client

    monkeypatch.setattr(
        simba.hooks._memory_client, "daemon_url", lambda: "http://daemon"
    )
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _StoreResp())

    trace_dir = tmp_path / "analysis-traces"
    rc = cli._cmd_codex_extract(["--run", "--trace-dir", str(trace_dir)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "[codex] analysis trace:" in out
    trace_files = list(trace_dir.glob("*.jsonl"))
    assert len(trace_files) == 1
    records = [
        json.loads(line)
        for line in trace_files[0].read_text(encoding="utf-8").splitlines()
    ]
    events = [record["event"] for record in records]
    assert events == [
        "run_started",
        "transcript_loaded",
        "candidate",
        "curator_decision",
        "store_result",
        "run_completed",
    ]

    candidate = next(record for record in records if record["event"] == "candidate")
    assert candidate["payload"]["type"] == "PREFERENCE"
    assert candidate["payload"]["reason"] == "matched preference transcript heuristic"
    assert candidate["payload"]["source_span"]
    assert "Always use uv run" in candidate["payload"]["evidence"]

    decision = next(
        record for record in records if record["event"] == "curator_decision"
    )
    assert decision["payload"]["decision"] == "keep"
    store = next(record for record in records if record["event"] == "store_result")
    assert store["payload"] == {
        "index": 0,
        "memory_id": "mem_1",
        "status": "stored",
        "superseded_id": None,
    }
    completed = records[-1]
    assert completed["payload"]["status"] == "stored"
    assert completed["payload"]["stored"] == 1


def _write_analysis_trace(
    trace_dir: pathlib.Path,
    *,
    name: str = "run.jsonl",
) -> pathlib.Path:
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace = trace_dir / name
    rows = [
        {
            "event": "candidate",
            "session_id": "session-1",
            "project_path": "/tmp/codex-project",
            "transcript_path": "/tmp/transcript.jsonl",
            "payload": {
                "index": 0,
                "type": "DECISION",
                "content": "Curator reports are review-only",
                "source_span": "message:2",
                "evidence": "no auto-store",
            },
        },
        {
            "event": "store_result",
            "session_id": "session-1",
            "project_path": "/tmp/codex-project",
            "transcript_path": "/tmp/transcript.jsonl",
            "payload": {"index": 0, "status": "duplicate", "memory_id": "mem-1"},
        },
        {
            "event": "run_completed",
            "session_id": "session-1",
            "project_path": "/tmp/codex-project",
            "transcript_path": "/tmp/transcript.jsonl",
            "payload": {"status": "stored"},
        },
    ]
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return trace


class _CodexCurateCfg:
    auto_extract_on_status = False
    auto_extract_max_items = 15
    extraction_trace_enabled = False
    extraction_trace_dir = ""
    curator_report_dir = ""
    curator_default_format = "markdown"
    curator_min_candidate_score = 0.0


def test_cmd_codex_curate_latest_uses_configured_trace_dir(
    tmp_path: pathlib.Path,
    monkeypatch,
    capsys,
) -> None:
    import simba.config

    trace_dir = tmp_path / "traces"
    report_dir = tmp_path / "reports"
    _write_analysis_trace(trace_dir)
    cfg = _CodexCurateCfg()
    cfg.extraction_trace_dir = str(trace_dir)
    cfg.curator_report_dir = str(report_dir)
    monkeypatch.setattr(simba.config, "load", lambda section: cfg)

    rc = cli._cmd_codex_curate(["--latest"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "[codex] curator report:" in out
    reports = list(report_dir.glob("*.md"))
    assert len(reports) == 1
    assert "Curator reports are review-only" in reports[0].read_text()


def test_cmd_codex_curate_trace_writes_default_curator_dir(
    tmp_path: pathlib.Path,
    monkeypatch,
    capsys,
) -> None:
    import simba.config

    monkeypatch.chdir(tmp_path)
    trace = _write_analysis_trace(tmp_path / "traces")
    monkeypatch.setattr(simba.config, "load", lambda section: _CodexCurateCfg())

    rc = cli._cmd_codex_curate(["--trace", str(trace)])

    assert rc == 0
    assert "candidates=1" in capsys.readouterr().out
    report = tmp_path / ".simba" / "curator_runs" / "run.md"
    assert report.exists()
    assert "message:2" in report.read_text()


def test_cmd_codex_curate_json_writes_json_report(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    import simba.config

    trace = _write_analysis_trace(tmp_path / "traces")
    out = tmp_path / "curator.json"
    monkeypatch.setattr(simba.config, "load", lambda section: _CodexCurateCfg())

    rc = cli._cmd_codex_curate(["--trace", str(trace), "--out", str(out), "--json"])

    assert rc == 0
    data = json.loads(out.read_text())
    assert data["metrics"]["duplicate_count"] == 1
    assert data["candidates"][0]["store_status"] == "duplicate"


def test_cmd_codex_curate_does_not_call_memory_store(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    import httpx

    import simba.config

    trace = _write_analysis_trace(tmp_path / "traces")
    monkeypatch.setattr(simba.config, "load", lambda section: _CodexCurateCfg())

    def _forbid_post(*args, **kwargs):
        raise AssertionError("codex-curate must not call memory store")

    monkeypatch.setattr(httpx, "post", _forbid_post)

    rc = cli._cmd_codex_curate(["--trace", str(trace), "--out", str(tmp_path)])

    assert rc == 0
    assert (tmp_path / "run.md").exists()


def test_cmd_codex_curate_review_appends_labels_and_prints_commands(
    tmp_path: pathlib.Path,
    monkeypatch,
    capsys,
) -> None:
    import httpx

    trace = _write_analysis_trace(tmp_path / "traces")
    report = tmp_path / "report.json"
    rc = cli._cmd_codex_curate(["--trace", str(trace), "--out", str(report), "--json"])
    assert rc == 0
    capsys.readouterr()

    def _forbid_post(*args, **kwargs):
        raise AssertionError("codex-curate review must not call memory store")

    monkeypatch.setattr(httpx, "post", _forbid_post)

    rc = cli._cmd_codex_curate(
        [
            "review",
            str(report),
            "--accept",
            "0",
            "--reason",
            "good evidence",
            "--reviewer",
            "tester",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "accepted promotion commands: 1 (not executed)" in out
    assert "simba memory store" in out
    rows = [
        json.loads(line)
        for line in (tmp_path / "report.review.jsonl").read_text().splitlines()
    ]
    assert rows[0]["label"] == "accepted"
    assert rows[0]["reason"] == "good evidence"
    assert rows[0]["reviewer"] == "tester"


def test_cmd_codex_curate_review_rejects_missing_labels(
    tmp_path: pathlib.Path,
    capsys,
) -> None:
    trace = _write_analysis_trace(tmp_path / "traces")

    rc = cli._cmd_codex_curate(["review", str(trace)])

    assert rc == 1
    assert "provide at least one review label" in capsys.readouterr().err


def test_cmd_codex_finalize_runs_signal_and_reflection(
    tmp_path: pathlib.Path,
    monkeypatch,
    capsys,
) -> None:
    transcript = tmp_path / "t.jsonl"
    transcript.write_text("{}\n")

    called: dict[str, object] = {}

    import simba.guardian.check_signal
    import simba.tailor.hook

    def _fake_signal(response: str, cwd: pathlib.Path | None = None) -> str:
        called["response"] = response
        called["cwd"] = cwd
        return ""

    def _fake_process(payload: str) -> None:
        called["payload"] = json.loads(payload)

    monkeypatch.setattr(simba.guardian.check_signal, "main", _fake_signal)
    monkeypatch.setattr(simba.tailor.hook, "process_hook", _fake_process)

    rc = cli._cmd_codex_finalize(
        ["--response", "hello [✓ rules]", "--transcript", str(transcript)]
    )
    assert rc == 0
    assert called["response"] == "hello [✓ rules]"
    assert called["payload"]["transcript_path"] == str(transcript)
    out = capsys.readouterr().out
    assert "signal check: ok" in out


def test_codex_config_visible_via_config_cli(
    tmp_path: pathlib.Path,
    capsys,
) -> None:
    import simba.config_cli

    rc = simba.config_cli.cmd_get("codex.auto_extract_on_status", tmp_path)
    assert rc == 0
    assert capsys.readouterr().out.strip() == "False"

    rc = simba.config_cli.cmd_get("codex.extraction_trace_enabled", tmp_path)
    assert rc == 0
    assert capsys.readouterr().out.strip() == "False"

    rc = simba.config_cli.cmd_get("codex.curator_default_format", tmp_path)
    assert rc == 0
    assert capsys.readouterr().out.strip() == "markdown"


def test_cmd_codex_recall_prints_memories(monkeypatch, capsys) -> None:
    import simba.hooks._memory_client

    monkeypatch.setattr(
        simba.hooks._memory_client,
        "recall_memories",
        lambda query, project_path=None, **kwargs: [
            {"type": "PATTERN", "similarity": 0.91, "content": "Use uv run for CLI"}
        ],
    )

    rc = cli._cmd_codex_recall(["uv", "command", "pattern"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "recall: 1 memories" in out
    assert "Use uv run for CLI" in out


def test_memory_store_allows_content_above_200_when_under_config_limit(
    monkeypatch,
    capsys,
) -> None:
    import httpx

    import simba.hooks._memory_client

    monkeypatch.setattr(cli, "_memory_max_content_length", lambda *_: 1000)
    monkeypatch.setattr(simba.hooks._memory_client, "daemon_url", lambda: "http://x")

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"status": "stored", "id": "mem_abc"}

    captured: dict[str, object] = {}

    def _fake_post(url: str, json=None, timeout: float = 0.0):
        captured["url"] = url
        captured["json"] = json
        return _Resp()

    monkeypatch.setattr(httpx, "post", _fake_post)

    long_content = "x" * 300
    rc = cli._memory_store(
        [
            "--type",
            "GOTCHA",
            "--content",
            long_content,
            "--context",
            "ctx",
            "--confidence",
            "0.9",
            "--occurred-at",
            "2026-06-01",
            "--source-file",
            "src/a.py",
            "--source-span",
            "10-12",
            "--extraction-agent",
            "test-agent",
            "--extraction-version",
            "1",
            "--anticipated-query",
            "How should this be found later?",
            "--anticipated-queries",
            "alternate phrase,second phrase",
        ]
    )
    assert rc == 0
    assert captured["json"]["content"] == long_content
    assert captured["json"]["occurredAt"] == "2026-06-01"
    assert captured["json"]["sourceFile"] == "src/a.py"
    assert captured["json"]["sourceSpan"] == "10-12"
    assert captured["json"]["extractionAgent"] == "test-agent"
    assert captured["json"]["extractionVersion"] == "1"
    assert captured["json"]["anticipatedQueries"] == [
        "How should this be found later?",
        "alternate phrase",
        "second phrase",
    ]
    out = capsys.readouterr().out
    assert "stored:" in out


def test_memory_store_rejects_content_above_config_limit(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli, "_memory_max_content_length", lambda *_: 250)
    long_content = "x" * 251
    rc = cli._memory_store(
        [
            "--type",
            "GOTCHA",
            "--content",
            long_content,
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    # Numeric facts: the configured cap (250, not the dataclass default of
    # 200) and the actual content length (251) both come from the mocked
    # _memory_max_content_length() / len(content) -- proving the message
    # is never hardcoded to 200.
    assert "exceeds 250 chars" in err
    assert "(251)" in err
    assert "200" not in err
    # Path A (recommended): trim --content and move detail into --context.
    assert "--context" in err
    assert "recommended" in err
    # Path B: the exact copy-pasteable command to raise the cap, pre-filled
    # with the actual content length so the same content would be admitted.
    assert "simba config set memory.max_content_length 251" in err
    # Caveat: raising the cap is global and also loosens the terseness
    # asked of auto-extraction/digest/episode/reflection prompts elsewhere.
    assert "global" in err


def test_memory_store_uses_project_scoped_cap_override(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    """--project-path with a local .simba/config.toml override raises the
    cap for THAT store -- proving the CLI resolves the cap per-project via
    the real resolver, not a single frozen/global value. A sibling project
    directory with no override still hits the dataclass default (200) at
    the same content length, proving the override doesn't leak globally."""
    import httpx

    import simba.config
    import simba.hooks._memory_client

    monkeypatch.setattr(simba.config, "_global_path", lambda: tmp_path / "global.toml")
    project_dir = tmp_path / "proj"
    (project_dir / ".simba").mkdir(parents=True)
    (project_dir / ".simba" / "config.toml").write_text(
        "[memory]\nmax_content_length = 500\n"
    )
    other_dir = tmp_path / "other"
    other_dir.mkdir()

    monkeypatch.setattr(simba.hooks._memory_client, "daemon_url", lambda: "http://x")

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"status": "stored", "id": "mem_abc"}

    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp())

    content = "x" * 300  # over the 200 default, under the project's 500
    rc = cli._memory_store(
        [
            "--type",
            "GOTCHA",
            "--content",
            content,
            "--project-path",
            str(project_dir),
        ]
    )
    assert rc == 0
    capsys.readouterr()  # drain stdout from the successful store above

    rc_other = cli._memory_store(
        [
            "--type",
            "GOTCHA",
            "--content",
            content,
            "--project-path",
            str(other_dir),
        ]
    )
    assert rc_other == 1
    err = capsys.readouterr().err
    assert "exceeds 200 chars" in err
    assert "simba config set memory.max_content_length 300" in err


# ---------- memory prune ----------


def test_parse_duration_seconds_units() -> None:
    assert cli._parse_duration_seconds("14d") == 14 * 86400
    assert cli._parse_duration_seconds("48h") == 48 * 3600
    assert cli._parse_duration_seconds("2w") == 2 * 604800
    assert cli._parse_duration_seconds("30m") == 30 * 60
    assert cli._parse_duration_seconds("45s") == 45
    assert cli._parse_duration_seconds("7") == 7 * 86400  # bare number = days


def test_parse_duration_seconds_invalid() -> None:
    assert cli._parse_duration_seconds("") is None
    assert cli._parse_duration_seconds("abc") is None
    assert cli._parse_duration_seconds("xd") is None


def test_rlm_complete_marks_done(monkeypatch, capsys):
    import simba.rlm.jobs

    calls = {}
    monkeypatch.setattr(
        simba.rlm.jobs,
        "complete",
        lambda tid, project, n, **k: calls.update(tid=tid, n=n),
    )
    rc = cli._cmd_rlm(["complete", "sess-1", "--stored", "5"])
    assert rc == 0
    assert calls == {"tid": "sess-1", "n": 5}
    assert "complete" in capsys.readouterr().out.lower()


def test_rlm_run_llm_invokes_worker(monkeypatch, capsys):
    import simba.rlm.engine

    seen = {}

    def fake_worker(prompt_file, *, cwd, session_source, mark_rlm):
        seen.update(
            prompt_file=prompt_file,
            cwd=cwd,
            session_source=session_source,
            mark_rlm=mark_rlm,
        )
        return 4

    monkeypatch.setattr(simba.rlm.engine, "run_completion_from_file", fake_worker)
    rc = cli._cmd_rlm(
        [
            "run-llm",
            "--prompt-file",
            "/tmp/p.txt",
            "--cwd",
            "/proj",
            "--session-source",
            "T1",
            "--mark-rlm-complete",
        ]
    )
    assert rc == 0
    assert seen == {
        "prompt_file": "/tmp/p.txt",
        "cwd": "/proj",
        "session_source": "T1",
        "mark_rlm": True,
    }
    assert "stored 4" in capsys.readouterr().out


def test_rlm_run_llm_requires_prompt_file(capsys):
    rc = cli._cmd_rlm(["run-llm", "--cwd", "/proj"])
    assert rc == 1
    assert "prompt-file" in capsys.readouterr().err


def test_memory_maintain_runs_shadow_pass(monkeypatch, capsys) -> None:
    import httpx

    import simba.hooks._memory_client

    monkeypatch.setattr(simba.hooks._memory_client, "daemon_url", lambda: "http://x")
    captured: dict[str, object] = {}

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "at": "2026-07-03T00:00:00Z",
                "apply": False,
                "decay": {
                    "processed": 5731,
                    "updated": 4200,
                    "newly_dormant": 1800,
                    "revived": 0,
                    "errors": 0,
                    "dry_run": True,
                },
                "hygiene": {
                    "expired_count": 3,
                    "checked_count": 60,
                    "errors": 0,
                    "dry_run": True,
                },
                "errors": 0,
            }

    def _fake_post(url, json=None, timeout=0.0):
        captured["url"] = url
        captured["json"] = json
        return _Resp()

    monkeypatch.setattr(httpx, "post", _fake_post)

    rc = cli._memory_maintain([])

    assert rc == 0
    assert captured["url"] == "http://x/maintenance/run"
    assert captured["json"] == {}
    out = capsys.readouterr().out
    assert "shadow" in out
    assert "newly_dormant=1800" in out
    assert "expired=3" in out


def test_memory_maintain_apply_flag(monkeypatch, capsys) -> None:
    import httpx

    import simba.hooks._memory_client

    monkeypatch.setattr(simba.hooks._memory_client, "daemon_url", lambda: "http://x")
    captured: dict[str, object] = {}

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "at": "2026-07-03T00:00:00Z",
                "apply": True,
                "decay": {"skipped": True},
                "hygiene": {"skipped": True},
                "errors": 0,
            }

    def _fake_post(url, json=None, timeout=0.0):
        captured["json"] = json
        return _Resp()

    monkeypatch.setattr(httpx, "post", _fake_post)

    rc = cli._memory_maintain(["--apply"])

    assert rc == 0
    assert captured["json"] == {"apply": True}
    out = capsys.readouterr().out
    assert "apply" in out
    assert "decay: skipped" in out


def test_memory_maintain_rejects_unknown_option(capsys) -> None:
    rc = cli._memory_maintain(["--bogus"])
    assert rc == 1
    assert "Usage" in capsys.readouterr().err


def test_memory_restart_success(monkeypatch, capsys) -> None:
    import httpx

    import simba.hooks._memory_client

    monkeypatch.setattr(simba.hooks._memory_client, "daemon_url", lambda: "http://x")
    captured: dict[str, object] = {}

    class _Resp:
        status_code = 202

        def json(self) -> dict:
            return {"restarting": True, "pid": 4242}

    def _fake_post(url, timeout=0.0, headers=None):
        captured["url"] = url
        captured["headers"] = headers
        return _Resp()

    monkeypatch.setattr(httpx, "post", _fake_post)

    rc = cli._memory_restart([])

    assert rc == 0
    assert captured["url"] == "http://x/restart"
    assert captured["headers"]["X-Simba-Client"]
    out = capsys.readouterr().out
    assert "4242" in out
    assert "/health" in out


def test_memory_restart_reports_daemon_error(monkeypatch, capsys) -> None:
    import httpx

    import simba.hooks._memory_client

    monkeypatch.setattr(simba.hooks._memory_client, "daemon_url", lambda: "http://x")

    class _Resp:
        status_code = 503
        text = '{"error": "restart unavailable: no boot argv"}'

        def json(self) -> dict:
            return {"error": "restart unavailable: no boot argv"}

    monkeypatch.setattr(httpx, "post", lambda *a, **kw: _Resp())

    rc = cli._memory_restart([])

    assert rc == 1
    assert "no boot argv" in capsys.readouterr().err


def test_memory_restart_rejects_unknown_option(capsys) -> None:
    rc = cli._memory_restart(["--bogus"])
    assert rc == 1
    assert "Usage" in capsys.readouterr().err


def test_memory_normalize_scopes_dry_run(monkeypatch, capsys) -> None:
    import httpx

    import simba.hooks._memory_client

    monkeypatch.setattr(simba.hooks._memory_client, "daemon_url", lambda: "http://x")
    captured: dict[str, object] = {}

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "run": False,
                "changed": 2,
                "folds": [{"from": "/repo/.worktrees/wt", "to": "/repo", "count": 2}],
            }

    def _fake_post(url, json=None, timeout=0.0):
        captured["url"] = url
        captured["json"] = json
        return _Resp()

    monkeypatch.setattr(httpx, "post", _fake_post)

    rc = cli._memory_normalize_scopes([])

    assert rc == 0
    assert captured["url"] == "http://x/scopes/normalize"
    assert captured["json"] == {}
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "/repo/.worktrees/wt" in out
    assert "2 memories" in out


def test_memory_normalize_scopes_run_flag(monkeypatch, capsys) -> None:
    import httpx

    import simba.hooks._memory_client

    monkeypatch.setattr(simba.hooks._memory_client, "daemon_url", lambda: "http://x")
    captured: dict[str, object] = {}

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"run": True, "changed": 0, "folds": []}

    def _fake_post(url, json=None, timeout=0.0):
        captured["json"] = json
        return _Resp()

    monkeypatch.setattr(httpx, "post", _fake_post)

    rc = cli._memory_normalize_scopes(["--run"])

    assert rc == 0
    assert captured["json"] == {"run": True}
    assert "nothing to fold" in capsys.readouterr().out


def test_memory_promote_lists_candidates(monkeypatch, capsys) -> None:
    import httpx

    import simba.hooks._memory_client

    monkeypatch.setattr(simba.hooks._memory_client, "daemon_url", lambda: "http://x")
    captured: dict[str, object] = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "total": 1,
                "minUses": 3,
                "maxNoiseRatio": 0.5,
                "candidates": [
                    {
                        "id": "mem_hot",
                        "type": "GOTCHA",
                        "useCount": 4,
                        "noiseCount": 1,
                        "content": "always run the docker test runner",
                    }
                ],
            }

    def _fake_get(url, params=None, timeout=0.0):
        captured["url"] = url
        captured["params"] = params
        return _Resp()

    monkeypatch.setattr(httpx, "get", _fake_get)

    rc = cli._memory_promote([])

    assert rc == 0
    assert captured["url"] == "http://x/promotions/candidates"
    out = capsys.readouterr().out
    assert "mem_hot" in out
    assert "use=4" in out
    assert "docker test runner" in out


def test_memory_gaps_lists_known_unknowns(monkeypatch, capsys) -> None:
    import httpx

    import simba.hooks._memory_client

    monkeypatch.setattr(simba.hooks._memory_client, "daemon_url", lambda: "http://x")
    captured: dict[str, object] = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "minAsks": 3,
                "maxBest": 0.5,
                "gaps": [
                    {
                        "query": "how do we rotate the staging certs",
                        "askCount": 5,
                        "zeroCount": 4,
                        "bestScoreMax": 0.31,
                        "avgBestScore": 0.2,
                        "lastAsked": 1000.0,
                    }
                ],
            }

    def _fake_get(url, params=None, timeout=0.0):
        captured["url"] = url
        return _Resp()

    monkeypatch.setattr(httpx, "get", _fake_get)
    rc = cli._memory_gaps([])
    assert rc == 0
    assert captured["url"] == "http://x/demand/gaps"
    out = capsys.readouterr().out
    assert "rotate the staging certs" in out
    assert "asked=5" in out
    assert "best=0.31" in out


def test_memory_gaps_empty(monkeypatch, capsys) -> None:
    import httpx

    import simba.hooks._memory_client

    monkeypatch.setattr(simba.hooks._memory_client, "daemon_url", lambda: "http://x")

    class _Resp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"minAsks": 3, "maxBest": 0.5, "gaps": []}

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _Resp())
    rc = cli._memory_gaps([])
    assert rc == 0
    assert "no knowledge gaps" in capsys.readouterr().out


def test_memory_promote_empty(monkeypatch, capsys) -> None:
    import httpx

    import simba.hooks._memory_client

    monkeypatch.setattr(simba.hooks._memory_client, "daemon_url", lambda: "http://x")

    class _Resp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"total": 0, "minUses": 3, "maxNoiseRatio": 0.5, "candidates": []}

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _Resp())
    rc = cli._memory_promote([])
    assert rc == 0
    assert "no promotion candidates" in capsys.readouterr().out


def _write_curated(tmp_path) -> None:
    (tmp_path / "MEMORY.md").write_text("# Memory Index\n- ignored\n")
    (tmp_path / "triage-table.md").write_text(
        "---\n"
        "name: triage-table\n"
        'description: "User requires a triage table before batch-acting"\n'
        "metadata:\n"
        "  type: user\n"
        "---\n\n"
        "Full details about the triage-table requirement.\n"
    )
    (tmp_path / "push-trap.md").write_text(
        "---\n"
        "name: push-trap\n"
        "description: git push.default=upstream lands on MAIN from worktrees\n"
        "metadata:\n"
        "  type: project\n"
        "---\n\n"
        "Always push branch:refs/heads/branch.\n"
    )


def test_memory_import_curated_dry_run(tmp_path, monkeypatch, capsys) -> None:
    import httpx

    _write_curated(tmp_path)

    def _boom(*a, **kw):  # pragma: no cover - must not be reached
        raise AssertionError("dry run must not POST")

    monkeypatch.setattr(httpx, "post", _boom)
    rc = cli._memory_import_curated(["--dir", str(tmp_path), "--project-path", "/p"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "PREFERENCE" in out and "triage table" in out
    assert "DECISION" in out and "push.default" in out
    assert "MEMORY.md" not in out


def test_memory_import_curated_vetoes_secret_files(
    tmp_path, monkeypatch, capsys
) -> None:
    """The hippo-memory exploit: credential-bearing memory files must never
    be ingested — vetoed in the plan, skipped on --run, loudly reported."""
    import httpx

    _write_curated(tmp_path)
    (tmp_path / "leaky.md").write_text(
        "---\n"
        "name: leaky\n"
        "description: deploy credentials for the staging box\n"
        "metadata:\n"
        "  type: project\n"
        "---\n\n"
        "aws_access_key_id = AKIAIOSFODNN7EXAMPLE\n"
    )
    posted: list[dict] = []

    class _Resp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"id": "mem_new", "status": "stored"}

    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, json=None, timeout=0.0: posted.append(json) or _Resp(),
    )
    rc = cli._memory_import_curated(
        ["--dir", str(tmp_path), "--project-path", "/p", "--run"]
    )
    assert rc == 0
    assert len(posted) == 2  # the two clean files only
    assert all(
        "AKIA" not in (p.get("content", "") + p.get("context", "")) for p in posted
    )
    out = capsys.readouterr().out
    assert "vetoed 1" in out
    assert "leaky.md" in out
    assert "aws-access-key" in out


def test_memory_import_curated_requires_explicit_scope(tmp_path, capsys) -> None:
    """No silent global sharing (the other half of the hippo hazard): scope
    must be an explicit choice — --project-path or --global."""
    _write_curated(tmp_path)
    rc = cli._memory_import_curated(["--dir", str(tmp_path)])
    assert rc == 1
    assert "--project-path" in capsys.readouterr().err


def test_memory_import_curated_global_flag_allows_empty_scope(
    tmp_path, monkeypatch, capsys
) -> None:
    import httpx

    _write_curated(tmp_path)

    def _boom(*a, **kw):  # pragma: no cover
        raise AssertionError("dry run must not POST")

    monkeypatch.setattr(httpx, "post", _boom)
    rc = cli._memory_import_curated(["--dir", str(tmp_path), "--global"])
    assert rc == 0
    assert "dry-run" in capsys.readouterr().out


def test_memory_import_curated_run_posts(tmp_path, monkeypatch, capsys) -> None:
    import httpx

    import simba.hooks._memory_client

    monkeypatch.setattr(simba.hooks._memory_client, "daemon_url", lambda: "http://x")
    _write_curated(tmp_path)
    posted: list[dict] = []

    class _Resp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"id": "mem_new", "status": "stored"}

    def _fake_post(url, json=None, timeout=0.0):
        posted.append({"url": url, "json": json})
        return _Resp()

    monkeypatch.setattr(httpx, "post", _fake_post)
    rc = cli._memory_import_curated(
        ["--dir", str(tmp_path), "--project-path", "/p", "--run"]
    )
    assert rc == 0
    assert len(posted) == 2
    payloads = {p["json"]["content"]: p["json"] for p in posted}
    pref = payloads["User requires a triage table before batch-acting"]
    assert pref["type"] == "PREFERENCE"
    assert pref["trustSource"] == "user_confirmed"
    assert pref["captureOrigin"] == "curated_import"
    assert pref["projectPath"] == "/p"
    assert "stored 2" in capsys.readouterr().out


def test_memory_import_curated_requires_dir(capsys) -> None:
    rc = cli._memory_import_curated([])
    assert rc == 1
    assert "Usage" in capsys.readouterr().err


def test_memory_import_curated_run_writes_marker(tmp_path, monkeypatch, capsys) -> None:
    """Spec 33 R4: a successful --run stamps .simba/curated-import.json with
    the curated dir + import time, so SessionStart can nudge a re-import once
    the curated MEMORY.md changes again."""
    import time as time_mod

    import httpx

    _write_curated(tmp_path)
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    class _Resp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"id": "mem_new", "status": "stored"}

    monkeypatch.setattr(httpx, "post", lambda url, json=None, timeout=0.0: _Resp())
    before = time_mod.time()
    rc = cli._memory_import_curated(
        ["--dir", str(tmp_path), "--project-path", str(project_dir), "--run"]
    )
    after = time_mod.time()
    assert rc == 0

    marker_path = project_dir / ".simba" / "curated-import.json"
    assert marker_path.exists()
    marker = json.loads(marker_path.read_text())
    assert marker["dir"] == str(tmp_path.resolve())
    assert before <= marker["last_import_at"] <= after


def test_memory_import_curated_dry_run_does_not_write_marker(
    tmp_path, monkeypatch, capsys
) -> None:
    import httpx

    _write_curated(tmp_path)
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    def _boom(*a, **kw):  # pragma: no cover - must not be reached
        raise AssertionError("dry run must not POST")

    monkeypatch.setattr(httpx, "post", _boom)
    rc = cli._memory_import_curated(
        ["--dir", str(tmp_path), "--project-path", str(project_dir)]
    )
    assert rc == 0
    assert not (project_dir / ".simba" / "curated-import.json").exists()


def test_memory_import_curated_partial_errors_does_not_write_marker(
    tmp_path, monkeypatch, capsys
) -> None:
    """A run with any store errors isn't "successful" — skip the marker so
    the SessionStart nudge keeps firing until a clean run actually lands."""
    import httpx

    _write_curated(tmp_path)
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, json=None, timeout=0.0: (_ for _ in ()).throw(
            httpx.ConnectError("refused")
        ),
    )
    rc = cli._memory_import_curated(
        ["--dir", str(tmp_path), "--project-path", str(project_dir), "--run"]
    )
    assert rc == 1
    out = capsys.readouterr().out
    assert "errors 2 (of 2)" in out
    assert not (project_dir / ".simba" / "curated-import.json").exists()


def test_memory_compact_dry_run_reports_snapshot(monkeypatch, capsys) -> None:
    import httpx

    import simba.hooks._memory_client

    monkeypatch.setattr(simba.hooks._memory_client, "daemon_url", lambda: "http://x")
    captured: dict[str, object] = {}

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "status": "dry_run",
                "retentionSeconds": 86400,
                "before": {
                    "rows": 6512,
                    "liveBytes": 30_013_613,
                    "onDiskBytes": 23 * 1024**3,
                    "versions": 21_592,
                    "fragments": 12,
                },
            }

    def _fake_post(url, params=None, timeout=0.0):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(httpx, "post", _fake_post)

    rc = cli._memory_compact([])

    assert rc == 0
    assert captured["url"] == "http://x/compact"
    assert captured["params"]["dry_run"] == "true"
    assert captured["params"]["older_than_seconds"] == 86400
    out = capsys.readouterr().out
    assert "dry-run:" in out
    assert "disk=23.0 GiB" in out
    assert "versions=21592" in out


def test_memory_compact_run_passes_options(monkeypatch, capsys) -> None:
    import httpx

    import simba.hooks._memory_client

    monkeypatch.setattr(simba.hooks._memory_client, "daemon_url", lambda: "http://x")
    captured: dict[str, object] = {}

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "status": "compacted",
                "before": {"rows": 1, "liveBytes": 1024, "onDiskBytes": 4096},
                "after": {"rows": 1, "liveBytes": 1024, "onDiskBytes": 2048},
            }

    def _fake_post(url, params=None, timeout=0.0):
        captured["params"] = params
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(httpx, "post", _fake_post)

    rc = cli._memory_compact(["--run", "--older-than", "2h", "--delete-unverified"])

    assert rc == 0
    assert captured["params"]["dry_run"] == "false"
    assert captured["params"]["older_than_seconds"] == 7200
    assert captured["params"]["delete_unverified"] == "true"
    assert captured["timeout"] == 3600.0
    assert "compacted:" in capsys.readouterr().out


def test_memory_compact_rejects_bad_retention(capsys) -> None:
    rc = cli._memory_compact(["--older-than", "bogus"])
    assert rc == 1
    assert "invalid --older-than" in capsys.readouterr().err


def test_memory_prune_requires_a_filter(capsys) -> None:
    rc = cli._memory_prune([])
    assert rc == 1
    assert "requires at least one filter" in capsys.readouterr().err


def test_memory_prune_invalid_older_than(capsys) -> None:
    rc = cli._memory_prune(["--older-than", "bogus"])
    assert rc == 1
    assert "invalid --older-than" in capsys.readouterr().err


def test_memory_prune_deletes_only_old_matches(monkeypatch, capsys) -> None:
    import time

    import httpx

    import simba.hooks._memory_client

    monkeypatch.setattr(simba.hooks._memory_client, "daemon_url", lambda: "http://x")

    old = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 30 * 86400))
    recent = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 86400))
    listing = {
        "memories": [
            {
                "id": "mem_old",
                "type": "TOOL_RULE",
                "content": "stale",
                "confidence": 0.85,
                "createdAt": old,
            },
            {
                "id": "mem_new",
                "type": "TOOL_RULE",
                "content": "fresh",
                "confidence": 0.85,
                "createdAt": recent,
            },
        ]
    }

    class _ListResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return listing

    captured: dict[str, object] = {}

    def _fake_get(url, params=None, timeout=0.0):
        captured["params"] = params
        return _ListResp()

    deleted: list[str] = []

    class _DelResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"status": "deleted"}

    def _fake_delete(url, timeout=0.0):
        deleted.append(url)
        return _DelResp()

    monkeypatch.setattr(httpx, "get", _fake_get)
    monkeypatch.setattr(httpx, "delete", _fake_delete)

    rc = cli._memory_prune(["--type", "TOOL_RULE", "--older-than", "14d"])
    assert rc == 0
    assert len(deleted) == 1
    assert "mem_old" in deleted[0]
    assert captured["params"]["type"] == "TOOL_RULE"
    assert "pruned 1/1" in capsys.readouterr().out


def test_memory_prune_dry_run_deletes_nothing(monkeypatch, capsys) -> None:
    import httpx

    import simba.hooks._memory_client

    monkeypatch.setattr(simba.hooks._memory_client, "daemon_url", lambda: "http://x")
    listing = {
        "memories": [
            {
                "id": "mem_a",
                "type": "TOOL_RULE",
                "content": "x",
                "confidence": 0.85,
                "createdAt": None,
            }
        ]
    }

    class _ListResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return listing

    called = {"deleted": False}

    def _no_delete(*a, **k):
        called["deleted"] = True

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _ListResp())
    monkeypatch.setattr(httpx, "delete", _no_delete)

    rc = cli._memory_prune(["--type", "TOOL_RULE", "--dry-run"])
    assert rc == 0
    assert called["deleted"] is False
    assert "dry-run" in capsys.readouterr().out


def test_memory_prune_max_confidence_filter(monkeypatch, capsys) -> None:
    import httpx

    import simba.hooks._memory_client

    monkeypatch.setattr(simba.hooks._memory_client, "daemon_url", lambda: "http://x")
    listing = {
        "memories": [
            {
                "id": "mem_lo",
                "type": "TOOL_RULE",
                "content": "lo",
                "confidence": 0.85,
                "createdAt": None,
            },
            {
                "id": "mem_hi",
                "type": "DECISION",
                "content": "hi",
                "confidence": 0.97,
                "createdAt": None,
            },
        ]
    }

    class _ListResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return listing

    deleted: list[str] = []

    class _DelResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {}

    def _fake_delete(url, timeout=0.0):
        deleted.append(url)
        return _DelResp()

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _ListResp())
    monkeypatch.setattr(httpx, "delete", _fake_delete)

    rc = cli._memory_prune(["--max-confidence", "0.9"])
    assert rc == 0
    assert len(deleted) == 1
    assert "mem_lo" in deleted[0]


def test_memory_supersession_prints_chain(
    tmp_path: pathlib.Path, monkeypatch, capsys
) -> None:
    import simba.db
    import simba.memory.supersession as supersession

    monkeypatch.chdir(tmp_path)
    with simba.db.connect(tmp_path):
        supersession.append_event(
            old_id="mem_old",
            new_id="mem_new",
            project_path="/repo",
            memory_type="PATTERN",
            similarity=0.91,
            reason="near_duplicate_same_type",
            provenance="{}",
            now=1000.0,
        )

    rc = cli._memory_supersession(["mem_old"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "mem_old -> mem_new" in out
    assert "sim=0.910" in out


def test_memory_supersession_confirm_pending(
    tmp_path: pathlib.Path, monkeypatch, capsys
) -> None:
    import simba.db
    import simba.memory.supersession as supersession

    monkeypatch.chdir(tmp_path)
    with simba.db.connect(tmp_path):
        pending = supersession.append_event(
            old_id="mem_old",
            new_id="mem_new",
            project_path="/repo",
            memory_type="PATTERN",
            similarity=0.91,
            reason="near_duplicate_same_type",
            provenance="{}",
            status=supersession.STATUS_PENDING,
            old_trust_score=1.2,
            new_trust_score=0.6,
            now=1000.0,
        )

    rc = cli._memory_supersession(["confirm", str(pending.id)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "confirmed supersession" in out
    with simba.db.connect(tmp_path):
        assert supersession.latest_successors(["mem_old"])["mem_old"].new_id == (
            "mem_new"
        )


def test_memory_supersession_reject_pending(
    tmp_path: pathlib.Path, monkeypatch, capsys
) -> None:
    import simba.db
    import simba.memory.supersession as supersession

    monkeypatch.chdir(tmp_path)
    with simba.db.connect(tmp_path):
        pending = supersession.append_event(
            old_id="mem_old",
            new_id="mem_new",
            project_path="/repo",
            memory_type="PATTERN",
            similarity=0.91,
            reason="near_duplicate_same_type",
            provenance="{}",
            status=supersession.STATUS_PENDING,
            old_trust_score=1.2,
            new_trust_score=0.6,
            now=1000.0,
        )

    rc = cli._memory_supersession(["reject", str(pending.id)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "rejected supersession" in out
    with simba.db.connect(tmp_path):
        assert supersession.latest_successors(["mem_old"]) == {}


def test_latest_transcript_metadata_prefers_codex_sessions(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    codex_home = tmp_path / ".codex"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    session_dir = codex_home / "sessions" / "2026" / "02" / "20"
    session_dir.mkdir(parents=True, exist_ok=True)
    codex_jsonl = session_dir / "rollout-2026-02-20T08-33-13-abc123.jsonl"
    codex_jsonl.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {"id": "abc123", "cwd": "/tmp/codex-project"},
            }
        )
        + "\n"
    )

    claude_latest = tmp_path / ".claude" / "transcripts" / "latest.json"
    claude_latest.parent.mkdir(parents=True, exist_ok=True)
    claude_latest.write_text(
        json.dumps(
            {
                "session_id": "claude-session",
                "project_path": "/tmp/claude-project",
                "transcript_path": "/tmp/claude.md",
                "status": "pending_extraction",
            }
        )
    )

    meta = cli._latest_transcript_metadata()
    assert meta is not None
    assert meta["source"] == "codex"
    assert meta["session_id"] == "abc123"
    assert meta["project_path"] == "/tmp/codex-project"
    assert meta["transcript_path"] == str(codex_jsonl)


def test_cmd_sessions_index_latest_and_search_json(
    tmp_path: pathlib.Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    codex_home = tmp_path / ".codex"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    transcript = _write_codex_session(
        codex_home,
        session_id="codex-session-2",
        cwd="/tmp/codex-project",
        text="Exact recovery marker: RuntimeError bad state in src/session.py:12",
    )

    rc = cli._cmd_sessions(["index", "--latest", "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["session_id"] == "codex-session-2"
    assert payload["transcript_path"] == str(transcript)
    assert payload["message_count"] == 1

    rc = cli._cmd_sessions(
        ["search", "RuntimeError", "bad", "state", "src/session.py:12", "--json"]
    )

    assert rc == 0
    hits = json.loads(capsys.readouterr().out)
    assert hits[0]["session_id"] == "codex-session-2"
    assert hits[0]["message_span"] == [0, 0]
    assert "src/session.py:12" in hits[0]["file_refs"]


def test_cmd_sessions_path_search_respects_project_filter(
    tmp_path: pathlib.Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.chdir(tmp_path)
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "s1", "cwd": "/repo/a"}})
        + "\n"
        + json.dumps({"message": {"role": "user", "content": "FILTER_TOKEN"}})
        + "\n"
    )

    assert cli._cmd_sessions(["index", "--path", str(transcript)]) == 0
    capsys.readouterr()

    assert (
        cli._cmd_sessions(
            ["search", "FILTER_TOKEN", "--project-path", "/repo/other", "--json"]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == []


def test_codex_extract_does_not_fallback_to_claude_metadata(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    codex_home = tmp_path / ".codex"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    codex_home.mkdir(parents=True, exist_ok=True)

    claude_latest = tmp_path / ".claude" / "transcripts" / "latest.json"
    claude_latest.parent.mkdir(parents=True, exist_ok=True)
    claude_latest.write_text(
        json.dumps(
            {
                "session_id": "claude-session",
                "project_path": "/tmp/claude-project",
                "transcript_path": "/tmp/claude.md",
                "status": "pending_extraction",
            }
        )
    )

    rc = cli._cmd_codex_extract([])
    assert rc == 1


def test_cmd_codex_extract_scopes_to_current_project(
    tmp_path: pathlib.Path,
    monkeypatch,
    capsys,
) -> None:
    # Cross-wiring guard: a NEWER session belonging to another project must not
    # be the one codex-extract surfaces. It scopes to the current project (cwd),
    # mirroring transcripts.find_pending on the Claude side.
    monkeypatch.setenv("HOME", str(tmp_path))
    codex_home = tmp_path / ".codex"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    here = tmp_path / "this-project"
    here.mkdir()
    other = tmp_path / "other-project"
    other.mkdir()

    # current project's session (written first -> older mtime)
    _write_codex_session(codex_home, session_id="mine", cwd=str(here), text="x")
    # another project's session, globally NEWER (written second -> newer mtime)
    _write_codex_session(codex_home, session_id="theirs", cwd=str(other), text="y")

    monkeypatch.chdir(here)
    rc = cli._cmd_codex_extract([])
    out = capsys.readouterr().out
    assert rc == 0
    assert 'session-source "mine"' in out  # current project's session
    assert "theirs" not in out  # never the newer cross-project one


def test_cmd_codex_extract_uses_codex_session_path(
    tmp_path: pathlib.Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    codex_home = tmp_path / ".codex"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    # Default session cwd == live cwd, so the project-scoped extract matches it.
    codex_jsonl = _write_codex_session(codex_home, session_id="abc123")

    rc = cli._cmd_codex_extract([])
    assert rc == 0
    out = capsys.readouterr().out
    assert str(codex_jsonl) in out
    assert 'session-source "abc123"' in out


def test_latest_codex_transcript_uses_rollout_filename_timestamp(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    codex_home = tmp_path / ".codex"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    session_dir = codex_home / "sessions" / "2026" / "02" / "20"
    session_dir.mkdir(parents=True, exist_ok=True)

    older = session_dir / "rollout-2026-02-20T08-33-13-old111.jsonl"
    newer = session_dir / "rollout-2026-02-20T19-40-53-new222.jsonl"
    older.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {"id": "old111", "cwd": "/tmp/codex-project"},
            }
        )
        + "\n"
    )
    newer.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {"id": "new222", "cwd": "/tmp/codex-project"},
            }
        )
        + "\n"
    )
    # Invert mtimes to ensure filename timestamp, not mtime, drives selection.
    os.utime(older, (2_000_000_000, 2_000_000_000))  # newer mtime
    os.utime(newer, (1_000_000_000, 1_000_000_000))  # older mtime

    meta = cli._latest_codex_transcript_metadata()
    assert meta is not None
    assert meta["session_id"] == "new222"
    assert meta["transcript_path"] == str(newer)


class TestCodexProjectHooks:
    def test_write_creates_hooks_json(self, tmp_path: pathlib.Path) -> None:
        cli._write_codex_project_hooks(tmp_path)
        hooks_path = tmp_path / ".codex" / "hooks.json"
        assert hooks_path.exists()
        data = json.loads(hooks_path.read_text())
        assert set(data["hooks"]) == set(cli._CODEX_HOOK_EVENTS)
        assert "PreCompact" in data["hooks"]

    def test_write_has_correct_matchers(self, tmp_path: pathlib.Path) -> None:
        cli._write_codex_project_hooks(tmp_path)
        data = json.loads((tmp_path / ".codex" / "hooks.json").read_text())
        hooks = data["hooks"]
        assert hooks["SessionStart"][0]["matcher"] == cli._CODEX_SESSION_MATCHER
        assert "compact" in hooks["SessionStart"][0]["matcher"]
        assert hooks["PreCompact"][0]["matcher"] == cli._CODEX_COMPACT_MATCHER
        assert hooks["PreToolUse"][0]["matcher"] == cli._CODEX_TOOL_MATCHER
        assert hooks["PermissionRequest"][0]["matcher"] == cli._CODEX_TOOL_MATCHER
        assert "matcher" not in hooks["UserPromptSubmit"][0]
        assert "matcher" not in hooks["Stop"][0]

    def test_write_timeouts_are_seconds(self, tmp_path: pathlib.Path) -> None:
        cli._write_codex_project_hooks(tmp_path)
        data = json.loads((tmp_path / ".codex" / "hooks.json").read_text())
        for entries in data["hooks"].values():
            for entry in entries:
                for h in entry["hooks"]:
                    assert h["timeout"] < 100, f"timeout looks like ms: {h}"

    def test_remove_deletes_file(self, tmp_path: pathlib.Path) -> None:
        cli._write_codex_project_hooks(tmp_path)
        changed = cli._write_codex_project_hooks(tmp_path, remove=True)
        assert changed
        assert not (tmp_path / ".codex" / "hooks.json").exists()

    def test_remove_missing_file_returns_false(self, tmp_path: pathlib.Path) -> None:
        changed = cli._write_codex_project_hooks(tmp_path, remove=True)
        assert not changed

    def test_install_writes_codex_hooks(
        self, tmp_path: pathlib.Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cli, "_install_skills", lambda d: 0)
        cli._cmd_install([])
        assert (tmp_path / ".codex" / "hooks.json").exists()

    def test_install_remove_deletes_codex_hooks(
        self, tmp_path: pathlib.Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cli, "_install_skills", lambda d: 0)
        monkeypatch.setattr(cli, "_remove_skills", lambda d: 0)
        cli._cmd_install([])
        assert (tmp_path / ".codex" / "hooks.json").exists()
        cli._cmd_install(["--remove"])
        assert not (tmp_path / ".codex" / "hooks.json").exists()

    def test_install_global_does_not_write_codex_hooks(
        self, tmp_path: pathlib.Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        global_settings = tmp_path / ".claude" / "settings.json"
        monkeypatch.setattr(cli, "_GLOBAL_SETTINGS", global_settings)
        monkeypatch.setattr(cli, "_install_skills", lambda d: 0)
        cli._cmd_install(["--global"])
        assert not (tmp_path / ".codex" / "hooks.json").exists()

    def test_install_migrates_project_local_codex_config(
        self, tmp_path: pathlib.Path, monkeypatch
    ) -> None:
        import tomllib

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cli, "_install_skills", lambda d: 0)
        project_cfg = tmp_path / ".codex" / "config.toml"
        project_cfg.parent.mkdir(parents=True)
        project_cfg.write_text("[features]\ncodex_hooks = true\n")
        cli._cmd_install([])
        cfg = tomllib.loads(project_cfg.read_text())
        assert cfg["features"] == {"hooks": True}

    def test_install_no_op_when_local_codex_config_absent(
        self, tmp_path: pathlib.Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cli, "_install_skills", lambda d: 0)
        cli._cmd_install([])
        assert not (tmp_path / ".codex" / "config.toml").exists()


class TestCodexProjectHooksHeal:
    """Re-running install over an existing hooks.json heals, not overwrites.

    An old ``simba codex-install``/``simba install`` (before ``--client
    codex`` existed) wrote flagless ``simba hook <Event>`` commands. Codex's
    hook subprocesses set no reliable client marker, so those commands
    resolve to the claude-code default and get claude's Stop shape, which
    Codex's schema rejects ("hook returned invalid stop hook JSON output").
    Re-installing must heal the missing flag in place without clobbering an
    already-flagged command or a non-simba command another tool registered.
    """

    def _seed_hooks_json(self, project_dir: pathlib.Path) -> pathlib.Path:
        hooks_path = project_dir / ".codex" / "hooks.json"
        hooks_path.parent.mkdir(parents=True, exist_ok=True)
        hooks_path.write_text(
            json.dumps(
                {
                    "hooks": {
                        "Stop": [
                            {
                                "hooks": [
                                    {
                                        "type": "command",
                                        "command": "simba hook Stop",
                                        "timeout": 5,
                                    }
                                ]
                            }
                        ],
                        "PreToolUse": [
                            {
                                "matcher": "Bash|apply_patch|Edit|Write",
                                "hooks": [
                                    {
                                        "type": "command",
                                        "command": (
                                            "simba hook PreToolUse --client codex"
                                        ),
                                        "timeout": 5,
                                    }
                                ],
                            }
                        ],
                        "OtherTool": [
                            {
                                "hooks": [
                                    {
                                        "type": "command",
                                        "command": "other-tool run --flag",
                                        "timeout": 10,
                                    }
                                ]
                            }
                        ],
                    }
                },
                indent=2,
            )
            + "\n"
        )
        return hooks_path

    def _commands_by_event(self, hooks_path: pathlib.Path) -> dict[str, str]:
        data = json.loads(hooks_path.read_text())
        return {
            event: entries[0]["hooks"][0]["command"]
            for event, entries in data["hooks"].items()
        }

    def test_flagless_simba_command_gains_flag(self, tmp_path: pathlib.Path) -> None:
        hooks_path = self._seed_hooks_json(tmp_path)
        cli._write_codex_project_hooks(tmp_path)
        cmds = self._commands_by_event(hooks_path)
        assert cmds["Stop"] == "simba hook Stop --client codex"

    def test_already_flagged_simba_command_unchanged(
        self, tmp_path: pathlib.Path
    ) -> None:
        hooks_path = self._seed_hooks_json(tmp_path)
        cli._write_codex_project_hooks(tmp_path)
        cmds = self._commands_by_event(hooks_path)
        assert cmds["PreToolUse"] == "simba hook PreToolUse --client codex"

    def test_non_simba_command_byte_identical(self, tmp_path: pathlib.Path) -> None:
        hooks_path = self._seed_hooks_json(tmp_path)
        cli._write_codex_project_hooks(tmp_path)
        cmds = self._commands_by_event(hooks_path)
        assert cmds["OtherTool"] == "other-tool run --flag"

    def test_heal_preserves_non_simba_event_key(self, tmp_path: pathlib.Path) -> None:
        # A merge must not drop structure simba didn't write, even if it
        # isn't one of simba's own recognized hook events.
        hooks_path = self._seed_hooks_json(tmp_path)
        cli._write_codex_project_hooks(tmp_path)
        data = json.loads(hooks_path.read_text())
        assert "OtherTool" in data["hooks"]

    def test_install_heals_legacy_flagless_command(
        self, tmp_path: pathlib.Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(cli, "_install_skills", lambda d: 0)
        hooks_path = self._seed_hooks_json(tmp_path)
        cli._cmd_install([])
        cmds = self._commands_by_event(hooks_path)
        assert cmds["Stop"] == "simba hook Stop --client codex"
        assert cmds["PreToolUse"] == "simba hook PreToolUse --client codex"
        assert cmds["OtherTool"] == "other-tool run --flag"


def test_apply_codex_feature_flag_skips_when_create_disabled(
    tmp_path: pathlib.Path,
) -> None:
    missing = tmp_path / "nonexistent" / "config.toml"
    status = cli._apply_codex_feature_flag(missing, create_if_missing=False)
    assert status == "not-present"
    assert not missing.exists()


def test_rlm_digest_dispatches(monkeypatch, capsys):
    import simba.rlm.engine
    import simba.rlm.jobs

    dispatched = {}

    class _Engine:
        def digest(self, tid, query, *, cwd):
            dispatched["tid"] = tid
            dispatched["cwd"] = cwd

    monkeypatch.setattr(simba.rlm.engine, "get_engine", lambda cfg: _Engine())
    monkeypatch.setattr(simba.rlm.jobs, "claim", lambda *a, **k: True)

    rc = cli._cmd_rlm(["digest", "sess-1"])
    assert rc == 0
    assert dispatched["tid"] == "sess-1"
    assert "dispatched" in capsys.readouterr().out


def test_rlm_digest_no_engine(monkeypatch, capsys):
    import simba.rlm.engine

    monkeypatch.setattr(simba.rlm.engine, "get_engine", lambda cfg: None)
    rc = cli._cmd_rlm(["digest", "sess-1"])
    assert rc == 1
    assert "engine" in capsys.readouterr().out.lower()


def test_rlm_digest_dedup_skips(monkeypatch, capsys):
    import simba.rlm.engine
    import simba.rlm.jobs

    class _Engine:
        def digest(self, tid, query, *, cwd):
            raise AssertionError("should not dispatch when already claimed")

    monkeypatch.setattr(simba.rlm.engine, "get_engine", lambda cfg: _Engine())
    monkeypatch.setattr(simba.rlm.jobs, "claim", lambda *a, **k: False)
    rc = cli._cmd_rlm(["digest", "sess-1"])
    assert rc == 0
    assert "already" in capsys.readouterr().out.lower()


def test_db_facts_shows_occurred_at(tmp_path, monkeypatch, capsys) -> None:
    import simba.db
    import simba.kg.store

    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)
    simba.kg.store.kg_add(
        "alpha", "rel", "beta", "proof", project_path="p1", occurred_at="2025-03-01"
    )

    rc = cli._db_facts(tmp_path, 10)
    out = capsys.readouterr().out
    assert rc == 0
    assert "alpha rel beta" in out
    assert "2025-03-01" in out


def test_eval_ambiguity_generate_dispatches_codegen(monkeypatch, capsys) -> None:
    import simba.eval.ambiguity_codegen as codegen

    calls: list[tuple[str, str]] = []

    def _fake_generate_and_run(case, *, language):
        calls.append((case.id, language))
        return (
            codegen.GeneratedProgram(
                case_id=case.id,
                language=language,
                code="ANSWER_SPACE = {'count': 1}",
            ),
            codegen.GeneratedRun(
                case_id=case.id,
                language=language,
                answer_space={"lower": 1, "upper": 1},
                ok=True,
            ),
        )

    monkeypatch.setattr(codegen, "generate_and_run", _fake_generate_and_run)

    rc = cli._eval_ambiguity(["--generate", "python"])
    out = capsys.readouterr().out

    assert rc == 0
    assert calls
    assert {language for _, language in calls} == {"python"}
    assert "generated python ok" in out


# ---------- simba db reconcile ----------


def _db_reconcile_mem_row(mid: str, content: str, **over) -> dict:
    base = {
        "id": mid,
        "type": "GOTCHA",
        "content": content,
        "context": "",
        "tags": "[]",
        "confidence": 0.85,
        "sessionSource": "",
        "projectPath": "proj-1",
        "createdAt": "2026-01-01T00:00:00Z",
        "lastAccessedAt": "2026-01-01T00:00:00Z",
        "accessCount": 0,
        "vector": [0.1] * 768,
    }
    base.update(over)
    return base


def test_db_reconcile_errors_when_no_lance_store(
    tmp_path: pathlib.Path, monkeypatch, capsys
) -> None:
    import simba.memory.reconcile

    data_dir = tmp_path / ".simba" / "memory"
    monkeypatch.setattr(
        simba.memory.reconcile, "resolve_data_dir", lambda cwd: data_dir
    )

    rc = cli._db_reconcile(tmp_path, run=False)

    assert rc == 1
    assert "no LanceDB store" in capsys.readouterr().err


def test_db_reconcile_dry_run_reports_drift_and_changes_nothing(
    tmp_path: pathlib.Path, monkeypatch, capsys
) -> None:
    import asyncio

    import simba.memory.fts as fts
    import simba.memory.reconcile

    # Isolation: this dev machine's global ~/.config/simba/config.toml sets
    # memory.db_path to the real project store -- resolve_data_dir must be
    # pinned to tmp_path or this test would touch live `.simba/` data.
    data_dir = tmp_path / ".simba" / "memory"
    data_dir.mkdir(parents=True)
    monkeypatch.setattr(
        simba.memory.reconcile, "resolve_data_dir", lambda cwd: data_dir
    )

    async def _seed() -> None:
        import lancedb

        db = await lancedb.connect_async(str(data_dir / "memories.lance"))
        await db.create_table(
            "memories",
            [
                _db_reconcile_mem_row("m1", "ruff lints the python code"),
                _db_reconcile_mem_row("m2", "pytest runs the suite"),
                _db_reconcile_mem_row("m3", "uv manages the venv"),
            ],
        )

    asyncio.run(_seed())
    with fts.connect(data_dir / fts.FTS_FILENAME):
        fts.upsert(_db_reconcile_mem_row("m1", "ruff lints the python code"))

    rc = cli._db_reconcile(tmp_path, run=False)

    assert rc == 0
    out = capsys.readouterr().out
    assert "missing-fts" in out
    assert "m2" in out
    assert "m3" in out
    with fts.connect(data_dir / fts.FTS_FILENAME):
        assert fts.count() == 1  # dry-run must change nothing


def test_db_reconcile_run_repairs_missing_fts_only(
    tmp_path: pathlib.Path, monkeypatch, capsys
) -> None:
    import asyncio

    import simba.memory.fts as fts
    import simba.memory.reconcile

    data_dir = tmp_path / ".simba" / "memory"
    data_dir.mkdir(parents=True)
    monkeypatch.setattr(
        simba.memory.reconcile, "resolve_data_dir", lambda cwd: data_dir
    )

    async def _seed() -> None:
        import lancedb

        db = await lancedb.connect_async(str(data_dir / "memories.lance"))
        await db.create_table(
            "memories",
            [
                _db_reconcile_mem_row("m1", "ruff lints the python code"),
                _db_reconcile_mem_row("m2", "pytest runs the suite"),
            ],
        )

    asyncio.run(_seed())
    with fts.connect(data_dir / fts.FTS_FILENAME):
        fts.upsert(_db_reconcile_mem_row("m1", "ruff lints the python code"))

    rc = cli._db_reconcile(tmp_path, run=True)

    assert rc == 0
    assert "repaired" in capsys.readouterr().out
    with fts.connect(data_dir / fts.FTS_FILENAME):
        assert fts.count() == 2

    rc2 = cli._db_reconcile(tmp_path, run=False)
    assert rc2 == 0
    assert "missing-fts: 0" in capsys.readouterr().out


def test_cmd_db_reconcile_dispatches_run_flag(monkeypatch) -> None:
    calls: list[tuple[pathlib.Path, bool]] = []
    monkeypatch.setattr(
        cli, "_db_reconcile", lambda cwd, run: calls.append((cwd, run)) or 0
    )

    rc = cli._cmd_db(["reconcile", "--run"])

    assert rc == 0
    assert calls[0][1] is True


def test_cmd_db_reconcile_defaults_to_dry_run(monkeypatch) -> None:
    calls: list[tuple[pathlib.Path, bool]] = []
    monkeypatch.setattr(
        cli, "_db_reconcile", lambda cwd, run: calls.append((cwd, run)) or 0
    )

    rc = cli._cmd_db(["reconcile"])

    assert rc == 0
    assert calls[0][1] is False
