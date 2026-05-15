"""Tests for top-level CLI helpers in simba.__main__."""

from __future__ import annotations

import json
import os
import pathlib

import simba.__main__ as cli


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
    config_path.write_text(
        'model = "gpt-5.5"\n\n[features]\nmulti_agent = true\n'
    )
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
    (codex_home / "config.toml").write_text(
        "[features]\nhooks = true\n"
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    assert cli._ensure_codex_feature_flag() == "already-set"


def test_ensure_codex_feature_flag_remove(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    import tomllib

    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    config_path = codex_home / "config.toml"
    config_path.write_text(
        "[features]\nmulti_agent = true\nhooks = true\n"
    )
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
    config_path.write_text(
        "[features]\nmulti_agent = true\ncodex_hooks = true\n"
    )
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
                "payload": {"id": "s1", "cwd": "/tmp/project"},
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
                "payload": {"id": "s1", "cwd": "/tmp/project"},
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

    rc = cli._cmd_codex_status([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "pending_extraction" in out
    assert "simba codex-extract" in out
    assert "transcript source: codex" in out


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


def test_cmd_codex_recall_prints_memories(monkeypatch, capsys) -> None:
    import simba.hooks._memory_client

    monkeypatch.setattr(
        simba.hooks._memory_client,
        "recall_memories",
        lambda query, project_path=None: [
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

    monkeypatch.setattr(cli, "_memory_max_content_length", lambda: 1000)
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
        ]
    )
    assert rc == 0
    assert captured["json"]["content"] == long_content
    out = capsys.readouterr().out
    assert "stored:" in out


def test_memory_store_rejects_content_above_config_limit(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli, "_memory_max_content_length", lambda: 250)
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
    assert "exceeds 250 chars" in err


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


def test_cmd_codex_extract_uses_codex_session_path(
    tmp_path: pathlib.Path,
    monkeypatch,
    capsys,
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

    def test_write_has_correct_matchers(self, tmp_path: pathlib.Path) -> None:
        cli._write_codex_project_hooks(tmp_path)
        data = json.loads((tmp_path / ".codex" / "hooks.json").read_text())
        hooks = data["hooks"]
        assert hooks["SessionStart"][0]["matcher"] == cli._CODEX_SESSION_MATCHER
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
