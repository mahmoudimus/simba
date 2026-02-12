"""Tests for top-level CLI helpers in simba.__main__."""

from __future__ import annotations

import json
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


def test_cmd_codex_extract_mark_done_updates_latest_target(
    tmp_path: pathlib.Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    meta_dir = tmp_path / ".claude" / "transcripts" / "s1"
    meta_dir.mkdir(parents=True, exist_ok=True)
    metadata = meta_dir / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "session_id": "s1",
                "project_path": "/tmp/project",
                "transcript_path": "/tmp/transcript.md",
                "status": "pending_extraction",
            }
        )
    )
    latest = tmp_path / ".claude" / "transcripts" / "latest.json"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.symlink_to(metadata)

    rc = cli._cmd_codex_extract(["--mark-done"])
    assert rc == 0

    data = json.loads(metadata.read_text())
    assert data["status"] == "extracted"


def test_cmd_codex_status_shows_pending_extraction(
    tmp_path: pathlib.Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    latest = tmp_path / ".claude" / "transcripts" / "latest.json"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(
        json.dumps(
            {
                "session_id": "s1",
                "transcript_path": "/tmp/transcript.md",
                "status": "pending_extraction",
            }
        )
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
