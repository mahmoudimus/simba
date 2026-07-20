"""Tests for the SessionStart hook module."""

from __future__ import annotations

import io
import json
import os
import subprocess
import unittest.mock

import pytest

import simba.hooks.session_start


class TestSessionStartHook:
    def test_returns_valid_json(self):
        with unittest.mock.patch(
            "simba.hooks.session_start._check_health", return_value=None
        ):
            result = json.loads(simba.hooks.session_start.main({}))
        assert "hookSpecificOutput" in result
        assert result["hookSpecificOutput"]["hookEventName"] == "SessionStart"

    def test_includes_tailor_context(self, tmp_path):
        with unittest.mock.patch(
            "simba.hooks.session_start._check_health", return_value=None
        ):
            result = json.loads(simba.hooks.session_start.main({"cwd": str(tmp_path)}))
        ctx = result["hookSpecificOutput"]["additionalContext"]
        # Tailor context includes time
        assert "Time:" in ctx

    def test_resets_signal_flag(self, tmp_path):
        """A fresh session clears any stale rules-signal flag so the first
        prompt re-injects the CORE block (spec 25)."""
        import simba.guardian.signal_flag as sf

        with unittest.mock.patch.object(sf, "_TMP_DIR", tmp_path):
            sf.record_signal("fresh-session", present=True)
            assert sf.flag_path("fresh-session").exists()
            with unittest.mock.patch(
                "simba.hooks.session_start._check_health", return_value=None
            ):
                simba.hooks.session_start.main(
                    {"cwd": str(tmp_path), "session_id": "fresh-session"}
                )
            assert not sf.flag_path("fresh-session").exists()

    def test_includes_memory_status_when_healthy(self):
        health = {"memoryCount": 42, "embeddingModel": "nomic-embed-text"}
        with unittest.mock.patch(
            "simba.hooks.session_start._check_health", return_value=health
        ):
            result = json.loads(simba.hooks.session_start.main({}))
        ctx = result["hookSpecificOutput"]["additionalContext"]
        assert "42 memories" in ctx
        assert "nomic-embed-text" in ctx

    def test_no_memory_status_when_unhealthy(self, tmp_path):
        with unittest.mock.patch(
            "simba.hooks.session_start._check_health", return_value=None
        ):
            result = json.loads(simba.hooks.session_start.main({"cwd": str(tmp_path)}))
        ctx = result["hookSpecificOutput"]["additionalContext"]
        assert "Semantic Memory" not in ctx

    def test_auto_starts_daemon_if_needed(self):
        health = {"memoryCount": 0, "embeddingModel": "nomic-embed-text"}
        with (
            unittest.mock.patch(
                "simba.hooks.session_start._check_health",
                side_effect=[None, health],
            ),
            unittest.mock.patch(
                "simba.hooks.session_start._auto_start_daemon",
                return_value=True,
            ),
        ):
            result = json.loads(simba.hooks.session_start.main({}))
        ctx = result["hookSpecificOutput"]["additionalContext"]
        assert "Semantic Memory" in ctx

    @pytest.mark.real_daemon_spawn
    def test_auto_start_daemon_redirects_stdio_to_log_file(self, tmp_path):
        """The spawned daemon's stdout/stderr must go to an append-mode
        .simba/memory/daemon.log --- never DEVNULL (live 2026-07-10: both
        the silent /restart failure and the 45GB /list blowup were buried
        until sampled via lsof, because both streams were discarded)."""
        captured: dict = {}

        def _fake_popen(args, **kwargs):
            captured.update(kwargs)
            return unittest.mock.MagicMock()

        with (
            unittest.mock.patch(
                "simba.hooks.session_start.subprocess.Popen",
                side_effect=_fake_popen,
            ),
            unittest.mock.patch(
                "simba.hooks.session_start._hooks_cfg",
                return_value=unittest.mock.MagicMock(
                    poll_attempts=1, poll_interval=0.0
                ),
            ),
            unittest.mock.patch(
                "simba.hooks.session_start._check_health", return_value=None
            ),
        ):
            simba.hooks.session_start._auto_start_daemon(tmp_path)

        log_path = tmp_path / ".simba" / "memory" / "daemon.log"
        assert log_path.exists()

        assert captured["stdout"] is not subprocess.DEVNULL
        assert captured["stderr"] is not subprocess.DEVNULL
        assert isinstance(captured["stdout"], io.IOBase)
        assert isinstance(captured["stderr"], io.IOBase)
        assert captured["stdout"].name == str(log_path)
        assert captured["stderr"].name == str(log_path)

    @pytest.mark.real_daemon_spawn
    def test_auto_start_daemon_appends_without_truncating_existing_log(self, tmp_path):
        """Append-only (CORE_INSTRUCTIONS.md): a pre-existing daemon.log ---
        from an earlier daemon lifetime --- must never be wiped by the next
        auto-start."""
        log_path = tmp_path / ".simba" / "memory" / "daemon.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("prior daemon output\n")

        with (
            unittest.mock.patch(
                "simba.hooks.session_start.subprocess.Popen",
                return_value=unittest.mock.MagicMock(),
            ),
            unittest.mock.patch(
                "simba.hooks.session_start._hooks_cfg",
                return_value=unittest.mock.MagicMock(
                    poll_attempts=1, poll_interval=0.0
                ),
            ),
            unittest.mock.patch(
                "simba.hooks.session_start._check_health", return_value=None
            ),
        ):
            simba.hooks.session_start._auto_start_daemon(tmp_path)

        assert log_path.read_text().startswith("prior daemon output\n")

    @pytest.mark.real_daemon_spawn
    def test_auto_start_daemon_injects_malloc_stack_logging_env(self, tmp_path):
        """2026-07-19: a 16.7GB RSS burst had no attributable allocator
        stacks because the daemon that ended up serving was an unarmed hook
        auto-start. When memory.malloc_stack_logging resolves True, the
        spawned daemon's env must carry MallocStackLogging=lite --- the env
        var macOS's malloc_history needs, and one that can only be set at
        process SPAWN, never injected into an already-running process."""
        captured: dict = {}

        def _fake_popen(args, **kwargs):
            captured.update(kwargs)
            return unittest.mock.MagicMock()

        with (
            unittest.mock.patch(
                "simba.hooks.session_start.subprocess.Popen",
                side_effect=_fake_popen,
            ),
            unittest.mock.patch(
                "simba.hooks.session_start._hooks_cfg",
                return_value=unittest.mock.MagicMock(
                    poll_attempts=1, poll_interval=0.0
                ),
            ),
            unittest.mock.patch(
                "simba.hooks.session_start._check_health", return_value=None
            ),
            unittest.mock.patch(
                "simba.memory.config.resolve_malloc_stack_logging",
                return_value=True,
            ),
        ):
            simba.hooks.session_start._auto_start_daemon(tmp_path)

        assert captured["env"]["MallocStackLogging"] == "lite"

    @pytest.mark.real_daemon_spawn
    def test_auto_start_daemon_does_not_inject_env_when_flag_false(self, tmp_path):
        """Default (malloc_stack_logging=False): no MallocStackLogging key at
        all --- not even set to something falsy --- so the spawned daemon's
        env is byte-identical to before this lever existed."""
        captured: dict = {}

        def _fake_popen(args, **kwargs):
            captured.update(kwargs)
            return unittest.mock.MagicMock()

        with (
            unittest.mock.patch(
                "simba.hooks.session_start.subprocess.Popen",
                side_effect=_fake_popen,
            ),
            unittest.mock.patch(
                "simba.hooks.session_start._hooks_cfg",
                return_value=unittest.mock.MagicMock(
                    poll_attempts=1, poll_interval=0.0
                ),
            ),
            unittest.mock.patch(
                "simba.hooks.session_start._check_health", return_value=None
            ),
            unittest.mock.patch(
                "simba.memory.config.resolve_malloc_stack_logging",
                return_value=False,
            ),
        ):
            simba.hooks.session_start._auto_start_daemon(tmp_path)

        assert "MallocStackLogging" not in captured.get("env", {})

    @pytest.mark.real_daemon_spawn
    def test_auto_start_daemon_never_mutates_parent_environ(self, tmp_path):
        """The child env is a COPY --- injecting MallocStackLogging for the
        spawned daemon must never leak into this (the hook's own) process."""
        assert "MallocStackLogging" not in os.environ

        def _fake_popen(args, **kwargs):
            return unittest.mock.MagicMock()

        with (
            unittest.mock.patch(
                "simba.hooks.session_start.subprocess.Popen",
                side_effect=_fake_popen,
            ),
            unittest.mock.patch(
                "simba.hooks.session_start._hooks_cfg",
                return_value=unittest.mock.MagicMock(
                    poll_attempts=1, poll_interval=0.0
                ),
            ),
            unittest.mock.patch(
                "simba.hooks.session_start._check_health", return_value=None
            ),
            unittest.mock.patch(
                "simba.memory.config.resolve_malloc_stack_logging",
                return_value=True,
            ),
        ):
            simba.hooks.session_start._auto_start_daemon(tmp_path)

        assert "MallocStackLogging" not in os.environ

    def test_auto_start_daemon_log_path_is_repo_root_aware(self, tmp_path):
        """The log lives next to the sqlite sidecar's `.simba`
        (`simba.db.get_db_path`'s resolution) --- not necessarily the raw
        cwd passed in --- matching the existing path-resolution helper."""
        resolved_db = tmp_path / "repo-root" / ".simba" / "simba.db"
        with unittest.mock.patch("simba.db.get_db_path", return_value=resolved_db):
            log_path = simba.hooks.session_start._daemon_log_path(tmp_path / "sub")
        assert log_path == tmp_path / "repo-root" / ".simba" / "memory" / "daemon.log"

    def test_empty_input_does_not_crash(self):
        with unittest.mock.patch(
            "simba.hooks.session_start._check_health", return_value=None
        ):
            result = json.loads(simba.hooks.session_start.main({}))
        assert "hookSpecificOutput" in result

    def test_includes_project_memory_stats(self, tmp_path):
        health = {"memoryCount": 5, "embeddingModel": "nomic-embed-text"}
        db_file = tmp_path / ".simba" / "simba.db"
        db_file.parent.mkdir(parents=True, exist_ok=True)
        db_file.write_text("")
        with (
            unittest.mock.patch(
                "simba.hooks.session_start._check_health", return_value=health
            ),
            unittest.mock.patch("simba.db.get_db_path", return_value=db_file),
            unittest.mock.patch(
                "simba.search.project_memory.get_stats",
                return_value={"sessions": 10, "knowledge": 3, "facts": 7},
            ),
        ):
            result = json.loads(simba.hooks.session_start.main({"cwd": str(tmp_path)}))
        ctx = result["hookSpecificOutput"]["additionalContext"]
        assert "10 sessions" in ctx
        assert "3 knowledge areas" in ctx
        assert "7 facts" in ctx

    def test_project_memory_error_does_not_crash(self, tmp_path):
        db_file = tmp_path / ".simba" / "simba.db"
        db_file.parent.mkdir(parents=True, exist_ok=True)
        db_file.write_text("")
        with (
            unittest.mock.patch(
                "simba.hooks.session_start._check_health", return_value=None
            ),
            unittest.mock.patch("simba.db.get_db_path", return_value=db_file),
            unittest.mock.patch(
                "simba.search.project_memory.get_stats",
                side_effect=OSError("db error"),
            ),
        ):
            result = json.loads(simba.hooks.session_start.main({"cwd": str(tmp_path)}))
        assert "hookSpecificOutput" in result

    def _write_meta(self, tmp_path, sid, project, transcript):
        # Project-scoped resolution reads <session>/metadata.json, NOT latest.json.
        d = tmp_path / ".claude" / "transcripts" / sid
        d.mkdir(parents=True, exist_ok=True)
        (d / "metadata.json").write_text(
            json.dumps(
                {
                    "status": "pending_extraction",
                    "session_id": sid,
                    "project_path": project,
                    "transcript_path": transcript,
                    "exported_at": "2026-06-05T01:00:00Z",
                }
            )
        )

    def test_pending_extraction_included(self, tmp_path):
        proj = str(tmp_path / "proj")
        self._write_meta(tmp_path, "sess-A", proj, "/tmp/a.md")

        with (
            unittest.mock.patch(
                "simba.hooks.session_start._check_health", return_value=None
            ),
            unittest.mock.patch("pathlib.Path.home", return_value=tmp_path),
        ):
            result = json.loads(
                simba.hooks.session_start.main({"session_id": "sess-A", "cwd": proj})
            )
        ctx = result["hookSpecificOutput"]["additionalContext"]
        assert "learning-extraction-required" in ctx
        assert "/tmp/a.md" in ctx  # this project's transcript
        assert proj in ctx  # --project-path is the resolved (correct) project
        # Extraction-quality rules borrowed from agent-oss.
        assert "Preserve proper nouns" in ctx
        assert "Preserve numeric precision" in ctx
        assert "Resolve relative dates" in ctx

    def test_pending_extraction_is_project_scoped(self, tmp_path):
        # A transcript pending for project B must NOT be offered to a session in
        # project A (the cross-wiring bug). A has nothing pending -> no reminder.
        self._write_meta(tmp_path, "sess-B", str(tmp_path / "projB"), "/tmp/b.md")

        with (
            unittest.mock.patch(
                "simba.hooks.session_start._check_health", return_value=None
            ),
            unittest.mock.patch("pathlib.Path.home", return_value=tmp_path),
        ):
            result = json.loads(
                simba.hooks.session_start.main(
                    {"session_id": "x", "cwd": str(tmp_path / "projA")}
                )
            )
        ctx = result["hookSpecificOutput"]["additionalContext"]
        assert "learning-extraction-required" not in ctx
        assert "/tmp/b.md" not in ctx  # never another project's transcript
