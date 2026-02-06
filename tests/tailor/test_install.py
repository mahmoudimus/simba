"""Tests for tailor install module — dir creation, file copy, hook registration."""

from __future__ import annotations

import json
import pathlib

import simba.tailor.install


class TestParseArgs:
    def test_defaults(self):
        opts = simba.tailor.install.parse_args([])
        assert opts.dry_run is False
        assert opts.force is False
        assert opts.claude_md is None

    def test_dry_run_long(self):
        assert simba.tailor.install.parse_args(["--dry-run"]).dry_run is True

    def test_dry_run_short(self):
        assert simba.tailor.install.parse_args(["-n"]).dry_run is True

    def test_force_long(self):
        assert simba.tailor.install.parse_args(["--force"]).force is True

    def test_force_short(self):
        assert simba.tailor.install.parse_args(["-f"]).force is True

    def test_claude_md_skip(self):
        assert simba.tailor.install.parse_args(["--claude-md=skip"]).claude_md == "skip"

    def test_claude_md_overwrite(self):
        assert (
            simba.tailor.install.parse_args(["--claude-md=overwrite"]).claude_md
            == "overwrite"
        )

    def test_claude_md_merge(self):
        assert (
            simba.tailor.install.parse_args(["--claude-md=merge"]).claude_md == "merge"
        )

    def test_claude_md_invalid_ignored(self):
        assert (
            simba.tailor.install.parse_args(["--claude-md=invalid"]).claude_md is None
        )


class TestInstallDirectories:
    def test_creates_memory_dir(self, tmp_path: pathlib.Path):
        simba.tailor.install.install(cwd=tmp_path)
        assert (tmp_path / ".claude-tailor" / "memory").is_dir()

    def test_creates_commands_dir(self, tmp_path: pathlib.Path):
        simba.tailor.install.install(cwd=tmp_path)
        assert (tmp_path / ".claude" / "commands").is_dir()


class TestInstallHookRegistration:
    def test_creates_settings_file(self, tmp_path: pathlib.Path):
        simba.tailor.install.install(cwd=tmp_path)
        settings_path = tmp_path / ".claude" / "settings.local.json"
        assert settings_path.exists()

    def test_registers_session_start_hook(self, tmp_path: pathlib.Path):
        simba.tailor.install.install(cwd=tmp_path)
        settings = json.loads(
            (tmp_path / ".claude" / "settings.local.json").read_text()
        )
        assert "hooks" in settings
        assert "SessionStart" in settings["hooks"]
        hooks = settings["hooks"]["SessionStart"]
        assert len(hooks) > 0
        commands = [
            h["command"]
            for entry in hooks
            for h in entry.get("hooks", [])
            if "command" in h
        ]
        assert any("session-start" in c or "session_start" in c for c in commands)

    def test_no_duplicate_hooks_on_rerun(self, tmp_path: pathlib.Path):
        simba.tailor.install.install(cwd=tmp_path)
        simba.tailor.install.install(cwd=tmp_path)
        settings = json.loads(
            (tmp_path / ".claude" / "settings.local.json").read_text()
        )
        session_hooks = [
            entry
            for entry in settings["hooks"]["SessionStart"]
            if any(
                "session-start" in h.get("command", "")
                or "session_start" in h.get("command", "")
                for h in entry.get("hooks", [])
            )
        ]
        assert len(session_hooks) == 1

    def test_preserves_existing_hooks(self, tmp_path: pathlib.Path):
        claude_dir = tmp_path / ".claude"
        claude_dir.mkdir(parents=True)
        settings_path = claude_dir / "settings.local.json"
        settings_path.write_text(
            json.dumps(
                {
                    "hooks": {
                        "PreToolUse": [
                            {
                                "hooks": [
                                    {
                                        "type": "command",
                                        "command": "./existing-hook.js",
                                    }
                                ]
                            }
                        ]
                    }
                },
                indent=2,
            )
        )

        simba.tailor.install.install(cwd=tmp_path)
        settings = json.loads(settings_path.read_text())
        assert "PreToolUse" in settings["hooks"]
        assert "SessionStart" in settings["hooks"]

    def test_creates_valid_json(self, tmp_path: pathlib.Path):
        simba.tailor.install.install(cwd=tmp_path)
        settings_path = tmp_path / ".claude" / "settings.local.json"
        # Should not raise
        json.loads(settings_path.read_text())


class TestInstallClaudeMd:
    def test_creates_claude_md_when_missing(self, tmp_path: pathlib.Path):
        simba.tailor.install.install(cwd=tmp_path)
        claude_md = tmp_path / "CLAUDE.md"
        assert claude_md.exists()
        assert "tailor-nano" in claude_md.read_text()

    def test_skip_does_not_modify(self, tmp_path: pathlib.Path):
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("Original content\n")
        simba.tailor.install.install(cwd=tmp_path, claude_md_action="skip")
        assert claude_md.read_text() == "Original content\n"

    def test_overwrite_replaces(self, tmp_path: pathlib.Path):
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("Original content\n")
        simba.tailor.install.install(cwd=tmp_path, claude_md_action="overwrite")
        content = claude_md.read_text()
        assert "Original content" not in content
        assert "tailor-nano" in content

    def test_merge_appends(self, tmp_path: pathlib.Path):
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("Original content\n")
        simba.tailor.install.install(cwd=tmp_path, claude_md_action="merge")
        content = claude_md.read_text()
        assert "Original content" in content
        assert "tailor-nano" in content

    def test_merge_no_duplicate(self, tmp_path: pathlib.Path):
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("Original content\n")
        simba.tailor.install.install(cwd=tmp_path, claude_md_action="merge")
        first_content = claude_md.read_text()
        count_first = first_content.count("tailor-nano")

        simba.tailor.install.install(cwd=tmp_path, claude_md_action="merge")
        second_content = claude_md.read_text()
        count_second = second_content.count("tailor-nano")
        assert count_second == count_first


class TestDryRun:
    def test_creates_no_files(self, tmp_path: pathlib.Path):
        simba.tailor.install.install(cwd=tmp_path, dry_run=True)
        assert not (tmp_path / ".claude-tailor").exists()
        assert not (tmp_path / ".claude").exists()
        assert not (tmp_path / "CLAUDE.md").exists()


class TestForceMode:
    def test_overwrites_existing_hook(self, tmp_path: pathlib.Path):
        simba.tailor.install.install(cwd=tmp_path)
        hook_path = tmp_path / ".claude-tailor" / "hook.py"
        hook_path.write_text("# modified")

        simba.tailor.install.install(cwd=tmp_path, force=True)
        content = hook_path.read_text()
        assert content != "# modified"

    def test_no_overwrite_without_force(self, tmp_path: pathlib.Path):
        simba.tailor.install.install(cwd=tmp_path)
        hook_path = tmp_path / ".claude-tailor" / "hook.py"
        hook_path.write_text("# modified")

        simba.tailor.install.install(cwd=tmp_path)
        assert hook_path.read_text() == "# modified"
