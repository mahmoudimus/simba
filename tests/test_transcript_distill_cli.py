"""Tests for `simba transcript distill <jsonl> [--session-id X] [--out DIR]
[--project-path P]` (src/simba/__main__.py's `_cmd_transcript` "distill"
subcommand).

Verifies the CLI layer's two jobs beyond the pure `distill_transcript` scan:
resolving `hooks.distill_max_output_mb` from config, and persisting the
returned arcs into the `failure_arc` sidecar table (transcripts/arcs.py) --
which distill_transcript itself never touches (no DB in the scan path).
"""

from __future__ import annotations

import json
import pathlib

import pytest

import simba.__main__ as cli
import simba.db
import simba.transcripts.arcs as arcs


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "proj" / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)


def _write_fixture(path: pathlib.Path) -> None:
    lines = [
        json.dumps({"type": "session_meta", "payload": {"id": "s1", "cwd": "/repo"}}),
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "please run pytest"}],
                },
            }
        ),
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call",
                    "call_id": "c1",
                    "name": "exec",
                    "input": "pytest -x",
                },
            }
        ),
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call_output",
                    "call_id": "c1",
                    "output": [
                        {
                            "type": "text",
                            "text": "Script failed\nWall time 1.0 seconds\nOutput:\n",
                        },
                        {"type": "text", "text": "AssertionError: boom"},
                    ],
                },
            }
        ),
    ]
    path.write_text("\n".join(lines) + "\n")


class TestCmdTranscriptDistill:
    def test_distill_writes_transcript_and_persists_arcs(
        self, tmp_path: pathlib.Path
    ) -> None:
        source = tmp_path / "rollout.jsonl"
        _write_fixture(source)
        out_dir = tmp_path / "out"
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        rc = cli._cmd_transcript(
            [
                "distill",
                str(source),
                "--session-id",
                "s1",
                "--out",
                str(out_dir),
                "--project-path",
                str(project_dir),
            ]
        )
        assert rc == 0
        assert (out_dir / "transcript.md").exists()
        assert (out_dir / "distill-meta.json").exists()

        rows = arcs.list_for_session("s1", cwd=project_dir)
        assert len(rows) == 1
        assert rows[0].tool == "exec"
        assert rows[0].resolved is False
        assert rows[0].project_path == str(project_dir)

    def test_distill_defaults_out_dir_to_session_transcripts_dir(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = tmp_path / "rollout.jsonl"
        _write_fixture(source)
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(pathlib.Path, "home", lambda: fake_home)

        rc = cli._cmd_transcript(["distill", str(source), "--session-id", "s2"])
        assert rc == 0
        assert (fake_home / ".claude" / "transcripts" / "s2" / "transcript.md").exists()

    def test_distill_usage_error_missing_source(self) -> None:
        rc = cli._cmd_transcript(["distill"])
        assert rc == 1

    def test_distill_second_run_is_idempotent_no_dup_arcs(
        self, tmp_path: pathlib.Path
    ) -> None:
        source = tmp_path / "rollout.jsonl"
        _write_fixture(source)
        out_dir = tmp_path / "out"
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        argv = [
            "distill",
            str(source),
            "--session-id",
            "s1",
            "--out",
            str(out_dir),
            "--project-path",
            str(project_dir),
        ]
        assert cli._cmd_transcript(argv) == 0
        assert cli._cmd_transcript(argv) == 0

        rows = arcs.list_for_session("s1", cwd=project_dir)
        assert len(rows) == 1
