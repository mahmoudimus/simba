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


def _write_two_arc_fixture(path: pathlib.Path) -> None:
    """A resolved "exec" arc (AssertionError) plus an unresolved "spawn_agent"
    arc (agent pool exhausted, never retried) -- the no-focus default order
    lists the resolved arc first (see distill.py's sort key); a focus
    mentioning the unresolved arc's words should move it first instead.
    """
    lines = [
        json.dumps({"type": "session_meta", "payload": {"id": "s4", "cwd": "/repo"}}),
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "call_id": "c-fail",
                    "name": "spawn_agent",
                    "arguments": '{"task":"risky"}',
                },
            }
        ),
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "c-fail",
                    "output": [
                        {"type": "text", "text": "Script failed\nWall time 1.0s\n"},
                        {
                            "type": "text",
                            "text": "AttributeError: agent pool exhausted",
                        },
                    ],
                },
            }
        ),
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call",
                    "call_id": "c-exec-fail",
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
                    "call_id": "c-exec-fail",
                    "output": [
                        {"type": "text", "text": "Script failed\nWall time 1.0s\n"},
                        {"type": "text", "text": "AssertionError: boom"},
                    ],
                },
            }
        ),
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call",
                    "call_id": "c-exec-fix",
                    "name": "exec",
                    "input": "pytest -k fixed",
                },
            }
        ),
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call_output",
                    "call_id": "c-exec-fix",
                    "output": [
                        {"type": "text", "text": "Script completed\nWall time 0.1s\n"},
                        {"type": "text", "text": "1 passed"},
                    ],
                },
            }
        ),
    ]
    path.write_text("\n".join(lines) + "\n")


class TestCmdTranscriptDistillFocus:
    def test_focus_flag_reorders_failure_arcs_section(
        self, tmp_path: pathlib.Path
    ) -> None:
        source = tmp_path / "rollout.jsonl"
        _write_two_arc_fixture(source)
        out_dir = tmp_path / "out"
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        rc = cli._cmd_transcript(
            [
                "distill",
                str(source),
                "--session-id",
                "s4",
                "--out",
                str(out_dir),
                "--project-path",
                str(project_dir),
                "--focus",
                "agent pool exhausted",
            ]
        )
        assert rc == 0
        text = (out_dir / "transcript.md").read_text()
        arcs_section = text.split("<failure-arcs")[1].split("</failure-arcs>")[0]
        assert arcs_section.index('tool="spawn_agent"') < arcs_section.index(
            'tool="exec"'
        )

    def test_no_focus_flag_keeps_default_order(self, tmp_path: pathlib.Path) -> None:
        source = tmp_path / "rollout.jsonl"
        _write_two_arc_fixture(source)
        out_dir = tmp_path / "out"
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        rc = cli._cmd_transcript(
            [
                "distill",
                str(source),
                "--session-id",
                "s4",
                "--out",
                str(out_dir),
                "--project-path",
                str(project_dir),
            ]
        )
        assert rc == 0
        text = (out_dir / "transcript.md").read_text()
        arcs_section = text.split("<failure-arcs")[1].split("</failure-arcs>")[0]
        # No focus -> resolved "exec" arc sorts first (default order).
        assert arcs_section.index('tool="exec"') < arcs_section.index(
            'tool="spawn_agent"'
        )


def _write_resolved_flag_drop_fixture(path: pathlib.Path) -> None:
    """Same failure ("rg -rn pattern") 3x, then a fix ("rg -n pattern") --
    resolved with repeat_count=3, meeting the default arc_promotion_min_evidence
    (3) so the post-distill rule-candidate scan produces a candidate."""
    lines = [
        json.dumps({"type": "session_meta", "payload": {"id": "s3", "cwd": "/repo"}}),
    ]

    def _call(call_id: str, command: str) -> str:
        return json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call",
                    "call_id": call_id,
                    "name": "exec",
                    "input": command,
                },
            }
        )

    def _failed(call_id: str) -> str:
        return json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call_output",
                    "call_id": call_id,
                    "output": [
                        {"type": "text", "text": "Script failed\nWall time 1.0s\n"},
                        {"type": "text", "text": "regex parse error: unsupported flag"},
                    ],
                },
            }
        )

    def _ok(call_id: str) -> str:
        return json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call_output",
                    "call_id": call_id,
                    "output": [
                        {"type": "text", "text": "Script completed\nWall time 0.1s\n"},
                        {"type": "text", "text": "no matches"},
                    ],
                },
            }
        )

    for i in range(3):
        cid = f"c{i}"
        lines.append(_call(cid, "rg -rn pattern"))
        lines.append(_failed(cid))
    lines.append(_call("c-fix", "rg -n pattern"))
    lines.append(_ok("c-fix"))
    path.write_text("\n".join(lines) + "\n")


class TestCmdTranscriptDistillRuleCandidateScan:
    """`simba transcript distill` also mines the failure_arc sidecar table it
    just fed for mechanical failed->fixed patterns (redirect/arc_promotion.py)
    -- fail-soft, one extra log line, after the arc upserts."""

    def test_distill_reports_rule_candidate_scan_line(
        self, tmp_path: pathlib.Path, capsys
    ) -> None:
        source = tmp_path / "rollout.jsonl"
        _write_resolved_flag_drop_fixture(source)
        out_dir = tmp_path / "out"
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        rc = cli._cmd_transcript(
            [
                "distill",
                str(source),
                "--session-id",
                "s3",
                "--out",
                str(out_dir),
                "--project-path",
                str(project_dir),
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "distill: rule-candidate scan -- 1 candidate(s) (1 new)" in out

        import simba.redirect.candidates as candidates

        pending = candidates.list_pending(cwd=project_dir)
        assert len(pending) == 1
        assert pending[0].rule_kind == "pattern"

    def test_distill_scan_failure_is_fail_soft(
        self, tmp_path: pathlib.Path, monkeypatch, capsys
    ) -> None:
        """A scan-time exception must never fail the distill itself, and the
        primary distill success line must still print."""
        source = tmp_path / "rollout.jsonl"
        _write_resolved_flag_drop_fixture(source)
        out_dir = tmp_path / "out"
        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        import simba.redirect.arc_promotion as arc_promotion

        def _boom(*a, **kw):
            raise RuntimeError("scan blew up")

        monkeypatch.setattr(arc_promotion, "scan", _boom)

        rc = cli._cmd_transcript(
            [
                "distill",
                str(source),
                "--session-id",
                "s3",
                "--out",
                str(out_dir),
                "--project-path",
                str(project_dir),
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "distill: rule-candidate scan" not in out
        assert "distill: wrote" in out
