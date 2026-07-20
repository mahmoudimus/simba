"""Tests for the bounded, single-pass transcript distiller
(transcripts/distill.py) -- the replacement for pre_compact's blind
over-cap skip.

The fixture below is a SYNTHETIC Codex-rollout-shaped JSONL (a few MB, not
the "few hundred MB" real incident file) built to exercise every required
behavior: giant tool outputs, a resolved failure->fix arc, an unresolved
arc, the same failure repeated x50 (must collapse to repeat_count=50), and
ordinary user/assistant messages.
"""

from __future__ import annotations

import json
import pathlib

import pytest

import simba.transcripts.distill as distill

GIANT_HEAD_MARK = "GIANT_HEAD_MARKER_TOKEN"
GIANT_MID_MARK = "GIANT_MIDDLE_MARKER_TOKEN_NEVER_KEPT"
GIANT_TAIL_MARK = "GIANT_TAIL_MARKER_TOKEN"

RESOLVED_TOOL = "exec"
RESOLVED_FAIL_ARGS = "pytest tests/test_foo.py::test_bar"
RESOLVED_FAIL_ERROR = "AssertionError: expected 200 got 404 in test_bar"
RESOLVED_FIX_ARGS = "pytest tests/test_foo.py::test_bar -k fixed"

UNRESOLVED_TOOL = "spawn_agent"
UNRESOLVED_ERROR = "AttributeError: agent pool exhausted"

REPEATED_TOOL = "lint_check"
REPEATED_ERROR = "LintError: unused import 'os' in target module"
REPEAT_N = 50


def _line(entry: dict) -> str:
    return json.dumps(entry)


def _session_meta(session_id: str, cwd: str) -> dict:
    return {"type": "session_meta", "payload": {"id": session_id, "cwd": cwd}}


def _noise(i: int) -> dict:
    return {"type": "event_msg", "payload": {"type": "token_count", "n": i}}


def _user(text: str) -> dict:
    return {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        },
    }


def _assistant(text: str) -> dict:
    return {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        },
    }


def _tool_call(call_id: str, name: str, args: str) -> dict:
    return {
        "type": "response_item",
        "payload": {
            "type": "custom_tool_call",
            "call_id": call_id,
            "name": name,
            "input": args,
        },
    }


def _tool_output(call_id: str, text: str, *, failed: bool) -> dict:
    header = "Script failed" if failed else "Script completed"
    header += "\nWall time 1.0 seconds\nOutput:\n"
    return {
        "type": "response_item",
        "payload": {
            "type": "custom_tool_call_output",
            "call_id": call_id,
            "output": [
                {"type": "text", "text": header},
                {"type": "text", "text": text},
            ],
        },
    }


def _func_call(call_id: str, name: str, args: str) -> dict:
    return {
        "type": "response_item",
        "payload": {
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": args,
        },
    }


def _func_output(call_id: str, text: str, *, failed: bool) -> dict:
    header = "Script failed" if failed else "Script completed"
    header += "\nWall time 0.5 seconds\nOutput:\n"
    return {
        "type": "response_item",
        "payload": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": [
                {"type": "text", "text": header},
                {"type": "text", "text": text},
            ],
        },
    }


def _giant_blob(size: int) -> str:
    filler = "x" * (size // 2)
    return f"{GIANT_HEAD_MARK}\n{filler}\n{GIANT_MID_MARK}\n{filler}\n{GIANT_TAIL_MARK}"


def build_codex_fixture(
    path: pathlib.Path, *, session_id="bloat-1", cwd="/repo/proj"
) -> int:
    """Write a synthetic, Codex-rollout-shaped JSONL to *path*. Returns the
    number of PLANTED giant-tool-output blocks (for size sanity checks)."""
    lines: list[str] = [_line(_session_meta(session_id, cwd))]

    # Bulk noise -- purely prefilterable telemetry (~few hundred KB).
    for i in range(6000):
        lines.append(_line(_noise(i)))

    lines.append(_line(_user("please run the test suite and fix any failures")))
    lines.append(_line(_assistant("I will run the tests now.")))

    # Giant SUCCESSFUL tool outputs (~300KB each) -- must be head+tail
    # truncated, never kept whole.
    giant_count = 3
    for i in range(giant_count):
        call_id = f"call-giant-{i}"
        lines.append(_line(_tool_call(call_id, "exec", f"cat big_log_{i}.txt")))
        lines.append(_line(_tool_output(call_id, _giant_blob(300_000), failed=False)))

    # Unresolved arc: spawn_agent fails and NEVER succeeds afterward.
    lines.append(
        _line(_func_call("call-unresolved-1", UNRESOLVED_TOOL, '{"task":"risky"}'))
    )
    lines.append(
        _line(_func_output("call-unresolved-1", UNRESOLVED_ERROR, failed=True))
    )

    # Resolved arc: exec fails, then a later exec call (different args) succeeds.
    lines.append(
        _line(_tool_call("call-resolved-fail", RESOLVED_TOOL, RESOLVED_FAIL_ARGS))
    )
    lines.append(
        _line(_tool_output("call-resolved-fail", RESOLVED_FAIL_ERROR, failed=True))
    )
    lines.append(
        _line(_tool_call("call-resolved-fix", RESOLVED_TOOL, RESOLVED_FIX_ARGS))
    )
    lines.append(_line(_tool_output("call-resolved-fix", "2 passed", failed=False)))

    # Repeated failure x50 -- byte-identical error text -- must collapse to
    # ONE arc with repeat_count == 50 (never resolved).
    for i in range(REPEAT_N):
        call_id = f"call-repeat-{i}"
        lines.append(_line(_tool_call(call_id, REPEATED_TOOL, "run_lint.sh")))
        lines.append(_line(_tool_output(call_id, REPEATED_ERROR, failed=True)))

    lines.append(_line(_assistant("Done investigating; summary follows.")))
    lines.append(_line(_user("thanks, looks good")))

    path.write_text("\n".join(lines) + "\n")
    return giant_count


@pytest.fixture
def fixture_path(tmp_path: pathlib.Path) -> pathlib.Path:
    p = tmp_path / "rollout.jsonl"
    build_codex_fixture(p)
    return p


class TestNoSlurp:
    def test_distill_never_slurps_the_source(
        self, tmp_path: pathlib.Path, fixture_path: pathlib.Path, monkeypatch
    ) -> None:
        real_read_text = pathlib.Path.read_text

        def guarded_read_text(self, *a, **k):
            if self == fixture_path:
                raise AssertionError("slurp")
            return real_read_text(self, *a, **k)

        monkeypatch.setattr(pathlib.Path, "read_text", guarded_read_text)

        out_dir = tmp_path / "out"
        result = distill.distill_transcript(
            fixture_path, out_dir=out_dir, session_id="s1", max_output_mb=12.0
        )
        assert result.md_path.exists()


class TestPrefilter:
    def test_prefilter_skips_noise_and_counts_bytes(
        self, tmp_path: pathlib.Path, fixture_path: pathlib.Path
    ) -> None:
        out_dir = tmp_path / "out"
        result = distill.distill_transcript(
            fixture_path, out_dir=out_dir, session_id="s1", max_output_mb=12.0
        )
        assert result.stats.prefiltered_lines >= 6000
        assert result.stats.prefiltered_bytes > 0


class TestHarnessDetection:
    def test_detects_codex_harness(
        self, tmp_path: pathlib.Path, fixture_path: pathlib.Path
    ) -> None:
        out_dir = tmp_path / "out"
        result = distill.distill_transcript(
            fixture_path, out_dir=out_dir, session_id="s1", max_output_mb=12.0
        )
        assert result.stats.harness == "codex"


class TestGiantToolOutputTruncation:
    def test_giant_output_truncated_head_and_tail_kept_middle_dropped(
        self, tmp_path: pathlib.Path, fixture_path: pathlib.Path
    ) -> None:
        out_dir = tmp_path / "out"
        distill.distill_transcript(
            fixture_path, out_dir=out_dir, session_id="s1", max_output_mb=12.0
        )
        text = (out_dir / "transcript.md").read_text()
        assert GIANT_HEAD_MARK in text
        assert GIANT_TAIL_MARK in text
        assert GIANT_MID_MARK not in text


class TestFailureFixArcs:
    def test_resolved_arc_detected(
        self, tmp_path: pathlib.Path, fixture_path: pathlib.Path
    ) -> None:
        out_dir = tmp_path / "out"
        result = distill.distill_transcript(
            fixture_path, out_dir=out_dir, session_id="s1", max_output_mb=12.0
        )
        resolved = [a for a in result.arcs if a.tool == RESOLVED_TOOL and a.resolved]
        assert len(resolved) == 1
        arc = resolved[0]
        assert RESOLVED_FAIL_ARGS in arc.failed_args_head
        assert arc.fix_args_head is not None
        assert RESOLVED_FIX_ARGS in arc.fix_args_head
        assert "expected 200 got 404" in arc.error_head

    def test_unresolved_arc_detected_as_dead_end(
        self, tmp_path: pathlib.Path, fixture_path: pathlib.Path
    ) -> None:
        out_dir = tmp_path / "out"
        result = distill.distill_transcript(
            fixture_path, out_dir=out_dir, session_id="s1", max_output_mb=12.0
        )
        unresolved = [a for a in result.arcs if a.tool == UNRESOLVED_TOOL]
        assert len(unresolved) == 1
        assert unresolved[0].resolved is False
        assert unresolved[0].fix_args_head is None

    def test_repeated_identical_failure_collapses_to_one_arc(
        self, tmp_path: pathlib.Path, fixture_path: pathlib.Path
    ) -> None:
        out_dir = tmp_path / "out"
        result = distill.distill_transcript(
            fixture_path, out_dir=out_dir, session_id="s1", max_output_mb=12.0
        )
        repeated = [a for a in result.arcs if a.tool == REPEATED_TOOL]
        assert len(repeated) == 1
        assert repeated[0].repeat_count == REPEAT_N
        assert repeated[0].resolved is False

    def test_arcs_written_into_transcript_md(
        self, tmp_path: pathlib.Path, fixture_path: pathlib.Path
    ) -> None:
        out_dir = tmp_path / "out"
        distill.distill_transcript(
            fixture_path, out_dir=out_dir, session_id="s1", max_output_mb=12.0
        )
        text = (out_dir / "transcript.md").read_text()
        assert "<failure-arcs" in text
        assert RESOLVED_FAIL_ARGS in text
        assert UNRESOLVED_ERROR in text
        assert f'repeat_count="{REPEAT_N}"' in text


class TestBoundedOutput:
    def test_output_stays_under_moderate_budget(
        self, tmp_path: pathlib.Path, fixture_path: pathlib.Path
    ) -> None:
        out_dir = tmp_path / "out"
        budget_mb = 0.2  # 200KB -- well under the ~1MB+ of giant tool output
        result = distill.distill_transcript(
            fixture_path,
            out_dir=out_dir,
            session_id="s1",
            max_output_mb=budget_mb,
        )
        size = result.md_path.stat().st_size
        # Generous slack over the nominal budget for header/tag overhead --
        # the point is "bounded", not byte-exact.
        assert size <= budget_mb * 1_000_000 * 1.5

    def test_arcs_survive_worst_case_budget_squeeze(
        self, tmp_path: pathlib.Path, fixture_path: pathlib.Path
    ) -> None:
        out_dir = tmp_path / "out"
        result = distill.distill_transcript(
            fixture_path,
            out_dir=out_dir,
            session_id="s1",
            max_output_mb=0.004,  # ~4KB -- far below the giant tool output alone
        )
        text = (out_dir / "transcript.md").read_text()
        # Arcs are the top-priority signal -- they must appear even when the
        # body is squeezed to nothing.
        assert "<failure-arcs" in text
        assert UNRESOLVED_TOOL in text
        assert f'repeat_count="{REPEAT_N}"' in text
        assert len(result.arcs) == 3
        # The giant tool-output body content must NOT have survived at this budget.
        assert GIANT_HEAD_MARK not in text


class TestDistillMeta:
    def test_meta_json_has_required_fields(
        self, tmp_path: pathlib.Path, fixture_path: pathlib.Path
    ) -> None:
        out_dir = tmp_path / "out"
        result = distill.distill_transcript(
            fixture_path, out_dir=out_dir, session_id="s1", max_output_mb=12.0
        )
        meta = json.loads(result.meta_path.read_text())
        assert meta["source_path"] == str(fixture_path)
        assert meta["source_bytes"] == fixture_path.stat().st_size
        assert meta["prefiltered_bytes"] > 0
        assert "kept_by_class" in meta
        assert "dropped_by_class" in meta
        assert meta["arc_resolved_count"] == 1
        assert meta["arc_unresolved_count"] == 2
        assert meta["elapsed_seconds"] >= 0


class TestIdempotence:
    def test_rerun_with_matching_marker_skips_rescan(
        self, tmp_path: pathlib.Path, fixture_path: pathlib.Path, monkeypatch
    ) -> None:
        out_dir = tmp_path / "out"
        first = distill.distill_transcript(
            fixture_path, out_dir=out_dir, session_id="s1", max_output_mb=12.0
        )
        assert first.skipped is False
        first_content = first.md_path.read_text()

        real_open = pathlib.Path.open

        def guarded_open(self, *a, **k):
            if self == fixture_path:
                raise AssertionError("must not re-scan when the marker matches")
            return real_open(self, *a, **k)

        monkeypatch.setattr(pathlib.Path, "open", guarded_open)

        second = distill.distill_transcript(
            fixture_path, out_dir=out_dir, session_id="s1", max_output_mb=12.0
        )
        assert second.skipped is True
        assert second.md_path.read_text() == first_content

    def test_rerun_does_not_duplicate_arc_rows_when_upserted(
        self, tmp_path: pathlib.Path, fixture_path: pathlib.Path
    ) -> None:
        """distill_transcript itself is DB-free; this proves its RETURNED
        arcs are stable/identical across a forced re-run, which is the
        precondition for arcs.upsert_arc's (session_source, signature) key
        to actually dedupe rather than accidentally diverge."""
        out_dir = tmp_path / "out"
        first = distill.distill_transcript(
            fixture_path,
            out_dir=out_dir,
            session_id="s1",
            max_output_mb=12.0,
            force=True,
        )
        second = distill.distill_transcript(
            fixture_path,
            out_dir=out_dir,
            session_id="s1",
            max_output_mb=12.0,
            force=True,
        )
        sigs_1 = sorted((a.tool, a.signature, a.repeat_count) for a in first.arcs)
        sigs_2 = sorted((a.tool, a.signature, a.repeat_count) for a in second.arcs)
        assert sigs_1 == sigs_2


class TestMarkerMatches:
    def test_marker_matches_true_after_distill(
        self, tmp_path: pathlib.Path, fixture_path: pathlib.Path
    ) -> None:
        out_dir = tmp_path / "out"
        distill.distill_transcript(
            fixture_path, out_dir=out_dir, session_id="s1", max_output_mb=12.0
        )
        assert distill.marker_matches(
            out_dir, fixture_path, fixture_path.stat().st_size
        )

    def test_marker_false_when_source_grows(
        self, tmp_path: pathlib.Path, fixture_path: pathlib.Path
    ) -> None:
        out_dir = tmp_path / "out"
        distill.distill_transcript(
            fixture_path, out_dir=out_dir, session_id="s1", max_output_mb=12.0
        )
        assert not distill.marker_matches(out_dir, fixture_path, 999999999)

    def test_marker_false_when_no_meta_exists(self, tmp_path: pathlib.Path) -> None:
        assert not distill.marker_matches(tmp_path / "nope", tmp_path / "src", 10)


class TestOrdinaryMessages:
    def test_user_and_assistant_text_present_under_generous_budget(
        self, tmp_path: pathlib.Path, fixture_path: pathlib.Path
    ) -> None:
        out_dir = tmp_path / "out"
        distill.distill_transcript(
            fixture_path, out_dir=out_dir, session_id="s1", max_output_mb=12.0
        )
        text = (out_dir / "transcript.md").read_text()
        assert "please run the test suite" in text
        assert "I will run the tests now." in text


class TestClaudeCodeShape:
    """Smoke test for the OTHER supported harness -- Claude Code's raw
    ~/.claude/projects/*/[session].jsonl shape (message.role/content, plus
    tool_use/tool_result content items and a toolUseResult sibling key)."""

    def _build(self, path: pathlib.Path) -> None:
        entries = [
            {
                "sessionId": "cc-1",
                "cwd": "/repo/proj",
                "type": "user",
                "message": {"role": "user", "content": "please fix the failing test"},
            },
            {
                "sessionId": "cc-1",
                "cwd": "/repo/proj",
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Investigating now."}],
                },
            },
            {
                "sessionId": "cc-1",
                "cwd": "/repo/proj",
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call-1",
                            "name": "Bash",
                            "input": {"command": "pytest -x"},
                        }
                    ],
                },
            },
            {
                "sessionId": "cc-1",
                "cwd": "/repo/proj",
                "type": "user",
                "toolUseResult": {"stdout": "", "stderr": "AssertionError: boom"},
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-1",
                            "content": "AssertionError: boom",
                            "is_error": True,
                        }
                    ],
                },
            },
            {
                "sessionId": "cc-1",
                "cwd": "/repo/proj",
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call-2",
                            "name": "Bash",
                            "input": {"command": "pytest -x -k fixed"},
                        }
                    ],
                },
            },
            {
                "sessionId": "cc-1",
                "cwd": "/repo/proj",
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call-2",
                            "content": "2 passed",
                            "is_error": False,
                        }
                    ],
                },
            },
        ]
        path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

    def test_claude_code_shape_detected_and_resolved_arc_found(
        self, tmp_path: pathlib.Path
    ) -> None:
        src = tmp_path / "session.jsonl"
        self._build(src)
        out_dir = tmp_path / "out"
        result = distill.distill_transcript(
            src, out_dir=out_dir, session_id="cc-1", max_output_mb=12.0
        )
        assert result.stats.harness == "claude-code"
        resolved = [a for a in result.arcs if a.resolved]
        assert len(resolved) == 1
        assert "boom" in resolved[0].error_head

        text = result.md_path.read_text()
        assert "please fix the failing test" in text
        assert "Investigating now." in text
