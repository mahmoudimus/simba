"""Tests for Codex trace curator reports."""

from __future__ import annotations

import json
import pathlib

import simba.codex.curator as curator


def _write_trace(path: pathlib.Path, rows: list[dict]) -> pathlib.Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def _row(event: str, payload: dict, *, line_id: str = "run1") -> dict:
    return {
        "event": event,
        "run_id": line_id,
        "session_id": "session-1",
        "project_path": "/tmp/project",
        "transcript_path": "/tmp/transcript.jsonl",
        "payload": payload,
    }


def test_summarize_trace_groups_store_outcomes(tmp_path: pathlib.Path) -> None:
    trace_path = _write_trace(
        tmp_path / "trace.jsonl",
        [
            _row("run_started", {}),
            _row(
                "candidate",
                {
                    "index": 0,
                    "type": "PREFERENCE",
                    "content": "Use uv run for Simba tests",
                    "context": "User confirmed this workflow.",
                    "confidence": 0.9,
                    "score": 0.9,
                    "reason": "matched preference transcript heuristic",
                    "source_span": "message:2",
                    "evidence": "Always use uv run",
                },
            ),
            _row(
                "curator_decision",
                {"index": 0, "decision": "keep", "reason": "good evidence"},
            ),
            _row(
                "store_result",
                {"index": 0, "status": "stored", "memory_id": "mem-1"},
            ),
            _row("run_completed", {"status": "stored"}),
        ],
    )

    report = curator.summarize_trace(curator.load_trace(trace_path))

    assert report.status == "stored"
    assert report.session_id == "session-1"
    assert report.metrics["candidate_count"] == 1
    assert report.metrics["stored_count"] == 1
    assert report.metrics["evidence_coverage"] == 1.0
    candidate = report.candidates[0]
    assert candidate.memory_type == "PREFERENCE"
    assert candidate.decision == "keep"
    assert candidate.store_status == "stored"
    assert candidate.memory_id == "mem-1"


def test_curator_keeps_negative_lessons_out_of_memory(
    tmp_path: pathlib.Path,
) -> None:
    trace_path = _write_trace(
        tmp_path / "trace.jsonl",
        [
            _row("candidate", {"index": 0, "type": "GOTCHA", "content": "bad"}),
            _row(
                "store_error",
                {"index": 0, "error_type": "HTTPError", "message": "boom"},
            ),
            _row(
                "negative_lesson",
                {"index": 0, "reason": "store_exception", "error_type": "HTTPError"},
            ),
            _row("run_completed", {"status": "store_errors"}),
        ],
    )

    report = curator.summarize_trace(curator.load_trace(trace_path))

    assert report.metrics["store_error_count"] == 1
    assert report.metrics["negative_lesson_count"] == 1
    assert report.candidates[0].store_status == "error"
    assert "store_exception" in report.negative_lessons[0]
    assert any("curator did not store memories" in a for a in report.suggested_actions)


def test_markdown_report_includes_evidence_and_source_span(
    tmp_path: pathlib.Path,
) -> None:
    trace_path = _write_trace(
        tmp_path / "trace.jsonl",
        [
            _row(
                "candidate",
                {
                    "index": 0,
                    "type": "DECISION",
                    "content": "Curator must be review-only",
                    "source_span": "message:4",
                    "evidence": "do not auto-store",
                },
            ),
        ],
    )
    report = curator.summarize_trace(curator.load_trace(trace_path))

    rendered = curator.render_markdown(report)

    assert "message:4" in rendered
    assert "do not auto-store" in rendered
    assert "Curator must be review-only" in rendered


def test_json_report_is_stable(tmp_path: pathlib.Path) -> None:
    trace_path = _write_trace(
        tmp_path / "trace.jsonl",
        [
            _row("candidate", {"index": 0, "type": "PATTERN", "content": "x"}),
            _row("run_completed", {"status": "no_candidates"}),
        ],
    )
    report = curator.summarize_trace(curator.load_trace(trace_path))
    out = tmp_path / "report.json"

    curator.write_json(report, out)

    data = json.loads(out.read_text())
    assert list(data.keys()) == sorted(data.keys())
    assert data["metrics"]["candidate_count"] == 1
    assert data["candidates"][0]["type"] == "PATTERN"


def test_incomplete_trace_still_writes_partial_report(
    tmp_path: pathlib.Path,
) -> None:
    trace_path = _write_trace(
        tmp_path / "trace.jsonl",
        [_row("candidate", {"index": 0, "type": "GOTCHA", "content": "partial"})],
    )

    report = curator.summarize_trace(curator.load_trace(trace_path))

    assert report.status == "unknown"
    assert report.metrics["candidate_count"] == 1


def test_unknown_events_are_reported_not_fatal(tmp_path: pathlib.Path) -> None:
    trace_path = _write_trace(
        tmp_path / "trace.jsonl",
        [_row("new_event", {"hello": "world"})],
    )

    report = curator.summarize_trace(curator.load_trace(trace_path))

    assert report.metrics["warning_count"] == 1
    assert "unknown event" in report.warnings[0]


def test_append_review_decisions_and_memory_store_commands(
    tmp_path: pathlib.Path,
) -> None:
    trace_path = _write_trace(
        tmp_path / "trace.jsonl",
        [
            _row(
                "candidate",
                {
                    "index": 0,
                    "type": "DECISION",
                    "content": "Curator review stays manual",
                    "context": "Accepted by reviewer.",
                    "confidence": 0.91,
                    "source_span": "message:7",
                    "evidence": "emit commands, do not execute",
                },
            ),
            _row("run_completed", {"status": "stored"}),
        ],
    )
    report = curator.summarize_trace(curator.load_trace(trace_path))
    decisions = (
        curator.ReviewDecision(
            candidate_index=0,
            label="accepted",
            reason="strong evidence",
            reviewer="test",
        ),
    )

    review_path = curator.append_review_decisions(
        report,
        decisions,
        tmp_path / "trace.review.jsonl",
    )
    commands = curator.memory_store_commands(report, decisions)

    rows = [json.loads(line) for line in review_path.read_text().splitlines()]
    assert rows[0]["event"] == "review_decision"
    assert rows[0]["label"] == "accepted"
    assert rows[0]["candidate"]["source_span"] == "message:7"
    assert commands == (
        "simba memory store --type DECISION --content "
        "'Curator review stays manual' --context 'Accepted by reviewer.' "
        "--confidence 0.91 --session-source session-1 --project-path /tmp/project",
    )


def test_review_commands_only_for_accepted_candidates(
    tmp_path: pathlib.Path,
) -> None:
    trace_path = _write_trace(
        tmp_path / "trace.jsonl",
        [
            _row("candidate", {"index": 0, "type": "PATTERN", "content": "keep"}),
            _row("candidate", {"index": 1, "type": "GOTCHA", "content": "drop"}),
        ],
    )
    report = curator.summarize_trace(curator.load_trace(trace_path))
    decisions = (
        curator.ReviewDecision(candidate_index=0, label="accepted"),
        curator.ReviewDecision(candidate_index=1, label="noisy"),
    )

    commands = curator.memory_store_commands(report, decisions)

    assert len(commands) == 1
    assert "--content keep" in commands[0]
    assert "drop" not in commands[0]


def test_load_markdown_report_reloads_trace_path(tmp_path: pathlib.Path) -> None:
    trace_path = _write_trace(
        tmp_path / "trace.jsonl",
        [_row("candidate", {"index": 0, "type": "PATTERN", "content": "from trace"})],
    )
    report = curator.summarize_trace(curator.load_trace(trace_path))
    report_path = curator.write_markdown(report, tmp_path / "report.md")

    loaded = curator.load_report_or_trace(report_path)

    assert loaded.trace_path == trace_path
    assert loaded.candidates[0].content == "from trace"
