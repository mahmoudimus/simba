"""Curator reports for Codex extraction analysis traces."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import shlex
import time
from typing import TYPE_CHECKING, Any

import simba.codex.analysis_runs as analysis_runs
import simba.db

if TYPE_CHECKING:
    from collections.abc import Mapping

KNOWN_EVENTS = {
    "run_started",
    "transcript_loaded",
    "candidate",
    "curator_decision",
    "store_result",
    "store_error",
    "negative_lesson",
    "run_completed",
}

VALID_REVIEW_LABELS = {
    "accepted",
    "rejected",
    "duplicate",
    "noisy",
    "needs_more_evidence",
}


@dataclasses.dataclass(frozen=True)
class TraceEvent:
    line_number: int
    event: str
    payload: Mapping[str, Any]
    session_id: str | None = None
    project_path: str | None = None
    transcript_path: str | None = None


@dataclasses.dataclass(frozen=True)
class CuratorTrace:
    path: pathlib.Path
    events: tuple[TraceEvent, ...]
    warnings: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class TraceCandidate:
    index: int
    memory_type: str
    content: str
    context: str
    confidence: float | None
    reason: str
    score: float | None
    source_span: str | None
    evidence: str | None
    decision: str | None
    store_status: str | None
    memory_id: str | None
    superseded_id: str | None
    line_number: int


@dataclasses.dataclass(frozen=True)
class PlaybookCandidate:
    kind: str
    summary: str
    evidence: tuple[str, ...]
    suggested_change: str
    risk: str


@dataclasses.dataclass(frozen=True)
class CuratorReport:
    trace_path: pathlib.Path
    session_id: str | None
    project_path: str | None
    transcript_path: str | None
    status: str
    candidates: tuple[TraceCandidate, ...]
    negative_lessons: tuple[str, ...]
    metrics: Mapping[str, int | float]
    suggested_actions: tuple[str, ...]
    playbook_candidates: tuple[PlaybookCandidate, ...]
    warnings: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class ReviewDecision:
    candidate_index: int
    label: str
    reason: str = ""
    reviewer: str = ""


def default_report_root(cwd: pathlib.Path | None = None) -> pathlib.Path:
    """Return the default append-only curator report directory."""
    return simba.db.get_db_path(cwd).parent / "curator_runs"


def find_latest_trace(trace_dir: pathlib.Path) -> pathlib.Path | None:
    """Return the newest JSONL trace in *trace_dir*, if one exists."""
    trace_dir = trace_dir.expanduser()
    if not trace_dir.exists():
        return None
    traces = [path for path in trace_dir.glob("*.jsonl") if path.is_file()]
    if not traces:
        return None
    return max(traces, key=lambda path: (path.stat().st_mtime_ns, path.name))


def load_trace(path: pathlib.Path) -> CuratorTrace:
    """Load a Codex analysis trace JSONL file."""
    events: list[TraceEvent] = []
    warnings: list[str] = []
    path = path.expanduser()
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            warnings.append(f"line {line_number}: malformed JSON ({exc.msg})")
            continue
        if not isinstance(row, dict):
            warnings.append(f"line {line_number}: expected object row")
            continue
        event = row.get("event")
        if not isinstance(event, str) or not event:
            warnings.append(f"line {line_number}: missing event")
            continue
        payload = row.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        if event not in KNOWN_EVENTS:
            warnings.append(f"line {line_number}: unknown event {event!r}")
        events.append(
            TraceEvent(
                line_number=line_number,
                event=event,
                payload=payload,
                session_id=_optional_str(row.get("session_id")),
                project_path=_optional_str(row.get("project_path")),
                transcript_path=_optional_str(row.get("transcript_path")),
            )
        )
    return CuratorTrace(path=path, events=tuple(events), warnings=tuple(warnings))


def summarize_trace(trace: CuratorTrace) -> CuratorReport:
    """Summarize one trace into a reviewable curator report."""
    candidate_rows: dict[int, dict[str, Any]] = {}
    negative_lessons: list[str] = []
    store_errors = 0
    final_status = "unknown"
    session_id: str | None = None
    project_path: str | None = None
    transcript_path: str | None = None

    for event in trace.events:
        session_id = session_id or event.session_id
        project_path = project_path or event.project_path
        transcript_path = transcript_path or event.transcript_path
        if event.event == "candidate":
            index = _event_index(event.payload, default=len(candidate_rows))
            row = candidate_rows.setdefault(index, {})
            row.update(event.payload)
            row["line_number"] = event.line_number
        elif event.event == "curator_decision":
            index = _event_index(event.payload)
            row = candidate_rows.setdefault(index, {})
            row["decision"] = event.payload.get("decision")
            row["decision_reason"] = event.payload.get("reason")
            if "score" in event.payload:
                row.setdefault("score", event.payload.get("score"))
        elif event.event == "store_result":
            index = _event_index(event.payload)
            row = candidate_rows.setdefault(index, {})
            row["store_status"] = event.payload.get("status")
            row["memory_id"] = event.payload.get("memory_id")
            row["superseded_id"] = event.payload.get("superseded_id")
        elif event.event == "store_error":
            store_errors += 1
            index = _event_index(event.payload)
            row = candidate_rows.setdefault(index, {})
            row["store_status"] = "error"
            row["store_error"] = event.payload.get("message") or event.payload.get(
                "error_type"
            )
        elif event.event == "negative_lesson":
            negative_lessons.append(_negative_lesson_text(event.payload))
        elif event.event == "run_completed":
            status = event.payload.get("status")
            if isinstance(status, str) and status:
                final_status = status
            session_id = session_id or _optional_str(event.payload.get("session_id"))
            project_path = project_path or _optional_str(
                event.payload.get("project_path")
            )
            transcript_path = transcript_path or _optional_str(
                event.payload.get("transcript_path")
            )

    candidates = tuple(
        _candidate_from_row(index, row)
        for index, row in sorted(candidate_rows.items(), key=lambda item: item[0])
    )
    metrics = _metrics(candidates, negative_lessons, trace.warnings, store_errors)
    playbooks = _playbook_candidates(candidates, negative_lessons)
    return CuratorReport(
        trace_path=trace.path,
        session_id=session_id,
        project_path=project_path,
        transcript_path=transcript_path,
        status=final_status,
        candidates=candidates,
        negative_lessons=tuple(negative_lessons),
        metrics=metrics,
        suggested_actions=_suggested_actions(metrics),
        playbook_candidates=playbooks,
        warnings=trace.warnings,
    )


def write_markdown(report: CuratorReport, path: pathlib.Path) -> pathlib.Path:
    """Write a markdown curator report."""
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(report), encoding="utf-8")
    return path


def write_json(report: CuratorReport, path: pathlib.Path) -> pathlib.Path:
    """Write a stable JSON curator report."""
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(report), indent=2, sort_keys=True) + "\n")
    return path


def load_report_or_trace(path: pathlib.Path) -> CuratorReport:
    """Load a curator JSON/markdown report or raw trace JSONL."""
    path = path.expanduser()
    if path.suffix == ".jsonl":
        return summarize_trace(load_trace(path))
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"expected JSON object in {path}")
        return _report_from_dict(data, fallback_path=path)

    trace_path = _trace_path_from_markdown(path)
    if trace_path is None:
        raise ValueError(f"could not find trace path in {path}")
    return summarize_trace(load_trace(trace_path))


def review_path_for(subject_path: pathlib.Path) -> pathlib.Path:
    """Return the append-only review-decision path for a report or trace."""
    subject_path = subject_path.expanduser()
    if subject_path.name.endswith(".review.jsonl"):
        return subject_path
    return subject_path.with_suffix(".review.jsonl")


def append_review_decisions(
    report: CuratorReport,
    decisions: tuple[ReviewDecision, ...],
    path: pathlib.Path,
) -> pathlib.Path:
    """Append reviewer labels for report candidates."""
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    candidates = {candidate.index: candidate for candidate in report.candidates}
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with path.open("a", encoding="utf-8") as fh:
        for decision in decisions:
            if decision.label not in VALID_REVIEW_LABELS:
                raise ValueError(f"invalid review label: {decision.label}")
            candidate = candidates.get(decision.candidate_index)
            row = {
                "ts": ts,
                "event": "review_decision",
                "trace_path": str(report.trace_path),
                "session_id": report.session_id,
                "project_path": report.project_path,
                "transcript_path": report.transcript_path,
                "candidate_index": decision.candidate_index,
                "label": decision.label,
                "reason": decision.reason,
                "reviewer": decision.reviewer,
                "candidate": (
                    _review_candidate_payload(candidate) if candidate else None
                ),
            }
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def memory_store_commands(
    report: CuratorReport,
    decisions: tuple[ReviewDecision, ...],
) -> tuple[str, ...]:
    """Build exact store commands for accepted review decisions."""
    candidates = {candidate.index: candidate for candidate in report.candidates}
    commands: list[str] = []
    for decision in decisions:
        if decision.label != "accepted":
            continue
        candidate = candidates.get(decision.candidate_index)
        if candidate is None:
            continue
        context = candidate.context or _fallback_context(candidate)
        confidence = candidate.confidence if candidate.confidence is not None else 0.85
        parts = [
            "simba",
            "memory",
            "store",
            "--type",
            candidate.memory_type or "PATTERN",
            "--content",
            candidate.content,
            "--context",
            context,
            "--confidence",
            f"{confidence:g}",
        ]
        if report.session_id:
            parts.extend(["--session-source", report.session_id])
        if report.project_path:
            parts.extend(["--project-path", report.project_path])
        commands.append(" ".join(shlex.quote(str(part)) for part in parts))
    return tuple(commands)


def render_markdown(report: CuratorReport) -> str:
    """Render a curator report as markdown."""
    lines = [
        "# Codex Trace Curator Report",
        "",
        "## Run",
        "",
        f"- status: `{report.status}`",
        f"- session: `{report.session_id or 'unknown'}`",
        f"- project: `{report.project_path or 'unknown'}`",
        f"- transcript: `{report.transcript_path or 'unknown'}`",
        f"- trace: `{report.trace_path}`",
        "",
        "## Metrics",
        "",
    ]
    for key, value in sorted(report.metrics.items()):
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Candidates", ""])
    if report.candidates:
        lines.extend(
            [
                "| # | type | status | decision | source | content | evidence |",
                "|---|------|--------|----------|--------|---------|----------|",
            ]
        )
        for candidate in report.candidates:
            lines.append(
                "| "
                f"{candidate.index} | "
                f"{_cell(candidate.memory_type)} | "
                f"{_cell(candidate.store_status or 'missing')} | "
                f"{_cell(candidate.decision or 'missing')} | "
                f"{_cell(candidate.source_span or 'missing')} | "
                f"{_cell(candidate.content)} | "
                f"{_cell(candidate.evidence or 'missing')} |"
            )
    else:
        lines.append("No candidate memories were recorded.")

    lines.extend(["", "## Negative Lessons", ""])
    if report.negative_lessons:
        lines.extend(f"- {_text(lesson)}" for lesson in report.negative_lessons)
    else:
        lines.append("None.")

    lines.extend(["", "## Playbook Candidates", ""])
    if report.playbook_candidates:
        for playbook in report.playbook_candidates:
            evidence = ", ".join(playbook.evidence)
            lines.extend(
                [
                    f"- **{_text(playbook.kind)}** ({playbook.risk}): "
                    f"{_text(playbook.summary)}",
                    f"  Evidence: {_text(evidence)}",
                    f"  Suggested change: {_text(playbook.suggested_change)}",
                ]
            )
    else:
        lines.append("None.")

    lines.extend(["", "## Suggested Actions", ""])
    lines.extend(f"- {_text(action)}" for action in report.suggested_actions)
    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {_text(warning)}" for warning in report.warnings)
    return "\n".join(lines) + "\n"


def to_dict(report: CuratorReport) -> dict[str, Any]:
    """Convert a report to stable JSON-compatible data."""
    return {
        "trace_path": str(report.trace_path),
        "session_id": report.session_id,
        "project_path": report.project_path,
        "transcript_path": report.transcript_path,
        "status": report.status,
        "metrics": dict(sorted(report.metrics.items())),
        "candidates": [
            {
                "index": candidate.index,
                "type": candidate.memory_type,
                "content": candidate.content,
                "context": candidate.context,
                "confidence": candidate.confidence,
                "reason": candidate.reason,
                "score": candidate.score,
                "source_span": candidate.source_span,
                "evidence": candidate.evidence,
                "decision": candidate.decision,
                "store_status": candidate.store_status,
                "memory_id": candidate.memory_id,
                "superseded_id": candidate.superseded_id,
                "line_number": candidate.line_number,
            }
            for candidate in report.candidates
        ],
        "negative_lessons": list(report.negative_lessons),
        "playbook_candidates": [
            {
                "kind": playbook.kind,
                "summary": playbook.summary,
                "evidence": list(playbook.evidence),
                "suggested_change": playbook.suggested_change,
                "risk": playbook.risk,
            }
            for playbook in report.playbook_candidates
        ],
        "suggested_actions": list(report.suggested_actions),
        "warnings": list(report.warnings),
    }


def filter_report(report: CuratorReport, *, min_score: float) -> CuratorReport:
    """Return a report with low-scoring candidates removed."""
    candidates = tuple(
        candidate
        for candidate in report.candidates
        if candidate.score is None or candidate.score >= min_score
    )
    negative_lessons = list(report.negative_lessons)
    metrics = _metrics(
        candidates,
        negative_lessons,
        report.warnings,
        store_errors=0,
    )
    return dataclasses.replace(
        report,
        candidates=candidates,
        metrics=metrics,
        suggested_actions=_suggested_actions(metrics),
        playbook_candidates=_playbook_candidates(candidates, negative_lessons),
    )


def resolve_trace_dir(raw: str | None, cwd: pathlib.Path | None = None) -> pathlib.Path:
    """Resolve a configured trace directory or return the analysis default."""
    if raw:
        return pathlib.Path(raw).expanduser()
    return analysis_runs.default_root(cwd)


def resolve_report_path(
    *,
    trace_path: pathlib.Path,
    raw_out: str | None,
    raw_report_dir: str | None,
    as_json: bool,
    cwd: pathlib.Path | None = None,
) -> pathlib.Path:
    """Resolve the output path for a curator report."""
    suffix = ".json" if as_json else ".md"
    if raw_out:
        out = pathlib.Path(raw_out).expanduser()
        if out.suffix:
            return out
        return out / f"{trace_path.stem}{suffix}"
    if raw_report_dir:
        report_dir = pathlib.Path(raw_report_dir).expanduser()
    else:
        report_dir = default_report_root(cwd)
    return report_dir / f"{trace_path.stem}{suffix}"


def _report_from_dict(
    data: Mapping[str, Any],
    fallback_path: pathlib.Path,
) -> CuratorReport:
    candidates = tuple(
        TraceCandidate(
            index=int(item.get("index", 0)),
            memory_type=_str(item.get("type") or item.get("memory_type")),
            content=_str(item.get("content")),
            context=_str(item.get("context")),
            confidence=_optional_float(item.get("confidence")),
            reason=_str(item.get("reason")),
            score=_optional_float(item.get("score")),
            source_span=_optional_str(item.get("source_span")),
            evidence=_optional_str(item.get("evidence")),
            decision=_optional_str(item.get("decision")),
            store_status=_optional_str(item.get("store_status")),
            memory_id=_optional_str(item.get("memory_id")),
            superseded_id=_optional_str(item.get("superseded_id")),
            line_number=int(item.get("line_number") or 0),
        )
        for item in data.get("candidates", [])
        if isinstance(item, dict)
    )
    negative_lessons = tuple(str(item) for item in data.get("negative_lessons", []))
    warnings = tuple(str(item) for item in data.get("warnings", []))
    metrics = data.get("metrics")
    if not isinstance(metrics, dict):
        metrics = _metrics(candidates, list(negative_lessons), warnings, 0)
    trace_path = pathlib.Path(str(data.get("trace_path") or fallback_path))
    return CuratorReport(
        trace_path=trace_path,
        session_id=_optional_str(data.get("session_id")),
        project_path=_optional_str(data.get("project_path")),
        transcript_path=_optional_str(data.get("transcript_path")),
        status=_str(data.get("status") or "unknown"),
        candidates=candidates,
        negative_lessons=negative_lessons,
        metrics=metrics,
        suggested_actions=tuple(
            str(item) for item in data.get("suggested_actions", [])
        ),
        playbook_candidates=tuple(
            PlaybookCandidate(
                kind=_str(item.get("kind")),
                summary=_str(item.get("summary")),
                evidence=tuple(str(e) for e in item.get("evidence", [])),
                suggested_change=_str(item.get("suggested_change")),
                risk=_str(item.get("risk")),
            )
            for item in data.get("playbook_candidates", [])
            if isinstance(item, dict)
        ),
        warnings=warnings,
    )


def _trace_path_from_markdown(path: pathlib.Path) -> pathlib.Path | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("- trace: `") or not stripped.endswith("`"):
            continue
        raw = stripped.removeprefix("- trace: `").removesuffix("`")
        if raw:
            return pathlib.Path(raw).expanduser()
    return None


def _review_candidate_payload(candidate: TraceCandidate) -> dict[str, Any]:
    return {
        "index": candidate.index,
        "type": candidate.memory_type,
        "content": candidate.content,
        "context": candidate.context,
        "confidence": candidate.confidence,
        "reason": candidate.reason,
        "score": candidate.score,
        "source_span": candidate.source_span,
        "evidence": candidate.evidence,
        "store_status": candidate.store_status,
        "memory_id": candidate.memory_id,
    }


def _fallback_context(candidate: TraceCandidate) -> str:
    parts = []
    if candidate.source_span:
        parts.append(f"source_span={candidate.source_span}")
    if candidate.evidence:
        parts.append(f"evidence={candidate.evidence}")
    if candidate.reason:
        parts.append(f"reason={candidate.reason}")
    return "; ".join(parts) or "Accepted from Codex trace curator review."


def _candidate_from_row(index: int, row: Mapping[str, Any]) -> TraceCandidate:
    return TraceCandidate(
        index=index,
        memory_type=_str(row.get("type") or row.get("memory_type")),
        content=_str(row.get("content")),
        context=_str(row.get("context")),
        confidence=_optional_float(row.get("confidence")),
        reason=_str(row.get("reason") or row.get("decision_reason")),
        score=_optional_float(row.get("score")),
        source_span=_optional_str(row.get("source_span")),
        evidence=_optional_str(row.get("evidence")),
        decision=_optional_str(row.get("decision")),
        store_status=_optional_str(row.get("store_status")),
        memory_id=_optional_str(row.get("memory_id")),
        superseded_id=_optional_str(row.get("superseded_id")),
        line_number=int(row.get("line_number") or 0),
    )


def _metrics(
    candidates: tuple[TraceCandidate, ...],
    negative_lessons: list[str],
    warnings: tuple[str, ...],
    store_errors: int,
) -> dict[str, int | float]:
    statuses = [candidate.store_status or "missing" for candidate in candidates]
    evidence_count = sum(
        1 for candidate in candidates if candidate.source_span and candidate.evidence
    )
    total = len(candidates)
    error_count = store_errors if store_errors else statuses.count("error")
    return {
        "candidate_count": total,
        "stored_count": statuses.count("stored"),
        "duplicate_count": statuses.count("duplicate"),
        "superseded_count": statuses.count("superseded"),
        "pending_confirmation_count": statuses.count("pending_confirmation"),
        "store_error_count": error_count,
        "negative_lesson_count": len(negative_lessons),
        "warning_count": len(warnings),
        "evidence_coverage": round(evidence_count / total, 4) if total else 0.0,
    }


def _suggested_actions(metrics: Mapping[str, int | float]) -> tuple[str, ...]:
    actions = ["Review candidates before promotion; curator did not store memories."]
    if metrics.get("pending_confirmation_count", 0):
        actions.append("Review pending confirmations before accepting successors.")
    if metrics.get("store_error_count", 0):
        actions.append("Inspect store errors before changing extraction heuristics.")
    if metrics.get("negative_lesson_count", 0):
        actions.append(
            "Cluster negative lessons across reports before playbook changes."
        )
    if not metrics.get("candidate_count", 0):
        actions.append("Inspect the source trace if extraction produced no candidates.")
    return tuple(actions)


def _playbook_candidates(
    candidates: tuple[TraceCandidate, ...],
    negative_lessons: list[str],
) -> tuple[PlaybookCandidate, ...]:
    playbooks: list[PlaybookCandidate] = []
    by_reason: dict[str, list[TraceCandidate]] = {}
    for candidate in candidates:
        if candidate.reason:
            by_reason.setdefault(candidate.reason, []).append(candidate)
    for reason, grouped in sorted(by_reason.items()):
        if len(grouped) < 2:
            continue
        evidence = tuple(f"candidate:{candidate.index}" for candidate in grouped[:5])
        playbooks.append(
            PlaybookCandidate(
                kind="extractor_pattern",
                summary=f"{len(grouped)} candidates shared reason: {reason}",
                evidence=evidence,
                suggested_change="Review whether this heuristic should be tightened.",
                risk="medium",
            )
        )
    if len(negative_lessons) >= 2:
        evidence = tuple(
            f"negative_lesson:{idx}" for idx in range(len(negative_lessons))
        )
        playbooks.append(
            PlaybookCandidate(
                kind="negative_lesson_cluster",
                summary=f"{len(negative_lessons)} negative lessons in one trace",
                evidence=evidence,
                suggested_change=(
                    "Aggregate across traces before changing storage policy."
                ),
                risk="high",
            )
        )
    return tuple(playbooks)


def _event_index(payload: Mapping[str, Any], default: int = 0) -> int:
    value = payload.get("index", default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _negative_lesson_text(payload: Mapping[str, Any]) -> str:
    parts = []
    if "index" in payload:
        parts.append(f"candidate {payload['index']}")
    for key in ("reason", "status", "error_type", "message"):
        value = payload.get(key)
        if value:
            parts.append(f"{key}={value}")
    return "; ".join(parts) if parts else json.dumps(payload, sort_keys=True)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _str(value: object) -> str:
    return "" if value is None else str(value)


def _cell(value: object, limit: int = 120) -> str:
    text = _text(value)
    if len(text) > limit:
        text = text[: limit - 1] + "..."
    return text.replace("|", "\\|")


def _text(value: object) -> str:
    return " ".join(str(value).split())
