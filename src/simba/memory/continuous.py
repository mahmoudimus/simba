"""Continuous extraction — the Stop-hook rails that make learning-extraction
incremental and per-turn (the "Continuous" gap).

Today extraction fires only at PreCompact and re-reads the *whole* transcript. This
moves the trigger to the Stop hook (every turn) and reads only the NEW window via the
incremental cursor (:mod:`simba.memory.transcript_cursor`), enqueueing it durably for
the (future, Evaluator-gated) scored extract→score→keep/drop worker. This module is
the cheap, no-LLM plumbing; **default-off** until that worker + its Importance rubric
+ a gold-set Evaluator exist. The enqueue is an append-only JSONL handoff next to the
control-plane DB (raw transcript stays the source of truth — we store byte ranges).
"""

from __future__ import annotations

import datetime
import json
import pathlib
import typing

import simba.db
import simba.memory.transcript_cursor as transcript_cursor

_DEFAULT_MAX_BYTES = 2_000_000


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def queue_path(cwd: pathlib.Path | None = None) -> pathlib.Path:
    """The append-only pending-window queue, next to ``.simba/simba.db``."""
    return simba.db.get_db_path(cwd).parent / "continuous" / "pending.jsonl"


def enqueue(cwd: pathlib.Path | None, record: dict) -> None:
    """Append one pending-window record (append-only; never rewrites)."""
    p = queue_path(cwd)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def drain(cwd: pathlib.Path | None = None) -> list[dict]:
    """Read all pending-window records (for the worker / tests). Non-destructive."""
    p = queue_path(cwd)
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def on_stop(
    hook_input: dict,
    cfg: typing.Any,
    *,
    cwd: pathlib.Path | None = None,
) -> int:
    """Stop-hook entry: enqueue the incremental transcript window, advance the cursor.

    Returns the number of windows enqueued (0 or 1). Default-off and fail-soft: the
    caller wraps it in suppression, but this also no-ops on missing fields. The
    cursor advances only after a durable enqueue, so a crash re-reads, not drops.
    """
    if not getattr(cfg, "continuous_extraction_enabled", False):
        return 0
    transcript_path = hook_input.get("transcript_path")
    session_id = hook_input.get("session_id")
    if not session_id and transcript_path:
        session_id = pathlib.Path(transcript_path).stem
    if not transcript_path or not session_id:
        return 0
    if cwd is None and hook_input.get("cwd"):
        cwd = pathlib.Path(hook_input["cwd"])
    max_bytes = getattr(cfg, "continuous_extraction_max_bytes", _DEFAULT_MAX_BYTES)

    window = transcript_cursor.next_window(
        transcript_path, session_id=session_id, cwd=cwd, max_bytes=max_bytes
    )
    if window is None:
        return 0
    if not window.content.strip():
        # whitespace/blank window — advance past it so we don't re-scan it forever
        transcript_cursor.advance(session_id, window.end, cwd=cwd)
        return 0

    # NOTE: no Importance gate yet (that's the scored worker, gated by the Evaluator).
    # The rails enqueue every non-blank window; keep/drop scoring comes later.
    enqueue(
        cwd,
        {
            "session_id": session_id,
            "project_path": hook_input.get("project_path", ""),
            "transcript_path": str(transcript_path),
            "start": window.start,
            "end": window.end,
            "n_bytes": window.end - window.start,
            "ts": _now(),
        },
    )
    transcript_cursor.advance(session_id, window.end, cwd=cwd)
    return 1
