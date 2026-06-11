"""Incremental transcript cursor — read only NEW transcript bytes since last seen.

The "Continuous" gap: the learning-extraction loop re-reads the *whole* session
transcript each pass. The transcript JSONL is append-only (it keeps appending even
across compactions), so a per-session **byte offset** lets each pass read only what
was appended — ``O(new)``, not ``O(whole)``. That is what makes a per-turn (Stop
hook) trigger affordable.

The offset is a checkpoint in the shared control-plane DB (``.simba/simba.db``), so
it persists across hook invocations. Append-only safe: if the file shrank/rotated
(offset > size) the cursor resets to 0. Pure and injectable (``cwd`` selects the DB,
``transcript_path`` the stream) — deterministically testable, no LLM.
"""

from __future__ import annotations

import dataclasses
import datetime
import pathlib

import simba._vendor.peewee as pw
import simba.db

_DEFAULT_MAX_BYTES = 2_000_000


class TranscriptCursor(simba.db.BaseModel):
    """Last byte offset consumed from a session's transcript JSONL."""

    session_id = pw.CharField(unique=True)
    offset = pw.IntegerField(default=0)
    updated_at = pw.CharField()

    class Meta:
        table_name = "transcript_cursor"


simba.db.register_model(TranscriptCursor)


@dataclasses.dataclass(frozen=True)
class Window:
    """A contiguous byte range of NEW transcript content."""

    start: int
    end: int
    content: str


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def peek_offset(session_id: str, *, cwd: pathlib.Path | None = None) -> int:
    """The last consumed offset for ``session_id`` (0 if unseen)."""
    with simba.db.connect(cwd):
        row = TranscriptCursor.get_or_none(TranscriptCursor.session_id == session_id)
        return int(row.offset) if row else 0


def advance(session_id: str, offset: int, *, cwd: pathlib.Path | None = None) -> None:
    """Set the cursor for ``session_id`` to ``offset`` (idempotent upsert)."""
    with simba.db.connect(cwd):
        row = TranscriptCursor.get_or_none(TranscriptCursor.session_id == session_id)
        if row is None:
            TranscriptCursor.create(
                session_id=session_id, offset=int(offset), updated_at=_now()
            )
        else:
            row.offset = int(offset)
            row.updated_at = _now()
            row.save()


def reset(session_id: str, *, cwd: pathlib.Path | None = None) -> None:
    """Rewind the cursor to the start of the stream."""
    advance(session_id, 0, cwd=cwd)


def next_window(
    transcript_path: str | pathlib.Path,
    *,
    session_id: str,
    cwd: pathlib.Path | None = None,
    max_bytes: int = _DEFAULT_MAX_BYTES,
) -> Window | None:
    """Return the NEW bytes appended since the cursor — WITHOUT advancing it.

    The caller advances explicitly (``advance(session_id, window.end)``) once it has
    durably handled the window, so a crash between read and handle re-reads rather
    than drops. Returns ``None`` when the file is missing or nothing is new. Caps the
    window at ``max_bytes`` (the caller advances past the cap, so a huge backlog is
    drained in bounded chunks rather than re-scanned forever).
    """
    p = pathlib.Path(transcript_path)
    try:
        size = p.stat().st_size
    except OSError:
        return None
    start = peek_offset(session_id, cwd=cwd)
    if start > size:  # file shrank / rotated → restart from the top
        start = 0
    if start >= size:
        return None
    end = min(size, start + max(1, int(max_bytes)))
    try:
        with p.open("rb") as fh:
            fh.seek(start)
            data = fh.read(end - start)
    except OSError:
        return None
    return Window(
        start=start,
        end=start + len(data),
        content=data.decode("utf-8", errors="replace"),
    )
