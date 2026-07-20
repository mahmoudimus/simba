"""Shared bounded-tail file reader.

Promoted (2026-07-20) from ``simba.hooks.pre_tool_use._read_tail_bytes``: the
same whole-file-read shape recurred in two more places that fire on EVERY
Stop hook (``simba.tailor.hook.process_hook`` and
``simba.hooks.usage_signals.extract_last_assistant_text``), one of which was
a co-culprit in a live 30GB daemon RSS balloon. Rather than three independent
copies of the same seek/discard-partial-line logic, this is now the single
shared primitive; ``pre_tool_use`` keeps its private name as an alias so
existing call sites and tests are unaffected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pathlib


def read_tail_bytes(path: pathlib.Path, cap_bytes: int) -> tuple[bytes, int]:
    """Read at most the last ``cap_bytes`` of ``path``, in binary mode.

    Returns ``(tail_bytes, tail_start_offset)`` where ``tail_start_offset`` is
    the absolute file offset of the first byte actually kept. When the file is
    within ``cap_bytes`` (or ``cap_bytes <= 0``, meaning uncapped), the whole
    file is returned with offset 0. Otherwise the seek lands mid-file, so the
    (possibly-partial) leading line is discarded -- up to and including its
    first ``\\n`` -- so callers always see whole JSONL lines; a window with no
    newline at all (a pathological single giant line) yields an empty tail
    rather than a corrupt partial line.

    May raise ``OSError`` -- callers decide the fallback.
    """
    size = path.stat().st_size
    start = max(0, size - cap_bytes) if cap_bytes > 0 else 0
    with path.open("rb") as fh:
        fh.seek(start)
        data = fh.read()
    if start == 0:
        return data, 0
    nl = data.find(b"\n")
    if nl == -1:
        return b"", size
    return data[nl + 1 :], start + nl + 1
