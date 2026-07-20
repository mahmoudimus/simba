"""Document store + search primitives for the RLM (Recursive Language Model)
memory layer.

Bounded/lazy by design (2026-07-20 incident): malloc_history on a live daemon
(36GB RSS, 50.9GB peak) attributed the heap to a single 2153MB transcript
retained whole -- a 2062MB read buffer, 16.5GB in the newline decoder, and
21.3GB across 894k live per-line strings from the old ``text.split("\\n")``
approach. Documents at or under ``rlm.max_document_mb`` keep a fast in-memory
path (``.text`` / ``.lines``, unchanged from before); anything larger is never
refused -- it is served through an offset-indexed lazy mode that seeks into
its source file and decodes only the bytes a given read actually needs. The
DocumentStore also caps its aggregate retained footprint (``rlm.store_budget_mb``)
and evicts least-recently-used documents' retained data under that cap.
"""

from __future__ import annotations

import array
import bisect
import codecs
import collections
import contextlib
import dataclasses
import logging
import os
import pathlib
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

logger = logging.getLogger(__name__)

_CHUNK_BYTES = 1 << 20  # 1 MiB streaming/decoding chunk -- bounds per-call memory
_BYTES_PER_MB = 1024 * 1024


class SearchError(Exception):
    """Raised on an unsafe/invalid pattern or a search timeout."""


class DocumentNotFoundError(KeyError):
    """Raised when a doc_id is not present in the store."""


@dataclasses.dataclass
class SearchMatch:
    doc_id: str
    line_number: int
    match_text: str
    start_char: int
    end_char: int
    context_before: str = ""
    context_after: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Streaming line index -- shared by ingest (build once) and grep (re-stream).
# ---------------------------------------------------------------------------


def _stream_lines(path: pathlib.Path, encoding: str, errors: str):
    """Yield ``(char_start, byte_start, line_text)`` for each line of *path*,
    reproducing ``text.split("\\n")`` semantics exactly (including a trailing
    empty element when the file ends with a newline) without ever holding
    more than one line -- plus a bounded read chunk -- in memory at a time.

    ``\\n`` (0x0A) is never a continuation byte in UTF-8 (or any ASCII-
    compatible encoding), so splitting the raw byte stream on it is always a
    safe decode boundary; a line's bytes are only decoded once the full line
    (up to and including its newline) has been accumulated.
    """
    char_pos = 0
    abs_pos = 0
    line_start_byte = 0
    pending = b""
    with open(path, "rb") as f:
        while True:
            raw = f.read(_CHUNK_BYTES)
            if not raw:
                break
            chunk_start_abs = abs_pos
            abs_pos += len(raw)
            start = 0
            while True:
                idx = raw.find(b"\n", start)
                if idx == -1:
                    pending += raw[start:]
                    break
                full = pending + raw[start:idx]
                pending = b""
                decoded = full.decode(encoding, errors=errors)
                yield char_pos, line_start_byte, decoded
                char_pos += len(decoded) + 1
                line_start_byte = chunk_start_abs + idx + 1
                start = idx + 1
        final = pending.decode(encoding, errors=errors)
        yield char_pos, line_start_byte, final


def _build_line_index(path: pathlib.Path, encoding: str, errors: str):
    """One streaming pass building only the offset index (char + byte start
    per line) -- the line text itself is discarded immediately, never
    accumulated. Memory cost is O(num_lines) for the index, not O(file size)."""
    char_starts = array.array("Q")
    byte_starts = array.array("Q")
    total_chars = 0
    for char_pos, byte_pos, line in _stream_lines(path, encoding, errors):
        char_starts.append(char_pos)
        byte_starts.append(byte_pos)
        total_chars = char_pos + len(line)
    return char_starts, byte_starts, total_chars


def _decode_from(
    path: pathlib.Path, byte_offset: int, needed_chars: int, encoding: str, errors: str
) -> str:
    """Seek to *byte_offset* (always a line boundary -- see ``_stream_lines``)
    and decode forward just far enough to cover *needed_chars* characters."""
    decoder = codecs.getincrementaldecoder(encoding)(errors=errors)
    collected: list[str] = []
    total = 0
    with open(path, "rb") as f:
        f.seek(byte_offset)
        while total < needed_chars:
            raw = f.read(_CHUNK_BYTES)
            if not raw:
                piece = decoder.decode(b"", final=True)
                if piece:
                    collected.append(piece)
                break
            piece = decoder.decode(raw)
            if piece:
                collected.append(piece)
                total += len(piece)
    return "".join(collected)


class _Document:
    """A document, either eager (small: retains ``.text`` / ``.lines`` / the
    original char-offset ``.line_starts`` list -- the pre-incident fast path,
    unchanged) or lazy (large: an offset-indexed view backed by a file on
    disk; never retains the full text)."""

    def __init__(
        self,
        doc_id: str,
        *,
        text: str | None = None,
        source_path: pathlib.Path | None = None,
        encoding: str = "utf-8",
        errors: str = "replace",
        owns_source: bool = False,
    ) -> None:
        self.doc_id = doc_id
        self.encoding = encoding
        self.errors = errors
        self.source_path = source_path
        self.owns_source = owns_source

        if text is not None:
            self.lazy = False
            self.text = text
            self.lines = text.split("\n")
            self.line_starts: list[int] = [0]
            pos = 0
            for line in self.lines[:-1]:
                pos += len(line) + 1  # +1 for the newline
                self.line_starts.append(pos)
            self._total_chars = len(text)
            self._line_char_starts: array.array | None = None
            self._line_byte_starts: array.array | None = None
            # Honest retained-bytes accounting, computed ONCE here (O(n_lines))
            # rather than on every budget check. A flat `sys.getsizeof(text) * 2`
            # estimate ignores the per-line string OBJECT overhead in `.lines`
            # and the per-int object overhead in `.line_starts` -- both scale
            # with line count, not text size, so many-short-line documents
            # under-counted 3-5x in production (see module docstring).
            self._eager_retained_bytes = (
                sys.getsizeof(self.text)
                + sys.getsizeof(self.lines)
                + sum(sys.getsizeof(line) for line in self.lines)
                + sys.getsizeof(self.line_starts)
                + sum(sys.getsizeof(n) for n in self.line_starts)
            )
        else:
            if source_path is None:
                raise ValueError("_Document requires either text or source_path")
            self.lazy = True
            self._line_char_starts, self._line_byte_starts, self._total_chars = (
                _build_line_index(source_path, encoding, errors)
            )

    # -- metadata -----------------------------------------------------------

    @property
    def char_length(self) -> int:
        return self._total_chars

    def is_index_resident(self) -> bool:
        """True unless this is a lazy doc whose offset index was freed by an
        LRU budget eviction (rebuilt transparently on the next read)."""
        return (not self.lazy) or self._line_char_starts is not None

    def retained_bytes(self) -> int:
        """Resident bytes -- used only to enforce ``store_budget_mb``, not
        required to be exact. Lazy: the two offset-index arrays (8
        bytes/entry each; 0 once evicted) -- these are packed C values with
        no per-entry Python object overhead, so the flat itemsize multiply
        is already honest. Eager: the honest sum computed once at ingest in
        ``__init__`` (see ``_eager_retained_bytes``) -- text + the per-line
        list (list object + every line's string object) + the per-line
        offset list (list object + every int's object). A prior ``~2x the
        text object's size`` shortcut ignored that per-object overhead,
        which scales with line count rather than text size and under-
        counted 3-5x for many-short-line documents in production."""
        if self.lazy:
            if self._line_char_starts is None:
                return 0
            return (
                len(self._line_char_starts) + len(self._line_byte_starts)
            ) * self._line_char_starts.itemsize
        return self._eager_retained_bytes

    def _ensure_index(self) -> None:
        if self.lazy and self._line_char_starts is None:
            self._line_char_starts, self._line_byte_starts, total = _build_line_index(
                self.source_path, self.encoding, self.errors
            )
            self._total_chars = total
            logger.debug(
                "rlm document %s: rebuilt offset index (%d lines) after LRU eviction",
                self.doc_id,
                len(self._line_char_starts),
            )

    # -- reads ----------------------------------------------------------

    def read_range(self, start_char: int, end_char: int) -> str:
        total = self._total_chars
        start = max(0, start_char)
        end = min(total, end_char)
        if start >= end:
            return ""
        if not self.lazy:
            return self.text[start:end]
        self._ensure_index()
        idx = max(0, bisect.bisect_right(self._line_char_starts, start) - 1)
        byte_offset = self._line_byte_starts[idx]
        char_offset = self._line_char_starts[idx]
        decoded = _decode_from(
            self.source_path, byte_offset, end - char_offset, self.encoding, self.errors
        )
        return decoded[start - char_offset : end - char_offset]

    def read_head(self, n_lines: int) -> str:
        if not self.lazy:
            return "\n".join(self.lines[:n_lines])
        if n_lines <= 0:
            return ""
        self._ensure_index()
        num_lines = len(self._line_char_starts)
        if n_lines >= num_lines:
            return self.read_range(0, self._total_chars)
        text = self.read_range(0, self._line_char_starts[n_lines])
        return text[:-1] if text.endswith("\n") else text

    def read_tail(self, n_lines: int) -> str:
        if not self.lazy:
            return "\n".join(self.lines[-n_lines:] if n_lines > 0 else [])
        if n_lines <= 0:
            return ""
        self._ensure_index()
        num_lines = len(self._line_char_starts)
        start_idx = max(0, num_lines - n_lines)
        return self.read_range(self._line_char_starts[start_idx], self._total_chars)

    def iter_lines(self):
        """Yield ``(line_number, char_start, line_text)`` triples, one line
        at a time. Lazy mode streams the source file in bounded chunks and
        never materializes the full text -- used by grep."""
        if not self.lazy:
            for i, line in enumerate(self.lines, start=1):
                yield i, self.line_starts[i - 1], line
            return
        for i, (char_pos, _byte_pos, line) in enumerate(
            _stream_lines(self.source_path, self.encoding, self.errors), start=1
        ):
            yield i, char_pos, line

    # -- eviction -------------------------------------------------------

    def evict_retained(self, spill_fn) -> bool:
        """Free this doc's retained payload for LRU budget eviction. Keeps
        doc_id + source identity so a later read transparently rebuilds it.
        Returns False if there was nothing left to free."""
        if self.lazy:
            if self._line_char_starts is None:
                return False
            self._line_char_starts = None
            self._line_byte_starts = None
            return True
        if self.source_path is None:
            self.source_path = spill_fn(self.text)
            self.owns_source = True
        self.lazy = True
        del self.text
        del self.lines
        del self.line_starts
        self._line_char_starts = None
        self._line_byte_starts = None
        return True


class DocumentStore:
    def __init__(
        self,
        *,
        max_document_mb: float = 64.0,
        store_budget_mb: float = 256.0,
        tmp_dir: pathlib.Path | str | None = None,
    ) -> None:
        self._docs: dict[str, _Document] = {}
        self._order: collections.OrderedDict[str, None] = collections.OrderedDict()
        self._max_document_mb = max_document_mb
        self._max_document_bytes = int(max_document_mb * _BYTES_PER_MB)
        self._store_budget_bytes = int(store_budget_mb * _BYTES_PER_MB)
        self._tmp_dir = pathlib.Path(tmp_dir) if tmp_dir is not None else None

    # -- spill dir --------------------------------------------------------

    def _ensure_tmp_dir(self) -> pathlib.Path:
        if self._tmp_dir is None:
            self._tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="simba-rlm-docstore-"))
        else:
            self._tmp_dir.mkdir(parents=True, exist_ok=True)
        return self._tmp_dir

    def _spill(self, text: str) -> pathlib.Path:
        """Write oversized text handed to add() directly to a temp file so it
        can be served lazily. Note: `text` is already fully resident in the
        caller's memory by the time it reaches us -- this can only stop US
        from retaining it further, not undo the caller's allocation.
        Production callers with a file on disk should prefer add_path(),
        which never materializes the full content at all."""
        tmp_dir = self._ensure_tmp_dir()
        fd, name = tempfile.mkstemp(dir=str(tmp_dir), suffix=".spill")
        with os.fdopen(fd, "w", encoding="utf-8", errors="replace") as fh:
            fh.write(text)
        return pathlib.Path(name)

    # -- ingest -------------------------------------------------------------

    def add(self, doc_id: str, text: str) -> None:
        if len(text) <= self._max_document_bytes:
            doc = _Document(doc_id, text=text)
        else:
            path = self._spill(text)
            doc = _Document(
                doc_id,
                source_path=path,
                encoding="utf-8",
                errors="replace",
                owns_source=True,
            )
            logger.info(
                "rlm document %s: %d chars exceeds rlm.max_document_mb (%.3fMB) -- "
                "offset-indexed lazy mode",
                doc_id,
                len(text),
                self._max_document_mb,
            )
        self._store(doc_id, doc)

    def add_path(
        self,
        doc_id: str,
        path: pathlib.Path,
        *,
        encoding: str = "utf-8",
        errors: str = "replace",
    ) -> None:
        """Ingest *path* without ever reading it whole unless it is small.
        This is the production path (TranscriptProvider) -- unlike add(), the
        caller never materializes the file's content itself."""
        size = path.stat().st_size
        if size <= self._max_document_bytes:
            text = path.read_text(encoding=encoding, errors=errors)
            doc = _Document(
                doc_id, text=text, source_path=path, encoding=encoding, errors=errors
            )
        else:
            doc = _Document(doc_id, source_path=path, encoding=encoding, errors=errors)
            logger.info(
                "rlm document %s: %d bytes exceeds rlm.max_document_mb (%.3fMB) -- "
                "offset-indexed lazy mode",
                doc_id,
                size,
                self._max_document_mb,
            )
        self._store(doc_id, doc)

    def _store(self, doc_id: str, doc: _Document) -> None:
        self._docs[doc_id] = doc
        self._touch(doc_id)
        self._enforce_budget()

    # -- LRU / budget ---------------------------------------------------

    def _touch(self, doc_id: str) -> None:
        self._order[doc_id] = None
        self._order.move_to_end(doc_id)

    def _total_retained_bytes(self) -> int:
        return sum(d.retained_bytes() for d in self._docs.values())

    def _enforce_budget(self) -> None:
        budget = self._store_budget_bytes
        if budget <= 0:
            return
        for doc_id in list(self._order.keys()):  # least-recently-used first
            if self._total_retained_bytes() <= budget:
                return
            doc = self._docs.get(doc_id)
            if doc is None:
                continue
            if doc.evict_retained(self._spill):
                logger.debug(
                    "rlm document store: evicted retained data for %s "
                    "(LRU, over rlm.store_budget_mb)",
                    doc_id,
                )

    # -- lookup -----------------------------------------------------------

    def get(self, doc_id: str) -> _Document:
        if doc_id not in self._docs:
            raise DocumentNotFoundError(doc_id)
        self._touch(doc_id)
        return self._docs[doc_id]

    def has(self, doc_id: str) -> bool:
        return doc_id in self._docs

    def remove(self, doc_id: str) -> None:
        doc = self._docs.pop(doc_id, None)
        self._order.pop(doc_id, None)
        if doc is not None and doc.owns_source and doc.source_path is not None:
            with contextlib.suppress(OSError):
                doc.source_path.unlink()


class DocumentSearcher:
    def __init__(self, store: DocumentStore, cfg) -> None:
        self._store = store
        self._cfg = cfg

    def _validate_regex(self, pattern: str) -> None:
        if not pattern or not pattern.strip():
            raise SearchError("pattern cannot be empty or whitespace-only")
        if len(pattern) > self._cfg.max_pattern_length:
            raise SearchError(
                f"pattern too long ({len(pattern)} > {self._cfg.max_pattern_length})"
            )
        if re.search(r"\([^)]*[+*]\)\s*[+*]", pattern):
            raise SearchError("nested quantifier detected (ReDoS risk)")
        if pattern.count(".*") + pattern.count(".+") > 3:
            raise SearchError("too many wildcard quantifiers (ReDoS risk)")

    def grep(
        self, doc_id: str, pattern: str, max_matches: int | None = None
    ) -> list[SearchMatch]:
        if max_matches is None:
            max_matches = self._cfg.max_search_matches
        self._validate_regex(pattern)
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise SearchError(f"invalid regex: {exc}") from exc

        doc = self._store.get(doc_id)
        ctx_chars = self._cfg.search_context_chars

        def _run() -> list[SearchMatch]:
            out: list[SearchMatch] = []
            for line_num, line_start, line in doc.iter_lines():
                if len(out) >= max_matches:
                    break
                m = regex.search(line)
                if not m:
                    continue
                start, end = line_start + m.start(), line_start + m.end()
                out.append(
                    SearchMatch(
                        doc_id=doc.doc_id,
                        line_number=line_num,
                        match_text=m.group(0),
                        start_char=start,
                        end_char=end,
                        context_before=doc.read_range(max(0, start - ctx_chars), start),
                        context_after=doc.read_range(end, end + ctx_chars),
                    )
                )
            return out

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            return executor.submit(_run).result(timeout=self._cfg.regex_timeout_seconds)
        except FutureTimeoutError as exc:
            raise SearchError(
                f"regex search timed out (>{self._cfg.regex_timeout_seconds}s)"
            ) from exc
        finally:
            executor.shutdown(wait=False)

    def peek(self, doc_id: str, start_char: int, end_char: int) -> str:
        return self._store.get(doc_id).read_range(start_char, end_char)

    def head(self, doc_id: str, n_lines: int = 20) -> str:
        return self._store.get(doc_id).read_head(n_lines)

    def tail(self, doc_id: str, n_lines: int = 20) -> str:
        return self._store.get(doc_id).read_tail(n_lines)

    def window(self, doc_id: str, around_char: int, radius: int) -> str:
        doc = self._store.get(doc_id)
        return doc.read_range(around_char - radius, around_char + radius)


class RLMContext:
    """Facade over a DocumentStore + DocumentSearcher."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self.documents = DocumentStore(
            max_document_mb=getattr(cfg, "max_document_mb", 64.0),
            store_budget_mb=getattr(cfg, "store_budget_mb", 256.0),
        )
        self.searcher = DocumentSearcher(self.documents, cfg)

    def add_document(self, doc_id: str, text: str) -> None:
        self.documents.add(doc_id, text)

    def grep(
        self, doc_id: str, pattern: str, max_matches: int | None = None
    ) -> list[SearchMatch]:
        return self.searcher.grep(doc_id, pattern, max_matches)

    def peek(self, doc_id: str, start_char: int, end_char: int) -> str:
        return self.searcher.peek(doc_id, start_char, end_char)

    def head(self, doc_id: str, n_lines: int = 20) -> str:
        return self.searcher.head(doc_id, n_lines)

    def tail(self, doc_id: str, n_lines: int = 20) -> str:
        return self.searcher.tail(doc_id, n_lines)

    def window(self, doc_id: str, around_char: int, radius: int) -> str:
        return self.searcher.window(doc_id, around_char, radius)
