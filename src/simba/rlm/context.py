from __future__ import annotations

import dataclasses
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError


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


class _Document:
    """Internal document with a line index for char-offset math."""

    def __init__(self, doc_id: str, text: str) -> None:
        self.doc_id = doc_id
        self.text = text
        self.lines = text.split("\n")
        self.line_starts: list[int] = [0]
        pos = 0
        for line in self.lines[:-1]:
            pos += len(line) + 1  # +1 for the newline
            self.line_starts.append(pos)


class DocumentStore:
    def __init__(self) -> None:
        self._docs: dict[str, _Document] = {}

    def add(self, doc_id: str, text: str) -> None:
        self._docs[doc_id] = _Document(doc_id, text)

    def get(self, doc_id: str) -> _Document:
        if doc_id not in self._docs:
            raise DocumentNotFoundError(doc_id)
        return self._docs[doc_id]

    def has(self, doc_id: str) -> bool:
        return doc_id in self._docs

    def remove(self, doc_id: str) -> None:
        self._docs.pop(doc_id, None)


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
            for line_num, line in enumerate(doc.lines, start=1):
                if len(out) >= max_matches:
                    break
                m = regex.search(line)
                if not m:
                    continue
                line_start = doc.line_starts[line_num - 1]
                start, end = line_start + m.start(), line_start + m.end()
                out.append(
                    SearchMatch(
                        doc_id=doc.doc_id,
                        line_number=line_num,
                        match_text=m.group(0),
                        start_char=start,
                        end_char=end,
                        context_before=doc.text[max(0, start - ctx_chars):start],
                        context_after=doc.text[end:end + ctx_chars],
                    )
                )
            return out

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            return executor.submit(_run).result(
                timeout=self._cfg.regex_timeout_seconds
            )
        except FutureTimeoutError as exc:
            raise SearchError(
                f"regex search timed out (>{self._cfg.regex_timeout_seconds}s)"
            ) from exc
        finally:
            executor.shutdown(wait=False)

    def peek(self, doc_id: str, start_char: int, end_char: int) -> str:
        doc = self._store.get(doc_id)
        return doc.text[max(0, start_char):min(len(doc.text), end_char)]

    def head(self, doc_id: str, n_lines: int = 20) -> str:
        return "\n".join(self._store.get(doc_id).lines[:n_lines])

    def tail(self, doc_id: str, n_lines: int = 20) -> str:
        lines = self._store.get(doc_id).lines
        return "\n".join(lines[-n_lines:] if n_lines > 0 else [])

    def window(self, doc_id: str, around_char: int, radius: int) -> str:
        doc = self._store.get(doc_id)
        lo = max(0, around_char - radius)
        hi = min(len(doc.text), around_char + radius)
        return doc.text[lo:hi]


class RLMContext:
    """Facade over a DocumentStore + DocumentSearcher."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self.documents = DocumentStore()
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
