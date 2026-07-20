"""Tests for RLM context primitives."""

from __future__ import annotations

import logging
import pathlib
import sys

import pytest

import simba.rlm.context as ctx


class TestDocumentStore:
    def test_add_get_has(self):
        store = ctx.DocumentStore()
        store.add("d1", "hello\nworld")
        assert store.has("d1")
        doc = store.get("d1")
        assert doc.text == "hello\nworld"
        assert doc.lines == ["hello", "world"]

    def test_line_starts(self):
        store = ctx.DocumentStore()
        store.add("d1", "ab\ncde\nf")
        # offsets: "ab"=0, "cde"=3 (after "ab\n"), "f"=7 (after "cde\n")
        assert store.get("d1").line_starts == [0, 3, 7]

    def test_get_missing_raises(self):
        store = ctx.DocumentStore()
        with pytest.raises(ctx.DocumentNotFoundError):
            store.get("nope")

    def test_remove(self):
        store = ctx.DocumentStore()
        store.add("d1", "x")
        store.remove("d1")
        assert not store.has("d1")
        store.remove("d1")  # idempotent, no raise


class TestSearchMatch:
    def test_to_dict(self):
        m = ctx.SearchMatch("d1", 2, "world", 6, 11, "hello\n", "")
        d = m.to_dict()
        assert d["doc_id"] == "d1"
        assert d["line_number"] == 2
        assert d["match_text"] == "world"
        assert d["start_char"] == 6
        assert d["end_char"] == 11


class _Cfg:
    max_search_matches = 20
    search_context_chars = 5
    regex_timeout_seconds = 2.0
    max_pattern_length = 500


class TestGrep:
    def _ctx(self):
        c = ctx.RLMContext(_Cfg())
        c.add_document("d1", "alpha beta\ngamma beta delta\nepsilon")
        return c

    def test_grep_finds_matches_with_offsets(self):
        c = self._ctx()
        matches = c.grep("d1", "beta")
        assert len(matches) == 2
        first = matches[0]
        assert first.line_number == 1
        # "beta" starts at char 6 on line 1
        assert first.start_char == 6
        assert first.end_char == 10
        assert first.match_text == "beta"

    def test_grep_respects_max_matches(self):
        c = self._ctx()
        assert len(c.grep("d1", "beta", max_matches=1)) == 1

    def test_grep_context_window(self):
        c = self._ctx()
        m = c.grep("d1", "gamma")[0]
        # context_before clamped to search_context_chars=5
        assert len(m.context_before) <= 5

    def test_grep_empty_pattern_raises(self):
        c = self._ctx()
        with pytest.raises(ctx.SearchError):
            c.grep("d1", "   ")

    def test_grep_pattern_too_long_raises(self):
        c = self._ctx()
        with pytest.raises(ctx.SearchError):
            c.grep("d1", "a" * 501)

    def test_grep_nested_quantifier_rejected(self):
        c = self._ctx()
        with pytest.raises(ctx.SearchError):
            c.grep("d1", "(a+)+")

    def test_grep_invalid_regex_raises(self):
        c = self._ctx()
        with pytest.raises(ctx.SearchError):
            c.grep("d1", "(unclosed")


class TestSlices:
    def _ctx(self):
        c = ctx.RLMContext(_Cfg())
        c.add_document("d1", "line0\nline1\nline2\nline3\nline4")
        return c

    def test_peek_clamps_bounds(self):
        c = self._ctx()
        assert c.peek("d1", 0, 5) == "line0"
        assert c.peek("d1", -10, 5) == "line0"  # clamps low
        assert c.peek("d1", 0, 10_000) == c.documents.get("d1").text  # clamps high

    def test_head(self):
        c = self._ctx()
        assert c.head("d1", 2) == "line0\nline1"

    def test_tail(self):
        c = self._ctx()
        assert c.tail("d1", 2) == "line3\nline4"
        assert c.tail("d1", 0) == ""

    def test_window(self):
        c = self._ctx()
        text = c.documents.get("d1").text
        idx = text.index("line2")
        w = c.window("d1", idx, 3)
        assert "ne2" in w or "lin" in w  # a slice around the offset
        assert len(w) <= 6


# ---------------------------------------------------------------------------
# Bounded/lazy DocumentStore (2026-07-20 RSS incident: a single huge document
# retained as one string + one str-per-line blew a daemon to 50.9GB peak).
# ---------------------------------------------------------------------------


class _LazyCfg:
    max_search_matches = 20
    search_context_chars = 5
    regex_timeout_seconds = 2.0
    max_pattern_length = 500


# Deliberately mixes: combining/precomposed accents, an astral-plane emoji
# (surrogate pair on some platforms), consecutive empty lines, and no
# trailing newline -- the awkward cases for a byte<->char offset index.
UNICODE_FIXTURE = (
    "line zero has café and é\n"
    "\n"
    "second line with emoji \U0001f600 here\n"
    "\n"
    "\n"
    "final line no trailing newline"
)


class TestLazyDocumentNeverRetainsText:
    def test_over_cap_doc_has_no_text_or_lines(self, tmp_path):
        path = tmp_path / "big.txt"
        path.write_text("x" * 5000, encoding="utf-8")
        store = ctx.DocumentStore(max_document_mb=0.001, tmp_dir=tmp_path / "spill")
        store.add_path("big", path)
        doc = store.get("big")
        assert doc.lazy is True
        assert not hasattr(doc, "text")
        assert not hasattr(doc, "lines")

    def test_over_cap_doc_still_serves_reads(self, tmp_path):
        path = tmp_path / "big.txt"
        path.write_text("y" * 5000, encoding="utf-8")
        store = ctx.DocumentStore(max_document_mb=0.001, tmp_dir=tmp_path / "spill")
        store.add_path("big", path)
        doc = store.get("big")
        assert doc.read_range(0, 5) == "yyyyy"

    def test_under_cap_doc_keeps_fast_path(self, tmp_path):
        path = tmp_path / "small.txt"
        path.write_text("small doc", encoding="utf-8")
        store = ctx.DocumentStore(max_document_mb=64.0, tmp_dir=tmp_path / "spill")
        store.add_path("small", path)
        doc = store.get("small")
        assert doc.lazy is False
        assert doc.text == "small doc"
        assert doc.lines == ["small doc"]

    def test_over_cap_text_add_spills_and_stays_lazy(self, tmp_path):
        store = ctx.DocumentStore(max_document_mb=0.001, tmp_dir=tmp_path / "spill")
        store.add("big", "z" * 5000)
        doc = store.get("big")
        assert doc.lazy is True
        assert not hasattr(doc, "text")
        assert not hasattr(doc, "lines")
        assert doc.read_range(0, 5) == "zzzzz"

    def test_lazy_path_logs_info(self, tmp_path, caplog):
        caplog.set_level(logging.INFO, logger="simba.rlm.context")
        path = tmp_path / "big.txt"
        path.write_text("z" * 5000, encoding="utf-8")
        store = ctx.DocumentStore(max_document_mb=0.001, tmp_dir=tmp_path / "spill")
        store.add_path("big", path)
        assert any(
            "big" in r.message and "lazy" in r.message.lower() for r in caplog.records
        )


class TestLazyEagerParity:
    """Same fixture text, one eager store and one forced-lazy store; every
    windowed op must return byte-identical results."""

    def _stores(self, tmp_path, text=UNICODE_FIXTURE):
        path = tmp_path / "fixture.txt"
        path.write_text(text, encoding="utf-8")
        eager = ctx.DocumentStore(max_document_mb=1000.0, tmp_dir=tmp_path / "e")
        eager.add_path("d", path)
        # Negative cap (not 0.0): 0.0 would still take the fast path for a
        # zero-byte file (0 <= 0), which defeats the empty-file parity case.
        lazy = ctx.DocumentStore(max_document_mb=-1.0, tmp_dir=tmp_path / "l")
        lazy.add_path("d", path)
        assert eager.get("d").lazy is False
        assert lazy.get("d").lazy is True
        return eager, lazy

    @pytest.mark.parametrize(
        "start,end",
        [
            (0, 5),
            (0, 10_000),
            (-10, 5),
            (5, 3),
            (10, 10),
            (7, 40),
            (0, 1),
            (0, 0),
            (1, 100),
            (20, 45),
            (58, 61),  # straddles the emoji
        ],
    )
    def test_peek_parity(self, tmp_path, start, end):
        eager, lazy = self._stores(tmp_path)
        se = ctx.DocumentSearcher(eager, _LazyCfg())
        sl = ctx.DocumentSearcher(lazy, _LazyCfg())
        assert sl.peek("d", start, end) == se.peek("d", start, end)

    @pytest.mark.parametrize(
        "around,radius", [(0, 5), (10, 3), (500, 1000), (1, 0), (60, 5)]
    )
    def test_window_parity(self, tmp_path, around, radius):
        eager, lazy = self._stores(tmp_path)
        se = ctx.DocumentSearcher(eager, _LazyCfg())
        sl = ctx.DocumentSearcher(lazy, _LazyCfg())
        assert sl.window("d", around, radius) == se.window("d", around, radius)

    @pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 6, 20, 1000])
    def test_head_parity(self, tmp_path, n):
        eager, lazy = self._stores(tmp_path)
        se = ctx.DocumentSearcher(eager, _LazyCfg())
        sl = ctx.DocumentSearcher(lazy, _LazyCfg())
        assert sl.head("d", n) == se.head("d", n)

    @pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 6, 20, 1000])
    def test_tail_parity(self, tmp_path, n):
        eager, lazy = self._stores(tmp_path)
        se = ctx.DocumentSearcher(eager, _LazyCfg())
        sl = ctx.DocumentSearcher(lazy, _LazyCfg())
        assert sl.tail("d", n) == se.tail("d", n)

    def test_head_tail_parity_no_trailing_newline_fixture(self, tmp_path):
        text = "one\ntwo\nthree"
        eager, lazy = self._stores(tmp_path, text=text)
        se = ctx.DocumentSearcher(eager, _LazyCfg())
        sl = ctx.DocumentSearcher(lazy, _LazyCfg())
        for n in range(0, 5):
            assert sl.head("d", n) == se.head("d", n)
            assert sl.tail("d", n) == se.tail("d", n)

    def test_head_tail_parity_empty_file(self, tmp_path):
        eager, lazy = self._stores(tmp_path, text="")
        se = ctx.DocumentSearcher(eager, _LazyCfg())
        sl = ctx.DocumentSearcher(lazy, _LazyCfg())
        for n in range(0, 3):
            assert sl.head("d", n) == se.head("d", n)
            assert sl.tail("d", n) == se.tail("d", n)

    def test_grep_parity(self, tmp_path):
        eager, lazy = self._stores(tmp_path)
        se = ctx.DocumentSearcher(eager, _LazyCfg())
        sl = ctx.DocumentSearcher(lazy, _LazyCfg())
        me = se.grep("d", "line")
        ml = sl.grep("d", "line")
        assert [m.to_dict() for m in me] == [m.to_dict() for m in ml]
        assert len(me) > 0

    def test_grep_parity_unicode_pattern(self, tmp_path):
        eager, lazy = self._stores(tmp_path)
        se = ctx.DocumentSearcher(eager, _LazyCfg())
        sl = ctx.DocumentSearcher(lazy, _LazyCfg())
        me = se.grep("d", "\U0001f600")
        ml = sl.grep("d", "\U0001f600")
        assert [m.to_dict() for m in me] == [m.to_dict() for m in ml]
        assert len(me) == 1


class TestNoSlurpGuard:
    def test_lazy_ops_never_call_path_read_text(self, tmp_path, monkeypatch):
        text = UNICODE_FIXTURE * 50
        path = tmp_path / "f.txt"
        path.write_text(text, encoding="utf-8")

        def _boom(*a, **k):
            raise AssertionError(
                "Path.read_text must not be called for a lazy document"
            )

        monkeypatch.setattr(pathlib.Path, "read_text", _boom)

        store = ctx.DocumentStore(max_document_mb=0.0, tmp_dir=tmp_path / "spill")
        store.add_path("d", path)  # ingest itself must not slurp
        searcher = ctx.DocumentSearcher(store, _LazyCfg())

        assert searcher.peek("d", 0, 10) == text[:10]
        assert searcher.window("d", 5, 3) == text[max(0, 2) : 8]
        assert searcher.head("d", 2) == "\n".join(text.split("\n")[:2])
        assert searcher.tail("d", 2) == "\n".join(text.split("\n")[-2:])
        assert len(searcher.grep("d", "line")) > 0


class TestBudgetEviction:
    def _many_line_file(self, path, n_lines=2000):
        # No trailing newline -- keeps "line" count == n_lines exactly, so
        # tail(1) is unambiguously "line" rather than a trailing empty line.
        path.write_text("\n".join(["line"] * n_lines), encoding="utf-8")

    def test_eviction_frees_lru_lazy_index(self, tmp_path):
        store = ctx.DocumentStore(
            max_document_mb=0.0, store_budget_mb=0.0002, tmp_dir=tmp_path / "spill"
        )
        for i in range(5):
            p = tmp_path / f"t{i}.txt"
            self._many_line_file(p)
            store.add_path(f"d{i}", p)
        doc0 = store._docs["d0"]
        assert doc0.lazy is True
        assert doc0.is_index_resident() is False  # LRU-evicted for budget

    def test_evicted_doc_still_serves_reads(self, tmp_path):
        store = ctx.DocumentStore(
            max_document_mb=0.0, store_budget_mb=0.0002, tmp_dir=tmp_path / "spill"
        )
        for i in range(5):
            p = tmp_path / f"t{i}.txt"
            self._many_line_file(p)
            store.add_path(f"d{i}", p)
        searcher = ctx.DocumentSearcher(store, _LazyCfg())
        assert searcher.head("d0", 2) == "line\nline"
        assert searcher.tail("d0", 1) == "line"
        # reading rebuilt the index
        assert store._docs["d0"].is_index_resident() is True

    def test_eviction_logs_debug(self, tmp_path, caplog):
        caplog.set_level(logging.DEBUG, logger="simba.rlm.context")
        store = ctx.DocumentStore(
            max_document_mb=0.0, store_budget_mb=0.0002, tmp_dir=tmp_path / "spill"
        )
        for i in range(5):
            p = tmp_path / f"t{i}.txt"
            self._many_line_file(p)
            store.add_path(f"d{i}", p)
        assert any("evict" in r.message.lower() for r in caplog.records)

    def test_within_budget_no_eviction(self, tmp_path):
        store = ctx.DocumentStore(
            max_document_mb=1000.0, store_budget_mb=256.0, tmp_dir=tmp_path / "spill"
        )
        p = tmp_path / "small.txt"
        p.write_text("hello world", encoding="utf-8")
        store.add_path("s1", p)
        assert store.get("s1").text == "hello world"


class TestEagerRetainedBytesAccounting:
    """The eager ``_Document.retained_bytes()`` must honestly account for
    the per-line string OBJECT overhead of ``.lines`` (and the per-int
    overhead of ``.line_starts``), not just double the text object's size.
    Under many-short-line documents the old ``sys.getsizeof(text) * 2``
    estimate under-counted real retention 3-5x in production (a live
    daemon capture showed ~4.7GB of split-line strings on a store that
    believed it was within budget), causing the DocumentStore LRU to
    under-evict."""

    @staticmethod
    def _many_short_lines_text(n_lines: int = 50_000, line: str = "abc") -> str:
        return "\n".join([line] * n_lines)

    @classmethod
    def _true_retained_bytes(cls, text: str) -> int:
        """Independently computed honest sum -- text + the per-line list
        (list object + every line's string object) + the per-line offset
        list (list object + every int's object), mirroring exactly what
        the eager ``_Document`` retains."""
        lines = text.split("\n")
        line_starts = [0]
        pos = 0
        for line in lines[:-1]:
            pos += len(line) + 1
            line_starts.append(pos)
        return (
            sys.getsizeof(text)
            + sys.getsizeof(lines)
            + sum(sys.getsizeof(line) for line in lines)
            + sys.getsizeof(line_starts)
            + sum(sys.getsizeof(n) for n in line_starts)
        )

    def test_retained_bytes_reflects_true_per_line_overhead(self):
        text = self._many_short_lines_text()
        store = ctx.DocumentStore()
        store.add("d1", text)
        doc = store.get("d1")

        true_sum = self._true_retained_bytes(text)
        old_estimate = sys.getsizeof(text) * 2

        accounted = doc.retained_bytes()
        assert accounted >= true_sum
        assert accounted > old_estimate * 1.5

    def test_eviction_budget_honors_honest_accounting(self, tmp_path):
        # Honest single-doc footprint here is ~4.7MB (~11x the old `* 2`
        # estimate of ~0.4MB). Size the budget so the OLD estimate would
        # have happily admitted two such docs (2 * ~0.4MB is well under
        # 5MB) but honest accounting only admits one (2 * ~4.7MB > 5MB).
        text = self._many_short_lines_text()
        store = ctx.DocumentStore(
            max_document_mb=1000.0, store_budget_mb=5.0, tmp_dir=tmp_path / "spill"
        )
        store.add("d1", text)
        assert store._docs["d1"].lazy is False  # still eager, nothing evicted yet

        store.add("d2", text)

        assert store._docs["d1"].lazy is True  # LRU-evicted once d2 hit the budget
        assert store._docs["d2"].lazy is False  # newest doc keeps its fast path
