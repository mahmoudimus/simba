"""Tests for RLM context primitives."""

from __future__ import annotations

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
