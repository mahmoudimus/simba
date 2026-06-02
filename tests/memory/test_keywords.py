"""Tests for high-signal term selection (src/simba/memory/keywords.py)."""

from __future__ import annotations

import simba.memory.keywords as keywords


class TestFocusTerms:
    def test_drops_stop_words(self) -> None:
        terms = keywords.focus_terms("the daemon stores vectors in the table")
        assert "the" not in terms
        assert "in" not in terms
        assert "daemon" in terms
        assert "vectors" in terms

    def test_identifiers_and_paths_rank_ahead_of_plain_words(self) -> None:
        terms = keywords.focus_terms(
            "update the hybrid_search function in routes.py module"
        )
        # hybrid_search (has _) and routes.py (has .) are entity-like -> ranked first.
        assert terms[0] in {"hybrid_search", "routes.py"}
        assert terms.index("hybrid_search") < terms.index("module")

    def test_proper_nouns_rank_ahead_of_plain_words(self) -> None:
        terms = keywords.focus_terms("ask LanceDB about the vector table")
        assert terms[0] == "LanceDB"

    def test_respects_max_terms_cap(self) -> None:
        terms = keywords.focus_terms(
            "alpha beta gamma delta epsilon zeta eta", max_terms=3
        )
        assert len(terms) == 3

    def test_dedup_is_case_insensitive_keeping_first_casing(self) -> None:
        assert keywords.focus_terms("Cache cache CACHE") == ["Cache"]

    def test_respects_min_len(self) -> None:
        assert keywords.focus_terms("a bb ccc dddd", min_len=3) == ["ccc", "dddd"]

    def test_empty_query_yields_no_terms(self) -> None:
        assert keywords.focus_terms("") == []
