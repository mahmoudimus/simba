"""Tests for query-intent classification (src/simba/memory/intent.py)."""

from __future__ import annotations

import pytest

import simba.memory.intent as intent


class TestClassify:
    @pytest.mark.parametrize(
        "query",
        [
            "list all the migrations we ran",
            "what is the history of the config schema",
            "summarize every decision about hooks",
            "compare the two embedding backends",
            "recurring patterns in the recall path",
        ],
    )
    def test_aggregation_queries_are_broad(self, query: str) -> None:
        assert intent.classify(query) == "broad"

    @pytest.mark.parametrize(
        "query",
        [
            "what port does the daemon listen on",
            "fix the RRF dedup bug in hybrid.py",
            "the embedding dim for nomic-embed",
            "why did /recall return zero memories",
        ],
    )
    def test_point_fact_queries_are_precise(self, query: str) -> None:
        assert intent.classify(query) == "precise"

    def test_long_thinking_block_without_markers_is_precise(self) -> None:
        # Thinking blocks are long but usually point-seeking; length alone must
        # NOT push them to broad — only explicit aggregation markers do.
        thinking = (
            "I need to open hybrid.py and trace how the keyword arm builds its "
            "MATCH expression so I can see why the bm25 score is dominated by "
            "stop words on a verbose query string before it reaches fusion. "
        ) * 4
        assert intent.classify(thinking) == "precise"

    def test_empty_query_is_precise(self) -> None:
        assert intent.classify("") == "precise"

    def test_marker_match_is_case_insensitive_and_whole_word(self) -> None:
        assert intent.classify("List ALL of them") == "broad"
        # substring of a marker must not trigger ("overviewer" != "overview").
        assert intent.classify("the listener callback") == "precise"


# ── count-intent detection (count candidate-depth PR) ─────────────────────────
def test_is_count_detects_instance_counting() -> None:
    assert intent.is_count("How many bikes do I own?") is True
    assert intent.is_count("number of korean restaurants I've tried") is True
    assert intent.is_count("what is the total number of trips I took") is True


def test_is_count_excludes_temporal_frequency_and_plain() -> None:
    assert intent.is_count("How many days between the trip and the move?") is False
    assert intent.is_count("how many times a week do I exercise") is False
    assert intent.is_count("what did the engineer say about the schema") is False
