"""Tests for the deterministic keyword-overlap scoring shared by the compact
relay's arc ranking (hooks/session_start.py) and the distiller's focus-based
output ordering (transcripts/distill.py). No LLM, no embeddings -- plain
lowercase word-token overlap.
"""

from __future__ import annotations

import simba.transcripts.focus as focus


class TestTokenize:
    def test_lowercases_and_splits_on_whitespace(self) -> None:
        assert focus.tokenize("Fix The Daemon Restart") == [
            "fix",
            "the",
            "daemon",
            "restart",
        ]

    def test_drops_tokens_shorter_than_three_chars(self) -> None:
        assert focus.tokenize("go fix it up now") == ["fix", "now"]

    def test_splits_on_punctuation(self) -> None:
        assert focus.tokenize("daemon-restart, RSS/watchdog!") == [
            "daemon",
            "restart",
            "rss",
            "watchdog",
        ]

    def test_empty_text_yields_no_tokens(self) -> None:
        assert focus.tokenize("") == []

    def test_none_like_falsy_yields_no_tokens(self) -> None:
        assert focus.tokenize(None) == []  # type: ignore[arg-type]


class TestScoreOverlap:
    def test_counts_distinct_matching_tokens(self) -> None:
        tokens = set(focus.tokenize("daemon restart watchdog"))
        assert focus.score_overlap(tokens, "the daemon needed a restart") == 2

    def test_repeated_matches_count_once(self) -> None:
        tokens = set(focus.tokenize("daemon"))
        assert focus.score_overlap(tokens, "daemon daemon daemon") == 1

    def test_no_overlap_scores_zero(self) -> None:
        tokens = set(focus.tokenize("daemon restart"))
        assert focus.score_overlap(tokens, "completely unrelated text here") == 0

    def test_empty_focus_tokens_scores_zero(self) -> None:
        assert focus.score_overlap(set(), "daemon restart watchdog") == 0

    def test_empty_candidate_text_scores_zero(self) -> None:
        tokens = set(focus.tokenize("daemon restart"))
        assert focus.score_overlap(tokens, "") == 0
