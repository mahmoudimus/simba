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


# ── knowledge-update / current-value intent (conflict 0.7.1 query-intent gate) ──
# These queries ask for the PRESENT value of a fact ("what is my X now?"). They
# routinely retrieve BOTH the old and the new value, which the pairwise conflict
# detector flags as a "conflict" and (wrongly) tells the answerer not to pick a
# side — when the correct behaviour is most-recent-wins. So conflict surfacing
# must SKIP its directive for these; recency handles them.
def test_is_knowledge_update_detects_current_value_queries() -> None:
    assert intent.is_knowledge_update("What is my current job title?") is True
    assert intent.is_knowledge_update("Where do I live now?") is True
    assert intent.is_knowledge_update("What is my latest car?") is True
    assert intent.is_knowledge_update("What is my most recent address?") is True
    assert intent.is_knowledge_update("Where am I working these days?") is True
    assert intent.is_knowledge_update("Do I still go to the gym?") is True
    assert intent.is_knowledge_update("What phone am I using nowadays?") is True
    assert intent.is_knowledge_update("As of now, what city am I in?") is True


def test_is_knowledge_update_excludes_genuine_conflict_and_plain() -> None:
    # Genuine preference / simultaneous-conflict queries: NO current-value marker,
    # so they must STAY on the strict surfacing path (return False here).
    assert intent.is_knowledge_update("Which coffee do I prefer?") is False
    assert intent.is_knowledge_update("Do I like cats or dogs better?") is False
    assert (
        intent.is_knowledge_update("what did the engineer say about the schema")
        is False
    )
    assert intent.is_knowledge_update("Why did /recall return zero memories") is False
    assert intent.is_knowledge_update("") is False
    # "current" as a substring of another word must not trigger.
    assert intent.is_knowledge_update("how does the concurrent queue drain") is False
# ── aggregation / multi-session detection (MS breadth PR) ─────────────────────
# Population drawn from real LongMemEval-S multi-session questions that the count
# predicate does NOT catch (no "how many X" instance-count shape) yet are still
# recall-BREADTH-bound: they sum / span events across sessions, so they want a
# wide candidate pool exactly like counting. Be conservative — these fire a wider,
# costlier retrieval, so a bounded "X and Y" arithmetic over two named items must
# NOT trigger it.
class TestIsAggregation:
    @pytest.mark.parametrize(
        "query",
        [
            # span/total over an open set of events across sessions
            "How many days did I take social media breaks in total?",
            "How many hours have I spent playing games in total?",
            "What is the total amount I spent on luxury items in the past few months?",
            "How many hours in total did I spend driving to my three road trip "
            "destinations combined?",
            # "across" + enumeration of all events
            "How many times did I ride rollercoasters across all the events I "
            "attended from July to October?",
            # "how often" / frequency-over-history phrasings
            "How often did I go to the gym this month?",
            # explicit list-all enumeration
            "list all the conferences I attended this year",
            "throughout the year, which restaurants did I revisit",
            # "every time" recurrence
            "what did I order every time I went to that cafe",
            # "which ... did I" enumeration shape
            "which museums did I visit in February",
        ],
    )
    def test_aggregation_queries_fire(self, query: str) -> None:
        assert intent.is_aggregation(query) is True

    @pytest.mark.parametrize(
        "query",
        [
            # bounded arithmetic over two/few NAMED items — pointwise, not breadth
            "What is the total cost of the car cover and detailing spray I purchased?",
            "What is the total cost of Lola's vet visit and flea medication?",
            "What is the difference in price between my luxury boots and the "
            "similar pair found at the budget store?",
            "What is the average GPA of my undergraduate and graduate studies?",
            "What is the total amount I spent on gifts for my coworker and brother?",
            # plain point facts
            "what port does the daemon listen on",
            "the embedding dim for nomic-embed",
            "When did I submit my research paper on sentiment analysis?",
            "At which university did I present a poster on my thesis research?",
        ],
    )
    def test_non_aggregation_queries_do_not_fire(self, query: str) -> None:
        assert intent.is_aggregation(query) is False

    def test_arithmetic_count_shape_excluded_here(self) -> None:
        # The frequency-RATE shape ("how many times a week") and temporal SPAN
        # ("how many days between …") are latest/state, not breadth — stay quiet.
        assert intent.is_aggregation(
            "How many days between the trip and the move?"
        ) is (False)
        assert intent.is_aggregation("how many times a week do I exercise") is False
        assert (
            intent.is_aggregation("How many days a week do I attend fitness classes?")
            is False
        )

    def test_cross_session_how_many_times_still_fires(self) -> None:
        # Overlap edge: the count predicate DROPS "how many times …" (not instance
        # counting), but "how many times … across all the events" IS a cross-session
        # aggregation, so is_aggregation must still fire. The two predicates answer
        # different questions about the same string.
        q = "How many times did I ride rollercoasters across all the events I attended?"
        assert intent.is_count(q) is False  # count drops "how many times"
        assert intent.is_aggregation(q) is True

    def test_both_predicates_can_fire_on_count_plus_span(self) -> None:
        # "How many X across all my trips" is genuinely count AND aggregation;
        # plan_recall resolves the tie by giving count precedence.
        q = "How many restaurants did I try across all my trips?"
        assert intent.is_count(q) is True
        assert intent.is_aggregation(q) is True

    def test_empty_query_is_not_aggregation(self) -> None:
        assert intent.is_aggregation("") is False

    def test_aggregation_is_case_insensitive(self) -> None:
        assert intent.is_aggregation("LIST ALL the trips") is True
        assert intent.is_aggregation("How OFTEN did I travel") is True
