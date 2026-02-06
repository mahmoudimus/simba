"""Tests for tailor status module — error counting, sorting, co-occurrence analysis."""

from __future__ import annotations

import datetime

import simba.tailor.status


class TestCountByErrorType:
    def test_counts_errors(self, mock_reflection):
        reflections = [
            mock_reflection({"error_type": "error"}),
            mock_reflection({"error_type": "error"}),
            mock_reflection({"error_type": "typeerror"}),
            mock_reflection({"error_type": "referenceerror"}),
        ]
        counts = simba.tailor.status.count_by_error_type(reflections)
        assert counts["error"] == 2
        assert counts["typeerror"] == 1
        assert counts["referenceerror"] == 1

    def test_empty_array(self):
        assert simba.tailor.status.count_by_error_type([]) == {}

    def test_single_reflection(self, mock_reflection):
        reflections = [mock_reflection({"error_type": "error"})]
        counts = simba.tailor.status.count_by_error_type(reflections)
        assert counts["error"] == 1

    def test_handles_missing_fields(self):
        reflections = [
            {"error_type": "error", "ts": "2024-01-01T00:00:00Z"},
            {"error_type": "typeerror", "ts": "2024-01-01T01:00:00Z"},
        ]
        counts = simba.tailor.status.count_by_error_type(reflections)
        assert counts["error"] == 1


class TestGetSortedCounts:
    def test_sorts_by_frequency(self, mock_reflection):
        reflections = [
            mock_reflection({"error_type": "error"}),
            mock_reflection({"error_type": "error"}),
            mock_reflection({"error_type": "error"}),
            mock_reflection({"error_type": "typeerror"}),
            mock_reflection({"error_type": "typeerror"}),
            mock_reflection({"error_type": "referenceerror"}),
        ]
        counts = simba.tailor.status.count_by_error_type(reflections)
        sorted_counts = simba.tailor.status.get_sorted_counts(counts)
        assert sorted_counts[0] == ("error", 3)
        assert sorted_counts[1] == ("typeerror", 2)


class TestGetTopN:
    def test_returns_top_5(self, mock_reflection):
        reflections = [
            mock_reflection({"error_type": t})
            for t in [
                "error",
                "typeerror",
                "referenceerror",
                "syntaxerror",
                "enoent",
                "eacces",
            ]
        ]
        counts = simba.tailor.status.count_by_error_type(reflections)
        sorted_counts = simba.tailor.status.get_sorted_counts(counts)
        top5 = simba.tailor.status.get_top_n(sorted_counts, 5)
        assert len(top5) == 5

    def test_returns_fewer_when_less_available(self, mock_reflection):
        reflections = [
            mock_reflection({"error_type": "error"}),
            mock_reflection({"error_type": "typeerror"}),
        ]
        counts = simba.tailor.status.count_by_error_type(reflections)
        sorted_counts = simba.tailor.status.get_sorted_counts(counts)
        top5 = simba.tailor.status.get_top_n(sorted_counts, 5)
        assert len(top5) == 2


class TestAnalyzeCoOccurrence:
    def test_detects_pairs(self, mock_reflection):
        reflections = [
            mock_reflection({"error_type": "error"}),
            mock_reflection({"error_type": "typeerror"}),
            mock_reflection({"error_type": "error"}),
            mock_reflection({"error_type": "typeerror"}),
        ]
        co = simba.tailor.status.analyze_co_occurrence(reflections)
        assert co.get("error|typeerror", 0) > 0

    def test_pairs_sorted_alphabetically(self, mock_reflection):
        reflections = [
            mock_reflection({"error_type": "zebra"}),
            mock_reflection({"error_type": "apple"}),
            mock_reflection({"error_type": "zebra"}),
            mock_reflection({"error_type": "apple"}),
        ]
        co = simba.tailor.status.analyze_co_occurrence(reflections)
        for key in co:
            parts = key.split("|")
            assert parts[0] <= parts[1]

    def test_filters_count_gt_1(self, mock_reflection):
        reflections = [
            mock_reflection({"error_type": "a"}),
            mock_reflection({"error_type": "b"}),
            mock_reflection({"error_type": "a"}),
            mock_reflection({"error_type": "b"}),
            mock_reflection({"error_type": "a"}),
        ]
        co = simba.tailor.status.analyze_co_occurrence(reflections)
        top = simba.tailor.status.get_top_co_occurrences(co, 3)
        for _, count in top:
            assert count > 1

    def test_returns_top_n_pairs(self, mock_reflection):
        reflections = []
        for _ in range(3):
            reflections.append(mock_reflection({"error_type": "error"}))
            reflections.append(mock_reflection({"error_type": "typeerror"}))
        for _ in range(2):
            reflections.append(mock_reflection({"error_type": "referenceerror"}))
            reflections.append(mock_reflection({"error_type": "syntaxerror"}))
        co = simba.tailor.status.analyze_co_occurrence(reflections)
        top = simba.tailor.status.get_top_co_occurrences(co, 3)
        assert len(top) <= 3


class TestGetLastReflection:
    def test_returns_last(self, mock_reflection):
        reflections = [
            mock_reflection({"error_type": "error", "ts": "2024-01-01T00:00:00Z"}),
            mock_reflection({"error_type": "error", "ts": "2024-01-01T01:00:00Z"}),
            mock_reflection({"error_type": "typeerror", "ts": "2024-01-01T02:00:00Z"}),
        ]
        last = simba.tailor.status.get_last_reflection(reflections)
        assert last is not None
        assert last["error_type"] == "typeerror"
        assert last["ts"] == "2024-01-01T02:00:00Z"

    def test_empty_returns_none(self):
        assert simba.tailor.status.get_last_reflection([]) is None

    def test_single_returns_that(self, mock_reflection):
        r = mock_reflection({"error_type": "error"})
        assert simba.tailor.status.get_last_reflection([r]) == r


class TestTotalCount:
    def test_total_count(self, mock_reflection):
        reflections = [
            mock_reflection({"error_type": "error"}),
            mock_reflection({"error_type": "error"}),
            mock_reflection({"error_type": "typeerror"}),
            mock_reflection({"error_type": "typeerror"}),
            mock_reflection({"error_type": "typeerror"}),
        ]
        assert len(reflections) == 5


class TestSignatureGrouping:
    def test_groups_by_signature(self, mock_reflection):
        reflections = [
            mock_reflection(
                {"error_type": "error", "signature": "error-module not found"}
            ),
            mock_reflection(
                {"error_type": "error", "signature": "error-module not found"}
            ),
            mock_reflection(
                {"error_type": "error", "signature": "error-different error"}
            ),
        ]
        sig_counts: dict[str, int] = {}
        for r in reflections:
            sig = r["signature"]
            sig_counts[sig] = sig_counts.get(sig, 0) + 1
        assert sig_counts["error-module not found"] == 2
        assert sig_counts["error-different error"] == 1


class TestLargeSet:
    def test_handles_large_sets(self, mock_reflection):
        types = ["error", "typeerror", "referenceerror"]
        reflections = [
            mock_reflection({"error_type": types[i % 3]}) for i in range(1000)
        ]
        counts = simba.tailor.status.count_by_error_type(reflections)
        sorted_counts = simba.tailor.status.get_sorted_counts(counts)
        top5 = simba.tailor.status.get_top_n(sorted_counts, 5)
        assert len(top5) > 0
        assert len(reflections) == 1000


class TestTimestampParsing:
    def test_last_error_timestamp(self, mock_reflection):

        now = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        past = (
            datetime.datetime.now(tz=datetime.timezone.utc)
            - datetime.timedelta(hours=1)
        ).isoformat()
        reflections = [
            mock_reflection({"error_type": "error", "ts": past}),
            mock_reflection({"error_type": "typeerror", "ts": now}),
        ]
        last = simba.tailor.status.get_last_reflection(reflections)
        assert last is not None
        assert last["ts"] == now
