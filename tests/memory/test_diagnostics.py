from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from simba.memory.diagnostics import DiagnosticsTracker


class TestDiagnosticsTracker:
    def test_initial_state(self) -> None:
        tracker = DiagnosticsTracker(report_interval=50)
        assert tracker._total_requests == 0
        assert tracker._recall_total == 0
        assert tracker._store_total == 0

    def test_record_request_increments_total(self) -> None:
        tracker = DiagnosticsTracker()
        tracker.record_request("POST /recall")
        tracker.record_request("POST /store")
        assert tracker._total_requests == 2
        assert tracker._endpoint_hits["POST /recall"] == 1
        assert tracker._endpoint_hits["POST /store"] == 1

    def test_record_client_counts_by_name(self) -> None:
        tracker = DiagnosticsTracker()
        tracker.record_client("claude-code")
        tracker.record_client("claude-code")
        tracker.record_client("pi")
        assert tracker._client_hits["claude-code"] == 2
        assert tracker._client_hits["pi"] == 1

    def test_client_hits_in_report(self) -> None:
        tracker = DiagnosticsTracker()
        tracker.record_client("pi")
        tracker.record_client("codex")
        import asyncio

        with (
            patch("simba.memory.vector_db.count_rows", new=AsyncMock(return_value=0)),
            patch.object(logging.getLogger("simba.memory"), "info") as mock_info,
        ):
            asyncio.run(tracker.emit_report(table=None))
        report = "\n".join(str(c.args[0]) for c in mock_info.call_args_list)
        assert "Client hits:" in report
        assert "pi" in report
        assert "codex" in report

    def test_client_hits_cumulative_survives_report_reset(self) -> None:
        tracker = DiagnosticsTracker()
        tracker.record_client("pi")
        tracker.record_client("pi")
        with patch("simba.memory.vector_db.count_rows", new=AsyncMock(return_value=0)):
            import asyncio

            asyncio.run(tracker.emit_report(table=None))
        # Per-interval window reset, but the cumulative view for /stats persists.
        assert tracker._client_hits == {}
        assert tracker.client_hits() == {"pi": 2}
        tracker.record_client("pi")
        assert tracker.client_hits() == {"pi": 3}

    def test_client_hits_returns_copy(self) -> None:
        tracker = DiagnosticsTracker()
        tracker.record_client("cli")
        snapshot = tracker.client_hits()
        snapshot["cli"] = 999
        assert tracker.client_hits() == {"cli": 1}

    def test_record_recall_successful(self) -> None:
        tracker = DiagnosticsTracker()
        tracker.record_recall("test query", 3)
        assert tracker._recall_total == 1
        assert tracker._recall_successful == 1
        assert tracker._recall_empty == 0

    def test_record_recall_empty(self) -> None:
        tracker = DiagnosticsTracker()
        tracker.record_recall("test query", 0)
        assert tracker._recall_total == 1
        assert tracker._recall_successful == 0
        assert tracker._recall_empty == 1

    def test_recall_queries_capped_at_10(self) -> None:
        tracker = DiagnosticsTracker()
        for i in range(15):
            tracker.record_recall(f"query {i}", 0)
        assert len(tracker._recall_queries) == 10

    def test_record_store_success(self) -> None:
        tracker = DiagnosticsTracker()
        tracker.record_store("GOTCHA", duplicate=False)
        tracker.record_store("GOTCHA", duplicate=False)
        tracker.record_store("PATTERN", duplicate=False)
        assert tracker._store_total == 3
        assert tracker._store_by_type["GOTCHA"] == 2
        assert tracker._store_by_type["PATTERN"] == 1

    def test_record_store_duplicate(self) -> None:
        tracker = DiagnosticsTracker()
        tracker.record_store("GOTCHA", duplicate=True)
        assert tracker._store_total == 1
        assert tracker._store_duplicates == 1

    def test_should_report_at_interval(self) -> None:
        tracker = DiagnosticsTracker(report_interval=5)
        for i in range(4):
            tracker.record_request(f"req{i}")
            assert not tracker.should_report()
        tracker.record_request("req4")
        assert tracker.should_report()

    def test_should_report_disabled_when_zero(self) -> None:
        tracker = DiagnosticsTracker(report_interval=0)
        for _ in range(100):
            tracker.record_request("req")
        assert not tracker.should_report()

    @pytest.mark.asyncio
    async def test_emit_report_resets_counters(self) -> None:
        tracker = DiagnosticsTracker(report_interval=5)
        tracker.record_request("POST /recall")
        tracker.record_recall("test", 1)
        tracker.record_store("GOTCHA", duplicate=False)

        mock_table = MagicMock()
        with patch(
            "simba.memory.vector_db.count_rows",
            new_callable=AsyncMock,
            return_value=42,
        ):
            await tracker.emit_report(table=mock_table)

        assert tracker._recall_total == 0
        assert tracker._store_total == 0
        assert tracker._endpoint_hits == {}
        # total_requests is NOT reset
        assert tracker._total_requests == 1

    @pytest.mark.asyncio
    async def test_emit_report_includes_all_sections(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        tracker = DiagnosticsTracker()
        tracker.record_request("POST /recall")
        tracker.record_recall("what is auth", 2)
        tracker.record_store("GOTCHA", duplicate=False)

        mock_table = MagicMock()
        with (
            caplog.at_level(logging.INFO, logger="simba.memory"),
            patch(
                "simba.memory.vector_db.count_rows",
                new_callable=AsyncMock,
                return_value=10,
            ),
        ):
            await tracker.emit_report(table=mock_table)

        output = caplog.text
        assert "[diagnostics]" in output
        assert "POST /recall" in output
        assert "Recall:" in output
        assert "what is auth" in output
        assert "Store:" in output
        assert "GOTCHA" in output
        assert "DB memory count: 10" in output


class TestLatencyMetrics:
    def test_record_latency_and_percentiles(self) -> None:
        tracker = DiagnosticsTracker(report_interval=50, reservoir_size=100)
        for i in range(20):
            tracker.record_latency("POST /recall", float(i))
        stats = tracker.latency_percentiles("POST /recall")
        assert stats["n"] == 20
        assert stats["p50"] > 0.0
        assert stats["p95"] >= stats["p50"]

    def test_reservoir_evicts_oldest(self) -> None:
        tracker = DiagnosticsTracker(reservoir_size=5)
        for i in range(10):
            tracker.record_latency("/store", float(i))
        assert len(tracker._latency_samples["/store"]) == 5

    def test_all_latency_stats_returns_all_endpoints(self) -> None:
        tracker = DiagnosticsTracker()
        tracker.record_latency("/recall", 12.0)
        tracker.record_latency("/recall", 15.0)
        tracker.record_latency("/store", 8.0)
        tracker.record_latency("/store", 9.0)
        stats = tracker.all_latency_stats()
        assert set(stats.keys()) == {"/recall", "/store"}
