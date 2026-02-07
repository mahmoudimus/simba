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
