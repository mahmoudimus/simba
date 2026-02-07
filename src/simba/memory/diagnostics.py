"""Request diagnostics tracker for the memory daemon."""

from __future__ import annotations

import collections
import logging

logger = logging.getLogger("simba.memory")


class DiagnosticsTracker:
    """Tracks per-endpoint hit counts, recall/store statistics.

    After every ``report_interval`` requests, prints a summary to the
    logger and resets all counters.
    """

    def __init__(self, report_interval: int = 50) -> None:
        self.report_interval = report_interval
        self._total_requests = 0
        self._reset()

    def _reset(self) -> None:
        self._endpoint_hits: dict[str, int] = collections.defaultdict(int)
        self._recall_total = 0
        self._recall_successful = 0
        self._recall_empty = 0
        self._recall_queries: list[str] = []
        self._store_total = 0
        self._store_duplicates = 0
        self._store_by_type: dict[str, int] = collections.defaultdict(int)

    def record_request(self, endpoint: str) -> None:
        """Record a hit to any endpoint."""
        self._endpoint_hits[endpoint] += 1
        self._total_requests += 1

    def record_recall(self, query: str, result_count: int) -> None:
        """Record a recall operation."""
        self._recall_total += 1
        if result_count > 0:
            self._recall_successful += 1
        else:
            self._recall_empty += 1
        if len(self._recall_queries) < 10:
            self._recall_queries.append(query[:50])

    def record_store(self, memory_type: str, *, duplicate: bool) -> None:
        """Record a store operation."""
        self._store_total += 1
        if duplicate:
            self._store_duplicates += 1
        else:
            self._store_by_type[memory_type] += 1

    def should_report(self) -> bool:
        """Return True if total requests have hit the report interval."""
        return (
            self.report_interval > 0
            and self._total_requests > 0
            and self._total_requests % self.report_interval == 0
        )

    async def emit_report(self, table: object = None) -> None:
        """Print diagnostics summary and reset counters."""
        import simba.memory.vector_db

        memory_count = 0
        if table is not None:
            memory_count = await simba.memory.vector_db.count_rows(table)
            # Compact fragments to keep vector search fast.
            await simba.memory.vector_db.compact_table(table)

        lines = [
            "=" * 60,
            f"[diagnostics] after {self._total_requests} total requests",
            "-" * 40,
            "Endpoint hits:",
        ]
        for ep, count in sorted(self._endpoint_hits.items()):
            lines.append(f"  {ep:<25s} {count:>5d}")

        lines.append("")
        lines.append(
            f"Recall: {self._recall_total} queries, "
            f"{self._recall_successful} with results, "
            f"{self._recall_empty} empty"
        )
        if self._recall_queries:
            lines.append("  Recent queries:")
            for q in self._recall_queries:
                lines.append(f'    "{q}"')

        lines.append("")
        lines.append(
            f"Store: {self._store_total} total, "
            f"{self._store_duplicates} duplicates"
        )
        if self._store_by_type:
            lines.append("  By type:")
            for t, c in sorted(self._store_by_type.items()):
                lines.append(f"    {t:<25s} {c:>5d}")

        lines.append("")
        lines.append(f"DB memory count: {memory_count}")
        lines.append("=" * 60)

        logger.info("\n".join(lines))
        self._reset()
