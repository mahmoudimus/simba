"""Request diagnostics tracker for the memory daemon."""

from __future__ import annotations

import collections
import logging
import statistics

logger = logging.getLogger("simba.memory")


class DiagnosticsTracker:
    """Tracks per-endpoint hit counts, recall/store statistics, and latency.

    After every ``report_interval`` requests, prints a summary to the
    logger and resets the rolling counters. Latency samples are kept in a
    fixed-size per-endpoint reservoir (oldest evicted at capacity) and survive
    report resets so percentiles reflect a stable window.
    """

    def __init__(
        self,
        report_interval: int = 50,
        reservoir_size: int = 1000,
        compact_cleanup_seconds: int = 0,
    ) -> None:
        self.report_interval = report_interval
        self._total_requests = 0
        self._reservoir_size = reservoir_size
        # Version-retention window for the periodic auto-compaction (0 = never
        # prune). Passed as optimize()'s cleanup_older_than so versions self-bound.
        self._compact_cleanup_seconds = compact_cleanup_seconds
        self.last_error: dict[str, str] | None = None
        self._latency_samples: dict[str, list[float]] = collections.defaultdict(list)
        # Cumulative client tallies survive report resets (like _total_requests)
        # so /stats can show an all-time breakdown; _client_hits is the rolling
        # per-interval window that feeds the periodic log report.
        self._client_hits_total: dict[str, int] = collections.defaultdict(int)
        self._reset()

    def _reset(self) -> None:
        self._endpoint_hits: dict[str, int] = collections.defaultdict(int)
        self._client_hits: dict[str, int] = collections.defaultdict(int)
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

    def record_client(self, client: str) -> None:
        """Record a hit attributed to a named client (``X-Simba-Client``)."""
        self._client_hits[client] += 1
        self._client_hits_total[client] += 1

    def client_hits(self) -> dict[str, int]:
        """Cumulative all-time hits per client (a copy; safe to mutate)."""
        return dict(self._client_hits_total)

    def record_latency(self, endpoint: str, latency_ms: float) -> None:
        """Record a latency sample, evicting the oldest when at capacity."""
        buf = self._latency_samples[endpoint]
        if len(buf) >= self._reservoir_size:
            buf.pop(0)
        buf.append(latency_ms)

    def record_error(self, endpoint: str, exc: BaseException, request_id: str) -> None:
        """Remember the latest request failure for readiness diagnostics."""
        self.last_error = {
            "endpoint": endpoint,
            "request_id": request_id,
            "type": type(exc).__name__,
            "message": str(exc),
        }

    def latency_percentiles(self, endpoint: str) -> dict[str, float]:
        """Return ``{"p50": float, "p95": float, "n": int}`` for one endpoint."""
        buf = self._latency_samples.get(endpoint, [])
        if len(buf) < 2:
            return {"p50": 0.0, "p95": 0.0, "n": len(buf)}
        qs = statistics.quantiles(buf, n=20)  # 5% increments
        return {"p50": qs[9], "p95": qs[18], "n": len(buf)}

    def all_latency_stats(self) -> dict[str, dict[str, float]]:
        """Return percentiles for all endpoints that have samples."""
        return {ep: self.latency_percentiles(ep) for ep in self._latency_samples}

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
        import datetime

        import simba.memory.vector_db

        memory_count = 0
        if table is not None:
            memory_count = await simba.memory.vector_db.count_rows(table)
            # Compact fragments AND prune old versions so the table self-bounds
            # (None retention = merge-only, the legacy 37GB-bloat behavior).
            cleanup = (
                datetime.timedelta(seconds=self._compact_cleanup_seconds)
                if self._compact_cleanup_seconds > 0
                else None
            )
            await simba.memory.vector_db.compact_table(
                table, cleanup_older_than=cleanup
            )

        lines = [
            "=" * 60,
            f"[diagnostics] after {self._total_requests} total requests",
            "-" * 40,
            "Endpoint hits:",
        ]
        for ep, count in sorted(self._endpoint_hits.items()):
            lines.append(f"  {ep:<25s} {count:>5d}")

        if self._client_hits:
            lines.append("")
            lines.append("Client hits:")
            for c, count in sorted(self._client_hits.items()):
                lines.append(f"  {c:<25s} {count:>5d}")

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
            f"Store: {self._store_total} total, {self._store_duplicates} duplicates"
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
