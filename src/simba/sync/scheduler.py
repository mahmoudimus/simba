"""Periodic sync scheduler — runs index + extract on an interval.

Designed for both standalone use (``simba sync schedule``) and embedding
into the memory daemon (``server.py`` can start a background task).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from simba.sync.extractor import run_extract
from simba.sync.indexer import run_index

logger = logging.getLogger("simba.sync.scheduler")


class SyncScheduler:
    """Run sync pipelines periodically in an asyncio loop."""

    def __init__(
        self,
        *,
        cwd: str | Path = ".",
        daemon_url: str = "http://localhost:8741",
        interval_seconds: int = 300,
        use_claude: bool = False,
    ) -> None:
        self.cwd = Path(cwd)
        self.daemon_url = daemon_url
        self.interval_seconds = interval_seconds
        self.use_claude = use_claude
        self._stop_event = asyncio.Event()
        self._cycle_count = 0
        self._running = False

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def running(self) -> bool:
        return self._running

    async def run_once(self) -> dict:
        """Run one sync cycle (index + extract) in a thread pool.

        Returns a summary dict with results from both pipelines.
        """
        loop = asyncio.get_running_loop()

        idx = await loop.run_in_executor(
            None,
            lambda: run_index(self.cwd, daemon_url=self.daemon_url),
        )
        ext = await loop.run_in_executor(
            None,
            lambda: run_extract(
                self.cwd,
                daemon_url=self.daemon_url,
                use_claude=self.use_claude,
            ),
        )

        self._cycle_count += 1
        total_errors = idx.errors + ext.errors

        summary = {
            "cycle": self._cycle_count,
            "index": {
                "rows_indexed": idx.rows_indexed,
                "duplicates": idx.duplicates,
                "exported": idx.rows_exported,
                "errors": idx.errors,
            },
            "extract": {
                "facts_extracted": ext.facts_extracted,
                "facts_duplicate": ext.facts_duplicate,
                "memories_processed": ext.memories_processed,
                "agent_dispatched": ext.agent_dispatched,
                "errors": ext.errors,
            },
            "total_errors": total_errors,
        }

        if total_errors:
            logger.warning("Sync cycle %d completed with %d errors",
                           self._cycle_count, total_errors)
        else:
            logger.info(
                "Sync cycle %d: indexed=%d, facts=%d",
                self._cycle_count,
                idx.rows_indexed,
                ext.facts_extracted,
            )

        return summary

    async def run_forever(self) -> None:
        """Run sync cycles until stopped."""
        self._running = True
        self._stop_event.clear()

        try:
            while not self._stop_event.is_set():
                try:
                    await self.run_once()
                except Exception:
                    logger.debug("Sync cycle failed", exc_info=True)

                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.interval_seconds,
                    )
                    break  # stop_event was set
                except TimeoutError:
                    continue  # timeout → next cycle
        finally:
            self._running = False

    def stop(self) -> None:
        """Signal the scheduler to stop after the current cycle."""
        self._stop_event.set()
