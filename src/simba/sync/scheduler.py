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

        epi = await loop.run_in_executor(None, self._maybe_consolidate)
        ref = await loop.run_in_executor(None, self._maybe_reflect)
        dist = await loop.run_in_executor(None, self._maybe_distill)
        hyg = await loop.run_in_executor(None, self._maybe_hygiene)

        decay_result = await loop.run_in_executor(None, self._maybe_decay)

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
            "episodes": {"dispatched": len(epi.get("dispatched", []))},
            "decay": (
                {
                    "processed": decay_result.processed,
                    "updated": decay_result.updated,
                    "newly_dormant": decay_result.newly_dormant,
                    "revived": decay_result.revived,
                }
                if decay_result
                else {"skipped": True}
            ),
            "reflection": ref,
            "distillation": dist,
            "hygiene": hyg,
            "total_errors": total_errors,
        }

        if total_errors:
            logger.warning(
                "Sync cycle %d completed with %d errors",
                self._cycle_count,
                total_errors,
            )
        else:
            logger.info(
                "Sync cycle %d: indexed=%d, facts=%d",
                self._cycle_count,
                idx.rows_indexed,
                ext.facts_extracted,
            )

        return summary

    def _maybe_consolidate(self) -> dict:
        """Consolidate eligible sessions for this project (engine-gated)."""
        import simba.config
        import simba.episodes.config  # registers the "episodes" section
        import simba.episodes.consolidate

        ecfg = simba.config.load("episodes")
        if not ecfg.enabled or not ecfg.scheduler_enabled:
            return {"dispatched": [], "skipped": 0}
        return simba.episodes.consolidate.consolidate_eligible(
            str(self.cwd), ecfg=ecfg, daemon_url=self.daemon_url
        )

    def _maybe_decay(self):  # type: ignore[no-untyped-def]
        """Run the memory decay/forgetting pass when enabled (fail-open).

        Fail-open like the other scheduler passes: a fresh project whose
        ``memory_usage`` table hasn't been initialised yet (the daemon imports
        ``memory.routes`` -> usage at startup, but the standalone scheduler may
        not) must not abort the whole sync cycle.
        """
        try:
            import time

            import simba.config
            import simba.memory.config  # registers the "memory" section
            import simba.memory.decay

            _ = simba.memory.config
            cfg = simba.config.load("memory")
            if not getattr(cfg, "decay_enabled", True):
                return None
            return simba.memory.decay.run_decay_pass(
                now=time.time(),
                cwd=self.cwd,
                cfg=cfg,
            )
        except Exception:
            logger.debug("decay pass failed", exc_info=True)
            return None

    def _maybe_reflect(self) -> dict:
        """Run a cross-session reflection pass (engine-gated, fail-open)."""
        try:
            import simba.config
            import simba.reflection.config  # registers section
            import simba.reflection.pass_

            _ = simba.reflection.config
            rcfg = simba.config.load("reflection")
            if not rcfg.enabled or not rcfg.scheduler_enabled:
                return {"status": "disabled"}
            result = simba.reflection.pass_.reflect_pass(
                cwd=str(self.cwd),
                cycle_count=self._cycle_count,
                rcfg=rcfg,
                daemon_url=self.daemon_url,
            )
            return {"status": result.status, "dispatched": result.dispatched}
        except Exception:
            logger.debug("reflection pass failed", exc_info=True)
            return {"status": "error", "dispatched": False}

    def _maybe_distill(self) -> dict:
        """Run the neuro-symbolic distillation pipeline (gated, fail-open)."""
        try:
            import simba.config
            import simba.neuron.config  # registers section
            import simba.neuron.pipeline

            _ = simba.neuron.config
            ncfg = simba.config.load("neuron")
            if not ncfg.enabled:
                return {"status": "disabled"}
            result = simba.neuron.pipeline.distillation_pass(
                project_path=str(self.cwd), cfg=ncfg
            )
            return {
                "status": result.status,
                "distilled": result.distilled,
                "dormant": result.dormant,
            }
        except Exception:
            logger.debug("distillation pass failed", exc_info=True)
            return {"status": "error"}

    def _maybe_hygiene(self) -> dict:
        """Expire stale TOOL_RULE memories (gated, fail-open)."""
        try:
            import simba.config
            import simba.memory.config  # registers section
            import simba.memory.hygiene

            _ = simba.memory.config
            cfg = simba.config.load("memory")
            if not cfg.hygiene_scheduler_enabled or cfg.tool_rule_max_age_days == 0:
                return {"status": "disabled"}
            result = simba.memory.hygiene.run_hygiene_pass(
                daemon_url=self.daemon_url, cfg=cfg
            )
            return {"status": "ok", "expired": result.expired_count}
        except Exception:
            logger.debug("hygiene pass failed", exc_info=True)
            return {"status": "error"}

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
