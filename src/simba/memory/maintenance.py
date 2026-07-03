"""Maintenance heartbeat (spec 33 Phase 0) — decay + hygiene, decoupled from sync.

The 2026-07-03 audit found both lifecycle passes hosted inside ``SyncScheduler``,
whose startup is gated on ``sync_interval > 0`` (default ``0``) — so neither had
EVER run against a live store (every ``memory_usage`` row at strength exactly
1.0, zero dormant, hygiene never pruning). This module gives them their own
driver: a dedicated scheduler started by the daemon lifespan, gated only on
``memory.maintenance_interval_hours``.

SHADOW by default: until ``memory.maintenance_apply`` flips ON (a measured
ranking change — strength feeds scoring, dormancy hides memories), every pass
runs with ``dry_run=True``, counting would-be changes and persisting nothing.
``POST /maintenance/run`` triggers the same pass by hand; the latest result is
surfaced in ``GET /stats`` as ``lastMaintenance``.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import time
import typing

from simba.memory.decay import run_decay_pass
from simba.memory.hygiene import run_hygiene_pass

if typing.TYPE_CHECKING:
    import pathlib

logger = logging.getLogger("simba.memory.maintenance")


def run_maintenance(
    *,
    now: float,
    cwd: pathlib.Path,
    cfg: typing.Any,
    daemon_url: str,
    apply: bool | None = None,
) -> dict:
    """One maintenance pass (decay + hygiene). Returns a JSON-able summary.

    ``apply=None`` defers to ``cfg.maintenance_apply`` (default False = shadow:
    both passes run dry). Each pass is independently fail-soft — one failing
    never blocks the other.
    """
    if apply is None:
        apply = bool(getattr(cfg, "maintenance_apply", False))
    dry_run = not apply
    result: dict = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "apply": apply,
        "errors": 0,
    }

    if getattr(cfg, "decay_enabled", True):
        try:
            # Type-aware half-lives need the id→type join (types live only in
            # LanceDB); fetched only when multipliers are configured.
            type_map = None
            if getattr(cfg, "decay_type_multipliers", ""):
                type_map = _fetch_type_map(daemon_url)
            decay = run_decay_pass(
                now=now, cwd=cwd, cfg=cfg, dry_run=dry_run, type_map=type_map
            )
            result["decay"] = (
                dataclasses.asdict(decay) if decay is not None else {"skipped": True}
            )
        except Exception:
            logger.debug("[maintenance] decay pass failed", exc_info=True)
            result["decay"] = {"error": True}
            result["errors"] += 1
    else:
        result["decay"] = {"skipped": True}

    if (
        getattr(cfg, "hygiene_scheduler_enabled", True)
        and getattr(cfg, "tool_rule_max_age_days", 0) > 0
    ):
        try:
            hygiene = run_hygiene_pass(
                daemon_url=daemon_url, cfg=cfg, dry_run=dry_run, cwd=cwd
            )
            result["hygiene"] = dataclasses.asdict(hygiene)
        except Exception:
            logger.debug("[maintenance] hygiene pass failed", exc_info=True)
            result["hygiene"] = {"error": True}
            result["errors"] += 1
    else:
        result["hygiene"] = {"skipped": True}

    return result


def _fetch_type_map(daemon_url: str) -> dict[str, str] | None:
    """``memory_id -> TYPE`` over the corpus via ``GET /list`` (the capacity-cap
    pattern: types live only in LanceDB). Fail-soft ``None`` → decay falls back
    to the base half-life for every row."""
    import httpx

    try:
        resp = httpx.get(f"{daemon_url}/list", params={"limit": 100_000}, timeout=15.0)
        resp.raise_for_status()
        memories = resp.json().get("memories", [])
    except (httpx.HTTPError, ValueError):
        logger.debug("[maintenance] type-map fetch failed", exc_info=True)
        return None
    return {
        m["id"]: str(m.get("type", ""))
        for m in memories
        if isinstance(m, dict) and m.get("id")
    }


def _default_config_loader() -> typing.Any:
    """Fresh ``memory`` config each cycle so ``simba config set`` takes effect live."""
    import simba.config
    import simba.memory.config

    _ = simba.memory.config  # registers the "memory" section
    return simba.config.load("memory")


class MaintenanceScheduler:
    """Run maintenance passes on an interval, with an interruptible startup delay.

    Mirrors ``SyncScheduler``'s stop-event shape but owns nothing besides
    decay + hygiene. Config is reloaded per cycle (``config_loader``) so knob
    changes apply without a daemon restart. Every pass is fail-open.
    """

    def __init__(
        self,
        *,
        cwd: pathlib.Path,
        daemon_url: str,
        interval_seconds: float,
        startup_delay_seconds: float = 0.0,
        config_loader: typing.Callable[[], typing.Any] | None = None,
        on_result: typing.Callable[[dict], None] | None = None,
    ) -> None:
        self.cwd = cwd
        self.daemon_url = daemon_url
        self.interval_seconds = interval_seconds
        self.startup_delay_seconds = startup_delay_seconds
        self._config_loader = config_loader or _default_config_loader
        self._on_result = on_result
        self._stop_event = asyncio.Event()
        self.last_result: dict | None = None

    async def run_once(self) -> dict | None:
        """One pass in a worker thread. Fail-open: any error → ``None``."""
        try:
            cfg = self._config_loader()
            result = await asyncio.to_thread(
                run_maintenance,
                now=time.time(),
                cwd=self.cwd,
                cfg=cfg,
                daemon_url=self.daemon_url,
            )
        except Exception:
            logger.debug("[maintenance] pass failed", exc_info=True)
            return None
        self.last_result = result
        if self._on_result is not None:
            with contextlib.suppress(Exception):
                self._on_result(result)
        return result

    async def _wait(self, seconds: float) -> bool:
        """Wait up to ``seconds``; True when the stop event fired."""
        if seconds <= 0:
            return self._stop_event.is_set()
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=seconds)
            return True
        except TimeoutError:
            return False

    async def run_forever(self) -> None:
        """Startup delay, then a pass every ``interval_seconds`` until stopped.

        The stop event is deliberately NOT cleared on entry: a ``stop()``
        issued between task creation and the first poll (the daemon's
        start-then-immediate-shutdown race) must not be lost.
        """
        if await self._wait(self.startup_delay_seconds):
            return
        while not self._stop_event.is_set():
            await self.run_once()
            if await self._wait(self.interval_seconds):
                return

    def stop(self) -> None:
        """Signal the scheduler to stop (interrupts delay and interval waits)."""
        self._stop_event.set()
