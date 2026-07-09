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
import datetime
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

    if getattr(cfg, "supersession_adjudication_enabled", False):
        try:
            result["adjudication"] = _adjudicate_supersessions(
                now=now, cwd=cwd, cfg=cfg, dry_run=dry_run
            )
        except Exception:
            logger.debug("[maintenance] adjudication failed", exc_info=True)
            result["adjudication"] = {"error": True}
            result["errors"] += 1
    else:
        result["adjudication"] = {"skipped": True}

    result["episodes"] = _run_episodes(cwd, daemon_url, dry_run)
    result["reflection"] = _run_reflection(cwd, daemon_url, dry_run)
    result["promotions"] = _count_promotion_candidates(cwd, cfg)
    result["repeat_failures"] = _cluster_repeat_failures(cwd, cfg)
    result["graduation"] = _graduation_readiness(
        now=now, cwd=cwd, cfg=cfg, daemon_url=daemon_url
    )
    result["tempfiles"] = _sweep_session_tempfiles(now=now, cfg=cfg, dry_run=dry_run)

    # Forgetting-run tracker (spec 33 v2 rule R5, hebb-mind borrow): every
    # pass appends its summary so health becomes a plottable trend, not just
    # the latest snapshot in /stats. Fail-soft.
    if getattr(cfg, "maintenance_log_enabled", True):
        try:
            import json
            import pathlib as _pathlib

            log_path = _pathlib.Path(cwd) / ".simba" / "memory"
            log_path.mkdir(parents=True, exist_ok=True)
            with (log_path / "maintenance-log.jsonl").open("a") as fh:
                fh.write(json.dumps(result) + "\n")
        except Exception:
            logger.debug("[maintenance] run log append failed", exc_info=True)

    return result


def _adjudicate_supersessions(
    *, now: float, cwd: pathlib.Path, cfg: typing.Any, dry_run: bool
) -> dict:
    """Resolve stale ``pending_confirmation`` supersessions (spec 33 Phase 4).

    The audit found 166 pendings and nothing that had ever adjudicated one —
    an inbox with no reader. Policy: a pending older than
    ``supersession_adjudication_max_age_days`` resolves NEWEST-WINS (lww): the
    supersession is CONFIRMED (append-only decision event) and the superseded
    memory goes dormant (reversible). Younger pendings are left for explicit
    review — age is the deterministic judge, not a substitute for one.
    """
    import simba.db
    import simba.memory.supersession as supersession
    import simba.memory.usage

    max_age_days = float(getattr(cfg, "supersession_adjudication_max_age_days", 30.0))
    cutoff = now - max_age_days * 86400.0

    with simba.db.connect(cwd):
        pendings = list(
            supersession.MemorySupersession.select().where(
                (supersession.MemorySupersession.status == supersession.STATUS_PENDING)
                & (supersession.MemorySupersession.created_at < cutoff)
            )
        )
        decided = supersession._decided_pending_ids([p.id for p in pendings])
        stale = [p for p in pendings if p.id not in decided]
        if not dry_run:
            for pending in stale:
                supersession.confirm(pending.id, now=now)
                simba.memory.usage.get_or_create(pending.old_id, now)
                simba.memory.usage.set_dormant(pending.old_id, dormant=True)

    return {
        "confirmed": len(stale),
        "max_age_days": max_age_days,
        "dry_run": dry_run,
    }


def _count_promotion_candidates(cwd: pathlib.Path, cfg: typing.Any) -> dict:
    """Stuck-candidate sweep (spec 33 v2, MemOS borrow).

    The promotion surface is otherwise query-time-only: a memory whose stored
    counters already qualify but that stops being recalled never re-enters
    any touch path and never surfaces. Recomputing from STORED counters every
    heartbeat closes that trap; the count lands in ``lastMaintenance``, the
    run log, and the boot digest. Read-only — safe in shadow.
    """
    try:
        import simba.db
        import simba.memory.usage

        min_uses = int(getattr(cfg, "promotion_min_uses", 3))
        max_ratio = float(getattr(cfg, "promotion_max_noise_ratio", 0.5))
        with simba.db.connect(cwd):
            rows = simba.memory.usage.MemoryUsage.select().where(
                (simba.memory.usage.MemoryUsage.use_count >= min_uses)
                & (simba.memory.usage.MemoryUsage.dormant == False)  # noqa: E712
            )
            candidates = sum(
                1
                for row in rows
                if not (
                    row.use_count > 0 and (row.noise_count / row.use_count) >= max_ratio
                )
            )
        return {"candidates": candidates}
    except Exception:
        logger.debug("[maintenance] promotion count failed", exc_info=True)
        return {"error": True}


def _parse_reflection_ts(ts: str) -> float | None:
    """Parse a reflections row's ``ts`` (``%Y-%m-%dT%H:%M:%SZ``) to epoch secs.

    ``None`` on any malformed value — the row still counts toward
    occurrences/sessions, it just can't contribute to the span computation.
    """
    try:
        return (
            datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
            .replace(tzinfo=datetime.UTC)
            .timestamp()
        )
    except (TypeError, ValueError):
        return None


def _cluster_repeat_failures(cwd: pathlib.Path, cfg: typing.Any) -> dict:
    """Reflections-ledger reader (spec 33 v2 rule R3).

    The tailor-style ``reflections`` table (normalized failure captures piped
    in by the Stop hook's error-capture pipeline) has been write-only since
    inception — 1,100+ rows, never read. READ-ONLY: groups rows by their
    already-normalized ``signature`` column and flags a REPEAT FAILURE when
    the same signature recurs across >= ``repeat_failure_min_sessions``
    distinct sessions AND its first/last occurrence spans
    >= ``repeat_failure_min_days_apart`` days. Rows with no attributed
    session (blank ``session_id`` — captures written before this reader
    existed) never count toward the distinct-session tally, so historical
    data can only under-count, never falsely cluster. Never writes anything
    (no store mutation, no auto-promotion) — runs on every heartbeat
    regardless of ``maintenance_apply`` (there is nothing to gate) and
    surfaces its top clusters as rule-DRAFT candidates only.
    """
    try:
        import simba.db
        import simba.tailor.hook as tailor_hook

        min_sessions = int(getattr(cfg, "repeat_failure_min_sessions", 2))
        min_days_apart = float(getattr(cfg, "repeat_failure_min_days_apart", 3.0))
        top_n = int(getattr(cfg, "repeat_failure_top_n", 3))

        with simba.db.connect(cwd):
            rows = list(tailor_hook.Reflection.select())

        groups: dict[str, list[typing.Any]] = {}
        for row in rows:
            signature = (row.signature or "").strip()
            if not signature:
                continue
            groups.setdefault(signature, []).append(row)

        clusters: list[dict] = []
        for signature, group_rows in groups.items():
            sessions = {
                row.session_id.strip()
                for row in group_rows
                if (row.session_id or "").strip()
            }
            timestamps = sorted(
                t
                for t in (_parse_reflection_ts(row.ts) for row in group_rows)
                if t is not None
            )
            span_days = (
                (timestamps[-1] - timestamps[0]) / 86400.0
                if len(timestamps) >= 2
                else 0.0
            )
            if len(sessions) >= min_sessions and span_days >= min_days_apart:
                clusters.append(
                    {
                        "signature": signature,
                        "error_type": group_rows[0].error_type,
                        "sessions": len(sessions),
                        "span_days": round(span_days, 2),
                        "occurrences": len(group_rows),
                    }
                )

        clusters.sort(key=lambda c: (-c["sessions"], -c["span_days"], c["signature"]))
        return {"clusters": len(clusters), "top": clusters[:top_n]}
    except Exception:
        logger.debug("[maintenance] repeat-failure clustering failed", exc_info=True)
        return {"error": True}


def _fetch_tool_rule_ids(daemon_url: str) -> list[str]:
    """TOOL_RULE memory ids via ``GET /list?type=TOOL_RULE`` — mirrors the
    hygiene pass's server-filtered fetch (``run_hygiene_pass``). Fail-soft
    ``[]`` on any HTTP error; never raises."""
    import simba.memory.background

    if simba.memory.background.is_shutting_down():
        # Handoff item 10: once the daemon starts shutting down, this
        # loopback GET can never complete (uvicorn has stopped serving) ---
        # skip it immediately instead of guaranteeing a graceful-shutdown
        # timeout breach.
        return []

    import httpx

    try:
        resp = httpx.get(
            f"{daemon_url}/list",
            params={"type": "TOOL_RULE", "limit": 10000},
            timeout=15.0,
        )
        resp.raise_for_status()
        memories = resp.json().get("memories", [])
    except (httpx.HTTPError, ValueError):
        logger.debug("[maintenance] graduation tool-rule fetch failed", exc_info=True)
        return []
    return [
        m["id"]
        for m in memories
        if isinstance(m, dict) and m.get("type") == "TOOL_RULE" and m.get("id")
    ]


def _graduation_readiness(
    *, now: float, cwd: pathlib.Path, cfg: typing.Any, daemon_url: str
) -> dict:
    """Data-side readiness for the ``maintenance_apply`` graduation gate
    (spec 33 Part 8 rule R1). READ-ONLY, like ``_cluster_repeat_failures`` —
    runs every heartbeat regardless of ``maintenance_apply`` and never
    writes or flips anything; a human flips the lever by hand, after the
    MANUAL bench guards (LME-S/LoCoMo/HaluMem — explicitly out of scope
    here, hence ``benchGuards`` always reports ``"manual"``).

    Two data criteria, both required for ``ready``:

    - ``signalDays``: days between ``now`` and the EARLIEST ``usage_events``
      row (any memory, session, or kind) — how long signals have been
      accumulating. No events yet -> 0.0 (not started). Compared against
      ``memory.maintenance_graduation_min_days``.
    - ``usedRatio``: among TOOL_RULE memories that FIRED, the fraction whose
      sidecar carries ``last_used > 0``. "Fired" here means
      ``memory_usage.match_count > 0`` — the counter ``POST /recall`` bumps
      every time ``_recall_tool_rules`` (PreToolUse) surfaces a TOOL_RULE as
      a gate candidate. ``inject_count`` is deliberately NOT used: that
      counter is wired only to the separate UserPromptSubmit
      ``/recall/ack`` lane (``hooks.recall_ack_enabled``, default off) and
      stays 0 for TOOL_RULE rows — the tool-rule gate never acks; it posts
      ``use`` feedback directly the moment it actually fires (see
      ``pre_tool_use.py``'s "Spec 33 Phase 1" comment). No fired rules yet
      -> 0.0. Compared against
      ``memory.maintenance_graduation_min_used_ratio``.

    Fail-soft: any error -> ``{"error": True}`` (matches
    ``_cluster_repeat_failures`` / ``_count_promotion_candidates``).
    """
    try:
        import simba.db
        import simba.memory.usage
        import simba.memory.usage_events

        min_days = float(getattr(cfg, "maintenance_graduation_min_days", 14.0))
        min_ratio = float(getattr(cfg, "maintenance_graduation_min_used_ratio", 0.6))
        tool_rule_ids = _fetch_tool_rule_ids(daemon_url)

        with simba.db.connect(cwd):
            earliest = simba.memory.usage_events.earliest_event_at()
            signal_days = (
                max(0.0, (now - earliest) / 86400.0) if earliest is not None else 0.0
            )

            fired = []
            if tool_rule_ids:
                rows = simba.memory.usage.MemoryUsage.select().where(
                    simba.memory.usage.MemoryUsage.memory_id.in_(tool_rule_ids)
                )
                fired = [row for row in rows if row.match_count > 0]
            used_ratio = (
                sum(1 for row in fired if row.last_used > 0) / len(fired)
                if fired
                else 0.0
            )

        days_met = signal_days >= min_days
        ratio_met = used_ratio >= min_ratio
        return {
            "signalDays": round(signal_days, 2),
            "usedRatio": round(used_ratio, 4),
            "daysMet": days_met,
            "ratioMet": ratio_met,
            "ready": days_met and ratio_met,
            "benchGuards": "manual",
        }
    except Exception:
        logger.debug("[maintenance] graduation readiness failed", exc_info=True)
        return {"error": True}


# Per-session flag files the hooks strew across the tempdir (guardian flags,
# usage-signal records). Anything not matching these prefixes is never touched.
_SESSION_FLAG_PREFIXES = (
    "claude-usage-turn-",
    "claude-usage-session-",
    "claude-engagement-",
    "claude-rules-signal-",
    "claude-preflight-",
    "claude-mandate-armed-",
)


def _session_tempdir() -> pathlib.Path:
    import pathlib as _pathlib
    import tempfile

    return _pathlib.Path(tempfile.gettempdir())


def _sweep_session_tempfiles(*, now: float, cfg: typing.Any, dry_run: bool) -> dict:
    """Session-tempfile TTL (spec 33 v2 rule R6).

    The audit found 827 stale engagement flags in the tempdir; the usage
    records now join them. Files with known session-flag prefixes older than
    ``session_tempfile_max_age_days`` are deleted on apply passes (shadow
    counts them). 0 disables.
    """
    max_age_days = float(getattr(cfg, "session_tempfile_max_age_days", 7.0) or 0.0)
    if max_age_days <= 0:
        return {"skipped": True}
    cutoff = now - max_age_days * 86400.0
    stale = 0
    deleted = 0
    try:
        for path in _session_tempdir().iterdir():
            if not any(path.name.startswith(p) for p in _SESSION_FLAG_PREFIXES):
                continue
            try:
                if path.stat().st_mtime >= cutoff:
                    continue
                stale += 1
                if not dry_run:
                    path.unlink(missing_ok=True)
                    deleted += 1
            except OSError:
                continue
    except OSError:
        logger.debug("[maintenance] tempfile sweep failed", exc_info=True)
        return {"error": True}
    return {"stale": stale, "deleted": deleted, "dry_run": dry_run}


def _run_episodes(cwd: pathlib.Path, daemon_url: str, dry_run: bool) -> dict:
    """Episode consolidation, re-armed (spec 33 Phase 4).

    ``episodes.scheduler_enabled`` pointed at the sync scheduler that never
    ran — the same dead-driver bug as decay (``episode_jobs`` froze
    2026-06-08). The heartbeat is its driver now. Dispatches (LLM digest
    jobs) only when applying; shadow reports the skip so the gap is visible.
    """
    try:
        import simba.config
        import simba.episodes.config  # registers the "episodes" section
        import simba.episodes.consolidate

        _ = simba.episodes.config
        ecfg = simba.config.load("episodes")
        if not (ecfg.enabled and ecfg.scheduler_enabled):
            return {"skipped": True, "reason": "disabled"}
        if dry_run:
            return {"skipped": True, "reason": "shadow"}
        return simba.episodes.consolidate.consolidate_eligible(
            str(cwd), ecfg=ecfg, daemon_url=daemon_url
        )
    except Exception:
        logger.debug("[maintenance] episode consolidation failed", exc_info=True)
        return {"error": True}


def _run_reflection(cwd: pathlib.Path, daemon_url: str, dry_run: bool) -> dict:
    """Cross-session reflection pass, re-armed (spec 33 Phase 4/6).

    Same dead-driver story as episodes: ``reflection.scheduler_enabled``
    default-True with no live scheduler. Runs (LLM synthesis) only when
    applying.
    """
    try:
        import simba.config
        import simba.reflection.config  # registers the "reflection" section
        import simba.reflection.pass_

        _ = simba.reflection.config
        rcfg = simba.config.load("reflection")
        if not (rcfg.enabled and rcfg.scheduler_enabled):
            return {"skipped": True, "reason": "disabled"}
        if dry_run:
            return {"skipped": True, "reason": "shadow"}
        outcome = simba.reflection.pass_.reflect_pass(
            cwd=str(cwd), rcfg=rcfg, daemon_url=daemon_url
        )
        return {"status": outcome.status, "dispatched": outcome.dispatched}
    except Exception:
        logger.debug("[maintenance] reflection pass failed", exc_info=True)
        return {"error": True}


def _fetch_type_map(daemon_url: str) -> dict[str, str] | None:
    """``memory_id -> TYPE`` over the corpus via ``GET /list`` (the capacity-cap
    pattern: types live only in LanceDB). Fail-soft ``None`` → decay falls back
    to the base half-life for every row."""
    import simba.memory.background

    if simba.memory.background.is_shutting_down():
        # Handoff item 10: same shutdown guard as _fetch_tool_rule_ids ---
        # this loopback GET can never complete once uvicorn stops serving.
        return None

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
