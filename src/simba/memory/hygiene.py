"""Memory-hygiene pass: TTL aging for stale ``TOOL_RULE`` memories (Task C.2).

Expires ``TOOL_RULE`` memories older than ``memory.tool_rule_max_age_days`` via
the daemon DELETE route (LanceDB is the source of truth for memories). This
solves the known stale false-warning injection from aged ``TOOL_RULE`` entries.
Fail-open: any HTTP error increments ``errors`` and leaves memories intact.
"""

from __future__ import annotations

import dataclasses
import logging
import typing
from datetime import UTC, datetime, timedelta

import httpx

if typing.TYPE_CHECKING:
    from simba.memory.config import MemoryConfig

logger = logging.getLogger("simba.memory.hygiene")


@dataclasses.dataclass
class HygieneResult:
    expired_count: int = 0
    checked_count: int = 0
    errors: int = 0
    dry_run: bool = False  # shadow pass: expired_count = WOULD-expire


def _memory_cfg() -> MemoryConfig:
    import simba.config
    import simba.memory.config  # registers section

    _ = simba.memory.config
    return simba.config.load("memory")


def run_hygiene_pass(
    *,
    daemon_url: str,
    cfg: MemoryConfig | None = None,
    dry_run: bool = False,
) -> HygieneResult:
    """Expire stale TOOL_RULE memories via daemon DELETE. Never raises.

    ``dry_run`` (spec 33 shadow mode) counts would-expire rules without
    issuing any DELETE.
    """
    cfg = cfg or _memory_cfg()
    if cfg.tool_rule_max_age_days == 0:
        return HygieneResult(dry_run=dry_run)

    cutoff = datetime.now(tz=UTC) - timedelta(days=cfg.tool_rule_max_age_days)
    cutoff_iso = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        resp = httpx.get(
            f"{daemon_url}/list",
            params={"type": "TOOL_RULE", "limit": 10000},
            timeout=15.0,
        )
        resp.raise_for_status()
        memories = resp.json().get("memories", [])
    except (httpx.HTTPError, ValueError):
        logger.debug("hygiene: list request failed", exc_info=True)
        return HygieneResult(errors=1, dry_run=dry_run)

    expired = 0
    errors = 0
    checked = 0
    for mem in memories:
        if mem.get("type") != "TOOL_RULE":
            continue
        checked += 1
        created = mem.get("createdAt") or ""
        if not created or created >= cutoff_iso:
            continue
        mid = mem.get("id")
        if not mid:
            continue
        if dry_run:
            expired += 1
            continue
        try:
            dresp = httpx.delete(f"{daemon_url}/memory/{mid}", timeout=15.0)
            dresp.raise_for_status()
            expired += 1
        except (httpx.HTTPError, ValueError):
            logger.debug("hygiene: delete failed for %s", mid, exc_info=True)
            errors += 1

    return HygieneResult(
        expired_count=expired, checked_count=checked, errors=errors, dry_run=dry_run
    )
