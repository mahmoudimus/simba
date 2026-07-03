"""Periodic decay / forgetting pass over the ``memory_usage`` sidecar.

``run_decay_pass`` recomputes every memory's ``strength`` via the pure
:func:`simba.memory.strength.compute_strength`, writes it back, and toggles the
``dormant`` flag against ``strength_dormancy_threshold``.  Forgetting is fully
reversible: a memory whose strength recovers (e.g. via outcome feedback) is
revived.  No row is ever deleted, and LanceDB is never mutated by this pass.

An optional capacity cap demotes the weakest memories per
``(type, project_path)`` bucket; because that metadata lives only in LanceDB,
the cap joins type/project via the daemon's ``GET /list`` and is skipped
silently when the daemon is unreachable or the cap is disabled (``0``).

The pass is synchronous and deterministic — the caller supplies ``now`` so the
clock never enters the computation.  The scheduler runs it once per sync cycle.
"""

from __future__ import annotations

import dataclasses
import logging
import typing

import simba.db
import simba.memory.strength
import simba.memory.usage

if typing.TYPE_CHECKING:
    import pathlib

logger = logging.getLogger("simba.memory")

_STRENGTH_TOLERANCE = 1e-6


@dataclasses.dataclass
class DecayResult:
    processed: int = 0  # total rows visited
    updated: int = 0  # rows where strength changed
    newly_dormant: int = 0  # rows transitioned to dormant=True this pass
    revived: int = 0  # rows transitioned dormant=True -> False this pass
    errors: int = 0
    dry_run: bool = False  # shadow pass: counts reflect WOULD-BE changes


def run_decay_pass(
    *,
    now: float,
    cwd: pathlib.Path,
    cfg: typing.Any,
    dry_run: bool = False,
) -> DecayResult | None:
    """Recompute strength + dormancy for every ``memory_usage`` row.

    Returns ``None`` when ``cfg.decay_enabled`` is false (the master switch).
    Otherwise returns a :class:`DecayResult` summarising the pass. Per-row
    failures are caught and counted in ``result.errors`` without aborting.

    ``dry_run`` (spec 33 shadow mode) computes and counts every would-be
    strength update / dormancy transition but persists nothing — the
    maintenance heartbeat runs shadow until ``maintenance_apply`` is measured.
    """
    if not getattr(cfg, "decay_enabled", True):
        return None

    result = DecayResult(dry_run=dry_run)
    half_life = float(getattr(cfg, "decay_half_life_days", 30.0))
    scale = float(getattr(cfg, "reinforcement_scale", 0.5))
    fb_weight = float(getattr(cfg, "feedback_weight", 0.2))
    outcome_weight = float(getattr(cfg, "outcome_quality_weight", 0.0))
    threshold = float(getattr(cfg, "strength_dormancy_threshold", 0.1))
    arousal_mult = float(getattr(cfg, "arousal_decay_multiplier", 1.0))

    with simba.db.connect(cwd):
        rows = simba.memory.usage.get_all_for_decay(include_dormant=True)
        for row in rows:
            result.processed += 1
            try:
                new_strength = simba.memory.strength.compute_strength(
                    created_at_epoch=row.created_at,
                    now=now,
                    access_count=row.access_count,
                    feedback_score=_effective_feedback(row, outcome_weight),
                    half_life=half_life,
                    reinforcement_scale=scale,
                    feedback_weight=fb_weight,
                    arousal_decay_multiplier=arousal_mult,
                )
                if abs(new_strength - row.strength) > _STRENGTH_TOLERANCE:
                    if not dry_run:
                        simba.memory.usage.set_strength(row.memory_id, new_strength)
                    result.updated += 1
                if new_strength < threshold and not row.dormant:
                    if not dry_run:
                        simba.memory.usage.set_dormant(row.memory_id, dormant=True)
                    result.newly_dormant += 1
                elif new_strength >= threshold and row.dormant:
                    if not dry_run:
                        simba.memory.usage.set_dormant(row.memory_id, dormant=False)
                    result.revived += 1
            except Exception:
                result.errors += 1
                logger.debug("[decay] row update failed", exc_info=True)

    if getattr(cfg, "decay_capacity_per_type", 0) > 0:
        try:
            _apply_capacity_cap(cfg, cwd, dry_run=dry_run)
        except Exception:
            logger.debug("[decay] capacity cap failed", exc_info=True)

    return result


def _effective_feedback(row: typing.Any, outcome_weight: float) -> float:
    """Blend explicit feedback with bounded use/noise outcome counters."""
    feedback = float(getattr(row, "feedback_score", 0.0) or 0.0)
    if outcome_weight <= 0:
        return feedback
    use_count = int(getattr(row, "use_count", 0) or 0)
    noise_count = int(getattr(row, "noise_count", 0) or 0)
    total = use_count + noise_count
    if total <= 0:
        return feedback
    outcome = (use_count - noise_count) / total
    blended = feedback + outcome_weight * outcome
    return max(-1.0, min(1.0, blended))


def _apply_capacity_cap(
    cfg: typing.Any, cwd: pathlib.Path, *, dry_run: bool = False
) -> tuple[int, int]:
    """Demote the weakest non-dormant memories per ``(type, project_path)``.

    ``memory_type`` and ``project_path`` live only in LanceDB, so this bulk-fetches
    them via the daemon's ``GET /list``. When the daemon is unreachable the step
    is skipped silently. Returns ``(groups_processed, memories_demoted)``.
    ``dry_run`` counts would-be demotions without writing.
    """
    cap = int(getattr(cfg, "decay_capacity_per_type", 0))
    if cap <= 0:
        return (0, 0)

    import httpx

    port = int(getattr(cfg, "port", 8741))
    try:
        resp = httpx.get(
            f"http://127.0.0.1:{port}/list",
            params={"limit": 100000},
            timeout=10.0,
        )
        resp.raise_for_status()
        memories = resp.json().get("memories", [])
    except httpx.HTTPError:
        return (0, 0)

    meta = {
        m["id"]: (m.get("type", ""), m.get("projectPath", ""))
        for m in memories
        if m.get("id")
    }

    with simba.db.connect(cwd):
        rows = simba.memory.usage.get_all_for_decay(include_dormant=False)
        groups: dict[tuple[str, str], list[typing.Any]] = {}
        for row in rows:
            key = meta.get(row.memory_id)
            if key is None:
                continue
            groups.setdefault(key, []).append(row)

        groups_processed = 0
        demoted = 0
        for members in groups.values():
            if len(members) <= cap:
                continue
            groups_processed += 1
            # Weakest first; demote everything beyond the cap.
            members.sort(key=lambda r: r.strength)
            for row in members[: len(members) - cap]:
                if not dry_run:
                    simba.memory.usage.set_dormant(row.memory_id, dormant=True)
                demoted += 1

    return (groups_processed, demoted)
