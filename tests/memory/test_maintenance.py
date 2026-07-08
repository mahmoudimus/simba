"""Tests for the maintenance heartbeat (spec 33 Phase 0).

The heartbeat drives decay + hygiene independent of ``sync_interval`` (the
gating that left both passes unreachable). Default is SHADOW: passes run
dry (log-only) until ``memory.maintenance_apply`` flips after measurement.
"""

from __future__ import annotations

import asyncio
import pathlib
import types

import pytest

import simba.db
import simba.memory.usage as usage
import simba.memory.usage_events as usage_events
from simba.memory.maintenance import MaintenanceScheduler, run_maintenance

_DAY = 86400.0
_NOW = 1_000_000_000.0


def _cfg(**kw):
    base = dict(
        decay_enabled=True,
        decay_half_life_days=30.0,
        reinforcement_scale=0.5,
        feedback_weight=0.2,
        outcome_quality_weight=0.0,
        strength_dormancy_threshold=0.1,
        decay_capacity_per_type=0,
        arousal_decay_multiplier=1.0,
        hygiene_scheduler_enabled=False,
        tool_rule_max_age_days=0,
        maintenance_apply=False,
        maintenance_interval_hours=24.0,
        maintenance_startup_delay_seconds=0.0,
        supersession_adjudication_enabled=False,
        supersession_adjudication_max_age_days=30.0,
        maintenance_log_enabled=False,
        promotion_min_uses=3,
        promotion_max_noise_ratio=0.5,
        session_tempfile_max_age_days=0.0,
        repeat_failure_min_sessions=2,
        repeat_failure_min_days_apart=3.0,
        repeat_failure_top_n=3,
        maintenance_graduation_min_days=14.0,
        maintenance_graduation_min_used_ratio=0.6,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def _seed_reflection(
    *,
    id_: str,
    ts_epoch: float,
    signature: str,
    session_id: str,
    error_type: str = "error",
) -> None:
    """Insert a reflections row directly (bypassing the Stop-hook capture
    pipeline) so clustering tests can control signature/session/timing."""
    import time as _time

    import simba.tailor.hook as tailor_hook

    tailor_hook.Reflection.create(
        id=id_,
        ts=_time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime(ts_epoch)),
        error_type=error_type,
        snippet="",
        context="{}",
        signature=signature,
        session_id=session_id,
    )


def test_maintenance_reports_promotion_candidates(tmp_path: pathlib.Path) -> None:
    """Stuck-candidate sweep (spec 33 v2, MemOS borrow): a memory whose stored
    counters already qualify surfaces on every heartbeat, not only when a
    recall happens to touch it."""
    with simba.db.connect(tmp_path):
        usage.bump_quality("mem_hot", _NOW, use=4, noise=1)
        usage.bump_quality("mem_cold", _NOW, use=1)
    result = run_maintenance(
        now=_NOW, cwd=tmp_path, cfg=_cfg(), daemon_url="http://unused"
    )
    assert result["promotions"]["candidates"] == 1


def test_maintenance_reports_repeat_failure_cluster(tmp_path: pathlib.Path) -> None:
    """Spec 33 v2 rule R3: the reflections ledger (tailor-style failure
    captures) has been write-only since inception. A normalized error
    (``signature``) recurring across >=2 distinct sessions >=3 days apart is
    a rule candidate by definition."""
    with simba.db.connect(tmp_path):
        _seed_reflection(
            id_="r1",
            ts_epoch=_NOW - 5 * _DAY,
            signature="error-cannot-find-module",
            session_id="session-a",
        )
        _seed_reflection(
            id_="r2",
            ts_epoch=_NOW - 1 * _DAY,
            signature="error-cannot-find-module",
            session_id="session-b",
        )
    result = run_maintenance(
        now=_NOW, cwd=tmp_path, cfg=_cfg(), daemon_url="http://unused"
    )
    assert result["repeat_failures"]["clusters"] == 1
    top = result["repeat_failures"]["top"][0]
    assert top["signature"] == "error-cannot-find-module"
    assert top["sessions"] == 2
    assert top["span_days"] >= 3.0
    assert top["occurrences"] == 2


def test_repeat_failure_excludes_single_session(tmp_path: pathlib.Path) -> None:
    """Same signature, same session, wide time span: still just ONE session
    reporting the failure — not a cross-session repeat."""
    with simba.db.connect(tmp_path):
        _seed_reflection(
            id_="r1",
            ts_epoch=_NOW - 10 * _DAY,
            signature="error-timeout",
            session_id="session-a",
        )
        _seed_reflection(
            id_="r2",
            ts_epoch=_NOW - 1 * _DAY,
            signature="error-timeout",
            session_id="session-a",
        )
    result = run_maintenance(
        now=_NOW, cwd=tmp_path, cfg=_cfg(), daemon_url="http://unused"
    )
    assert result["repeat_failures"]["clusters"] == 0
    assert result["repeat_failures"]["top"] == []


def test_repeat_failure_excludes_short_span(tmp_path: pathlib.Path) -> None:
    """Two distinct sessions but under the 3-day span floor: not yet a
    REPEAT failure (could just be one bad afternoon)."""
    with simba.db.connect(tmp_path):
        _seed_reflection(
            id_="r1",
            ts_epoch=_NOW - 1 * _DAY,
            signature="error-timeout",
            session_id="session-a",
        )
        _seed_reflection(
            id_="r2",
            ts_epoch=_NOW - 0.5 * _DAY,
            signature="error-timeout",
            session_id="session-b",
        )
    result = run_maintenance(
        now=_NOW, cwd=tmp_path, cfg=_cfg(), daemon_url="http://unused"
    )
    assert result["repeat_failures"]["clusters"] == 0


def test_repeat_failure_clustering_ignores_maintenance_apply(
    tmp_path: pathlib.Path,
) -> None:
    """READ-ONLY: clustering never writes, so (unlike decay/hygiene/
    adjudication) it must report identically in shadow and apply passes."""
    with simba.db.connect(tmp_path):
        _seed_reflection(
            id_="r1",
            ts_epoch=_NOW - 5 * _DAY,
            signature="error-timeout",
            session_id="session-a",
        )
        _seed_reflection(
            id_="r2",
            ts_epoch=_NOW - 1 * _DAY,
            signature="error-timeout",
            session_id="session-b",
        )
    shadow = run_maintenance(
        now=_NOW, cwd=tmp_path, cfg=_cfg(), daemon_url="http://unused"
    )
    applied = run_maintenance(
        now=_NOW,
        cwd=tmp_path,
        cfg=_cfg(maintenance_apply=True),
        daemon_url="http://unused",
    )
    assert shadow["repeat_failures"] == applied["repeat_failures"]
    assert shadow["repeat_failures"]["clusters"] == 1


def test_repeat_failure_top_bounded(tmp_path: pathlib.Path) -> None:
    """``top`` is capped at repeat_failure_top_n; ``clusters`` always reports
    the true total so nothing is silently hidden."""
    with simba.db.connect(tmp_path):
        for i in range(5):
            sig = f"error-case-{i}"
            _seed_reflection(
                id_=f"{sig}-a",
                ts_epoch=_NOW - 10 * _DAY,
                signature=sig,
                session_id="session-a",
            )
            _seed_reflection(
                id_=f"{sig}-b",
                ts_epoch=_NOW - 1 * _DAY,
                signature=sig,
                session_id="session-b",
            )
    result = run_maintenance(
        now=_NOW,
        cwd=tmp_path,
        cfg=_cfg(repeat_failure_top_n=3),
        daemon_url="http://unused",
    )
    assert result["repeat_failures"]["clusters"] == 5
    assert len(result["repeat_failures"]["top"]) == 3


def test_repeat_failure_no_reflections_reports_zero(tmp_path: pathlib.Path) -> None:
    result = run_maintenance(
        now=_NOW, cwd=tmp_path, cfg=_cfg(), daemon_url="http://unused"
    )
    assert result["repeat_failures"] == {"clusters": 0, "top": []}


# --- Graduation readiness (spec 33 Part 8 rule R1) --------------------------
#
# The DATA half of the criteria gating `maintenance_apply`'s flip to True:
# >= maintenance_graduation_min_days of accumulated usage-signal history
# (usage_events, earliest to now) AND >= maintenance_graduation_min_used_ratio
# of TOOL_RULE memories that FIRED carrying a `last_used` stamp. "Fired" here
# means `memory_usage.match_count > 0` (see test_graduation_fired_is_match_
# count_not_inject_count for why, not inject_count). The bench guards (LME-S/
# LoCoMo/HaluMem) are explicitly manual and never automated by this check —
# it only ever informs, never flips the lever.


def test_graduation_ready_when_days_and_ratio_met(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    """Both DATA criteria met -> ready. 2/3 fired TOOL_RULEs carry last_used
    (0.667 >= the 0.6 floor); 20d of signal history clears the 14d floor."""
    import simba.memory.maintenance as maint

    monkeypatch.setattr(maint, "_fetch_tool_rule_ids", lambda url: ["r1", "r2", "r3"])
    with simba.db.connect(tmp_path):
        usage_events.record("mem_x", "session-a", "use", now=_NOW - 20 * _DAY)
        usage.bump_quality("r1", _NOW, match=1, use=1)
        usage.bump_quality("r2", _NOW, match=1, use=1)
        usage.bump_quality("r3", _NOW, match=1)  # fired, never used
    result = run_maintenance(
        now=_NOW, cwd=tmp_path, cfg=_cfg(), daemon_url="http://unused"
    )
    grad = result["graduation"]
    assert grad["signalDays"] == pytest.approx(20.0, abs=0.01)
    assert grad["usedRatio"] == pytest.approx(2 / 3, abs=0.001)
    assert grad["daysMet"] is True
    assert grad["ratioMet"] is True
    assert grad["ready"] is True
    assert grad["benchGuards"] == "manual"


def test_graduation_not_ready_when_only_days_met(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    import simba.memory.maintenance as maint

    monkeypatch.setattr(maint, "_fetch_tool_rule_ids", lambda url: ["r1", "r2", "r3"])
    with simba.db.connect(tmp_path):
        usage_events.record("mem_x", "session-a", "use", now=_NOW - 20 * _DAY)
        usage.bump_quality("r1", _NOW, match=1, use=1)
        usage.bump_quality("r2", _NOW, match=1)
        usage.bump_quality("r3", _NOW, match=1)
    result = run_maintenance(
        now=_NOW, cwd=tmp_path, cfg=_cfg(), daemon_url="http://unused"
    )
    grad = result["graduation"]
    assert grad["daysMet"] is True
    assert grad["ratioMet"] is False
    assert grad["ready"] is False


def test_graduation_not_ready_when_only_ratio_met(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    import simba.memory.maintenance as maint

    monkeypatch.setattr(maint, "_fetch_tool_rule_ids", lambda url: ["r1", "r2"])
    with simba.db.connect(tmp_path):
        usage_events.record("mem_x", "session-a", "use", now=_NOW - 5 * _DAY)
        usage.bump_quality("r1", _NOW, match=1, use=1)
        usage.bump_quality("r2", _NOW, match=1, use=1)
    result = run_maintenance(
        now=_NOW, cwd=tmp_path, cfg=_cfg(), daemon_url="http://unused"
    )
    grad = result["graduation"]
    assert grad["daysMet"] is False
    assert grad["ratioMet"] is True
    assert grad["ready"] is False


def test_graduation_not_ready_when_neither_met(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    import simba.memory.maintenance as maint

    monkeypatch.setattr(maint, "_fetch_tool_rule_ids", lambda url: ["r1", "r2"])
    with simba.db.connect(tmp_path):
        usage_events.record("mem_x", "session-a", "use", now=_NOW - 5 * _DAY)
        usage.bump_quality("r1", _NOW, match=1)
        usage.bump_quality("r2", _NOW, match=1)
    result = run_maintenance(
        now=_NOW, cwd=tmp_path, cfg=_cfg(), daemon_url="http://unused"
    )
    grad = result["graduation"]
    assert grad["daysMet"] is False
    assert grad["ratioMet"] is False
    assert grad["ready"] is False


def test_graduation_no_events_reports_zero_days_not_started(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    import simba.memory.maintenance as maint

    monkeypatch.setattr(maint, "_fetch_tool_rule_ids", lambda url: [])
    result = run_maintenance(
        now=_NOW, cwd=tmp_path, cfg=_cfg(), daemon_url="http://unused"
    )
    grad = result["graduation"]
    assert grad["signalDays"] == 0.0
    assert grad["daysMet"] is False


def test_graduation_no_fired_rules_reports_zero_ratio(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    """No TOOL_RULE has ever fired (matched) yet -> ratio 0.0, not a division
    error and not vacuously 1.0."""
    import simba.memory.maintenance as maint

    monkeypatch.setattr(maint, "_fetch_tool_rule_ids", lambda url: ["r1"])
    with simba.db.connect(tmp_path):
        usage.bump_quality("r1", _NOW, inject=1, use=1)  # never matched
    result = run_maintenance(
        now=_NOW, cwd=tmp_path, cfg=_cfg(), daemon_url="http://unused"
    )
    grad = result["graduation"]
    assert grad["usedRatio"] == 0.0
    assert grad["ratioMet"] is False


def test_graduation_fired_is_match_count_not_inject_count(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    """'Fired' = memory_usage.match_count > 0 (bumped every time
    `_recall_tool_rules` surfaces a TOOL_RULE as a gate candidate via the
    daemon's POST /recall), NOT inject_count > 0 (wired only to the separate
    UserPromptSubmit /recall/ack lane, which the tool-rule gate never calls
    — it posts `use` feedback directly the moment it actually fires). A
    TOOL_RULE with inject+use but zero match must not count as fired."""
    import simba.memory.maintenance as maint

    monkeypatch.setattr(maint, "_fetch_tool_rule_ids", lambda url: ["r1", "r2"])
    with simba.db.connect(tmp_path):
        usage.bump_quality("r1", _NOW, match=1, use=1)  # fired + used
        usage.bump_quality("r2", _NOW, inject=1, use=1)  # injected, never matched
    result = run_maintenance(
        now=_NOW, cwd=tmp_path, cfg=_cfg(), daemon_url="http://unused"
    )
    grad = result["graduation"]
    # Only r1 counts as fired; it is used -> ratio 1.0, not diluted by r2.
    assert grad["usedRatio"] == 1.0


def test_graduation_readiness_ignores_maintenance_apply(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    """READ-ONLY: reports identically in shadow and apply passes. This check
    never itself flips maintenance_apply — a human does, after the manual
    bench guards."""
    import simba.memory.maintenance as maint

    monkeypatch.setattr(maint, "_fetch_tool_rule_ids", lambda url: ["r1"])
    with simba.db.connect(tmp_path):
        usage_events.record("mem_x", "session-a", "use", now=_NOW - 20 * _DAY)
        usage.bump_quality("r1", _NOW, match=1, use=1)
    shadow = run_maintenance(
        now=_NOW, cwd=tmp_path, cfg=_cfg(), daemon_url="http://unused"
    )
    applied = run_maintenance(
        now=_NOW,
        cwd=tmp_path,
        cfg=_cfg(maintenance_apply=True),
        daemon_url="http://unused",
    )
    assert shadow["graduation"] == applied["graduation"]


def test_maintenance_tempfile_sweep_shadow_then_apply(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    """Session-tempfile TTL (spec 33 v2 rule R6): stale per-session flag files
    (827 engagement files observed in the audit) age out via the heartbeat;
    unknown files are never touched."""
    import os

    import simba.memory.maintenance as maint

    flags = tmp_path / "flags"
    flags.mkdir()
    monkeypatch.setattr(maint, "_session_tempdir", lambda: flags)

    old_flag = flags / "claude-engagement-dead.json"
    old_flag.write_text("{}")
    os.utime(old_flag, (_NOW - 10 * _DAY, _NOW - 10 * _DAY))
    old_usage = flags / "claude-usage-session-dead.json"
    old_usage.write_text("{}")
    os.utime(old_usage, (_NOW - 10 * _DAY, _NOW - 10 * _DAY))
    fresh = flags / "claude-usage-turn-live.json"
    fresh.write_text("{}")
    os.utime(fresh, (_NOW - 1 * _DAY, _NOW - 1 * _DAY))
    unrelated = flags / "somebody-elses-file.json"
    unrelated.write_text("{}")
    os.utime(unrelated, (_NOW - 30 * _DAY, _NOW - 30 * _DAY))

    cfg = _cfg(session_tempfile_max_age_days=7.0)
    shadow = run_maintenance(
        now=_NOW, cwd=tmp_path, cfg=cfg, daemon_url="http://unused"
    )
    assert shadow["tempfiles"]["stale"] == 2
    assert shadow["tempfiles"]["deleted"] == 0
    assert old_flag.exists()

    applied = run_maintenance(
        now=_NOW, cwd=tmp_path, cfg=cfg, daemon_url="http://unused", apply=True
    )
    assert applied["tempfiles"]["deleted"] == 2
    assert not old_flag.exists()
    assert not old_usage.exists()
    assert fresh.exists()
    assert unrelated.exists()


def test_maintenance_tempfile_sweep_disabled(tmp_path: pathlib.Path) -> None:
    result = run_maintenance(
        now=_NOW, cwd=tmp_path, cfg=_cfg(), daemon_url="http://unused"
    )
    assert result["tempfiles"] == {"skipped": True}


def test_maintenance_log_appends_jsonl(tmp_path: pathlib.Path) -> None:
    """Forgetting-run tracker (spec 33 v2, hebb-mind borrow / rule R5): every
    pass appends its summary so health trends are plottable, not just the
    latest snapshot."""
    import json

    log_path = tmp_path / ".simba" / "memory" / "maintenance-log.jsonl"
    for _ in range(2):
        run_maintenance(
            now=_NOW,
            cwd=tmp_path,
            cfg=_cfg(maintenance_log_enabled=True),
            daemon_url="http://unused",
        )
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2
    entry = json.loads(lines[0])
    assert entry["apply"] is False
    assert "decay" in entry


def test_maintenance_log_disabled_writes_nothing(tmp_path: pathlib.Path) -> None:
    run_maintenance(now=_NOW, cwd=tmp_path, cfg=_cfg(), daemon_url="http://unused")
    assert not (tmp_path / ".simba" / "memory" / "maintenance-log.jsonl").exists()


def _pending_event(old_id: str, new_id: str, *, age_days: float):
    import simba.memory.supersession as ss

    return ss.append_event(
        old_id=old_id,
        new_id=new_id,
        project_path="",
        memory_type="GOTCHA",
        similarity=0.9,
        reason="near_duplicate_same_type",
        provenance="{}",
        status=ss.STATUS_PENDING,
        now=_NOW - age_days * _DAY,
    )


def test_adjudication_confirms_stale_pendings_on_apply(
    tmp_path: pathlib.Path,
) -> None:
    """Spec 33 Phase 4: pendings older than the max age resolve newest-wins —
    the supersession is confirmed and the superseded memory goes dormant."""
    import simba.memory.supersession as ss

    with simba.db.connect(tmp_path):
        _pending_event("mem_old", "mem_new", age_days=40)
        _pending_event("mem_old2", "mem_new2", age_days=5)
    result = run_maintenance(
        now=_NOW,
        cwd=tmp_path,
        cfg=_cfg(maintenance_apply=True, supersession_adjudication_enabled=True),
        daemon_url="http://unused",
    )
    assert result["adjudication"]["confirmed"] == 1
    with simba.db.connect(tmp_path):
        assert usage.get_many(["mem_old"])["mem_old"].dormant is True
        assert usage.get_many(["mem_old2"]) == {}  # fresh pending untouched
        # Penalty-inversion audit (animaworks F1): the SURVIVOR must carry no
        # penalty from the adjudication — no usage row is created for it, so
        # nothing (dormancy, noise) can land on the wrong side.
        assert usage.get_many(["mem_new"]) == {}
        assert ss.latest_successors(["mem_old"])["mem_old"].new_id == "mem_new"
        assert "mem_old2" in ss.latest_pending(["mem_old2"])


def test_adjudication_shadow_counts_without_writing(
    tmp_path: pathlib.Path,
) -> None:
    import simba.memory.supersession as ss

    with simba.db.connect(tmp_path):
        _pending_event("mem_old", "mem_new", age_days=40)
    result = run_maintenance(
        now=_NOW,
        cwd=tmp_path,
        cfg=_cfg(supersession_adjudication_enabled=True),  # shadow default
        daemon_url="http://unused",
    )
    assert result["adjudication"]["confirmed"] == 1
    assert result["adjudication"]["dry_run"] is True
    with simba.db.connect(tmp_path):
        assert usage.get_many(["mem_old"]) == {}
        assert "mem_old" in ss.latest_pending(["mem_old"])


def test_adjudication_disabled_reports_skip(tmp_path: pathlib.Path) -> None:
    result = run_maintenance(
        now=_NOW, cwd=tmp_path, cfg=_cfg(), daemon_url="http://unused"
    )
    assert result["adjudication"] == {"skipped": True}


def test_maintenance_runs_episodes_and_reflection_on_apply(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    """Spec 33 Phase 4 re-arm: episodes + reflection knobs pointed at the sync
    scheduler that never ran (episode_jobs froze 2026-06-08). The heartbeat is
    their driver now — dispatching only when applying."""
    import simba.episodes.consolidate
    import simba.reflection.pass_

    monkeypatch.setattr(
        simba.episodes.consolidate,
        "consolidate_eligible",
        lambda cwd, **kw: {"dispatched": ["s1"], "skipped": 0},
    )
    monkeypatch.setattr(
        simba.reflection.pass_,
        "reflect_pass",
        lambda **kw: types.SimpleNamespace(status="ok", dispatched=True),
    )
    result = run_maintenance(
        now=_NOW,
        cwd=tmp_path,
        cfg=_cfg(maintenance_apply=True),
        daemon_url="http://unused",
    )
    assert result["episodes"]["dispatched"] == ["s1"]
    assert result["reflection"]["status"] == "ok"


def test_maintenance_shadow_skips_episodes_and_reflection(
    tmp_path: pathlib.Path,
) -> None:
    result = run_maintenance(
        now=_NOW, cwd=tmp_path, cfg=_cfg(), daemon_url="http://unused"
    )
    assert result["episodes"]["skipped"] is True
    assert result["reflection"]["skipped"] is True


def test_run_maintenance_is_shadow_by_default(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        usage.get_or_create("mem_a", now=_NOW - 200 * _DAY)
    result = run_maintenance(
        now=_NOW, cwd=tmp_path, cfg=_cfg(), daemon_url="http://unused"
    )
    assert result["apply"] is False
    assert result["decay"]["newly_dormant"] == 1
    with simba.db.connect(tmp_path):
        row = usage.get_many(["mem_a"])["mem_a"]
    assert row.strength == 1.0
    assert row.dormant is False


def test_run_maintenance_apply_persists(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        usage.get_or_create("mem_b", now=_NOW - 200 * _DAY)
    result = run_maintenance(
        now=_NOW,
        cwd=tmp_path,
        cfg=_cfg(maintenance_apply=True),
        daemon_url="http://unused",
    )
    assert result["apply"] is True
    with simba.db.connect(tmp_path):
        row = usage.get_many(["mem_b"])["mem_b"]
    assert row.strength < 0.1
    assert row.dormant is True


def test_run_maintenance_explicit_apply_overrides_config(
    tmp_path: pathlib.Path,
) -> None:
    with simba.db.connect(tmp_path):
        usage.get_or_create("mem_c", now=_NOW - 200 * _DAY)
    result = run_maintenance(
        now=_NOW,
        cwd=tmp_path,
        cfg=_cfg(maintenance_apply=False),
        daemon_url="http://unused",
        apply=True,
    )
    assert result["apply"] is True
    with simba.db.connect(tmp_path):
        assert usage.get_many(["mem_c"])["mem_c"].dormant is True


def test_run_maintenance_decay_disabled_reports_skip(
    tmp_path: pathlib.Path,
) -> None:
    result = run_maintenance(
        now=_NOW,
        cwd=tmp_path,
        cfg=_cfg(decay_enabled=False),
        daemon_url="http://unused",
    )
    assert result["decay"] == {"skipped": True}


def test_run_maintenance_hygiene_disabled_reports_skip(
    tmp_path: pathlib.Path,
) -> None:
    result = run_maintenance(
        now=_NOW, cwd=tmp_path, cfg=_cfg(), daemon_url="http://unused"
    )
    assert result["hygiene"] == {"skipped": True}


def test_run_maintenance_fetches_type_map_for_multipliers(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    """Type-aware decay (spec 33 Phase 2): configured multipliers pull the
    id→type map from the daemon and slow the configured types."""
    import simba.memory.maintenance as maint

    calls: dict = {}

    def _fake_fetch(daemon_url: str):
        calls["url"] = daemon_url
        return {"mem_p": "PATTERN"}

    monkeypatch.setattr(maint, "_fetch_type_map", _fake_fetch)
    with simba.db.connect(tmp_path):
        usage.get_or_create("mem_p", now=_NOW - 45 * _DAY)
        usage.get_or_create("mem_g", now=_NOW - 45 * _DAY)
    cfg = _cfg(maintenance_apply=True, decay_type_multipliers="PATTERN:4")
    run_maintenance(now=_NOW, cwd=tmp_path, cfg=cfg, daemon_url="http://x")
    with simba.db.connect(tmp_path):
        rows = usage.get_many(["mem_p", "mem_g"])
    assert calls["url"] == "http://x"
    assert rows["mem_p"].strength > rows["mem_g"].strength


def test_fetch_type_map_skips_http_when_shutting_down(monkeypatch) -> None:
    """Root cause (b), handoff item 10: once uvicorn stops serving, a
    self-HTTP loopback call can never complete --- guaranteeing the graceful
    shutdown timeout is breached if a maintenance pass is mid-flight. The
    shutdown flag must short-circuit before any HTTP attempt."""
    import httpx

    import simba.memory.background as background
    import simba.memory.maintenance as maint

    def _boom(*a, **kw):
        raise AssertionError("httpx.get must not be called while shutting down")

    monkeypatch.setattr(httpx, "get", _boom)
    background.mark_shutting_down()

    assert maint._fetch_type_map("http://x") is None


def test_fetch_tool_rule_ids_skips_http_when_shutting_down(monkeypatch) -> None:
    """Same guard as ``_fetch_type_map`` for the graduation-readiness fetch
    (also reachable from ``GET /digest``'s ``_graduation_readiness`` call)."""
    import httpx

    import simba.memory.background as background
    import simba.memory.maintenance as maint

    def _boom(*a, **kw):
        raise AssertionError("httpx.get must not be called while shutting down")

    monkeypatch.setattr(httpx, "get", _boom)
    background.mark_shutting_down()

    assert maint._fetch_tool_rule_ids("http://x") == []


def test_fetch_type_map_calls_http_when_not_shutting_down(monkeypatch) -> None:
    """Sanity check: the guard is additive --- normal operation is unaffected."""
    import httpx

    import simba.memory.background as background
    import simba.memory.maintenance as maint

    assert background.is_shutting_down() is False

    calls: dict = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"memories": [{"id": "mem_a", "type": "PATTERN"}]}

    def _fake_get(url, **kw):
        calls["url"] = url
        return _FakeResponse()

    monkeypatch.setattr(httpx, "get", _fake_get)

    assert maint._fetch_type_map("http://x") == {"mem_a": "PATTERN"}
    assert calls["url"] == "http://x/list"


@pytest.mark.asyncio
async def test_scheduler_run_once_stores_and_reports(tmp_path: pathlib.Path) -> None:
    with simba.db.connect(tmp_path):
        usage.get_or_create("mem_s", now=_NOW - 200 * _DAY)
    seen: list[dict] = []
    sched = MaintenanceScheduler(
        cwd=tmp_path,
        daemon_url="http://unused",
        interval_seconds=3600,
        startup_delay_seconds=0.0,
        config_loader=lambda: _cfg(),
        on_result=seen.append,
    )
    result = await sched.run_once()
    assert result["decay"]["processed"] == 1
    assert seen == [result]
    assert sched.last_result is result


@pytest.mark.asyncio
async def test_scheduler_stops_during_startup_delay(tmp_path: pathlib.Path) -> None:
    sched = MaintenanceScheduler(
        cwd=tmp_path,
        daemon_url="http://unused",
        interval_seconds=3600,
        startup_delay_seconds=60.0,
        config_loader=lambda: _cfg(),
    )
    task = asyncio.create_task(sched.run_forever())
    await asyncio.sleep(0.05)
    sched.stop()
    await asyncio.wait_for(task, timeout=2.0)
    assert sched.last_result is None  # never got past the startup delay


@pytest.mark.asyncio
async def test_scheduler_config_loader_failure_is_fail_open(
    tmp_path: pathlib.Path,
) -> None:
    def _boom():
        raise RuntimeError("config unavailable")

    sched = MaintenanceScheduler(
        cwd=tmp_path,
        daemon_url="http://unused",
        interval_seconds=3600,
        startup_delay_seconds=0.0,
        config_loader=_boom,
    )
    result = await sched.run_once()
    assert result is None
    assert sched.last_result is None


@pytest.mark.asyncio
async def test_server_gates_scheduler_on_interval(tmp_path: pathlib.Path) -> None:
    import simba.memory.config
    import simba.memory.server

    app = simba.memory.server.create_app(
        simba.memory.config.MemoryConfig(maintenance_interval_hours=0)
    )
    app.state.cwd = tmp_path
    task = await simba.memory.server._start_maintenance_scheduler(app)
    assert task is None
    assert app.state.maintenance_scheduler is None


@pytest.mark.asyncio
async def test_server_starts_scheduler_when_enabled(tmp_path: pathlib.Path) -> None:
    import simba.memory.config
    import simba.memory.server

    # Default startup delay keeps the first pass parked, so the test can stop
    # the scheduler before any pass touches the (unconfigured) daemon URL.
    app = simba.memory.server.create_app(
        simba.memory.config.MemoryConfig(
            maintenance_interval_hours=24.0,
            maintenance_startup_delay_seconds=300.0,
        )
    )
    app.state.cwd = tmp_path
    task = await simba.memory.server._start_maintenance_scheduler(app)
    assert task is not None
    assert app.state.maintenance_scheduler is not None
    app.state.maintenance_scheduler.stop()
    await asyncio.wait_for(task, timeout=2.0)


@pytest.mark.asyncio
async def test_maintenance_endpoint_and_stats(
    async_client, monkeypatch, tmp_path
) -> None:
    import simba.memory.hygiene
    import simba.memory.maintenance

    monkeypatch.setattr(
        simba.memory.maintenance,
        "run_hygiene_pass",
        lambda **kw: simba.memory.hygiene.HygieneResult(),
    )
    with simba.db.connect(tmp_path):
        usage.get_or_create("mem_e", now=_NOW - 200 * _DAY)

    resp = await async_client.post("/maintenance/run", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["apply"] is False  # config default = shadow
    assert data["decay"]["processed"] >= 1

    stats = (await async_client.get("/stats")).json()
    assert stats["lastMaintenance"]["at"] == data["at"]


@pytest.mark.asyncio
async def test_maintenance_endpoint_apply_override(
    async_client, monkeypatch, tmp_path
) -> None:
    import simba.memory.hygiene
    import simba.memory.maintenance

    monkeypatch.setattr(
        simba.memory.maintenance,
        "run_hygiene_pass",
        lambda **kw: simba.memory.hygiene.HygieneResult(),
    )
    with simba.db.connect(tmp_path):
        usage.get_or_create("mem_f", now=_NOW - 200 * _DAY)

    resp = await async_client.post("/maintenance/run", json={"apply": True})
    assert resp.status_code == 200
    assert resp.json()["apply"] is True
    with simba.db.connect(tmp_path):
        assert usage.get_many(["mem_f"])["mem_f"].dormant is True
