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
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


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
