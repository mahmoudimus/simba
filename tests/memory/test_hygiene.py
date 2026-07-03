"""Memory-hygiene TTL aging for stale TOOL_RULE memories (Task C.2)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _now() -> datetime:
    return datetime.now(tz=UTC)


def test_hygiene_disabled_when_zero_age() -> None:
    from simba.memory.config import MemoryConfig
    from simba.memory.hygiene import run_hygiene_pass

    result = run_hygiene_pass(
        daemon_url="http://x", cfg=MemoryConfig(tool_rule_max_age_days=0)
    )
    assert result.expired_count == 0
    assert result.errors == 0


def test_hygiene_deletes_stale_tool_rules(monkeypatch) -> None:
    from simba.memory.config import MemoryConfig
    from simba.memory.hygiene import run_hygiene_pass

    old = _iso(_now() - timedelta(days=40))
    fresh = _iso(_now() - timedelta(days=5))
    memories = [
        {"id": "m1", "type": "TOOL_RULE", "createdAt": old},
        {"id": "m2", "type": "TOOL_RULE", "createdAt": fresh},
    ]

    deleted: list[str] = []

    def fake_get(url, **kw):
        r = MagicMock()
        r.raise_for_status = lambda: None
        r.json.return_value = {"memories": memories, "total": 2}
        return r

    def fake_delete(url, **kw):
        mid = url.split("/")[-1]
        deleted.append(mid)
        r = MagicMock()
        r.raise_for_status = lambda: None
        return r

    with (
        patch("httpx.get", side_effect=fake_get),
        patch("httpx.delete", side_effect=fake_delete),
    ):
        result = run_hygiene_pass(
            daemon_url="http://localhost:8741",
            cfg=MemoryConfig(tool_rule_max_age_days=30),
        )
    assert result.expired_count == 1
    assert deleted == ["m1"]


def test_hygiene_http_error_is_fail_open(monkeypatch) -> None:
    import httpx

    from simba.memory.config import MemoryConfig
    from simba.memory.hygiene import run_hygiene_pass

    with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
        result = run_hygiene_pass(
            daemon_url="http://localhost:8741",
            cfg=MemoryConfig(tool_rule_max_age_days=30),
        )
    assert result.errors >= 1
    assert result.expired_count == 0


def test_hygiene_skips_recently_used_rules(tmp_path, monkeypatch) -> None:
    """Use-it-and-keep-it (spec 33 Phase 2): a stale-created rule with a fresh
    last_used survives; expiry keys off consumption, not creation."""
    import time as _time

    import simba.db
    import simba.memory.usage as usage_mod
    from simba.memory.config import MemoryConfig
    from simba.memory.hygiene import run_hygiene_pass

    old = _iso(_now() - timedelta(days=40))
    memories = [
        {"id": "m_used", "type": "TOOL_RULE", "createdAt": old},
        {"id": "m_stale", "type": "TOOL_RULE", "createdAt": old},
    ]
    with simba.db.connect(tmp_path):
        usage_mod.bump_quality("m_used", _time.time() - 86400, use=1)

    deleted: list[str] = []

    def fake_get(url, **kw):
        r = MagicMock()
        r.raise_for_status = lambda: None
        r.json.return_value = {"memories": memories, "total": 2}
        return r

    def fake_delete(url, **kw):
        deleted.append(url.split("/")[-1])
        r = MagicMock()
        r.raise_for_status = lambda: None
        return r

    with (
        patch("httpx.get", side_effect=fake_get),
        patch("httpx.delete", side_effect=fake_delete),
    ):
        result = run_hygiene_pass(
            daemon_url="http://localhost:8741",
            cfg=MemoryConfig(tool_rule_max_age_days=30),
            cwd=tmp_path,
        )
    assert deleted == ["m_stale"]
    assert result.expired_count == 1


def test_hygiene_dry_run_counts_without_delete(monkeypatch) -> None:
    """Shadow mode (spec 33): count would-expire rules, never DELETE."""
    from simba.memory.config import MemoryConfig
    from simba.memory.hygiene import run_hygiene_pass

    old = _iso(_now() - timedelta(days=40))
    memories = [{"id": "m1", "type": "TOOL_RULE", "createdAt": old}]
    deleted: list[str] = []

    def fake_get(url, **kw):
        r = MagicMock()
        r.raise_for_status = lambda: None
        r.json.return_value = {"memories": memories, "total": 1}
        return r

    def fake_delete(url, **kw):
        deleted.append(url.split("/")[-1])
        r = MagicMock()
        r.raise_for_status = lambda: None
        return r

    with (
        patch("httpx.get", side_effect=fake_get),
        patch("httpx.delete", side_effect=fake_delete),
    ):
        result = run_hygiene_pass(
            daemon_url="http://localhost:8741",
            cfg=MemoryConfig(tool_rule_max_age_days=30),
            dry_run=True,
        )
    assert result.dry_run is True
    assert result.expired_count == 1
    assert deleted == []
