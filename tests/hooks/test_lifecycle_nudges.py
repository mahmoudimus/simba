"""SessionStart lifecycle nudges (spec 33 Phase 5).

One glanceable line each for the latest maintenance heartbeat and the
promotion candidates awaiting review. Default-off → "" and zero HTTP.
"""

from __future__ import annotations

import simba.hooks.config as hooks_config
import simba.hooks.session_start as session_start


class _Resp:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def test_nudges_prefer_boot_digest(monkeypatch) -> None:
    """Spec 33 v2: one /digest call is the consumption surface — heartbeat,
    promotion inbox, supersession pendings, and knowledge gaps in one glance."""
    cfg = hooks_config.HooksConfig(session_start_lifecycle_nudges=True)

    def fake_get(url, params=None, timeout=0.0):
        assert url.endswith("/digest")
        return _Resp(
            {
                "heartbeat": {
                    "at": "2026-07-04T00:00:00Z",
                    "apply": False,
                    "decay": {"updated": 4245, "newly_dormant": 0},
                },
                "promotions": {"total": 3, "top": []},
                "supersessions": {"pending": 170},
                "gaps": {"total": 2, "top": ["how do we rotate staging certs"]},
            }
        )

    monkeypatch.setattr(session_start.httpx, "get", fake_get)
    text = session_start._lifecycle_nudges(cfg)
    assert "2026-07-04T00:00:00Z" in text
    assert "shadow" in text
    assert "3 promotion candidate" in text
    assert "170 pending supersession" in text
    assert "2 knowledge gap" in text
    assert "simba memory promote" in text


def test_nudges_render_heartbeat_and_candidates(monkeypatch) -> None:
    """Fallback path (older daemon without /digest): /stats + /promotions."""
    cfg = hooks_config.HooksConfig(session_start_lifecycle_nudges=True)

    def fake_get(url, params=None, timeout=0.0):
        if url.endswith("/digest"):
            return _Resp({}, status_code=404)
        if url.endswith("/stats"):
            return _Resp(
                {
                    "lastMaintenance": {
                        "at": "2026-07-03T21:00:00Z",
                        "apply": False,
                        "decay": {"updated": 4200, "newly_dormant": 1800},
                    }
                }
            )
        return _Resp({"total": 2, "candidates": []})

    monkeypatch.setattr(session_start.httpx, "get", fake_get)
    text = session_start._lifecycle_nudges(cfg)
    assert "2026-07-03T21:00:00Z" in text
    assert "shadow" in text
    assert "dormant=1800" in text
    assert "2 promotion candidate" in text
    assert "simba memory promote" in text


def test_nudges_off_by_default_no_http(monkeypatch) -> None:
    def _boom(*a, **kw):  # pragma: no cover - must not be reached
        raise AssertionError("no HTTP when nudges disabled")

    monkeypatch.setattr(session_start.httpx, "get", _boom)
    assert session_start._lifecycle_nudges(hooks_config.HooksConfig()) == ""


def test_nudges_fail_soft_on_daemon_error(monkeypatch) -> None:
    import httpx

    cfg = hooks_config.HooksConfig(session_start_lifecycle_nudges=True)

    def fake_get(*a, **kw):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(session_start.httpx, "get", fake_get)
    assert session_start._lifecycle_nudges(cfg) == ""


def test_nudges_quiet_when_nothing_to_say(monkeypatch) -> None:
    cfg = hooks_config.HooksConfig(session_start_lifecycle_nudges=True)

    def fake_get(url, params=None, timeout=0.0):
        if url.endswith("/digest"):
            return _Resp({}, status_code=404)
        if url.endswith("/stats"):
            return _Resp({"lastMaintenance": None})
        return _Resp({"total": 0, "candidates": []})

    monkeypatch.setattr(session_start.httpx, "get", fake_get)
    assert session_start._lifecycle_nudges(cfg) == ""


def test_nudges_digest_quiet_when_empty(monkeypatch) -> None:
    cfg = hooks_config.HooksConfig(session_start_lifecycle_nudges=True)

    def fake_get(url, params=None, timeout=0.0):
        return _Resp(
            {
                "heartbeat": None,
                "promotions": {"total": 0, "top": []},
                "supersessions": {"pending": 0},
                "gaps": {"total": 0, "top": []},
            }
        )

    monkeypatch.setattr(session_start.httpx, "get", fake_get)
    assert session_start._lifecycle_nudges(cfg) == ""
