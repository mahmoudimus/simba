"""SessionStart lifecycle nudges (spec 33 Phase 5).

One glanceable line each for the latest maintenance heartbeat and the
promotion candidates awaiting review. Default-off → "" and zero HTTP.
"""

from __future__ import annotations

import simba.hooks.config as hooks_config
import simba.hooks.session_start as session_start


class _Resp:
    def __init__(self, payload: dict) -> None:
        self.status_code = 200
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def test_nudges_render_heartbeat_and_candidates(monkeypatch) -> None:
    cfg = hooks_config.HooksConfig(session_start_lifecycle_nudges=True)

    def fake_get(url, params=None, timeout=0.0):
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
        if url.endswith("/stats"):
            return _Resp({"lastMaintenance": None})
        return _Resp({"total": 0, "candidates": []})

    monkeypatch.setattr(session_start.httpx, "get", fake_get)
    assert session_start._lifecycle_nudges(cfg) == ""
