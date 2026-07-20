"""SessionStart lifecycle nudges (spec 33 Phase 5).

One glanceable line each for the latest maintenance heartbeat and the
promotion candidates awaiting review. Default-off → "" and zero HTTP.
"""

from __future__ import annotations

import json
import pathlib

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


def test_nudges_graduation_ready_and_not_applying(monkeypatch) -> None:
    """Spec 33 Part 8 rule R1: nudge fires ONLY when the DATA criteria are
    met AND maintenance_apply hasn't already flipped. The bench guards
    (LME-S/LoCoMo/HaluMem) stay manual — this only informs, never flips."""
    cfg = hooks_config.HooksConfig(session_start_lifecycle_nudges=True)

    def fake_get(url, params=None, timeout=0.0):
        assert url.endswith("/digest")
        return _Resp(
            {
                "heartbeat": {
                    "at": "2026-07-04T00:00:00Z",
                    "apply": False,
                    "decay": {},
                },
                "promotions": {"total": 0, "top": []},
                "supersessions": {"pending": 0},
                "gaps": {"total": 0, "top": []},
                "graduation": {
                    "signalDays": 15.0,
                    "usedRatio": 0.75,
                    "daysMet": True,
                    "ratioMet": True,
                    "ready": True,
                    "benchGuards": "manual",
                },
            }
        )

    monkeypatch.setattr(session_start.httpx, "get", fake_get)
    text = session_start._lifecycle_nudges(cfg)
    assert "maintenance_apply data criteria MET" in text
    assert "15d signals" in text
    assert "75% fired rules used" in text
    assert "simba config set memory.maintenance_apply true" in text


def test_nudges_graduation_silent_when_not_ready(monkeypatch) -> None:
    cfg = hooks_config.HooksConfig(session_start_lifecycle_nudges=True)

    def fake_get(url, params=None, timeout=0.0):
        return _Resp(
            {
                "heartbeat": {
                    "at": "2026-07-04T00:00:00Z",
                    "apply": False,
                    "decay": {},
                },
                "promotions": {"total": 0, "top": []},
                "supersessions": {"pending": 0},
                "gaps": {"total": 0, "top": []},
                "graduation": {
                    "signalDays": 5.0,
                    "usedRatio": 0.2,
                    "daysMet": False,
                    "ratioMet": False,
                    "ready": False,
                    "benchGuards": "manual",
                },
            }
        )

    monkeypatch.setattr(session_start.httpx, "get", fake_get)
    text = session_start._lifecycle_nudges(cfg)
    assert "maintenance_apply data criteria MET" not in text


def test_nudges_graduation_silent_when_already_applying(monkeypatch) -> None:
    """ready=True but maintenance_apply already flipped (heartbeat.apply is
    True) -> no nudge; the lever's already turned, nagging would be noise."""
    cfg = hooks_config.HooksConfig(session_start_lifecycle_nudges=True)

    def fake_get(url, params=None, timeout=0.0):
        return _Resp(
            {
                "heartbeat": {
                    "at": "2026-07-04T00:00:00Z",
                    "apply": True,
                    "decay": {},
                },
                "promotions": {"total": 0, "top": []},
                "supersessions": {"pending": 0},
                "gaps": {"total": 0, "top": []},
                "graduation": {
                    "signalDays": 20.0,
                    "usedRatio": 0.9,
                    "daysMet": True,
                    "ratioMet": True,
                    "ready": True,
                    "benchGuards": "manual",
                },
            }
        )

    monkeypatch.setattr(session_start.httpx, "get", fake_get)
    text = session_start._lifecycle_nudges(cfg)
    assert "maintenance_apply data criteria MET" not in text


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


def _quiet_digest_get(url, params=None, timeout=0.0):
    """A /digest response with nothing to report — isolates the curated-import
    nudge (spec 33 R4) from the heartbeat/promotion lines in the tests below."""
    return _Resp(
        {
            "heartbeat": None,
            "promotions": {"total": 0, "top": []},
            "supersessions": {"pending": 0},
            "gaps": {"total": 0, "top": []},
        }
    )


def test_curated_nudge_present_when_marker_stale(tmp_path, monkeypatch) -> None:
    """Spec 33 R4: MEMORY.md touched after the marker's last_import_at means
    the curated bridge is out of date — nudge a re-import."""
    cfg = hooks_config.HooksConfig(session_start_lifecycle_nudges=True)
    monkeypatch.setattr(session_start.httpx, "get", _quiet_digest_get)

    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()
    memory_md = curated_dir / "MEMORY.md"
    memory_md.write_text("# Memory Index\n")
    stamp = memory_md.stat().st_mtime

    project_dir = tmp_path / "project"
    simba_dir = project_dir / ".simba"
    simba_dir.mkdir(parents=True)
    (simba_dir / "curated-import.json").write_text(
        json.dumps({"dir": str(curated_dir), "last_import_at": stamp - 100})
    )

    text = session_start._lifecycle_nudges(cfg, cwd=project_dir)
    assert "curated memory changed since last import" in text
    assert str(curated_dir) in text
    assert "import-curated --dir" in text


def test_curated_nudge_absent_when_marker_fresh(tmp_path, monkeypatch) -> None:
    """The mirror image: last_import_at AFTER MEMORY.md's mtime -> no nudge."""
    cfg = hooks_config.HooksConfig(session_start_lifecycle_nudges=True)
    monkeypatch.setattr(session_start.httpx, "get", _quiet_digest_get)

    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()
    memory_md = curated_dir / "MEMORY.md"
    memory_md.write_text("# Memory Index\n")
    stamp = memory_md.stat().st_mtime

    project_dir = tmp_path / "project"
    simba_dir = project_dir / ".simba"
    simba_dir.mkdir(parents=True)
    (simba_dir / "curated-import.json").write_text(
        json.dumps({"dir": str(curated_dir), "last_import_at": stamp + 100})
    )

    assert session_start._lifecycle_nudges(cfg, cwd=project_dir) == ""


def test_curated_nudge_absent_when_marker_missing(tmp_path, monkeypatch) -> None:
    cfg = hooks_config.HooksConfig(session_start_lifecycle_nudges=True)
    monkeypatch.setattr(session_start.httpx, "get", _quiet_digest_get)

    assert session_start._lifecycle_nudges(cfg, cwd=tmp_path) == ""


def test_curated_nudge_absent_when_marker_corrupt(tmp_path, monkeypatch) -> None:
    cfg = hooks_config.HooksConfig(session_start_lifecycle_nudges=True)
    monkeypatch.setattr(session_start.httpx, "get", _quiet_digest_get)

    simba_dir = tmp_path / ".simba"
    simba_dir.mkdir()
    (simba_dir / "curated-import.json").write_text("not json{{{")

    assert session_start._lifecycle_nudges(cfg, cwd=tmp_path) == ""


def test_curated_nudge_absent_when_marker_missing_keys(tmp_path, monkeypatch) -> None:
    """Structurally valid JSON but missing the expected fields -> no crash."""
    cfg = hooks_config.HooksConfig(session_start_lifecycle_nudges=True)
    monkeypatch.setattr(session_start.httpx, "get", _quiet_digest_get)

    simba_dir = tmp_path / ".simba"
    simba_dir.mkdir()
    (simba_dir / "curated-import.json").write_text(json.dumps({"unexpected": True}))

    assert session_start._lifecycle_nudges(cfg, cwd=tmp_path) == ""


def test_rule_candidates_inbox_line_present_when_pending(tmp_path, monkeypatch) -> None:
    """N pending rule candidate(s) shows up in the same joined inbox line as
    promotions/supersessions/gaps -- purely local (.simba/simba.db), no
    daemon dependency beyond the /digest call other inbox lines already make.
    """
    import simba.db
    import simba.redirect.candidates as candidates

    cfg = hooks_config.HooksConfig(session_start_lifecycle_nudges=True)
    monkeypatch.setattr(session_start.httpx, "get", _quiet_digest_get)

    with simba.db.connect(tmp_path):
        candidates.RuleCandidate.create(
            signature="sig-1",
            tool="Bash",
            rule_kind="pattern",
            status="pending",
            created_at="2026-07-20T00:00:00Z",
        )

    text = session_start._lifecycle_nudges(cfg, cwd=tmp_path)
    assert "1 pending rule candidate(s)" in text
    assert "simba rule promote" in text
    assert "[Lifecycle] inbox:" in text


def test_rule_candidates_inbox_line_absent_when_none_pending(
    tmp_path, monkeypatch
) -> None:
    cfg = hooks_config.HooksConfig(session_start_lifecycle_nudges=True)
    monkeypatch.setattr(session_start.httpx, "get", _quiet_digest_get)

    assert session_start._lifecycle_nudges(cfg, cwd=tmp_path) == ""


def test_rule_candidates_inbox_line_fail_soft_on_bad_db(monkeypatch) -> None:
    """A DB/filesystem error while counting must never break the whole
    lifecycle nudge -- resolves to "" for this line, same as everywhere
    else in this module."""
    cfg = hooks_config.HooksConfig(session_start_lifecycle_nudges=True)
    monkeypatch.setattr(session_start.httpx, "get", _quiet_digest_get)

    bogus_cwd = pathlib.Path("/nonexistent/not-a-real-path-xyz")
    assert session_start._lifecycle_nudges(cfg, cwd=bogus_cwd) == ""


def test_rule_candidates_inbox_line_in_fallback_path(tmp_path, monkeypatch) -> None:
    """Older-daemon fallback path (no /digest): the rule-candidate count is
    local-only, so it still surfaces even when the daemon can't serve
    /digest."""
    import simba.db
    import simba.redirect.candidates as candidates

    cfg = hooks_config.HooksConfig(session_start_lifecycle_nudges=True)

    def fake_get(url, params=None, timeout=0.0):
        if url.endswith("/digest"):
            return _Resp({}, status_code=404)
        if url.endswith("/stats"):
            return _Resp({"lastMaintenance": None})
        return _Resp({"total": 0, "candidates": []})

    monkeypatch.setattr(session_start.httpx, "get", fake_get)

    with simba.db.connect(tmp_path):
        candidates.RuleCandidate.create(
            signature="sig-1",
            tool="Bash",
            rule_kind="pattern",
            status="pending",
            created_at="2026-07-20T00:00:00Z",
        )

    text = session_start._lifecycle_nudges(cfg, cwd=tmp_path)
    assert "1 pending rule candidate(s)" in text
    assert "simba rule promote" in text


def test_curated_nudge_absent_when_cwd_is_none(monkeypatch) -> None:
    """No cwd (e.g. dispatch inside the daemon with no payload cwd) -> no nudge,
    no filesystem access attempted."""
    cfg = hooks_config.HooksConfig(session_start_lifecycle_nudges=True)
    monkeypatch.setattr(session_start.httpx, "get", _quiet_digest_get)

    assert session_start._lifecycle_nudges(cfg, cwd=None) == ""
