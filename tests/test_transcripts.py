"""Tests for project-scoped transcript resolution (fixes /memories-learn cross-wiring).

Bug: ~/.claude/transcripts/latest.json is a single GLOBAL symlink overwritten by
whichever session compacts last, across all projects — so /memories-learn extracted
the wrong project's transcript. find_pending resolves by the CURRENT project +
pending status instead; mark_extracted flips status so it isn't re-extracted.
"""

from __future__ import annotations

import json
import pathlib

import simba.transcripts as tr


def _meta(
    d: pathlib.Path, sid: str, project: str, status: str, exported_at: str
) -> pathlib.Path:
    sdir = d / sid
    sdir.mkdir(parents=True, exist_ok=True)
    mp = sdir / "metadata.json"
    mp.write_text(
        json.dumps(
            {
                "session_id": sid,
                "project_path": project,
                "status": status,
                "exported_at": exported_at,
                "transcript_path": str(sdir / "transcript.md"),
            }
        )
    )
    return mp


def test_find_pending_matches_project_and_status(tmp_path: pathlib.Path) -> None:
    _meta(tmp_path, "s-simba", "/p/simba", "pending_extraction", "2026-06-04T01:00:00Z")
    _meta(tmp_path, "s-acme", "/p/acme", "pending_extraction", "2026-06-04T02:00:00Z")
    # newest overall is acme, but we ask for simba -> must get simba, not acme
    got = tr.find_pending("/p/simba", transcripts_dir=tmp_path)
    assert got is not None
    assert got["session_id"] == "s-simba"


def test_find_pending_ignores_already_extracted(tmp_path: pathlib.Path) -> None:
    _meta(tmp_path, "s1", "/p/simba", "extracted", "2026-06-04T01:00:00Z")
    assert tr.find_pending("/p/simba", transcripts_dir=tmp_path) is None


def test_find_pending_unknown_project_is_none(tmp_path: pathlib.Path) -> None:
    _meta(tmp_path, "s1", "/p/simba", "pending_extraction", "2026-06-04T01:00:00Z")
    assert tr.find_pending("/p/other", transcripts_dir=tmp_path) is None


def test_find_pending_picks_newest_for_project(tmp_path: pathlib.Path) -> None:
    _meta(tmp_path, "old", "/p/simba", "pending_extraction", "2026-06-04T01:00:00Z")
    _meta(tmp_path, "new", "/p/simba", "pending_extraction", "2026-06-04T03:00:00Z")
    got = tr.find_pending("/p/simba", transcripts_dir=tmp_path)
    assert got["session_id"] == "new"


def test_find_pending_normalizes_paths(tmp_path: pathlib.Path) -> None:
    _meta(tmp_path, "s1", "/p/simba", "pending_extraction", "2026-06-04T01:00:00Z")
    # trailing slash / dot segments must still match
    assert tr.find_pending("/p/simba/", transcripts_dir=tmp_path) is not None
    assert tr.find_pending("/p/./simba", transcripts_dir=tmp_path) is not None


def test_mark_extracted_flips_status_on_disk(tmp_path: pathlib.Path) -> None:
    mp = _meta(tmp_path, "s1", "/p/simba", "pending_extraction", "2026-06-04T01:00:00Z")
    tr.mark_extracted(mp)
    assert json.loads(mp.read_text())["status"] == "extracted"
    # and it's no longer returned as pending
    assert tr.find_pending("/p/simba", transcripts_dir=tmp_path) is None


def test_cli_pending_scoped_by_project(tmp_path, monkeypatch, capsys) -> None:
    import simba.__main__ as m

    _meta(tmp_path, "s-simba", "/p/simba", "pending_extraction", "2026-06-04T01:00:00Z")
    _meta(tmp_path, "s-acme", "/p/acme", "pending_extraction", "2026-06-04T02:00:00Z")
    monkeypatch.setattr(tr, "default_transcripts_dir", lambda: tmp_path)
    rc = m._cmd_transcript(["pending", "--project", "/p/acme"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "s-acme" in out and "s-simba" not in out  # never the wrong project


def test_cli_pending_none_returns_nonzero(tmp_path, monkeypatch) -> None:
    import simba.__main__ as m

    monkeypatch.setattr(tr, "default_transcripts_dir", lambda: tmp_path)
    assert m._cmd_transcript(["pending", "--project", "/p/none"]) == 1


def test_cli_mark_extracted(tmp_path, monkeypatch) -> None:
    import simba.__main__ as m

    _meta(tmp_path, "s1", "/p/simba", "pending_extraction", "2026-06-04T01:00:00Z")
    monkeypatch.setattr(tr, "default_transcripts_dir", lambda: tmp_path)
    assert m._cmd_transcript(["mark-extracted", "s1"]) == 0
    assert tr.find_pending("/p/simba", transcripts_dir=tmp_path) is None
