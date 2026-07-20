"""Tests for the PreCompact over-cap path's DETACHED distiller spawn
(2026-07-20): a bounded replacement for the blind over-cap skip -- see
``transcripts/distill.py`` and ``hooks/pre_compact.py``'s over-cap branch.

The in-daemon code path must only ever do a size check, a marker check, and
a ``Popen`` -- it must NEVER parse the transcript itself. These tests mock
``subprocess.Popen`` throughout so no real subprocess (and no real parse)
ever runs during the unit suite.
"""

from __future__ import annotations

import json
import pathlib
import sys
import unittest.mock

import simba.config
import simba.hooks.config
import simba.hooks.pre_compact as pc
import simba.transcripts.distill as distill

_LOGGER_NAME = "simba.hooks.pre_compact"


def _write_transcript(path: pathlib.Path, n: int) -> None:
    lines = [
        json.dumps(
            {
                "session_id": "s1",
                "cwd": "/proj",
                "message": {"role": "user", "content": f"m{i}"},
            }
        )
        for i in range(n)
    ]
    path.write_text("\n".join(lines) + "\n")


def _patch_cap_and_distill(monkeypatch, *, cap_mb: float, distill_enabled: bool = True):
    real_load = simba.config.load

    def fake_load(section, *a, **k):
        if section == "hooks":
            return simba.hooks.config.HooksConfig(
                pre_compact_max_transcript_mb=cap_mb,
                pre_compact_distill_enabled=distill_enabled,
            )
        return real_load(section, *a, **k)

    monkeypatch.setattr(simba.config, "load", fake_load)


def _run_main(fake_home, session_id, transcript, cwd) -> dict:
    with unittest.mock.patch("pathlib.Path.home", return_value=fake_home):
        return json.loads(
            pc.main(
                {
                    "session_id": session_id,
                    "transcript_path": str(transcript),
                    "cwd": str(cwd),
                }
            )
        )


class TestOverCapSpawnsDistillerDetached:
    def test_spawns_popen_with_expected_argv_and_log_redirect(
        self, tmp_path: pathlib.Path, monkeypatch
    ) -> None:
        transcript = tmp_path / "t.jsonl"
        _write_transcript(transcript, 10)
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        _patch_cap_and_distill(monkeypatch, cap_mb=1e-6)

        captured = {}

        def fake_popen(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return unittest.mock.MagicMock()

        monkeypatch.setattr(pc.subprocess, "Popen", fake_popen)

        result = _run_main(fake_home, "cap-session", transcript, tmp_path)
        assert result == {}

        argv = captured["argv"]
        assert argv[:4] == [sys.executable, "-m", "simba", "transcript"]
        assert argv[4] == "distill"
        assert str(transcript) in argv
        assert "--session-id" in argv
        assert argv[argv.index("--session-id") + 1] == "cap-session"
        assert "--out" in argv
        assert "--project-path" in argv

        kwargs = captured["kwargs"]
        assert kwargs.get("start_new_session") is True
        assert kwargs.get("stdin") == pc.subprocess.DEVNULL
        # stdout/stderr must be redirected to a real file (a log under the
        # session export dir), never inherited/DEVNULL'd -- a distiller
        # crash must leave a trace (mirrors session_start.py's daemon-log
        # rationale).
        assert kwargs.get("stdout") is not None
        assert kwargs.get("stdout") != pc.subprocess.DEVNULL
        assert kwargs["stdout"] is kwargs["stderr"]

        session_dir = fake_home / ".claude" / "transcripts" / "cap-session"
        assert (session_dir / "distill.log").exists()

    def test_disabled_by_flag(self, tmp_path: pathlib.Path, monkeypatch) -> None:
        transcript = tmp_path / "t.jsonl"
        _write_transcript(transcript, 10)
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        _patch_cap_and_distill(monkeypatch, cap_mb=1e-6, distill_enabled=False)

        popen_mock = unittest.mock.MagicMock()
        monkeypatch.setattr(pc.subprocess, "Popen", popen_mock)

        _run_main(fake_home, "cap-session-off", transcript, tmp_path)
        popen_mock.assert_not_called()

    def test_skips_spawn_when_marker_already_matches(
        self, tmp_path: pathlib.Path, monkeypatch
    ) -> None:
        transcript = tmp_path / "t.jsonl"
        _write_transcript(transcript, 10)
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        _patch_cap_and_distill(monkeypatch, cap_mb=1e-6)

        session_dir = fake_home / ".claude" / "transcripts" / "cap-session-idem"
        session_dir.mkdir(parents=True)
        (session_dir / "distill-meta.json").write_text(
            json.dumps(
                {
                    "source_path": str(transcript),
                    "source_bytes": transcript.stat().st_size,
                }
            )
        )

        popen_mock = unittest.mock.MagicMock()
        monkeypatch.setattr(pc.subprocess, "Popen", popen_mock)

        _run_main(fake_home, "cap-session-idem", transcript, tmp_path)
        popen_mock.assert_not_called()

    def test_under_cap_never_spawns_distiller(
        self, tmp_path: pathlib.Path, monkeypatch
    ) -> None:
        transcript = tmp_path / "t.jsonl"
        _write_transcript(transcript, 3)
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        _patch_cap_and_distill(monkeypatch, cap_mb=256.0)  # comfortably under

        popen_mock = unittest.mock.MagicMock()
        monkeypatch.setattr(pc.subprocess, "Popen", popen_mock)

        _run_main(fake_home, "under-cap-session", transcript, tmp_path)
        popen_mock.assert_not_called()

    def test_never_parses_transcript_in_process(
        self, tmp_path: pathlib.Path, monkeypatch
    ) -> None:
        """The daemon code path must do size check + marker check + Popen
        only -- never actually distill. Proven by asserting
        ``distill.distill_transcript`` is never called from this process."""
        transcript = tmp_path / "t.jsonl"
        _write_transcript(transcript, 10)
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        _patch_cap_and_distill(monkeypatch, cap_mb=1e-6)

        monkeypatch.setattr(pc.subprocess, "Popen", unittest.mock.MagicMock())

        spy = unittest.mock.MagicMock(
            side_effect=AssertionError("must not parse in-process")
        )
        monkeypatch.setattr(distill, "distill_transcript", spy)

        _run_main(fake_home, "no-inprocess-parse-session", transcript, tmp_path)
        spy.assert_not_called()
