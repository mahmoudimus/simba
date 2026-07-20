"""Streaming/idempotence/cap/single-flight tests for the PreCompact export
path.

2026-07-17 incident: ``read_text().strip().split("\\n")`` on 10-25MB
transcripts (several concurrent, re-exported on every re-compaction) decoded
the whole file into one string while holding the GIL, copied it again on
``.strip()``, then exploded it into tens of thousands of line strings —
several transient full-file copies per PreCompact firing, several firings in
flight at once — and the daemon's RSS ballooned to 13GB within 30-60s of
every boot. See ``src/simba/hooks/pre_compact.py``.
"""

from __future__ import annotations

import json
import logging
import pathlib
import threading
import unittest.mock

import simba.config
import simba.hooks.config
import simba.hooks.pre_compact as pc

_LOGGER_NAME = "simba.hooks.pre_compact"


def _write_transcript(
    path: pathlib.Path, n: int, *, session_id: str = "s1", cwd: str = "/proj"
) -> None:
    lines = [
        json.dumps(
            {
                "session_id": session_id,
                "cwd": cwd,
                "message": {"role": "user", "content": f"message {i}"},
            }
        )
        for i in range(n)
    ]
    path.write_text("\n".join(lines) + "\n")


def _run_main(
    fake_home: pathlib.Path,
    session_id: str,
    transcript: pathlib.Path,
    cwd: pathlib.Path,
) -> dict:
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


class TestConfigDefault:
    def test_pre_compact_max_transcript_mb_default(self) -> None:
        assert simba.hooks.config.HooksConfig().pre_compact_max_transcript_mb == 256.0


class TestStreamingByteIdentical:
    def test_streamed_export_matches_legacy_join(self, tmp_path):
        """The streaming writer must produce byte-identical markdown to the
        old slurp-based ``_parse_transcript_to_markdown`` for the same input.
        """
        transcript = tmp_path / "t.jsonl"
        _write_transcript(transcript, 5)
        lines = transcript.read_text().strip().split("\n")
        expected_md, expected_count = pc._parse_transcript_to_markdown(lines)

        dest_md = tmp_path / "transcript.md"
        session_id, cwd, msg_count = pc._stream_transcript_to_markdown(
            transcript, dest_md
        )

        assert dest_md.read_text() == expected_md
        assert msg_count == expected_count
        assert session_id == "s1"
        assert cwd == "/proj"

    def test_streamed_export_matches_legacy_join_with_thinking_and_empty(
        self, tmp_path
    ):
        """Cover assistant thinking blocks and an entry with no messages at
        all (empty transcript), which exercise the header-only formatting
        edge (no trailing separator drift between the two code paths).
        """
        transcript = tmp_path / "t2.jsonl"
        entries = [
            {
                "session_id": "s2",
                "cwd": "/proj2",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "hmm"},
                        {"type": "text", "text": "answer"},
                    ],
                },
            },
            {
                "session_id": "s2",
                "cwd": "/proj2",
                "message": {"role": "user", "content": "follow up"},
            },
        ]
        transcript.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
        lines = transcript.read_text().strip().split("\n")
        expected_md, expected_count = pc._parse_transcript_to_markdown(lines)

        dest_md = tmp_path / "transcript2.md"
        session_id, cwd, msg_count = pc._stream_transcript_to_markdown(
            transcript, dest_md
        )

        assert dest_md.read_text() == expected_md
        assert msg_count == expected_count == 2
        assert session_id == "s2"
        assert cwd == "/proj2"

    def test_streamed_export_empty_transcript_matches_legacy(self, tmp_path):
        transcript = tmp_path / "empty.jsonl"
        transcript.write_text("")
        expected_md, expected_count = pc._parse_transcript_to_markdown([])

        dest_md = tmp_path / "empty.md"
        session_id, cwd, msg_count = pc._stream_transcript_to_markdown(
            transcript, dest_md
        )

        assert dest_md.read_text() == expected_md
        assert msg_count == expected_count == 0
        assert session_id == ""
        assert cwd == ""


class TestIdempotentReexport:
    def test_unchanged_size_skips_reexport(self, tmp_path, caplog):
        transcript = tmp_path / "t.jsonl"
        _write_transcript(transcript, 3)
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        _run_main(fake_home, "idem-session", transcript, tmp_path)
        session_dir = fake_home / ".claude" / "transcripts" / "idem-session"
        first_content = (session_dir / "transcript.md").read_text()

        with (
            unittest.mock.patch.object(
                pc,
                "_stream_transcript_to_markdown",
                side_effect=AssertionError(
                    "should not re-export an unchanged transcript"
                ),
            ),
            caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME),
        ):
            result = _run_main(fake_home, "idem-session", transcript, tmp_path)

        assert result.get("suppressOutput") is True
        assert (session_dir / "transcript.md").read_text() == first_content
        assert any("skip" in rec.message.lower() for rec in caplog.records)

    def test_grown_transcript_reexports(self, tmp_path):
        transcript = tmp_path / "t.jsonl"
        _write_transcript(transcript, 2)
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        _run_main(fake_home, "grow-session", transcript, tmp_path)
        session_dir = fake_home / ".claude" / "transcripts" / "grow-session"

        _write_transcript(transcript, 5)  # transcript grew
        _run_main(fake_home, "grow-session", transcript, tmp_path)

        dest_jsonl = session_dir / "transcript.jsonl"
        assert dest_jsonl.stat().st_size == transcript.stat().st_size
        md_text = (session_dir / "transcript.md").read_text()
        assert md_text.count("<user>") == 5


class TestSizeCap:
    def _patch_cap(self, monkeypatch, cap_mb: float):
        real_load = simba.config.load

        def fake_load(section, *a, **k):
            if section == "hooks":
                return simba.hooks.config.HooksConfig(
                    pre_compact_max_transcript_mb=cap_mb
                )
            return real_load(section, *a, **k)

        monkeypatch.setattr(simba.config, "load", fake_load)

    def test_over_cap_skips_export_with_warning(self, tmp_path, caplog, monkeypatch):
        transcript = tmp_path / "t.jsonl"
        _write_transcript(transcript, 50)
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        self._patch_cap(monkeypatch, cap_mb=1e-6)  # ~1 byte cap
        # 2026-07-20: the over-cap path now ALSO spawns a detached distiller
        # (see test_pre_compact_distill.py for that behavior in isolation) --
        # this test is about the EXPORT skip/warning specifically, so stub
        # Popen out rather than let a real `simba transcript distill`
        # subprocess spawn during the unit suite.
        monkeypatch.setattr(pc.subprocess, "Popen", unittest.mock.MagicMock())

        with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
            result = _run_main(fake_home, "cap-session", transcript, tmp_path)

        assert result.get("suppressOutput") is True
        session_dir = fake_home / ".claude" / "transcripts" / "cap-session"
        assert not (session_dir / "transcript.md").exists()
        assert not (session_dir / "transcript.jsonl").exists()

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1
        msg = warnings[0].message
        assert "pre_compact_max_transcript_mb" in msg
        assert str(transcript) in msg

    def test_cap_zero_disables_check(self, tmp_path, monkeypatch):
        transcript = tmp_path / "t.jsonl"
        _write_transcript(transcript, 5)
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        self._patch_cap(monkeypatch, cap_mb=0)

        result = _run_main(fake_home, "cap-zero-session", transcript, tmp_path)

        assert result.get("suppressOutput") is True
        session_dir = fake_home / ".claude" / "transcripts" / "cap-zero-session"
        assert (session_dir / "transcript.md").exists()


class TestSingleFlight:
    def test_concurrent_same_session_only_exports_once(self, tmp_path):
        transcript = tmp_path / "t.jsonl"
        _write_transcript(transcript, 3)
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        enter_evt = threading.Event()
        release_evt = threading.Event()
        call_count = {"n": 0}
        real_stream = pc._stream_transcript_to_markdown

        def slow_stream(transcript_path, dest_md):
            call_count["n"] += 1
            enter_evt.set()
            release_evt.wait(timeout=5)
            return real_stream(transcript_path, dest_md)

        results: list[dict] = []
        errors: list[BaseException] = []

        def worker():
            # A bare append swallows any exception with the thread, turning a
            # real failure into an opaque `len(results) == 1` flake later.
            try:
                results.append(
                    _run_main(fake_home, "concurrent-session", transcript, tmp_path)
                )
            except BaseException as exc:
                errors.append(exc)

        with unittest.mock.patch.object(
            pc, "_stream_transcript_to_markdown", side_effect=slow_stream
        ):
            t1 = threading.Thread(target=worker)
            t1.start()
            assert enter_evt.wait(timeout=5), "thread1 never entered the export"

            t2 = threading.Thread(target=worker)
            t2.start()
            # Generous join: thread2 skips the export but still runs the rest
            # of the hook pipeline, which can take >5s on a loaded machine
            # (e.g. a parallel full-suite run) — the tight timeout was a flake.
            t2.join(timeout=30)
            assert not t2.is_alive(), "thread2 did not finish within 30s"

            # thread2 must have returned WITHOUT ever calling the streaming
            # function — the single-flight guard, not a race, explains this.
            assert call_count["n"] == 1

            release_evt.set()
            t1.join(timeout=30)
            assert not t1.is_alive(), "thread1 did not finish after release"

        assert errors == []
        assert len(results) == 2
        assert all(r.get("suppressOutput") is True for r in results)

        # Guard released in `finally` — no leaked in-flight state.
        assert "concurrent-session" not in pc._INFLIGHT_SESSIONS

        # A third call proceeds normally (doesn't deadlock / stay locked out).
        result3 = _run_main(fake_home, "concurrent-session", transcript, tmp_path)
        assert result3.get("suppressOutput") is True


class TestNoSlurp:
    def test_export_never_slurps_the_transcript(self, tmp_path, monkeypatch):
        """Memory-shape guard: prove the export path never calls
        ``Path.read_text()`` on the transcript file (the root cause of the
        2026-07-17 incident) — it must stream line-by-line instead.
        """
        transcript = tmp_path / "t.jsonl"
        _write_transcript(transcript, 20)
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        real_read_text = pathlib.Path.read_text

        def guarded_read_text(self, *a, **k):
            if self == transcript:
                raise AssertionError("slurp")
            return real_read_text(self, *a, **k)

        monkeypatch.setattr(pathlib.Path, "read_text", guarded_read_text)

        result = _run_main(fake_home, "no-slurp-session", transcript, tmp_path)

        assert result.get("suppressOutput") is True
        session_dir = fake_home / ".claude" / "transcripts" / "no-slurp-session"
        assert (session_dir / "transcript.md").exists()
        assert (session_dir / "transcript.jsonl").exists()
