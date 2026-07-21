"""Tests for /compact focus double duty (docs/plans -- compact relay follow-up):

1. PreCompact persists a non-empty ``custom_instructions`` (the ``/compact
   <text>`` focus Claude Code passes on PreCompact's stdin) into the
   session's exported ``metadata.json`` as ``compactFocus``, truncated to
   ~300 chars. Empty/auto-compact -> no key at all.
2. The over-cap detached distiller spawn forwards the same (truncated)
   focus as a ``--focus`` argv arg.
3. SessionStart(source="compact") reads the persisted focus for THIS
   session and uses it to rank the compact-relay arc block -- matching
   arcs first, a "Compaction focus: <text>" line prepended. No persisted
   focus -> today's pure-recency behavior, byte-identical (no focus line).
"""

from __future__ import annotations

import json
import pathlib
import unittest.mock

import pytest

import simba.config
import simba.db
import simba.hooks.config as hooks_config
import simba.hooks.pre_compact as pc
import simba.hooks.session_start as session_start
import simba.transcripts.arcs as arcs

# ── shared helpers ──────────────────────────────────────────────────────────


def _run_precompact(fake_home, session_id, transcript, cwd, **extra) -> dict:
    with unittest.mock.patch("pathlib.Path.home", return_value=fake_home):
        payload = {
            "session_id": session_id,
            "transcript_path": str(transcript),
            "cwd": str(cwd),
            **extra,
        }
        return json.loads(pc.main(payload))


def _write_transcript(path: pathlib.Path, n: int = 3) -> None:
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


def _arc(**overrides) -> dict:
    base = dict(
        session_source="sess-1",
        harness="claude-code",
        tool="Bash",
        signature="sig-abc",
        error_head="boom",
        failed_args_head="pytest -x",
        fix_args_head="pytest -k test_foo",
        resolved=True,
        repeat_count=1,
        project_path="/repo",
    )
    base.update(overrides)
    return base


@pytest.fixture
def _tmp_db(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)


# ── 1. PreCompact persists compactFocus in metadata.json ───────────────────


class TestFocusPersistence:
    def test_custom_instructions_persisted_as_compact_focus(
        self, tmp_path: pathlib.Path
    ) -> None:
        transcript = tmp_path / "t.jsonl"
        _write_transcript(transcript)
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        _run_precompact(
            fake_home,
            "focus-session",
            transcript,
            tmp_path,
            custom_instructions="focus on the daemon restart bug",
        )

        meta_path = (
            fake_home / ".claude" / "transcripts" / "focus-session" / "metadata.json"
        )
        meta = json.loads(meta_path.read_text())
        assert meta["compactFocus"] == "focus on the daemon restart bug"

    def test_empty_custom_instructions_no_focus_key(
        self, tmp_path: pathlib.Path
    ) -> None:
        transcript = tmp_path / "t.jsonl"
        _write_transcript(transcript)
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        _run_precompact(
            fake_home, "no-focus-session", transcript, tmp_path, custom_instructions=""
        )

        meta_path = (
            fake_home / ".claude" / "transcripts" / "no-focus-session" / "metadata.json"
        )
        meta = json.loads(meta_path.read_text())
        assert "compactFocus" not in meta

    def test_missing_custom_instructions_no_focus_key(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Auto-compact never sends custom_instructions at all."""
        transcript = tmp_path / "t.jsonl"
        _write_transcript(transcript)
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        _run_precompact(fake_home, "auto-session", transcript, tmp_path)

        meta_path = (
            fake_home / ".claude" / "transcripts" / "auto-session" / "metadata.json"
        )
        meta = json.loads(meta_path.read_text())
        assert "compactFocus" not in meta

    def test_custom_instructions_truncated_to_300_chars(
        self, tmp_path: pathlib.Path
    ) -> None:
        transcript = tmp_path / "t.jsonl"
        _write_transcript(transcript)
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        long_focus = "x" * 500

        _run_precompact(
            fake_home,
            "long-focus-session",
            transcript,
            tmp_path,
            custom_instructions=long_focus,
        )

        meta_path = (
            fake_home
            / ".claude"
            / "transcripts"
            / "long-focus-session"
            / "metadata.json"
        )
        meta = json.loads(meta_path.read_text())
        assert len(meta["compactFocus"]) == 300
        assert meta["compactFocus"] == "x" * 300

    def test_camelcase_custom_instructions_supported(
        self, tmp_path: pathlib.Path
    ) -> None:
        transcript = tmp_path / "t.jsonl"
        _write_transcript(transcript)
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        _run_precompact(
            fake_home,
            "camel-session",
            transcript,
            tmp_path,
            customInstructions="camelCase focus text",
        )

        meta_path = (
            fake_home / ".claude" / "transcripts" / "camel-session" / "metadata.json"
        )
        meta = json.loads(meta_path.read_text())
        assert meta["compactFocus"] == "camelCase focus text"


# ── 2. Over-cap detached distiller spawn forwards --focus ──────────────────


def _patch_cap_and_distill(monkeypatch, *, cap_mb: float):
    real_load = simba.config.load

    def fake_load(section, *a, **k):
        if section == "hooks":
            return hooks_config.HooksConfig(
                pre_compact_max_transcript_mb=cap_mb,
                pre_compact_distill_enabled=True,
            )
        return real_load(section, *a, **k)

    monkeypatch.setattr(simba.config, "load", fake_load)


class TestSpawnDistillerFocusArg:
    def test_focus_forwarded_to_argv_when_present(
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
            return unittest.mock.MagicMock()

        monkeypatch.setattr(pc.subprocess, "Popen", fake_popen)

        _run_precompact(
            fake_home,
            "cap-focus-session",
            transcript,
            tmp_path,
            custom_instructions="fix the flaky test",
        )

        argv = captured["argv"]
        assert "--focus" in argv
        assert argv[argv.index("--focus") + 1] == "fix the flaky test"

    def test_no_focus_omits_focus_arg(
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
            return unittest.mock.MagicMock()

        monkeypatch.setattr(pc.subprocess, "Popen", fake_popen)

        _run_precompact(fake_home, "cap-nofocus-session", transcript, tmp_path)

        assert "--focus" not in captured["argv"]


# ── 3. SessionStart(compact) reads the persisted focus + ranks arcs ────────


class TestReadCompactFocus:
    def test_reads_persisted_focus(self, tmp_path: pathlib.Path) -> None:
        transcripts_dir = tmp_path / "transcripts"
        session_dir = transcripts_dir / "s1"
        session_dir.mkdir(parents=True)
        (session_dir / "metadata.json").write_text(
            json.dumps({"session_id": "s1", "compactFocus": "daemon restart"})
        )
        focus = session_start._read_compact_focus("s1", transcripts_dir=transcripts_dir)
        assert focus == "daemon restart"

    def test_no_session_id_returns_empty(self, tmp_path: pathlib.Path) -> None:
        assert session_start._read_compact_focus("", transcripts_dir=tmp_path) == ""

    def test_missing_metadata_returns_empty(self, tmp_path: pathlib.Path) -> None:
        assert session_start._read_compact_focus("nope", transcripts_dir=tmp_path) == ""

    def test_missing_key_returns_empty(self, tmp_path: pathlib.Path) -> None:
        session_dir = tmp_path / "s2"
        session_dir.mkdir(parents=True)
        (session_dir / "metadata.json").write_text(json.dumps({"session_id": "s2"}))
        assert session_start._read_compact_focus("s2", transcripts_dir=tmp_path) == ""

    def test_corrupt_json_returns_empty(self, tmp_path: pathlib.Path) -> None:
        session_dir = tmp_path / "s3"
        session_dir.mkdir(parents=True)
        (session_dir / "metadata.json").write_text("{not json")
        assert session_start._read_compact_focus("s3", transcripts_dir=tmp_path) == ""


class TestCompactRelayArcBlockFocusRanking:
    pytestmark = pytest.mark.usefixtures("_tmp_db")

    def _seed_metadata(self, fake_home, session_id, focus_text) -> None:
        session_dir = fake_home / ".claude" / "transcripts" / session_id
        session_dir.mkdir(parents=True)
        (session_dir / "metadata.json").write_text(
            json.dumps({"session_id": session_id, "compactFocus": focus_text})
        )

    def test_focus_reorders_matching_arc_first_despite_recency(
        self, tmp_path: pathlib.Path
    ) -> None:
        # Newest-first recency order would put "sig-unrelated" ahead of
        # "sig-daemon-restart" -- the focus should override that.
        arcs.upsert_arc(
            **_arc(
                signature="sig-daemon-restart",
                fix_args_head="restart the daemon process",
                now="2026-07-01T00:00:00Z",
            )
        )
        arcs.upsert_arc(
            **_arc(
                signature="sig-unrelated",
                fix_args_head="bump the dependency version",
                now="2026-07-02T00:00:00Z",
            )
        )

        fake_home = tmp_path / "home"
        fake_home.mkdir()
        self._seed_metadata(fake_home, "rank-session", "daemon restart keeps failing")

        cfg = hooks_config.HooksConfig(compact_relay_arcs_k=5)
        with unittest.mock.patch("pathlib.Path.home", return_value=fake_home):
            block = session_start._compact_relay_arc_block(
                "/repo", cfg, session_id="rank-session"
            )

        assert "Compaction focus: daemon restart keeps failing" in block
        assert block.index("sig-daemon-restart") < block.index("sig-unrelated")

    def test_no_persisted_focus_is_pure_recency_and_no_focus_line(
        self, tmp_path: pathlib.Path
    ) -> None:
        arcs.upsert_arc(**_arc(signature="sig-old", now="2026-07-01T00:00:00Z"))
        arcs.upsert_arc(**_arc(signature="sig-new", now="2026-07-02T00:00:00Z"))
        fake_home = tmp_path / "home"
        fake_home.mkdir()

        cfg = hooks_config.HooksConfig(compact_relay_arcs_k=5)
        with unittest.mock.patch("pathlib.Path.home", return_value=fake_home):
            block = session_start._compact_relay_arc_block(
                "/repo", cfg, session_id="never-compacted-session"
            )

        assert "Compaction focus:" not in block
        assert block.index("sig-new") < block.index("sig-old")

    def test_no_session_id_matches_current_behavior(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Callers that don't pass session_id (back-compat) get exactly
        today's block -- no focus line, pure recency."""
        arcs.upsert_arc(**_arc(signature="sig-only"))
        cfg = hooks_config.HooksConfig(compact_relay_arcs_k=5)
        block = session_start._compact_relay_arc_block("/repo", cfg)
        assert "Compaction focus:" not in block
        assert "sig-only" in block

    def test_k_cap_still_respected_with_focus(self, tmp_path: pathlib.Path) -> None:
        for i in range(10):
            arcs.upsert_arc(
                **_arc(
                    signature=f"sig-match-{i}",
                    fix_args_head="daemon restart fix",
                    now=f"2026-07-01T00:00:{i:02d}Z",
                )
            )
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        self._seed_metadata(fake_home, "capped-session", "daemon restart")

        cfg = hooks_config.HooksConfig(compact_relay_arcs_k=3)
        with unittest.mock.patch("pathlib.Path.home", return_value=fake_home):
            block = session_start._compact_relay_arc_block(
                "/repo", cfg, session_id="capped-session"
            )
        arc_lines = [line for line in block.split("\n") if line.startswith("- ")]
        assert len(arc_lines) == 3

    def test_line_and_block_char_caps_respected_with_focus(
        self, tmp_path: pathlib.Path
    ) -> None:
        for i in range(20):
            arcs.upsert_arc(
                **_arc(
                    signature=f"sig-{i} " + "e" * 190,
                    fix_args_head="f" * 190,
                    now=f"2026-07-01T00:{i:02d}:00Z",
                )
            )
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        self._seed_metadata(fake_home, "caps-session", "sig-5 focus")

        cfg = hooks_config.HooksConfig(compact_relay_arcs_k=20)
        with unittest.mock.patch("pathlib.Path.home", return_value=fake_home):
            block = session_start._compact_relay_arc_block(
                "/repo", cfg, session_id="caps-session"
            )
        arc_lines = [line for line in block.split("\n") if line.startswith("- ")]
        for line in arc_lines:
            assert len(line) <= 210
        assert len(block) <= 1600


class TestSessionStartRunIntegrationFocus:
    pytestmark = pytest.mark.usefixtures("_tmp_db")

    def test_compact_source_with_persisted_focus_injects_focus_line(
        self, tmp_path: pathlib.Path
    ) -> None:
        arcs.upsert_arc(
            **_arc(
                signature="sig-daemon-restart",
                fix_args_head="restart daemon",
                project_path=str(tmp_path),
            )
        )
        fake_home = tmp_path / "home"
        session_dir = fake_home / ".claude" / "transcripts" / "int-session"
        session_dir.mkdir(parents=True)
        (session_dir / "metadata.json").write_text(
            json.dumps({"session_id": "int-session", "compactFocus": "daemon restart"})
        )

        with (
            unittest.mock.patch(
                "simba.hooks.session_start._check_health", return_value=None
            ),
            unittest.mock.patch("pathlib.Path.home", return_value=fake_home),
        ):
            result = session_start.run(
                {
                    "cwd": str(tmp_path),
                    "source": "compact",
                    "session_id": "int-session",
                }
            )

        assert "Compaction focus: daemon restart" in result.additional_context
