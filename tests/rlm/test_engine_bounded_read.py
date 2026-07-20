"""`_load_transcript_text` must never slurp an over-cap transcript (2026-07-20)."""

from __future__ import annotations

import pathlib

import simba.config
import simba.rlm.config as rlm_config
import simba.rlm.engine as engine


def _fake_home(tmp_path, monkeypatch, tid: str, content: str) -> pathlib.Path:
    d = tmp_path / ".claude" / "transcripts" / tid
    d.mkdir(parents=True)
    (d / "transcript.md").write_text(content)
    monkeypatch.setattr(pathlib.Path, "home", lambda: tmp_path)
    return d


def test_under_cap_reads_whole_file(tmp_path, monkeypatch) -> None:
    _fake_home(tmp_path, monkeypatch, "t1", "line1\nline2\n")
    assert engine._load_transcript_text("t1") == "line1\nline2\n"


def test_over_cap_reads_bounded_tail(tmp_path, monkeypatch) -> None:
    body = "".join(f"line{i:06d}\n" for i in range(200_000))  # ~2.2MB
    _fake_home(tmp_path, monkeypatch, "t2", body)
    tiny = rlm_config.RlmConfig(digest_max_transcript_mb=0.5)
    monkeypatch.setattr(simba.config, "load", lambda *a, **k: tiny)
    text = engine._load_transcript_text("t2")
    assert 0 < len(text) <= int(0.5 * 1024 * 1024)
    assert text.endswith("line199999\n")  # the TAIL survives
    assert not text.startswith("line000000")  # the head is dropped
    assert text.startswith("line")  # partial first line was discarded


def test_config_default() -> None:
    assert rlm_config.RlmConfig().digest_max_transcript_mb == 24.0
