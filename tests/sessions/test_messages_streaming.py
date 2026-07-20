"""The sessions indexer must never slurp transcripts (2026-07-20 incident).

A post-#103 daemon ballooned to 32GB MALLOC_LARGE; `malloc_history` attributed
it to whole-file `read_text()` + split in `sessions/messages.py`, fed by
`sync.scheduler.run_index` chewing legacy multi-GB `transcript.jsonl` copies
in an executor thread. `_parse_jsonl` must stream line-by-line, and both
parsers must skip sources over ``sessions.max_parse_mb``.
"""

from __future__ import annotations

import json
import logging
import pathlib

import simba.config
import simba.sessions.messages as sm


def _write_jsonl(path: pathlib.Path, n: int = 3) -> None:
    with path.open("w") as fh:
        for i in range(n):
            fh.write(
                json.dumps(
                    {
                        "type": "message",
                        "payload": {"role": "user", "content": f"hello {i}"},
                    }
                )
                + "\n"
            )


def test_parse_jsonl_never_calls_read_text(tmp_path, monkeypatch) -> None:
    p = tmp_path / "t.jsonl"
    _write_jsonl(p)
    real_read_text = pathlib.Path.read_text

    def _boom(self, *a, **k):
        # Config loading legitimately read_text()s TOML files — only the
        # transcript itself must never be slurped.
        if self == p:
            raise AssertionError("slurp: read_text() called on the transcript")
        return real_read_text(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "read_text", _boom)
    msgs = sm._parse_jsonl(p, default_session_id="s1")
    assert isinstance(msgs, list)


def test_parse_jsonl_over_cap_skips_with_warning(tmp_path, monkeypatch, caplog) -> None:
    p = tmp_path / "t.jsonl"
    _write_jsonl(p, n=50)
    tiny = sm.SessionsConfig(max_parse_mb=0.000001)
    monkeypatch.setattr(simba.config, "load", lambda *a, **k: tiny)
    with caplog.at_level(logging.WARNING):
        msgs = sm._parse_jsonl(p, default_session_id="s1")
    assert msgs == []
    assert any("max_parse_mb" in r.message for r in caplog.records)


def test_parse_markdown_over_cap_skips_with_warning(
    tmp_path, monkeypatch, caplog
) -> None:
    p = tmp_path / "transcript.md"
    p.write_text("<user>\nhello\n</user>\n" * 200)
    tiny = sm.SessionsConfig(max_parse_mb=0.000001)
    monkeypatch.setattr(simba.config, "load", lambda *a, **k: tiny)
    with caplog.at_level(logging.WARNING):
        msgs = sm._parse_markdown(p)
    assert msgs == []
    assert any("max_parse_mb" in r.message for r in caplog.records)


def test_sessions_config_has_max_parse_mb_default() -> None:
    assert sm.SessionsConfig().max_parse_mb == 384.0
