"""Tests for context-low detection: post-compaction tail, not cumulative size.

The transcript JSONL is append-only — it keeps growing after Claude Code compacts,
so total file size massively overcounts the *live* context once a session has
compacted. We measure bytes since the last `isCompactSummary` line instead, and
the threshold is calibrated for the (large) current context window + configurable.
"""

from __future__ import annotations

import json
import pathlib
import types

import simba.hooks.pre_tool_use as ptu


def _write(path: pathlib.Path, pre: int, post: int, *, compacted: bool) -> None:
    lines = [json.dumps({"type": "user", "m": "x" * pre})]
    if compacted:
        lines.append(json.dumps({"type": "user", "isCompactSummary": True, "m": "s"}))
    lines.append(json.dumps({"type": "assistant", "m": "y" * post}))
    path.write_text("\n".join(lines) + "\n")


def test_tail_is_small_after_compaction(tmp_path: pathlib.Path) -> None:
    f = tmp_path / "t.jsonl"
    _write(f, pre=100_000, post=500, compacted=True)
    tail, off = ptu._post_compaction_tail_bytes(f)
    assert off > 0  # found a compaction boundary
    assert tail < 5_000  # tail ≪ total (which is >100k)
    assert tail < f.stat().st_size


def test_tail_equals_total_when_never_compacted(tmp_path: pathlib.Path) -> None:
    f = tmp_path / "t.jsonl"
    _write(f, pre=1_000, post=1_000, compacted=False)
    tail, off = ptu._post_compaction_tail_bytes(f)
    assert off == 0 and tail == f.stat().st_size


def _patch(monkeypatch, tmp_path, threshold: int) -> None:
    monkeypatch.setattr(ptu, "_CONTEXT_LOW_FLAG", tmp_path / "flag.json")
    monkeypatch.setattr(
        ptu, "_hooks_cfg", lambda: types.SimpleNamespace(context_low_bytes=threshold)
    )


def test_no_warning_when_compaction_shrank_live_context(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    f = tmp_path / "t.jsonl"
    _write(f, pre=50_000, post=200, compacted=True)  # huge total, tiny tail
    _patch(monkeypatch, tmp_path, threshold=10_000)
    # total ≫ threshold, but post-compaction tail ≪ threshold -> no false alarm
    assert ptu._check_context_low(f) is None


def test_warns_when_live_tail_exceeds_threshold(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    f = tmp_path / "t.jsonl"
    _write(f, pre=1_000, post=50_000, compacted=True)  # big tail
    _patch(monkeypatch, tmp_path, threshold=10_000)
    out = ptu._check_context_low(f)
    assert out is not None and "context-low-warning" in out
    assert "since last compaction" in out


def test_rearms_after_a_new_compaction(tmp_path: pathlib.Path, monkeypatch) -> None:
    f = tmp_path / "t.jsonl"
    _write(f, pre=1_000, post=50_000, compacted=True)
    _patch(monkeypatch, tmp_path, threshold=10_000)
    assert ptu._check_context_low(f) is not None  # fires once
    assert ptu._check_context_low(f) is None  # same boundary -> silent
    # a NEW compaction (different offset) re-arms the warning
    _write(f, pre=80_000, post=50_000, compacted=True)
    assert ptu._check_context_low(f) is not None
