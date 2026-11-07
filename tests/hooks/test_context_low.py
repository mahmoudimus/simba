"""Tests for context-low detection: post-compaction tail, not cumulative size.

The transcript JSONL is append-only — it keeps growing after Claude Code compacts,
so total file size massively overcounts the *live* context once a session has
compacted. We measure bytes since the last `isCompactSummary` line instead, and
the threshold is calibrated for the (large) current context window + configurable.
"""

from __future__ import annotations

import json
import pathlib
import threading
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

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


def _patch(
    monkeypatch, tmp_path, threshold: int, *, pre_tool_tail_mb: float = 16.0
) -> None:
    monkeypatch.setattr(ptu, "_CONTEXT_LOW_FLAG", tmp_path / "flag.json")
    monkeypatch.setattr(
        ptu,
        "_hooks_cfg",
        lambda: types.SimpleNamespace(
            context_low_bytes=threshold, pre_tool_tail_mb=pre_tool_tail_mb
        ),
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


def test_interleaved_chats_keep_their_warning_state(tmp_path, monkeypatch) -> None:
    _patch(monkeypatch, tmp_path, threshold=100)
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    for transcript in (a, b):
        _write(transcript, pre=200, post=200, compacted=False)
    assert ptu._check_context_low(a) is not None
    assert ptu._check_context_low(b) is not None
    assert ptu._check_context_low(a) is None
    assert ptu._check_context_low(b) is None


def test_simultaneous_hooks_emit_only_one_notice(tmp_path, monkeypatch) -> None:
    _patch(monkeypatch, tmp_path, threshold=100)
    transcript = tmp_path / "t.jsonl"
    _write(transcript, pre=200, post=200, compacted=False)
    workers = 8
    barrier = threading.Barrier(workers, timeout=10)
    legacy_flag = tmp_path / "flag.json"
    read_text = pathlib.Path.read_text

    def synchronized_legacy_read(path, *args, **kwargs):
        # Force the old read-then-write implementation to observe the missing
        # flag in every worker before any worker can write it. Exclusive-create
        # claims require no reads and therefore do not use this legacy path.
        if path == legacy_flag:
            try:
                return read_text(path, *args, **kwargs)
            finally:
                barrier.wait()
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", synchronized_legacy_read)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(ptu._check_context_low, [transcript] * workers))
    assert sum(result is not None for result in results) == 1


def test_unavailable_warning_storage_stays_quiet(tmp_path, monkeypatch) -> None:
    _patch(monkeypatch, tmp_path, threshold=100)
    monkeypatch.setattr(
        ptu, "_CONTEXT_LOW_FLAG", tmp_path / "missing-parent" / "flag.json"
    )
    transcript = tmp_path / "t.jsonl"
    _write(transcript, pre=200, post=200, compacted=False)
    assert ptu._check_context_low(transcript) is None
    assert ptu._check_context_low(transcript) is None


@pytest.mark.parametrize("threshold", [0, -1])
def test_nonpositive_threshold_disables_notices(tmp_path, monkeypatch, threshold):
    _patch(monkeypatch, tmp_path, threshold=threshold)
    transcript = tmp_path / "t.jsonl"
    _write(transcript, pre=200, post=200, compacted=False)
    assert ptu._check_context_low(transcript) is None


def test_path_aliases_share_warning_state(tmp_path, monkeypatch) -> None:
    _patch(monkeypatch, tmp_path, threshold=100)
    transcript = tmp_path / "t.jsonl"
    _write(transcript, pre=200, post=200, compacted=False)
    alias = tmp_path / "alias.jsonl"
    alias.symlink_to(transcript)
    assert ptu._check_context_low(transcript) is not None
    assert ptu._check_context_low(alias) is None


def test_marker_aging_out_of_scan_does_not_rearm(tmp_path, monkeypatch) -> None:
    _patch(monkeypatch, tmp_path, threshold=100, pre_tool_tail_mb=0.001)
    transcript = tmp_path / "t.jsonl"
    _write(transcript, pre=200, post=200, compacted=True)
    assert ptu._check_context_low(transcript) is not None
    with transcript.open("a") as stream:
        stream.write(json.dumps({"type": "assistant", "m": "z" * 2_000}) + "\n")
    assert ptu._check_context_low(transcript) is None


def test_unknown_boundary_stays_quiet(tmp_path, monkeypatch) -> None:
    _patch(monkeypatch, tmp_path, threshold=100, pre_tool_tail_mb=0.001)
    transcript = tmp_path / "t.jsonl"
    _write(transcript, pre=200, post=2_000, compacted=True)
    assert ptu._check_context_low(transcript) is None


def test_interleaved_scan_snapshots_emit_only_one_notice(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path, threshold=100, pre_tool_tail_mb=0.001)
    transcript = tmp_path / "t.jsonl"
    _write(transcript, pre=200, post=200, compacted=True)
    read_segment = ptu._post_compaction_tail_bytes
    notices = []

    def pause_detected_scan(path):
        snapshot = read_segment(path)
        # Schedule a second hook after the first read a known segment, but
        # before it claims the notice. Its scan can no longer see the marker.
        monkeypatch.setattr(ptu, "_post_compaction_tail_bytes", read_segment)
        with transcript.open("a") as stream:
            stream.write(json.dumps({"type": "assistant", "m": "z" * 2_000}) + "\n")
        notices.append(ptu._check_context_low(path))
        return snapshot

    monkeypatch.setattr(ptu, "_post_compaction_tail_bytes", pause_detected_scan)
    notices.append(ptu._check_context_low(transcript))
    assert sum(notice is not None for notice in notices) == 1


def test_growth_between_size_check_and_scan_stays_quiet(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path, threshold=100, pre_tool_tail_mb=0.001)
    transcript = tmp_path / "t.jsonl"
    _write(transcript, pre=200, post=200, compacted=True)
    read_segment = ptu._post_compaction_tail_bytes

    def grow_then_scan(path):
        with transcript.open("a") as stream:
            stream.write(json.dumps({"type": "assistant", "m": "z" * 2_000}) + "\n")
        return read_segment(path)

    monkeypatch.setattr(ptu, "_post_compaction_tail_bytes", grow_then_scan)
    assert ptu._check_context_low(transcript) is None


def test_notice_describes_a_heuristic_and_optional_compaction(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path, threshold=100)
    transcript = tmp_path / "t.jsonl"
    _write(transcript, pre=200, post=200, compacted=False)
    notice = ptu._check_context_low(transcript)
    assert notice is not None
    assert "heuristic" in notice
    assert "convenient stopping point" in notice
    assert "actual context usage is unknown" in notice
    assert "RECOMMENDED" not in notice


def _cap_bytes(mb: float) -> int:
    """Mirror the production mb->bytes conversion exactly, so test file
    layouts land at precise offsets regardless of float rounding."""
    return int(mb * 1_000_000)


class TestReadTailBytes:
    """Unit tests for the bounded-tail helper both inspectors share
    (2026-07-20 fix): whole-file reads on the every-tool-call PreToolUse hook
    were the recurring driver of multi-GB daemon RSS balloons.
    """

    def test_whole_file_when_under_cap(self, tmp_path: pathlib.Path) -> None:
        f = tmp_path / "t.jsonl"
        f.write_text("hello\nworld\n")
        tail, start = ptu._read_tail_bytes(f, 10_000)
        assert start == 0
        assert tail == b"hello\nworld\n"

    def test_discards_partial_leading_line(self, tmp_path: pathlib.Path) -> None:
        line1 = "a" * 100
        line2 = "b" * 50
        line3 = "c" * 50
        f = tmp_path / "t.jsonl"
        f.write_text(f"{line1}\n{line2}\n{line3}\n")
        total = f.stat().st_size
        seek_offset = 50  # lands inside line1's body, not on a line boundary
        tail, start = ptu._read_tail_bytes(f, total - seek_offset)
        # The partial tail of line1 is discarded; the first FULL line kept
        # begins right after line1's own newline.
        assert start == len(line1) + 1
        assert tail == f"{line2}\n{line3}\n".encode()

    def test_no_newline_in_window_yields_empty(self, tmp_path: pathlib.Path) -> None:
        # Pathological: the whole cap window is one giant partial line with no
        # newline at all -- nothing usable can be kept from it.
        f = tmp_path / "t.jsonl"
        f.write_text("z" * 1000)  # no trailing newline
        tail, start = ptu._read_tail_bytes(f, 200)
        assert tail == b""
        assert start == f.stat().st_size


class TestPostCompactionTailBounded:
    """``_post_compaction_tail_bytes`` is bounded to ``hooks.pre_tool_tail_mb``
    (2026-07-20): it used to ``read_bytes()`` the WHOLE transcript just to
    ``rfind`` the last compaction marker; its only caller already skips small
    files (the cheap path in ``_check_context_low``), so that whole-file read
    fired precisely and only for the giant files where it hurt most.
    """

    @staticmethod
    def _patch_cap(monkeypatch, mb: float) -> None:
        monkeypatch.setattr(
            ptu, "_hooks_cfg", lambda: types.SimpleNamespace(pre_tool_tail_mb=mb)
        )

    def test_marker_within_tail_window_matches_uncapped(
        self, tmp_path: pathlib.Path, monkeypatch
    ) -> None:
        pre = "x" * 2_000_000  # no marker here, just pushes the file over cap
        marker_line = json.dumps({"type": "user", "isCompactSummary": True, "m": "s"})
        post = "z" * 2_000
        f = tmp_path / "t.jsonl"
        f.write_text(f"{pre}\n{marker_line}\n{post}\n")
        total = f.stat().st_size

        cap_mb = 0.01  # 10_000 bytes: over the file's ~2MB, but still reaches
        # back past the marker line into the tail of `pre`.
        self._patch_cap(monkeypatch, cap_mb)
        assert total > _cap_bytes(cap_mb)  # sanity: file is genuinely over cap
        capped = ptu._post_compaction_tail_bytes(f)

        # Ground truth: a cap comfortably bigger than the whole file reduces to
        # the old unbounded full-file scan.
        self._patch_cap(monkeypatch, (total + 10_000_000) / 1_000_000)
        uncapped = ptu._post_compaction_tail_bytes(f)

        assert capped == uncapped
        expected_line_start = len(pre) + 1
        assert capped == (total - expected_line_start, expected_line_start)

    def test_marker_before_tail_window_has_unknown_boundary(
        self, tmp_path: pathlib.Path, monkeypatch
    ) -> None:
        # `lead` keeps the marker OFF the file's first line -- if it were on
        # line 1, an unbounded (old) scan would also report line_start == 0,
        # reason. With `lead` present, an unbounded scan reports a nonzero
        # line_start, distinguishing a known boundary from a clipped scan.
        lead = "a" * 500
        marker_line = json.dumps({"type": "user", "isCompactSummary": True, "m": "s"})
        post = "y" * 2_000_000  # marker is far outside the last cap_bytes
        f = tmp_path / "t.jsonl"
        f.write_text(f"{lead}\n{marker_line}\n{post}\n")
        total = f.stat().st_size

        cap_mb = 0.01  # 10_000 bytes -- nowhere near back to the marker
        self._patch_cap(monkeypatch, cap_mb)
        assert total > _cap_bytes(cap_mb)

        # A clipped scan cannot distinguish an old marker from a session that
        # never compacted. Keep that uncertainty explicit so it cannot rearm.
        assert ptu._post_compaction_tail_bytes(f) == (total, None)

    def test_no_marker_over_cap_returns_unknown_boundary(
        self, tmp_path: pathlib.Path, monkeypatch
    ) -> None:
        f = tmp_path / "t.jsonl"
        f.write_text("n" * 3_000_000 + "\n")
        total = f.stat().st_size
        cap_mb = 0.01
        self._patch_cap(monkeypatch, cap_mb)
        assert total > _cap_bytes(cap_mb)
        assert ptu._post_compaction_tail_bytes(f) == (total, None)

    def test_partial_line_discard_yields_correct_absolute_offset(
        self, tmp_path: pathlib.Path, monkeypatch
    ) -> None:
        cap_mb = 0.0005
        cap_bytes = _cap_bytes(cap_mb)  # 500

        pre = "x" * (cap_bytes + 50)  # tail seek point lands inside `pre`'s body
        marker_line = json.dumps({"type": "user", "isCompactSummary": True, "m": "s"})
        post = "y" * 20
        f = tmp_path / "t.jsonl"
        f.write_text(f"{pre}\n{marker_line}\n{post}\n")
        total = f.stat().st_size

        self._patch_cap(monkeypatch, cap_mb)
        result = ptu._post_compaction_tail_bytes(f)

        expected_line_start = len(pre) + 1
        assert result == (total - expected_line_start, expected_line_start)
