"""Tests for the shared bounded-tail reader (``simba.hooks._tail``).

Promoted (2026-07-20) from ``simba.hooks.pre_tool_use._read_tail_bytes`` so
``tailor.hook`` and ``usage_signals`` can share the same primitive instead of
each carrying its own whole-file-read copy. These mirror the semantics tests
already covering the old private location (``tests/hooks/test_context_low.py
::TestReadTailBytes``) but exercise the new shared module directly, and check
that ``pre_tool_use`` still exposes the old name as a working alias.
"""

from __future__ import annotations

import pathlib

import simba.hooks._tail as tail
import simba.hooks.pre_tool_use as ptu


class TestReadTailBytes:
    def test_whole_file_when_under_cap(self, tmp_path: pathlib.Path) -> None:
        f = tmp_path / "t.jsonl"
        f.write_text("hello\nworld\n")
        data, start = tail.read_tail_bytes(f, 10_000)
        assert start == 0
        assert data == b"hello\nworld\n"

    def test_discards_partial_leading_line(self, tmp_path: pathlib.Path) -> None:
        line1 = "a" * 100
        line2 = "b" * 50
        line3 = "c" * 50
        f = tmp_path / "t.jsonl"
        f.write_text(f"{line1}\n{line2}\n{line3}\n")
        total = f.stat().st_size
        seek_offset = 50  # lands inside line1's body, not on a line boundary
        data, start = tail.read_tail_bytes(f, total - seek_offset)
        assert start == len(line1) + 1
        assert data == f"{line2}\n{line3}\n".encode()

    def test_no_newline_in_window_yields_empty(self, tmp_path: pathlib.Path) -> None:
        f = tmp_path / "t.jsonl"
        f.write_text("z" * 1000)  # no trailing newline
        data, start = tail.read_tail_bytes(f, 200)
        assert data == b""
        assert start == f.stat().st_size

    def test_cap_zero_is_uncapped(self, tmp_path: pathlib.Path) -> None:
        f = tmp_path / "t.jsonl"
        content = "x" * 500 + "\n"
        f.write_text(content)
        data, start = tail.read_tail_bytes(f, 0)
        assert start == 0
        assert data == content.encode()


def test_pre_tool_use_alias_is_the_shared_function() -> None:
    """``pre_tool_use._read_tail_bytes`` must keep working — existing tests
    and call sites reference it under that name."""
    assert ptu._read_tail_bytes is tail.read_tail_bytes
