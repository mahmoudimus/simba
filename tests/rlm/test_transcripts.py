"""Tests for the lazy LRU transcript provider."""

from __future__ import annotations

import pathlib

import pytest

import simba.rlm.context as ctx
import simba.rlm.transcripts as tr


class _Cfg:
    transcript_source = "md"
    lru_documents = 2


def _make_transcript(root: pathlib.Path, sid: str, body: str) -> None:
    d = root / sid
    d.mkdir(parents=True)
    (d / "transcript.md").write_text(body)


def test_available_and_path(tmp_path):
    _make_transcript(tmp_path, "s1", "hello")
    p = tr.TranscriptProvider(_Cfg(), root=tmp_path)
    assert p.available("s1")
    assert not p.available("missing")
    assert p.path_for("s1").name == "transcript.md"
    assert p.path_for("missing") is None


def test_load_caches_into_store(tmp_path):
    _make_transcript(tmp_path, "s1", "hello world")
    p = tr.TranscriptProvider(_Cfg(), root=tmp_path)
    p.load("s1")
    assert p.store.has("s1")
    assert p.store.get("s1").text == "hello world"


def test_load_missing_raises(tmp_path):
    p = tr.TranscriptProvider(_Cfg(), root=tmp_path)
    with pytest.raises(ctx.DocumentNotFoundError):
        p.load("missing")


def test_lru_eviction(tmp_path):
    for sid in ("s1", "s2", "s3"):
        _make_transcript(tmp_path, sid, sid)
    p = tr.TranscriptProvider(_Cfg(), root=tmp_path)  # lru_documents=2
    p.load("s1")
    p.load("s2")
    p.load("s3")  # evicts s1 (least recently used)
    assert not p.store.has("s1")
    assert p.store.has("s2")
    assert p.store.has("s3")


class _CfgForcedLazy(_Cfg):
    max_document_mb = 0.0  # force every transcript into lazy/offset-index mode


def test_load_large_transcript_never_slurps(tmp_path, monkeypatch):
    """The real 2026-07-20 RSS incident: TranscriptProvider.load() used to
    read_text() the whole transcript before handing it to the DocumentStore.
    A huge transcript must never be fully materialized -- not even
    transiently -- on the way into the store."""
    body = "line\n" * 500
    _make_transcript(tmp_path, "big", body)

    def _boom(*a, **k):
        raise AssertionError("Path.read_text must not be called for a lazy transcript")

    monkeypatch.setattr(pathlib.Path, "read_text", _boom)

    p = tr.TranscriptProvider(_CfgForcedLazy(), root=tmp_path)
    p.load("big")
    doc = p.store.get("big")
    assert doc.lazy is True
    assert not hasattr(doc, "text")
    assert not hasattr(doc, "lines")
    # still fully readable
    assert doc.read_range(0, 4) == "line"


def test_load_small_transcript_keeps_fast_path(tmp_path):
    _make_transcript(tmp_path, "s1", "hello world")
    p = tr.TranscriptProvider(_Cfg(), root=tmp_path)  # default max_document_mb (64MB)
    p.load("s1")
    doc = p.store.get("s1")
    assert doc.lazy is False
    assert doc.text == "hello world"
