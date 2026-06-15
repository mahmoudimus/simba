from __future__ import annotations

import pytest

import simba.harness.core as core


def test_canonical_result_defaults():
    r = core.CanonicalResult()
    assert r.additional_context == ""
    assert r.suppress_output is False
    assert r.block_reason is None


def test_dispatch_unknown_event_raises():
    with pytest.raises(KeyError):
        core.dispatch("not_an_event", {})


def test_dispatch_prompt_submit_returns_canonical(monkeypatch):
    # No daemon needed: recall failure returns empty, run() still produces a result.
    r = core.dispatch("prompt_submit", {"prompt": "", "cwd": "/tmp"})
    assert isinstance(r, core.CanonicalResult)
