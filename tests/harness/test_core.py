from __future__ import annotations

import pytest

import simba.harness.core as core


def test_canonical_result_defaults():
    r = core.CanonicalResult()
    assert r.additional_context == ""
    assert r.suppress_output is False
    assert r.memory_count == 0
    assert r.block_reason is None
    assert r.transform is None
    assert r.escalated_block is None


def test_dispatch_unknown_event_raises():
    with pytest.raises(KeyError):
        core.dispatch("not_an_event", {})


def test_dispatch_prompt_submit_returns_canonical():
    # No daemon needed: recall failure returns empty, run() still produces a result.
    r = core.dispatch("prompt_submit", {"prompt": "", "cwd": "/tmp"})
    assert isinstance(r, core.CanonicalResult)
    assert r.suppress_output is False
    assert r.block_reason is None


def test_dispatch_session_start_returns_canonical():
    r = core.dispatch("session_start", {"cwd": "/tmp"})
    assert isinstance(r, core.CanonicalResult)


def test_dispatch_stop_returns_canonical():
    r = core.dispatch("stop", {"cwd": "/tmp"})
    assert isinstance(r, core.CanonicalResult)


def test_dispatch_pre_compact_returns_canonical():
    r = core.dispatch("pre_compact", {})
    assert isinstance(r, core.CanonicalResult)


def test_dispatch_pre_tool_returns_canonical():
    # No transcript/rules → empty context path, no block/transform.
    r = core.dispatch(
        "pre_tool", {"tool_name": "Read", "tool_input": {}, "cwd": "/tmp"}
    )
    assert isinstance(r, core.CanonicalResult)
    assert r.block_reason is None
    assert r.transform is None
    assert r.escalated_block is None
