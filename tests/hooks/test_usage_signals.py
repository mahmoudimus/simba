"""Tests for spec 33 Phase 1 usage signals (hooks side).

Turn/session records live in session-scoped tempfiles (the guardian-flag
pattern); ``use`` = gate fire or citation overlap in the Stop response;
``noise`` = injected repeatedly and never used. All behind
``hooks.usage_signals_enabled`` (default off → byte-identical).
"""

from __future__ import annotations

import json
import time
import unittest.mock

import pytest

import simba.hooks.config as hooks_config
import simba.hooks.usage_signals as us


@pytest.fixture(autouse=True)
def _tmp_flags(tmp_path, monkeypatch):
    monkeypatch.setattr(us, "_TMP_DIR", tmp_path)


# ---------------------------------------------------------------------------
# Module unit tests
# ---------------------------------------------------------------------------


def test_record_turn_and_read_back() -> None:
    us.record_turn_injections(
        "s1", [{"id": "m1", "content": "Use the INTERR 50815 handler", "context": ""}]
    )
    turn = us.read_turn("s1")
    assert turn[0]["id"] == "m1"
    assert "50815" in turn[0]["terms"]


def test_record_accumulates_session_counts() -> None:
    mems = [{"id": "m1", "content": "zzz_qqq flag"}]
    us.record_turn_injections("s2", mems)
    us.record_turn_injections("s2", mems)
    assert us._read_session("s2")["counts"]["m1"] == 2


def test_distinctive_terms_skips_common_english() -> None:
    terms = us.distinctive_terms(
        "Fix the error in run_system_tests_docker.sh with docker"
    )
    assert "run_system_tests_docker.sh" in terms
    assert "the" not in terms
    assert "with" not in terms


def test_detect_used_requires_min_overlap() -> None:
    turn = [
        {"id": "m1", "terms": ["50815", "interr"]},
        {"id": "m2", "terms": ["zzz_qqq", "yyy_www"]},
    ]
    used = us.detect_used(
        "Fixed by handling INTERR 50815 explicitly", turn, min_overlap=2
    )
    assert used == ["m1"]


def test_detect_used_short_term_list_requires_all_terms() -> None:
    turn = [{"id": "m1", "terms": ["hodur"]}]
    assert us.detect_used("The Hodur pass ran clean", turn, min_overlap=2) == ["m1"]
    assert us.detect_used("Nothing relevant here", turn, min_overlap=2) == []


def test_detect_used_no_terms_is_never_used() -> None:
    assert us.detect_used("anything", [{"id": "m1", "terms": []}], min_overlap=2) == []


def test_detect_used_whole_token_only() -> None:
    turn = [{"id": "m1", "terms": ["50815"]}]
    assert us.detect_used("code 508150 differs", turn, min_overlap=1) == []


def test_sweep_noise_flags_twice_injected_unused_once() -> None:
    mems = [{"id": "m9", "content": "qqq_zzz marker"}]
    us.record_turn_injections("s3", mems)
    us.record_turn_injections("s3", mems)
    assert us.sweep_noise("s3", min_injects=2) == ["m9"]
    assert us.sweep_noise("s3", min_injects=2) == []  # noised exactly once


def test_sweep_noise_ignores_single_injection() -> None:
    us.record_turn_injections("s5", [{"id": "m5", "content": "qqq_zzz"}])
    assert us.sweep_noise("s5", min_injects=2) == []


def test_mark_used_prevents_noise() -> None:
    mems = [{"id": "m8", "content": "qqq_zzz marker"}]
    us.record_turn_injections("s4", mems)
    us.record_turn_injections("s4", mems)
    us.mark_used("s4", ["m8"])
    assert us.sweep_noise("s4", min_injects=2) == []


def test_reset_turn_clears_record() -> None:
    us.record_turn_injections("s6", [{"id": "m6", "content": "qqq_zzz"}])
    us.reset_turn("s6")
    assert us.read_turn("s6") == []


def test_empty_session_id_is_noop() -> None:
    us.record_turn_injections("", [{"id": "m1", "content": "x"}])
    assert us.read_turn("") == []
    assert us.sweep_noise("") == []


# ---------------------------------------------------------------------------
# Memory-client helpers
# ---------------------------------------------------------------------------


def test_post_feedback_posts_to_daemon(monkeypatch) -> None:
    import simba.hooks._memory_client as mc

    captured: dict = {}

    class _Resp:
        status_code = 200

    def fake_post(url, json=None, timeout=None, headers=None):
        captured["url"] = url
        captured["json"] = json
        return _Resp()

    monkeypatch.setattr(mc.httpx, "post", fake_post)
    assert mc.post_feedback("m1", "good", weight=0.3) is True
    assert captured["url"].endswith("/memory/m1/feedback")
    assert captured["json"] == {"signal": "good", "weight": 0.3}


def test_post_feedback_fails_soft(monkeypatch) -> None:
    import httpx

    import simba.hooks._memory_client as mc

    def fake_post(*a, **kw):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(mc.httpx, "post", fake_post)
    assert mc.post_feedback("m1", "good") is False


def test_ack_injected_posts_ids(monkeypatch) -> None:
    import simba.hooks._memory_client as mc

    captured: dict = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"acked": 2}

    def fake_post(url, json=None, timeout=None, headers=None):
        captured["url"] = url
        captured["json"] = json
        return _Resp()

    monkeypatch.setattr(mc.httpx, "post", fake_post)
    assert mc.ack_injected(["a", "b"]) == 2
    assert captured["url"].endswith("/recall/ack")
    assert captured["json"] == {"ids": ["a", "b"]}


def test_ack_injected_empty_skips_request(monkeypatch) -> None:
    import simba.hooks._memory_client as mc

    def fake_post(*a, **kw):  # pragma: no cover - must not be reached
        raise AssertionError("should not POST for empty ids")

    monkeypatch.setattr(mc.httpx, "post", fake_post)
    assert mc.ack_injected([]) == 0


# ---------------------------------------------------------------------------
# Hook wiring
# ---------------------------------------------------------------------------


def _fresh_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def test_user_prompt_submit_records_and_acks(tmp_path, monkeypatch) -> None:
    import simba.hooks.user_prompt_submit as ups

    cfg = hooks_config.HooksConfig(
        usage_signals_enabled=True, recall_ack_enabled=True, prompt_min_length=1
    )
    monkeypatch.setattr(ups, "_cfg", lambda: cfg)

    acked: dict = {}
    monkeypatch.setattr(
        "simba.hooks._memory_client.ack_injected",
        lambda ids, **kw: acked.setdefault("ids", list(ids)) and len(ids),
    )
    mems = [
        {
            "id": "m1",
            "type": "GOTCHA",
            "content": "INTERR 50815 fix",
            "similarity": 0.8,
        }
    ]
    with unittest.mock.patch(
        "simba.hooks._memory_client.recall_memories", return_value=mems
    ):
        ups.main(
            {
                "prompt": "a sufficiently long prompt",
                "cwd": str(tmp_path),
                "session_id": "sess-u",
            }
        )
    assert acked["ids"] == ["m1"]
    turn = us.read_turn("sess-u")
    assert turn and turn[0]["id"] == "m1"


def test_user_prompt_submit_signals_off_is_silent(tmp_path, monkeypatch) -> None:
    import simba.hooks.user_prompt_submit as ups

    cfg = hooks_config.HooksConfig(prompt_min_length=1)  # both levers default-off
    monkeypatch.setattr(ups, "_cfg", lambda: cfg)

    def _no_ack(*a, **kw):  # pragma: no cover - must not be reached
        raise AssertionError("ack must not fire when disabled")

    monkeypatch.setattr("simba.hooks._memory_client.ack_injected", _no_ack)
    mems = [{"id": "m1", "type": "GOTCHA", "content": "x", "similarity": 0.8}]
    with unittest.mock.patch(
        "simba.hooks._memory_client.recall_memories", return_value=mems
    ):
        ups.main(
            {
                "prompt": "a sufficiently long prompt",
                "cwd": str(tmp_path),
                "session_id": "sess-off",
            }
        )
    assert us.read_turn("sess-off") == []


def test_stop_posts_good_feedback_for_cited_memory(tmp_path, monkeypatch) -> None:
    import simba.hooks.stop as stop

    cfg = hooks_config.HooksConfig(usage_signals_enabled=True)
    monkeypatch.setattr(stop, "_hooks_cfg", lambda: cfg)

    posted: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "simba.hooks._memory_client.post_feedback",
        lambda mid, sig, **kw: posted.append((mid, sig)) or True,
    )
    us.record_turn_injections(
        "sess-s", [{"id": "m1", "content": "the INTERR 50815 fix"}]
    )
    stop.main(
        {
            "response": "Handled INTERR 50815 by widening the guard.",
            "cwd": str(tmp_path),
            "session_id": "sess-s",
        }
    )
    assert ("m1", "good") in posted
    assert us.read_turn("sess-s") == []  # turn record consumed


def test_stop_noise_sweep_posts_bad_after_repeat_unused(tmp_path, monkeypatch) -> None:
    import simba.hooks.stop as stop

    cfg = hooks_config.HooksConfig(usage_signals_enabled=True)
    monkeypatch.setattr(stop, "_hooks_cfg", lambda: cfg)

    posted: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "simba.hooks._memory_client.post_feedback",
        lambda mid, sig, **kw: posted.append((mid, sig)) or True,
    )
    mems = [{"id": "m2", "content": "qqq_zzz marker"}]
    for _ in range(2):
        us.record_turn_injections("sess-n", mems)
        stop.main(
            {
                "response": "unrelated answer text",
                "cwd": str(tmp_path),
                "session_id": "sess-n",
            }
        )
    assert ("m2", "bad") in posted
    assert ("m2", "good") not in posted


def test_pre_tool_use_gate_fire_posts_use_feedback(tmp_path, monkeypatch) -> None:
    import simba.hooks.pre_tool_use as ptu

    cfg = hooks_config.HooksConfig(usage_signals_enabled=True)
    monkeypatch.setattr(ptu, "_hooks_cfg", lambda: cfg)
    monkeypatch.setattr(ptu, "_check_truth_constraints", lambda *a, **kw: None)

    rule = {
        "id": "mem_rule",
        "type": "TOOL_RULE",
        "content": "never run raw pytest for system tests",
        "similarity": 0.9,
        "context": json.dumps({"correction": "use the docker runner"}),
        "createdAt": _fresh_iso(),
    }
    monkeypatch.setattr(ptu, "_recall_tool_rules", lambda *a, **kw: [rule])

    posted: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "simba.hooks._memory_client.post_feedback",
        lambda mid, sig, **kw: posted.append((mid, sig)) or True,
    )
    ptu.main(
        {
            "tool_name": "Bash",
            "tool_input": {"command": "pytest tests/system"},
            "cwd": str(tmp_path),
            "session_id": "sess-g",
        }
    )
    assert ("mem_rule", "good") in posted


def _iso_days_ago(days: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - days * 86400))


def test_rule_ttl_refresh_keeps_recently_used_rule(tmp_path, monkeypatch) -> None:
    """Spec 33 Phase 2: freshness = max(createdAt, lastUsedAt) when the
    refresh lever is on — a rule stays alive by firing, not by re-learning."""
    import simba.hooks.pre_tool_use as ptu

    cfg = hooks_config.HooksConfig(rule_ttl_refresh_enabled=True, rule_count_ttl=0)
    monkeypatch.setattr(ptu, "_hooks_cfg", lambda: cfg)
    rule = {
        "id": "r1",
        "type": "TOOL_RULE",
        "content": "x",
        "similarity": 0.9,
        "createdAt": _iso_days_ago(30),
        "lastUsedAt": _iso_days_ago(1),
    }
    monkeypatch.setattr(
        "simba.hooks._memory_client.recall_memories", lambda *a, **kw: [rule]
    )
    mems = ptu._recall_tool_rules("Bash", {"command": "x"}, str(tmp_path))
    assert [m["id"] for m in mems] == ["r1"]


def test_rule_ttl_without_refresh_drops_stale_created(tmp_path, monkeypatch) -> None:
    import simba.hooks.pre_tool_use as ptu

    cfg = hooks_config.HooksConfig(rule_count_ttl=0)  # refresh lever default-off
    monkeypatch.setattr(ptu, "_hooks_cfg", lambda: cfg)
    rule = {
        "id": "r1",
        "type": "TOOL_RULE",
        "content": "x",
        "similarity": 0.9,
        "createdAt": _iso_days_ago(30),
        "lastUsedAt": _iso_days_ago(1),
    }
    monkeypatch.setattr(
        "simba.hooks._memory_client.recall_memories", lambda *a, **kw: [rule]
    )
    assert ptu._recall_tool_rules("Bash", {"command": "x"}, str(tmp_path)) == []


def test_pre_tool_use_gate_fire_off_by_default(tmp_path, monkeypatch) -> None:
    import simba.hooks.pre_tool_use as ptu

    cfg = hooks_config.HooksConfig()
    monkeypatch.setattr(ptu, "_hooks_cfg", lambda: cfg)
    monkeypatch.setattr(ptu, "_check_truth_constraints", lambda *a, **kw: None)

    rule = {
        "id": "mem_rule",
        "type": "TOOL_RULE",
        "content": "never do X",
        "similarity": 0.9,
        "context": json.dumps({"correction": "do Y"}),
        "createdAt": _fresh_iso(),
    }
    monkeypatch.setattr(ptu, "_recall_tool_rules", lambda *a, **kw: [rule])

    def _no_feedback(*a, **kw):  # pragma: no cover - must not be reached
        raise AssertionError("feedback must not fire when disabled")

    monkeypatch.setattr("simba.hooks._memory_client.post_feedback", _no_feedback)
    ptu.main(
        {
            "tool_name": "Bash",
            "tool_input": {"command": "do X"},
            "cwd": str(tmp_path),
            "session_id": "sess-d",
        }
    )
