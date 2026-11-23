"""Tests for integrations.billion_context — the Billion Context adapter."""

from __future__ import annotations

import json
import pathlib
import sqlite3
import time

import pytest

import simba.integrations.billion_context as bili

SID = "01a1f0ef-d6f6-767e-b106-8f25c856f1e2"


def _db(path: pathlib.Path, rows: list[tuple[str, str, int]], convs=()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE blocks (session_id TEXT, block_id TEXT, active INT)")
    conn.execute(
        "CREATE TABLE conversations (namespace TEXT, conversation_id TEXT,"
        " session_id TEXT)"
    )
    conn.executemany("INSERT INTO blocks VALUES (?, ?, ?)", rows)
    conn.executemany("INSERT INTO conversations VALUES ('proxy', ?, ?)", convs)
    conn.commit()
    conn.close()


@pytest.fixture
def calls(monkeypatch):
    """Record CORE resets and export spawns instead of running them."""
    seen: dict[str, list] = {"reset": [], "export": []}
    import simba.guardian.signal_flag as signal_flag

    monkeypatch.setattr(signal_flag, "reset_signal", seen["reset"].append)
    monkeypatch.setattr(bili, "_spawn_export", seen["export"].append)
    return seen


# -- discovery --------------------------------------------------------------


def test_origin_prefers_payload_then_env_then_state_file(monkeypatch) -> None:
    assert bili.proxy_origin({}) is None
    state = bili.bili_state_dir()
    state.mkdir(parents=True)
    (state / "proxy-origin").write_text(
        json.dumps({"origin": "http://127.0.0.1:8787", "instanceId": "x", "pid": 1})
    )
    assert bili.proxy_origin({}) == "http://127.0.0.1:8787"
    (state / "proxy-origin").write_text("http://127.0.0.1:9000/\n")
    assert bili.proxy_origin({}) == "http://127.0.0.1:9000"
    monkeypatch.setenv("BILLION_CONTEXT_PROXY", "http://127.0.0.1:63437")
    assert bili.proxy_origin({}) == "http://127.0.0.1:63437"
    payload = {"bili_origin": "http://127.0.0.1:1234/"}
    assert bili.proxy_origin(payload) == "http://127.0.0.1:1234"
    assert bili.proxy_origin({"bili_origin": "not a url"}) == "http://127.0.0.1:63437"


# -- signatures -------------------------------------------------------------


def test_http_signature_lists_active_blocks_sorted() -> None:
    st = {
        "blocks": [
            {"id": "b3", "active": True},
            {"id": "b1"},
            {"id": "b2", "active": False},
        ]
    }
    assert bili.http_signature(st) == "b1,b3"
    assert bili.http_signature({"ok": True}) is None


def test_db_signature_reads_active_blocks_and_maps_conversations() -> None:
    _db(
        bili.bili_db_path(),
        [(SID, "b1", 1), (SID, "b2", 0), ("server-9", "b7", 1)],
        convs=[("claude-session", "server-9")],
    )
    assert bili.db_signature(SID) == "b1"
    assert bili.db_signature("claude-session") == "b7"
    assert bili.db_signature("unknown") is None


def test_db_signature_without_a_store_is_none() -> None:
    assert bili.db_signature(SID) is None


def test_codex_marker_count_reads_only_the_tail(tmp_path) -> None:
    rollout = tmp_path / "rollout.jsonl"
    early = b'{"text": "' + bili.CODEX_FOLD_MARKER + b'"}\n'
    rollout.write_bytes(early + b"x" * (bili.TAIL_BYTES + 10) + b"\n" + early * 2)
    assert bili.codex_marker_count(rollout) == 2
    assert bili.codex_marker_count(tmp_path / "missing") is None


# -- turn end ---------------------------------------------------------------


def test_no_bili_is_a_no_op(calls, tmp_path) -> None:
    result = bili.on_turn_end({"session_id": SID, "cwd": str(tmp_path)})
    assert result == {"bili": False}
    assert calls == {"reset": [], "export": []}
    assert not bili.is_active(SID)


def test_first_sighting_is_a_baseline_then_a_fold_triggers(calls) -> None:
    db = bili.bili_db_path()
    _db(db, [(SID, "b1", 1)])
    hook = {"session_id": SID, "transcript_path": "", "cwd": ""}
    assert bili.on_turn_end(hook)["folded"] is False
    assert bili.is_active(SID)
    assert bili.on_turn_end(hook)["folded"] is False  # unchanged
    db.unlink()
    _db(db, [(SID, "b1", 1), (SID, "b2", 1)])
    assert bili.on_turn_end(hook)["folded"] is True
    assert calls["reset"] == [SID]
    assert calls["export"][0]["session_id"] == SID


def test_http_status_is_preferred_over_the_store(calls, monkeypatch) -> None:
    blocks = [[{"id": "b1"}]]
    monkeypatch.setattr(bili, "status", lambda origin, cid: {"blocks": blocks[0]})
    monkeypatch.setattr(bili, "sync_rules", lambda *a: (0, "acp_rule disabled"))
    hook = {"session_id": SID, "bili_origin": "http://127.0.0.1:1"}
    bili.on_turn_end(hook)
    blocks[0] = [{"id": "b1"}, {"id": "b2"}]
    assert bili.on_turn_end(hook)["folded"] is True


def test_codex_compaction_marker_counts_as_a_fold(calls, tmp_path) -> None:
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_bytes(b"{}\n")
    hook = {"session_id": "codex-1", "transcript_path": str(rollout)}
    bili.on_turn_end(hook)
    with open(rollout, "ab") as handle:
        handle.write(b'{"text":"' + bili.CODEX_FOLD_MARKER + b'"}\n')
    assert bili.on_turn_end(hook)["folded"] is True
    assert calls["reset"] == ["codex-1"]


# -- CORE rules -------------------------------------------------------------


@pytest.fixture
def project(tmp_path) -> pathlib.Path:
    (tmp_path / "CLAUDE.md").write_text(
        "# Project\n\n<!-- BEGIN SIMBA:core -->\n"
        "- **Never** walk SMB network mounts such as /Volumes/public.\n"
        "- Use `>|` to overwrite files because zsh has noclobber set.\n"
        "- " + "long rule " * 60 + "\n"
        "<!-- END SIMBA:core -->\n"
    )
    return tmp_path


def test_core_rules_are_prefixed_and_within_bili_limits(project) -> None:
    rules = bili.core_rules(project)
    assert len(rules) == 3
    assert all(r.startswith(bili.RULE_PREFIX) for r in rules)
    assert all(len(r) <= bili.RULE_MAX_CHARS for r in rules)
    assert "Never walk SMB network mounts" in rules[0]


class FakeProxy:
    def __init__(self, tools=("compress", "acp_rule"), existing=""):
        self.tools, self.existing, self.added = list(tools), existing, []
        self.manifest_calls = 0

    def http(self, method, origin, path, *, params=None, body=None):
        if path.endswith("/manifest"):
            self.manifest_calls += 1
            return {"ok": True, "toolNames": self.tools}
        if path.endswith("/tool"):
            rule = body["args"].get("rule")
            if rule is None:
                return {"ok": True, "result": self.existing}
            self.added.append(rule)
            return {"ok": True, "outcome": "success", "result": f"Recorded {rule}"}
        if path.endswith("/status"):
            return {"ok": True, "blocks": []}
        return None


def test_sync_adds_only_missing_rules(monkeypatch) -> None:
    proxy = FakeProxy(existing="rule1: [simba] keep A")
    monkeypatch.setattr(bili, "_http", proxy.http)
    added, skipped = bili.sync_rules(
        "http://x", SID, ["[simba] keep A", "[simba] keep B"]
    )
    assert (added, skipped) == (1, None)
    assert proxy.added == ["[simba] keep B"]


def test_sync_skips_when_acp_rule_is_disabled(monkeypatch) -> None:
    proxy = FakeProxy(tools=("compress",))
    monkeypatch.setattr(bili, "_http", proxy.http)
    assert bili.sync_rules("http://x", SID, ["[simba] a"]) == (
        0,
        "acp_rule disabled (compress.rules is off)",
    )


def test_turn_end_syncs_rules_once_and_rechecks_disabled_hourly(
    calls, monkeypatch, project
) -> None:
    proxy = FakeProxy()
    monkeypatch.setattr(bili, "_http", proxy.http)
    hook = {"session_id": SID, "bili_origin": "http://x", "cwd": str(project)}
    assert bili.on_turn_end(hook)["rules_added"] == 3
    assert bili.on_turn_end(hook)["rules_added"] == 0  # digest unchanged
    assert proxy.manifest_calls == 1

    off = FakeProxy(tools=("compress",))
    monkeypatch.setattr(bili, "_http", off.http)
    other = {**hook, "session_id": "other"}
    bili.on_turn_end(other)
    bili.on_turn_end(other)
    assert off.manifest_calls == 1  # throttled for an hour
    state = bili._load("other")
    state["rules_checked_at"] = time.time() - 3700
    bili._save("other", state)
    bili.on_turn_end(other)
    assert off.manifest_calls == 2


# -- hooks ------------------------------------------------------------------


def test_context_low_notice_is_skipped_while_bili_manages(monkeypatch) -> None:
    import simba.hooks.pre_tool_use as ptu

    monkeypatch.setattr(bili, "is_active", lambda sid: sid == SID)
    assert ptu._bili_manages(SID) is True
    assert ptu._bili_manages("other") is False


def test_stop_hook_runs_the_bili_step(monkeypatch) -> None:
    import simba.hooks.stop as stop

    seen = []
    monkeypatch.setattr(bili, "on_turn_end", lambda hook: seen.append(hook))
    stop.run({"session_id": SID, "response": "", "cwd": ""})
    assert seen and seen[0]["session_id"] == SID


def test_http_and_store_signatures_match(monkeypatch, calls) -> None:
    """Falling back from HTTP to the store between turns is not a fold."""
    _db(bili.bili_db_path(), [(SID, "b1", 1), (SID, "b2", 1), (SID, "b3", 0)])
    http_st = {"blocks": [{"id": "b2"}, {"id": "b1"}, {"id": "b3", "active": False}]}
    assert bili.http_signature(http_st) == bili.db_signature(SID)
    reachable = [True]
    monkeypatch.setattr(bili, "status", lambda o, c: http_st if reachable[0] else None)
    monkeypatch.setattr(bili, "sync_rules", lambda *a: (0, "off"))
    hook = {"session_id": SID, "bili_origin": "http://x"}
    bili.on_turn_end(hook)
    reachable[0] = False
    assert bili.on_turn_end(hook)["folded"] is False
