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


# -- setup ------------------------------------------------------------------


@pytest.fixture
def home(tmp_path) -> pathlib.Path:
    h = tmp_path / "home"
    (h / ".pi" / "agent").mkdir(parents=True)
    (h / ".pi" / "agent" / "settings.json").write_text(
        json.dumps({"packages": ["npm:pi-subagents", "npm:billion-context"]})
    )
    return h


@pytest.mark.bili_setup
def test_detects_agents_that_route_through_bili(home, monkeypatch) -> None:
    monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
    assert bili.detect_agents(home) == ["pi"]
    (home / ".codex").mkdir()
    (home / ".codex" / "config.toml").write_text('[mcp_servers.bili]\ncommand = "x"\n')
    (home / ".claude" / "plugins").mkdir(parents=True)
    (home / ".claude" / "settings.json").write_text(
        json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:8787/bili/x"}})
    )
    (home / ".claude" / "plugins" / "installed_plugins.json").write_text("{}")
    assert bili.detect_agents(home) == ["pi", "claude", "codex"]


@pytest.fixture
def setup_mod():
    import simba.integrations.billion_context_setup as setup

    return setup


@pytest.mark.bili_setup
def test_auto_setup_enables_rules_once_and_notifies(home, setup_mod, monkeypatch):
    monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
    monkeypatch.setattr(bili, "detect_agents", lambda h=None: ["pi"])
    config = bili.bili_config_path()
    config.write_text(json.dumps({"providers": {}, "compress": {"tiers": True}}))
    result = setup_mod.apply(auto=True)
    assert result["changes"] == ["compress.rules"]
    assert "Detected Billion Context (pi)" in result["notice"]
    assert "run `simba bili restore` to restore settings" in result["notice"]
    assert "--summary-key-from-pi" in result["notice"]
    data = json.loads(config.read_text())
    assert data == {"providers": {}, "compress": {"tiers": True, "rules": True}}
    assert json.loads(pathlib.Path(result["backup"]).read_text())["providers"] == {}
    # Never again automatically, even if the user turns it off.
    data["compress"]["rules"] = False
    config.write_text(json.dumps(data))
    again = setup_mod.apply(auto=True)
    assert again["status"] == "already handled" and again["notice"] == ""
    assert json.loads(config.read_text())["compress"]["rules"] is False


@pytest.mark.bili_setup
def test_restore_reverts_exactly_simbas_keys(home, setup_mod, monkeypatch):
    monkeypatch.setattr(bili, "detect_agents", lambda h=None: ["pi"])
    config = bili.bili_config_path()
    config.write_text(json.dumps({"providers": {}, "compress": {"rules": False}}))
    setup_mod.apply()
    data = json.loads(config.read_text())
    data["compress"]["tiers"] = True  # a later change of the user's
    config.write_text(json.dumps(data))
    result = setup_mod.restore()
    assert result["status"] == "restored compress.rules"
    assert json.loads(config.read_text()) == {
        "providers": {},
        "compress": {"rules": False, "tiers": True},
    }
    assert setup_mod.apply(auto=True)["status"] == "already handled"
    assert setup_mod.restore()["status"] == "nothing to restore"


@pytest.mark.bili_setup
def test_summary_key_from_pi_adds_the_chain_and_restore_removes_it(
    home, setup_mod, monkeypatch
):
    import os
    import stat

    monkeypatch.setattr(bili, "detect_agents", lambda h=None: ["pi"])
    setup_mod.pi_auth_path().write_text(
        json.dumps({"deepseek": {"type": "api_key", "key": "sk-test-123"}})
    )
    config = bili.bili_config_path()
    config.write_text(json.dumps({"providers": {}}))
    result = setup_mod.apply(summary_key_from_pi=True)
    assert result["credential"] == "copied"
    data = json.loads(config.read_text())
    assert data["compress"]["externalSummary"] == setup_mod.SUMMARY_CHAIN
    assert data["providers"]["simba-deepseek"]["credentialRef"] == "simba-deepseek"
    store = setup_mod.credentials_path(config)
    assert json.loads(store.read_text()) == {"simba-deepseek": "sk-test-123"}
    assert stat.S_IMODE(os.stat(store).st_mode) == 0o600
    setup_mod.restore()
    assert json.loads(config.read_text()) == {"providers": {}, "compress": {}}
    assert json.loads(store.read_text()) == {}


@pytest.mark.bili_setup
def test_setup_dry_run_and_guards(home, setup_mod, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bili, "detect_agents", lambda h=None: ["pi"])
    config = bili.bili_config_path()
    config.write_text('{"providers": {}, // a comment\n}')
    assert "not plain JSON" in setup_mod.apply()["status"]
    assert "// a comment" in config.read_text()
    config.write_text("{}")
    assert setup_mod.apply(dry_run=True)["status"] == "would change compress.rules"
    assert json.loads(config.read_text()) == {}
    monkeypatch.setattr(bili, "detect_agents", lambda h=None: [])
    assert "not used" in setup_mod.apply()["status"]


def test_legacy_marker_is_restorable(setup_mod) -> None:
    setup_mod.state_path().parent.mkdir(parents=True, exist_ok=True)
    setup_mod.state_path().write_text(json.dumps({"rules_enabled_at": 1.0}))
    bili.bili_config_path().write_text(json.dumps({"compress": {"rules": True}}))
    assert setup_mod.restore()["status"] == "restored compress.rules"
    assert json.loads(bili.bili_config_path().read_text()) == {"compress": {}}


# -- summaries ----------------------------------------------------------------


def _summary_db(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE blocks (session_id TEXT, block_id TEXT, ord INT, active INT,"
        " summary_hash TEXT, payload_json TEXT)"
    )
    conn.execute("CREATE TABLE content (hash TEXT, bytes BLOB)")
    conn.execute(
        "CREATE TABLE conversations (namespace TEXT, conversation_id TEXT,"
        " session_id TEXT)"
    )
    rows = [
        (SID, "b1", 1, 1, None, json.dumps({"topic": "ports", "summary": "port 8741"})),
        (SID, "b2", 2, 1, "h2", json.dumps({"topic": "uv"})),
        (SID, "b3", 3, 0, None, json.dumps({"summary": "inactive"})),
    ]
    conn.executemany("INSERT INTO blocks VALUES (?, ?, ?, ?, ?, ?)", rows)
    conn.execute(
        "INSERT INTO content VALUES ('h2', ?)",
        (json.dumps({"text": "use uv run", "count": 3}).encode(),),
    )
    conn.commit()
    conn.close()


def test_summary_rows_read_active_blocks_with_content_fallback() -> None:
    _summary_db(bili.bili_db_path())
    rows = list(bili.summary_rows(SID, set(), 1000))
    assert rows == [("b1", "ports", "port 8741"), ("b2", "uv", "use uv run")]
    assert list(bili.summary_rows(SID, {"b1"}, 1000)) == [("b2", "uv", "use uv run")]
    assert next(iter(bili.summary_rows(SID, set(), 4)))[2] == "port"


def test_summary_chunks_pack_and_record_ids() -> None:
    rows = [("b1", "t", "x" * 50), ("b2", "t", "y" * 50), ("b3", "t", "z" * 50)]
    seen: list[str] = []
    chunks = list(bili.summary_chunks(rows, 200, seen))
    assert seen == ["b1", "b2", "b3"]
    assert all(len(c) <= 200 for c in chunks) and len(chunks) == 2
    assert 'block="b1"' in chunks[0]


def test_extracted_blocks_accumulate() -> None:
    bili.mark_extracted_blocks(SID, ["b1"])
    bili.mark_extracted_blocks(SID, ["b2"])
    assert bili.extracted_blocks(SID) == {"b1", "b2"}


@pytest.mark.background_extraction
def test_fold_and_first_sighting_queue_summary_extraction(calls, monkeypatch):
    queued: list[str] = []
    monkeypatch.setattr(bili, "_queue_summary_extraction", queued.append)
    db = bili.bili_db_path()
    _db(db, [(SID, "b1", 1)])
    hook = {"session_id": SID, "transcript_path": "", "cwd": "/p", "source": "pi"}
    bili.on_turn_end(hook)
    assert queued == [SID]  # first sighting covers earlier folds
    bili.on_turn_end(hook)
    assert queued == [SID]
    db.unlink()
    _db(db, [(SID, "b1", 1), (SID, "b2", 1)])
    bili.on_turn_end(hook)
    assert queued == [SID, SID]
    assert bili.session_info(SID) == {"source": "pi", "cwd": "/p"}


def test_session_start_never_touches_the_real_config() -> None:
    """The autouse fixture stubs agent detection, so setup is a no-op here."""
    import simba.hooks.session_start as session_start

    result = session_start.run({"session_id": "s", "cwd": ""})
    assert not bili.bili_config_path().exists()
    assert result.system_message == ""


def test_session_start_surfaces_the_setup_notice(monkeypatch) -> None:
    import simba.hooks.session_start as session_start
    import simba.integrations.billion_context_setup as setup

    monkeypatch.setattr(setup, "auto_setup", lambda: "Detected Billion Context")
    result = session_start.run({"session_id": "s", "cwd": ""})
    assert result.system_message == "Detected Billion Context"
