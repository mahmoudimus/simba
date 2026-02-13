"""Tests for the auto-learning tool rules system.

Tests PostToolUse failure detection, PreToolUse rule checking,
memory client filters, and the rules CLI.
"""

from __future__ import annotations

import json
import pathlib

import simba.hooks.post_tool_use as post_hook
import simba.hooks.pre_tool_use as pre_hook


# ---------- PostToolUse: error detection ----------


class TestErrorPatternDetection:
    def test_import_error_detected(self):
        assert post_hook._has_error_pattern(
            "ModuleNotFoundError: No module named 'ida_hexrays'"
        )

    def test_file_not_found_detected(self):
        assert post_hook._has_error_pattern(
            "FileNotFoundError: [Errno 2] No such file or directory: '/foo'"
        )

    def test_permission_denied_detected(self):
        assert post_hook._has_error_pattern("Permission denied")

    def test_command_not_found_detected(self):
        assert post_hook._has_error_pattern("bash: foo: command not found")

    def test_syntax_error_detected(self):
        assert post_hook._has_error_pattern(
            "SyntaxError: invalid syntax"
        )

    def test_clean_output_not_detected(self):
        assert not post_hook._has_error_pattern("Hello world\nDone.")

    def test_empty_string_not_detected(self):
        assert not post_hook._has_error_pattern("")


class TestExtractErrorLine:
    def test_extracts_first_error_line(self):
        output = "Loading...\nImportError: No module named 'foo'\nDone"
        result = post_hook._extract_error_line(output)
        assert "ImportError" in result

    def test_fallback_to_last_line(self):
        result = post_hook._extract_error_line("line1\nline2\nlast line")
        assert result == "last line"

    def test_empty_string(self):
        assert post_hook._extract_error_line("") == ""

    def test_truncates_long_lines(self):
        long_error = "ImportError: " + "x" * 300
        result = post_hook._extract_error_line(long_error)
        assert len(result) <= 200


class TestNormalizeCommand:
    def test_normalizes_absolute_paths(self):
        cmd = "python3 /Users/mahmoud/src/test.py"
        result = post_hook._normalize_command(cmd)
        assert "/Users/mahmoud" not in result
        assert "/PATH/" in result

    def test_normalizes_home_paths(self):
        cmd = "cat /home/user/file.txt"
        result = post_hook._normalize_command(cmd)
        assert "/home/user" not in result

    def test_normalizes_hex_addresses(self):
        cmd = "at 0x7fff5fbff8c0"
        result = post_hook._normalize_command(cmd)
        assert "0xADDR" in result

    def test_normalizes_line_col(self):
        cmd = "error at file.py:42:10"
        result = post_hook._normalize_command(cmd)
        assert ":LINE:COL" in result

    def test_normalizes_uuids(self):
        cmd = "session 550e8400-e29b-41d4-a716-446655440000"
        result = post_hook._normalize_command(cmd)
        assert "UUID" in result

    def test_preserves_simple_commands(self):
        cmd = "pytest tests/ -v"
        result = post_hook._normalize_command(cmd)
        assert result == cmd


class TestDetectFailure:
    def test_bash_import_error(self):
        result = post_hook._detect_failure(
            "Bash",
            {"command": "python3 -c 'from d810 import foo'"},
            {"output": "ModuleNotFoundError: No module named 'ida_hexrays'"},
        )
        assert result is not None
        assert result["tool"] == "Bash"
        assert "python3 -c" in result["command"]
        assert "ModuleNotFoundError" in result["error"]

    def test_bash_success_no_failure(self):
        result = post_hook._detect_failure(
            "Bash",
            {"command": "echo hello"},
            {"output": "hello"},
        )
        assert result is None

    def test_non_bash_ignored(self):
        result = post_hook._detect_failure(
            "Read",
            {"file_path": "/nonexistent"},
            {"output": "FileNotFoundError"},
        )
        assert result is None

    def test_stderr_error_detected(self):
        result = post_hook._detect_failure(
            "Bash",
            {"command": "bad-cmd"},
            {"stderr": "command not found: bad-cmd"},
        )
        assert result is not None

    def test_empty_response(self):
        result = post_hook._detect_failure("Bash", {"command": "ls"}, {})
        assert result is None


class TestRuleDedup:
    def test_dedup_round_trip(self, tmp_path, monkeypatch):
        cache = tmp_path / "dedup.json"
        monkeypatch.setattr(post_hook, "_RULE_DEDUP_CACHE", cache)

        assert not post_hook._check_rule_dedup("abc123")
        post_hook._save_rule_dedup("abc123")
        assert post_hook._check_rule_dedup("abc123")

    def test_dedup_cache_missing(self, tmp_path, monkeypatch):
        cache = tmp_path / "nonexistent.json"
        monkeypatch.setattr(post_hook, "_RULE_DEDUP_CACHE", cache)
        assert not post_hook._check_rule_dedup("anything")


# ---------- PostToolUse: main hook ----------


class TestPostToolUseMain:
    def test_empty_tool_name(self):
        result = json.loads(post_hook.main({}))
        assert "hookSpecificOutput" in result

    def test_activity_tracking_bash(self, monkeypatch):
        logged = []
        monkeypatch.setattr(
            "simba.search.activity_tracker.log_activity",
            lambda cwd, name, detail: logged.append((name, detail)),
        )
        result = json.loads(
            post_hook.main({
                "tool_name": "Bash",
                "tool_input": {"command": "echo hello"},
                "cwd": "/tmp",
            })
        )
        assert result["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
        assert len(logged) == 1
        assert logged[0][0] == "Bash"


# ---------- PreToolUse: tool rule check ----------


class TestCheckToolRules:
    def test_returns_none_when_disabled(self, monkeypatch):
        class FakeCfg:
            rule_check_enabled = False
            rule_min_similarity = 0.6
        monkeypatch.setattr(pre_hook, "_hooks_cfg", lambda: FakeCfg())
        result = pre_hook._check_tool_rules("Bash", {"command": "ls"}, "/tmp")
        assert result is None

    def test_returns_none_for_unsupported_tool(self, monkeypatch):
        class FakeCfg:
            rule_check_enabled = True
            rule_min_similarity = 0.6
        monkeypatch.setattr(pre_hook, "_hooks_cfg", lambda: FakeCfg())
        result = pre_hook._check_tool_rules("Glob", {"pattern": "*.py"}, "/tmp")
        assert result is None

    def test_returns_none_when_no_memories(self, monkeypatch):
        class FakeCfg:
            rule_check_enabled = True
            rule_min_similarity = 0.6
        monkeypatch.setattr(pre_hook, "_hooks_cfg", lambda: FakeCfg())
        monkeypatch.setattr(
            "simba.hooks._memory_client.recall_memories",
            lambda *a, **kw: [],
        )
        result = pre_hook._check_tool_rules("Bash", {"command": "ls"}, "/tmp")
        assert result is None

    def test_returns_warning_when_rule_matches(self, monkeypatch):
        class FakeCfg:
            rule_check_enabled = True
            rule_min_similarity = 0.6
        monkeypatch.setattr(pre_hook, "_hooks_cfg", lambda: FakeCfg())
        monkeypatch.setattr(
            "simba.hooks._memory_client.recall_memories",
            lambda *a, **kw: [
                {
                    "content": "Bash: ImportError — avoid: python3 -c",
                    "context": json.dumps({
                        "tool": "Bash",
                        "correction": "Use pytest instead",
                    }),
                    "similarity": 0.85,
                }
            ],
        )
        result = pre_hook._check_tool_rules(
            "Bash", {"command": "python3 -c 'from d810 import x'"}, "/tmp"
        )
        assert result is not None
        assert "tool-rule-warning" in result
        assert "ImportError" in result
        assert "pytest instead" in result


# ---------- PreToolUse: truth constraints ----------


class TestCheckTruthConstraints:
    def test_non_bash_returns_none(self):
        result = pre_hook._check_truth_constraints("Read", {"file_path": "x"})
        assert result is None

    def test_empty_command_returns_none(self):
        result = pre_hook._check_truth_constraints("Bash", {"command": ""})
        assert result is None

    def test_no_facts_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            "simba.hooks._truth_client.query_truth_db",
            lambda text: "",
        )
        result = pre_hook._check_truth_constraints("Bash", {"command": "ls"})
        assert result is None

    def test_facts_returned(self, monkeypatch):
        monkeypatch.setattr(
            "simba.hooks._truth_client.query_truth_db",
            lambda text: "<proven-facts><fact>test</fact></proven-facts>",
        )
        result = pre_hook._check_truth_constraints(
            "Bash", {"command": "pytest d810"}
        )
        assert result is not None
        assert "proven-facts" in result


# ---------- PreToolUse: main hook ----------


class TestPreToolUseMain:
    def test_empty_input(self):
        result = json.loads(pre_hook.main({}))
        assert "hookSpecificOutput" in result

    def test_no_context_when_nothing_fires(self, monkeypatch):
        monkeypatch.setattr(
            "simba.hooks._memory_client.recall_memories",
            lambda *a, **kw: [],
        )
        monkeypatch.setattr(
            "simba.hooks._truth_client.query_truth_db",
            lambda text: "",
        )
        result = json.loads(pre_hook.main({
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
            "cwd": "/tmp",
        }))
        # No transcript path → no thinking recall, no rule/truth match
        assert result.get("hookSpecificOutput", {}).get("additionalContext") is None


# ---------- Memory client: filters ----------


class TestMemoryClientFilters:
    def test_filters_included_in_payload(self, monkeypatch):
        sent_payloads = []

        class FakeResp:
            status_code = 200
            def json(self):
                return {"memories": []}

        def fake_post(url, json=None, timeout=None):
            sent_payloads.append(json)
            return FakeResp()

        import simba.hooks._memory_client as mc
        monkeypatch.setattr("httpx.post", fake_post)
        monkeypatch.setattr(mc, "_cfg", type("C", (), {
            "daemon_host": "localhost",
            "daemon_port": 8741,
            "min_similarity": 0.35,
            "default_max_results": 3,
            "default_timeout": 2.0,
        })())

        mc.recall_memories(
            "test query",
            filters={"types": ["TOOL_RULE"]},
        )

        assert len(sent_payloads) == 1
        assert sent_payloads[0]["filters"] == {"types": ["TOOL_RULE"]}

    def test_no_filters_omitted(self, monkeypatch):
        sent_payloads = []

        class FakeResp:
            status_code = 200
            def json(self):
                return {"memories": []}

        def fake_post(url, json=None, timeout=None):
            sent_payloads.append(json)
            return FakeResp()

        import simba.hooks._memory_client as mc
        monkeypatch.setattr("httpx.post", fake_post)
        monkeypatch.setattr(mc, "_cfg", type("C", (), {
            "daemon_host": "localhost",
            "daemon_port": 8741,
            "min_similarity": 0.35,
            "default_max_results": 3,
            "default_timeout": 2.0,
        })())

        mc.recall_memories("test query")

        assert "filters" not in sent_payloads[0]


# ---------- Memory client: store ----------


class TestMemoryClientStore:
    def test_store_sends_correct_payload(self, monkeypatch):
        sent_payloads = []

        class FakeResp:
            status_code = 200
            def json(self):
                return {"status": "stored", "id": "mem_test"}

        def fake_post(url, json=None, timeout=None):
            sent_payloads.append((url, json))
            return FakeResp()

        import simba.hooks._memory_client as mc
        monkeypatch.setattr("httpx.post", fake_post)
        monkeypatch.setattr(mc, "_cfg", type("C", (), {
            "daemon_host": "localhost",
            "daemon_port": 8741,
            "default_timeout": 2.0,
        })())

        result = mc.store_memory(
            memory_type="TOOL_RULE",
            content="test rule",
            context='{"tool":"Bash"}',
            tags=["Bash"],
            project_path="/tmp/project",
        )

        assert result["status"] == "stored"
        url, payload = sent_payloads[0]
        assert "/store" in url
        assert payload["type"] == "TOOL_RULE"
        assert payload["content"] == "test rule"
        assert payload["tags"] == ["Bash"]
        assert payload["projectPath"] == "/tmp/project"
