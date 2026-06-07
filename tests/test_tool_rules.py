"""Tests for the auto-learning tool rules system.

Tests PostToolUse failure detection, PreToolUse rule checking,
memory client filters, and the rules CLI.
"""

from __future__ import annotations

import json
import time

import pytest

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
        assert post_hook._has_error_pattern("SyntaxError: invalid syntax")

    def test_clean_output_not_detected(self):
        assert not post_hook._has_error_pattern("Hello world\nDone.")

    def test_empty_string_not_detected(self):
        assert not post_hook._has_error_pattern("")


class TestExtractErrorLine:
    def test_extracts_first_error_line(self):
        output = "Loading...\nImportError: No module named 'foo'\nDone"
        result = post_hook._extract_error_line(output)
        assert "ImportError" in result

    def test_no_error_line_returns_empty(self):
        # No line matches an error pattern -> nothing worth learning (previously
        # this fell back to the last line, which manufactured false-positive
        # rules from non-error output).
        assert post_hook._extract_error_line("line1\nline2\nlast line") == ""

    def test_skips_source_and_doc_mentions(self):
        # Lines that merely mention an error word but are source/comment/doc are
        # skipped; only a genuine error line is returned.
        text = "except ImportError:\n# handles ImportError\nImportError: boom"
        assert post_hook._extract_error_line(text) == "ImportError: boom"

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


class TestLeadingVerb:
    def test_simple_command(self):
        assert post_hook._leading_verb("ls -la /x") == "ls"

    def test_skips_env_assignment(self):
        assert post_hook._leading_verb("FOO=bar ls x") == "ls"

    def test_strips_path_prefix(self):
        assert post_hook._leading_verb("/usr/bin/find . -name x") == "find"

    def test_empty(self):
        assert post_hook._leading_verb("") == ""

    def test_only_env_assignment(self):
        assert post_hook._leading_verb("FOO=bar") == ""


class TestProbeNotFoundSkipping:
    _PROBES = frozenset({"ls", "bfs", "find", "stat"})

    def test_skips_ls_no_such_file(self):
        result = post_hook._detect_failure(
            "Bash",
            {"command": "ls src/d810/families/flow_automaton"},
            {
                "stderr": (
                    "ls: src/d810/families/flow_automaton: No such file or directory"
                )
            },
            skip_probe_not_found=True,
            probe_verbs=self._PROBES,
        )
        assert result is None

    def test_skips_bfs_no_such_file(self):
        result = post_hook._detect_failure(
            "Bash",
            {"command": "bfs src/d810/recon"},
            {"stderr": "bfs: 'src/d810/recon': No such file or directory"},
            skip_probe_not_found=True,
            probe_verbs=self._PROBES,
        )
        assert result is None

    def test_does_not_skip_when_disabled(self):
        result = post_hook._detect_failure(
            "Bash",
            {"command": "ls missing"},
            {"stderr": "ls: missing: No such file or directory"},
            skip_probe_not_found=False,
            probe_verbs=self._PROBES,
        )
        assert result is not None

    def test_does_not_skip_non_probe_command(self):
        # python failing with FileNotFoundError is a real mistake worth learning.
        result = post_hook._detect_failure(
            "Bash",
            {"command": "python3 build.py"},
            {
                "output": (
                    "FileNotFoundError: [Errno 2] No such file or "
                    "directory: 'config.yml'"
                )
            },
            skip_probe_not_found=True,
            probe_verbs=self._PROBES,
        )
        assert result is not None

    def test_does_not_skip_probe_with_other_error(self):
        # ls hitting a permission error IS worth learning.
        result = post_hook._detect_failure(
            "Bash",
            {"command": "ls /root/secret"},
            {"stderr": "ls: /root/secret: Permission denied"},
            skip_probe_not_found=True,
            probe_verbs=self._PROBES,
        )
        assert result is not None

    def test_skips_path_prefixed_probe(self):
        result = post_hook._detect_failure(
            "Bash",
            {"command": "/usr/bin/find . -name nope"},
            {"stderr": "find: './nope': No such file or directory"},
            skip_probe_not_found=True,
            probe_verbs=self._PROBES,
        )
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
            post_hook.main(
                {
                    "tool_name": "Bash",
                    "tool_input": {"command": "echo hello"},
                    "cwd": "/tmp",
                }
            )
        )
        assert result["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
        assert len(logged) == 1
        assert logged[0][0] == "Bash"


# ---------- PreToolUse: tool rule check ----------


class TestCheckToolRules:
    @pytest.fixture(autouse=True)
    def _bypass_project_gate(self, monkeypatch):
        # These tests exercise the rule-matching path; force the project gate True
        # so the ruleless-project short-circuit doesn't intercept them.
        monkeypatch.setattr(pre_hook, "_project_has_tool_rules", lambda *a, **k: True)

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
                    "context": json.dumps(
                        {
                            "tool": "Bash",
                            "correction": "Use pytest instead",
                        }
                    ),
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


class TestToolRuleRecencyGate:
    """Stale TOOL_RULE matches age out of the warning injection (gate B)."""

    @pytest.fixture(autouse=True)
    def _bypass_project_gate(self, monkeypatch):
        monkeypatch.setattr(pre_hook, "_project_has_tool_rules", lambda *a, **k: True)

    @staticmethod
    def _cfg(max_age_days):
        class FakeCfg:
            rule_check_enabled = True
            rule_min_similarity = 0.6
            rule_max_age_days = max_age_days

        return FakeCfg()

    @staticmethod
    def _iso(days_ago):
        return time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - days_ago * 86400)
        )

    def test_drops_stale_rule(self, monkeypatch):
        monkeypatch.setattr(pre_hook, "_hooks_cfg", lambda: self._cfg(14))
        monkeypatch.setattr(
            "simba.hooks._memory_client.recall_memories",
            lambda *a, **kw: [
                {
                    "content": "Bash: ls: No such file or directory",
                    "context": "{}",
                    "createdAt": self._iso(30),
                    "similarity": 0.8,
                }
            ],
        )
        result = pre_hook._check_tool_rules("Bash", {"command": "ls x"}, "/tmp")
        assert result is None

    def test_keeps_fresh_rule(self, monkeypatch):
        monkeypatch.setattr(pre_hook, "_hooks_cfg", lambda: self._cfg(14))
        monkeypatch.setattr(
            "simba.hooks._memory_client.recall_memories",
            lambda *a, **kw: [
                {
                    "content": "Bash: real recent rule",
                    "context": json.dumps({"correction": "do X"}),
                    "createdAt": self._iso(1),
                    "similarity": 0.8,
                }
            ],
        )
        result = pre_hook._check_tool_rules("Bash", {"command": "ls x"}, "/tmp")
        assert result is not None
        assert "real recent rule" in result

    def test_gate_disabled_keeps_ancient_rule(self, monkeypatch):
        monkeypatch.setattr(pre_hook, "_hooks_cfg", lambda: self._cfg(0))
        monkeypatch.setattr(
            "simba.hooks._memory_client.recall_memories",
            lambda *a, **kw: [
                {
                    "content": "Bash: ancient rule",
                    "context": json.dumps({"correction": "x"}),
                    "createdAt": self._iso(365),
                    "similarity": 0.8,
                }
            ],
        )
        result = pre_hook._check_tool_rules("Bash", {"command": "ls x"}, "/tmp")
        assert result is not None

    def test_missing_created_at_is_kept(self, monkeypatch):
        monkeypatch.setattr(pre_hook, "_hooks_cfg", lambda: self._cfg(14))
        monkeypatch.setattr(
            "simba.hooks._memory_client.recall_memories",
            lambda *a, **kw: [
                {
                    "content": "Bash: rule without timestamp",
                    "context": json.dumps({"correction": "x"}),
                    "similarity": 0.8,
                }
            ],
        )
        result = pre_hook._check_tool_rules("Bash", {"command": "ls x"}, "/tmp")
        assert result is not None


class TestWithinMaxAge:
    def test_recent_within_age(self):
        recent = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 86400))
        assert pre_hook._within_max_age(recent, 14) is True

    def test_old_outside_age(self):
        old = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 30 * 86400))
        assert pre_hook._within_max_age(old, 14) is False

    def test_missing_timestamp_kept(self):
        assert pre_hook._within_max_age(None, 14) is True

    def test_unparseable_timestamp_kept(self):
        assert pre_hook._within_max_age("not-a-date", 14) is True


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
            "simba.hooks._kg_client.query_kg",
            lambda *a, **k: "",
        )
        result = pre_hook._check_truth_constraints("Bash", {"command": "ls"})
        assert result is None

    def test_facts_returned(self, monkeypatch):
        monkeypatch.setattr(
            "simba.hooks._kg_client.query_kg",
            lambda *a, **k: "<kg-facts><fact>test</fact></kg-facts>",
        )
        result = pre_hook._check_truth_constraints("Bash", {"command": "pytest d810"})
        assert result is not None
        assert "kg-facts" in result


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
            "simba.hooks._kg_client.query_kg",
            lambda *a, **k: "",
        )
        result = json.loads(
            pre_hook.main(
                {
                    "tool_name": "Bash",
                    "tool_input": {"command": "ls"},
                    "cwd": "/tmp",
                }
            )
        )
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
        monkeypatch.setattr(
            mc,
            "_cfg",
            type(
                "C",
                (),
                {
                    "daemon_host": "localhost",
                    "daemon_port": 8741,
                    "min_similarity": 0.35,
                    "default_max_results": 3,
                    "default_timeout": 2.0,
                },
            )(),
        )

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
        monkeypatch.setattr(
            mc,
            "_cfg",
            type(
                "C",
                (),
                {
                    "daemon_host": "localhost",
                    "daemon_port": 8741,
                    "min_similarity": 0.35,
                    "default_max_results": 3,
                    "default_timeout": 2.0,
                },
            )(),
        )

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
        monkeypatch.setattr(
            mc,
            "_cfg",
            type(
                "C",
                (),
                {
                    "daemon_host": "localhost",
                    "daemon_port": 8741,
                    "default_timeout": 2.0,
                },
            )(),
        )

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


# ---------- Memory client: project-scoped count ----------


class TestMemoryClientCount:
    @staticmethod
    def _fake_cfg(mc):
        return type(
            "C",
            (),
            {"daemon_host": "localhost", "daemon_port": 8741, "default_timeout": 2.0},
        )()

    def test_count_uses_list_with_project(self, monkeypatch):
        captured = {}

        class FakeResp:
            status_code = 200

            def json(self):
                return {"total": 3, "memories": []}

        def fake_get(url, params=None, timeout=None):
            captured["url"] = url
            captured["params"] = params
            return FakeResp()

        import simba.hooks._memory_client as mc

        monkeypatch.setattr("httpx.get", fake_get)
        monkeypatch.setattr(mc, "_cfg", self._fake_cfg(mc))

        n = mc.count_memories(memory_type="TOOL_RULE", project_path="/p1")

        assert n == 3
        assert "/list" in captured["url"]
        assert captured["params"]["type"] == "TOOL_RULE"
        assert captured["params"]["projectPath"] == "/p1"

    def test_count_none_on_error(self, monkeypatch):
        import httpx

        import simba.hooks._memory_client as mc

        def fake_get(*a, **k):
            raise httpx.HTTPError("boom")

        monkeypatch.setattr("httpx.get", fake_get)
        monkeypatch.setattr(mc, "_cfg", self._fake_cfg(mc))

        assert mc.count_memories(memory_type="TOOL_RULE", project_path="/p") is None


# ---------- PreToolUse: ruleless-project short-circuit ----------


class TestToolRuleProjectGate:
    """Skip the per-tool-call embed+recall when the project has no TOOL_RULE."""

    def test_skips_recall_when_project_has_no_rules(self, monkeypatch, tmp_path):
        class FakeCfg:
            rule_check_enabled = True
            rule_min_similarity = 0.6
            rule_count_ttl = 300

        monkeypatch.setattr(pre_hook, "_hooks_cfg", lambda: FakeCfg())
        monkeypatch.setattr(pre_hook, "_TOOL_RULE_COUNT_CACHE", tmp_path / "trc.json")
        monkeypatch.setattr("simba.hooks._memory_client.count_memories", lambda **k: 0)

        def boom(*a, **k):
            raise AssertionError("recall must be skipped for a ruleless project")

        monkeypatch.setattr("simba.hooks._memory_client.recall_memories", boom)

        assert pre_hook._check_tool_rules("Bash", {"command": "ls"}, "/tmp") is None

    def test_proceeds_when_project_has_rules(self, monkeypatch, tmp_path):
        class FakeCfg:
            rule_check_enabled = True
            rule_min_similarity = 0.6
            rule_count_ttl = 300
            rule_max_age_days = 0

        monkeypatch.setattr(pre_hook, "_hooks_cfg", lambda: FakeCfg())
        monkeypatch.setattr(pre_hook, "_TOOL_RULE_COUNT_CACHE", tmp_path / "trc.json")
        monkeypatch.setattr("simba.hooks._memory_client.count_memories", lambda **k: 2)
        called = {}

        def fake_recall(*a, **k):
            called["yes"] = True
            return [
                {
                    "content": "Bash: rule",
                    "context": json.dumps({"correction": "do x"}),
                    "similarity": 0.8,
                }
            ]

        monkeypatch.setattr("simba.hooks._memory_client.recall_memories", fake_recall)

        result = pre_hook._check_tool_rules("Bash", {"command": "ls"}, "/tmp")
        assert called.get("yes") is True
        assert result is not None

    def test_gate_fail_open_on_count_error(self, monkeypatch, tmp_path):
        # count_memories returns None (daemon unreachable) -> proceed (do the
        # check) so a real rule is never silently suppressed.
        class FakeCfg:
            rule_count_ttl = 300

        monkeypatch.setattr(pre_hook, "_TOOL_RULE_COUNT_CACHE", tmp_path / "trc.json")
        monkeypatch.setattr(
            "simba.hooks._memory_client.count_memories", lambda **k: None
        )
        assert pre_hook._project_has_tool_rules("/proj", FakeCfg()) is True

    def test_gate_caches_within_ttl(self, monkeypatch, tmp_path):
        calls = {"n": 0}

        def fake_count(**k):
            calls["n"] += 1
            return 0

        monkeypatch.setattr(pre_hook, "_TOOL_RULE_COUNT_CACHE", tmp_path / "trc.json")
        monkeypatch.setattr("simba.hooks._memory_client.count_memories", fake_count)

        class FakeCfg:
            rule_count_ttl = 300

        cfg = FakeCfg()
        assert pre_hook._project_has_tool_rules("/proj", cfg) is False
        assert pre_hook._project_has_tool_rules("/proj", cfg) is False
        assert calls["n"] == 1  # second lookup served from the TTL cache
