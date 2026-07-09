"""Tests for TOOL_RULE auto-learn capture filters (post_tool_use)."""

from __future__ import annotations

import simba.db
import simba.hooks._memory_client
import simba.hooks.post_tool_use as ptu

READERS = frozenset({"grep", "rg", "ugrep", "cat", "echo", "sed", "awk"})
PROBES = frozenset({"ls", "find", "fd"})


def _detect(command, response, **kw):
    kw.setdefault("reader_verbs", READERS)
    kw.setdefault("probe_verbs", PROBES)
    kw.setdefault("skip_probe_not_found", True)
    kw.setdefault("require_nonzero_exit", True)
    return ptu._detect_failure("Bash", {"command": command}, response, **kw)


class TestExitAndStderrGate:
    def test_zero_exit_not_learned(self) -> None:
        # error word present, but the command succeeded -> not a failure.
        assert (
            _detect(
                "python run.py",
                {"stdout": "note: ImportError was handled", "exit_code": 0},
            )
            is None
        )

    def test_stdout_error_without_exit_not_learned(self) -> None:
        # No exit code; the only error mention is on stdout (pytest collection).
        assert (
            _detect(
                "uv run pytest",
                {
                    "stdout": "ImportError while importing test module test_config.py",
                    "stderr": "",
                },
            )
            is None
        )

    def test_nonzero_exit_with_stderr_is_learned(self) -> None:
        failure = _detect(
            "python app.py",
            {
                "stderr": "ModuleNotFoundError: No module named 'foo'",
                "exit_code": 1,
            },
        )
        assert failure is not None
        assert failure["error"] == "ModuleNotFoundError: No module named 'foo'"

    def test_no_exit_but_stderr_error_is_learned(self) -> None:
        failure = _detect(
            "python app.py",
            {
                "stderr": "PermissionError: [Errno 13] Permission denied",
            },
        )
        assert failure is not None


class TestReaderVerbSkip:
    def test_grep_failure_not_learned(self) -> None:
        # grep emits file content; an error word there is not grep's own failure.
        assert (
            _detect(
                "grep -n 'except ImportError' src/x.py",
                {
                    "stderr": "grep: src/x.py: No such file or directory",
                    "exit_code": 2,
                },
            )
            is None
        )

    def test_ugrep_warning_not_learned(self) -> None:
        assert (
            _detect(
                "ugrep -rn foo src/",
                {
                    "stderr": "ugrep: warning: src/x.py: No such file or directory",
                    "exit_code": 1,
                },
            )
            is None
        )

    def test_echo_header_not_learned(self) -> None:
        assert (
            _detect(
                "echo '=== ImportError traceback ==='",
                {"stdout": "=== ImportError traceback ===", "exit_code": 0},
            )
            is None
        )


class TestLineShapeReject:
    def test_source_line_not_learned(self) -> None:
        # nonzero exit + non-reader verb, but the only error line is source.
        assert (
            _detect(
                "python -c 'import x'",
                {"stderr": "  except ImportError:\n      pass", "exit_code": 1},
            )
            is None
        )

    def test_comment_line_not_learned(self) -> None:
        assert (
            _detect(
                "python build.py",
                {
                    "stderr": "# this would cause ImportError in headless mode",
                    "exit_code": 1,
                },
            )
            is None
        )

    def test_doc_arrow_line_not_learned(self) -> None:
        assert (
            _detect(
                "python gen.py",
                {
                    "stderr": "switch_case_analysis -> ImportError (risk register)",
                    "exit_code": 1,
                },
            )
            is None
        )

    def test_real_error_after_noise_is_learned(self) -> None:
        failure = _detect(
            "python app.py",
            {
                "stderr": "except ImportError:\nImportError: cannot import name 'z'",
                "exit_code": 1,
            },
        )
        assert failure is not None
        assert failure["error"] == "ImportError: cannot import name 'z'"


class TestHelpers:
    def test_is_noise_line(self) -> None:
        assert ptu._is_noise_line("except ImportError:")
        assert ptu._is_noise_line("# a comment about ImportError")
        assert ptu._is_noise_line("foo `bar` baz")
        assert ptu._is_noise_line("a -> b")
        assert not ptu._is_noise_line("ImportError: real failure")

    def test_exit_code_variants(self) -> None:
        assert ptu._exit_code({"exit_code": 2}) == 2
        assert ptu._exit_code({"returncode": "0"}) == 0
        assert ptu._exit_code({}) is None


class TestStoreUsesResolvedProjectId:
    def test_store_failure_rule_resolves_project_id(self, monkeypatch) -> None:
        captured = {}

        def fake_store(**kwargs):
            captured.update(kwargs)
            return {"status": "stored", "id": "mem_x"}

        monkeypatch.setattr(ptu, "_check_rule_dedup", lambda h: False)
        monkeypatch.setattr(ptu, "_save_rule_dedup", lambda h: None)
        monkeypatch.setattr(simba.db, "resolve_project_id", lambda p=None: "PROJ-ID")
        monkeypatch.setattr(simba.hooks._memory_client, "store_memory", fake_store)

        ptu._store_failure_rule(
            {"tool": "Bash", "command": "python app.py", "error": "Boom: bad"},
            "/some/cwd",
        )
        assert captured["project_path"] == "PROJ-ID"
        assert captured["memory_type"] == "TOOL_RULE"
