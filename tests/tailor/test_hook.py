"""Tests for tailor hook module."""

from __future__ import annotations

import json
import pathlib

import pytest

import simba.db
import simba.tailor.hook


class TestDetectError:
    def test_empty_input(self):
        assert simba.tailor.hook.detect_error("") is False

    def test_short_input(self):
        assert simba.tailor.hook.detect_error("Short") is False

    def test_detects_error_pattern(self):
        assert simba.tailor.hook.detect_error("Error: something went wrong") is True

    def test_detects_type_error(self):
        assert (
            simba.tailor.hook.detect_error("TypeError: null is not an object") is True
        )

    def test_detects_reference_error(self):
        assert (
            simba.tailor.hook.detect_error("ReferenceError: x is not defined") is True
        )

    def test_detects_syntax_error(self):
        assert simba.tailor.hook.detect_error("SyntaxError: unexpected token") is True

    def test_detects_assertion_error(self):
        assert simba.tailor.hook.detect_error("AssertionError: expected true") is True

    def test_detects_failed(self):
        assert simba.tailor.hook.detect_error("Test failed at line 5") is True

    def test_detects_enoent(self):
        assert (
            simba.tailor.hook.detect_error("ENOENT: no such file or directory") is True
        )

    def test_detects_eacces(self):
        assert simba.tailor.hook.detect_error("EACCES: permission denied") is True

    def test_detects_cannot_find_module(self):
        assert simba.tailor.hook.detect_error("Cannot find module 'express'") is True

    def test_detects_cannot_read_properties(self):
        assert (
            simba.tailor.hook.detect_error("Cannot read properties of undefined")
            is True
        )

    def test_detects_uncaught(self):
        assert simba.tailor.hook.detect_error("Uncaught exception in handler") is True

    def test_detects_exception(self):
        assert simba.tailor.hook.detect_error("Exception thrown in test runner") is True

    def test_no_error_in_clean_content(self):
        assert simba.tailor.hook.detect_error("All tests passed successfully") is False

    def test_case_insensitive(self):
        assert simba.tailor.hook.detect_error("error: lowercase too") is True


class TestExtractErrorType:
    def test_extracts_error(self):
        assert simba.tailor.hook.extract_error_type("Error: something") == "error"

    def test_extracts_type_error(self):
        # TypeError: contains Error: so the first pattern matches
        result = simba.tailor.hook.extract_error_type("TypeError: null")
        assert "error" in result

    def test_extracts_enoent(self):
        assert simba.tailor.hook.extract_error_type("ENOENT: no such file") == "enoent"

    def test_first_match_wins(self):
        result = simba.tailor.hook.extract_error_type(
            "At line 50: TypeError: expected string"
        )
        assert "error" in result

    def test_unknown_for_no_match(self):
        assert simba.tailor.hook.extract_error_type("Clean output") == "unknown"

    def test_no_colon_in_result(self):
        result = simba.tailor.hook.extract_error_type("TypeError: details")
        assert ":" not in result


class TestExtractSnippet:
    def test_captures_context_around_error(self):
        content = "Before " + "X" * 100 + " Error: failed " + "Y" * 500
        snippet = simba.tailor.hook.extract_snippet(content)
        assert "Error" in snippet
        assert len(snippet) > 50

    def test_returns_empty_for_no_error(self):
        assert simba.tailor.hook.extract_snippet("Clean output") == ""

    def test_captures_up_to_100_chars_before(self):
        prefix = "A" * 200
        content = prefix + "Error: oops"
        snippet = simba.tailor.hook.extract_snippet(content)
        # Should not contain the entire prefix
        assert len(snippet) <= 611  # 100 before + "Error: oops" + 500 after max

    def test_captures_up_to_500_chars_after(self):
        suffix = "Z" * 1000
        content = "Error: fail" + suffix
        snippet = simba.tailor.hook.extract_snippet(content)
        assert len(snippet) <= 611


class TestExtractContext:
    def test_extracts_file(self):
        ctx = simba.tailor.hook.extract_context("at function() in test.js:42")
        assert ctx.get("file") == "test.js"

    def test_extracts_operation(self):
        ctx = simba.tailor.hook.extract_context("at processData() in test.js:42")
        assert ctx.get("operation") == "processData"

    def test_extracts_module(self):
        ctx = simba.tailor.hook.extract_context('from "express" Error: failed')
        assert ctx.get("module") == "express"

    def test_handles_missing_context(self):
        ctx = simba.tailor.hook.extract_context("Error occurred somewhere")
        assert len(ctx) <= 3

    def test_extracts_all_three(self):
        snippet = (
            'at buildComponent() in src/components/app.tsx:100:20 from "react-router"'
        )
        ctx = simba.tailor.hook.extract_context(snippet)
        assert ctx.get("file")
        assert ctx.get("operation")
        assert ctx.get("module")
        assert len(ctx) == 3

    def test_excludes_node_modules(self):
        ctx = simba.tailor.hook.extract_context(
            "at func() in node_modules/pkg/index.js:1"
        )
        assert "file" not in ctx

    def test_tsx_file_extension(self):
        ctx = simba.tailor.hook.extract_context("at render() in App.tsx:15")
        assert ctx.get("file") == "App.tsx"

    def test_jsx_file_extension(self):
        ctx = simba.tailor.hook.extract_context("at render() in App.jsx:15")
        assert ctx.get("file") == "App.jsx"


class TestNormalizeSnippet:
    def test_normalizes_line_column(self):
        result = simba.tailor.hook.normalize_snippet("Error at file.js:42:15")
        assert ":LINE:COL" in result

    def test_normalizes_paths(self):
        result = simba.tailor.hook.normalize_snippet(
            "Error at /home/user/project/test.js"
        )
        assert "/PATH/" in result

    def test_normalizes_hex_addresses(self):
        result = simba.tailor.hook.normalize_snippet("code 0x12345abc")
        assert "0xADDR" in result

    def test_normalizes_large_numbers(self):
        result = simba.tailor.hook.normalize_snippet("token 9999999999")
        assert "NUM" in result

    def test_deterministic(self):
        snippet = "Error at file.js:10:5"
        assert simba.tailor.hook.normalize_snippet(
            snippet
        ) == simba.tailor.hook.normalize_snippet(snippet)


class TestGenerateSignature:
    def test_starts_with_error_type(self):
        sig = simba.tailor.hook.generate_signature("error", "Cannot read property")
        assert sig.startswith("error-")

    def test_includes_content(self):
        sig = simba.tailor.hook.generate_signature(
            "error", simba.tailor.hook.normalize_snippet("Cannot read property")
        )
        assert "Cannot" in sig


class TestCreateReflectionEntry:
    def test_has_required_fields(self):
        entry = simba.tailor.hook.create_reflection_entry(
            "error", "Error: test", {"file": "test.js"}
        )
        assert "id" in entry
        assert "ts" in entry
        assert "error_type" in entry
        assert "snippet" in entry
        assert "context" in entry
        assert "signature" in entry

    def test_id_starts_with_nano(self):
        entry = simba.tailor.hook.create_reflection_entry("error", "Error: test", {})
        assert entry["id"].startswith("nano-")

    def test_preserves_error_type(self):
        entry = simba.tailor.hook.create_reflection_entry(
            "typeerror", "TypeError: x", {}
        )
        assert entry["error_type"] == "typeerror"

    def test_unique_ids(self):
        e1 = simba.tailor.hook.create_reflection_entry("error", "Error 1", {})
        e2 = simba.tailor.hook.create_reflection_entry("error", "Error 2", {})
        assert e1["id"] != e2["id"]

    def test_snippet_is_trimmed(self):
        entry = simba.tailor.hook.create_reflection_entry(
            "error", "  Error: test  ", {}
        )
        assert entry["snippet"] == "Error: test"


class TestParseTranscriptContent:
    def test_extracts_tool_use_result(self):
        lines = [json.dumps({"toolUseResult": "Error: something failed"})]
        content = simba.tailor.hook.parse_transcript_content(lines)
        assert "Error: something failed" in content

    def test_extracts_tool_result_message(self):
        lines = [
            json.dumps(
                {
                    "message": {
                        "content": [
                            {"type": "tool_result", "content": "ENOENT: not found"}
                        ]
                    }
                }
            )
        ]
        content = simba.tailor.hook.parse_transcript_content(lines)
        assert "ENOENT: not found" in content

    def test_skips_malformed_lines(self):
        lines = ["{invalid json}", json.dumps({"toolUseResult": "Error: ok"})]
        content = simba.tailor.hook.parse_transcript_content(lines)
        assert "Error: ok" in content

    def test_empty_lines(self):
        content = simba.tailor.hook.parse_transcript_content([])
        assert content == ""

    def test_handles_dict_tool_use_result(self):
        lines = [json.dumps({"toolUseResult": {"output": "Error: failed", "code": 1}})]
        content = simba.tailor.hook.parse_transcript_content(lines)
        assert "Error: failed" in content

    def test_handles_dict_tool_result_content(self):
        lines = [
            json.dumps(
                {
                    "message": {
                        "content": [
                            {
                                "type": "tool_result",
                                "content": {"text": "TypeError: x is not a function"},
                            }
                        ]
                    }
                }
            )
        ]
        content = simba.tailor.hook.parse_transcript_content(lines)
        assert "TypeError" in content


class TestProcessHook:
    def test_stores_reflection_in_db(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ):
        db_path = tmp_path / ".simba" / "simba.db"
        monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)

        transcript_path = tmp_path / "transcript.jsonl"
        error_msg = (
            "Error: something went wrong here in the application "
            "while processing the request handler for the main module"
        )
        transcript_path.write_text(json.dumps({"toolUseResult": error_msg}) + "\n")

        hook_input = json.dumps(
            {"transcript_path": str(transcript_path), "cwd": str(tmp_path)}
        )
        simba.tailor.hook.process_hook(hook_input)

        with simba.db.get_db(tmp_path) as conn:
            rows = conn.execute("SELECT * FROM reflections").fetchall()
        assert len(rows) == 1
        row = rows[0]
        assert row["id"].startswith("nano-")
        assert row["ts"]
        assert row["error_type"] in ("error", "failed")
        assert row["snippet"]
        assert row["signature"]
        context = json.loads(row["context"])
        assert isinstance(context, dict)

    def test_exits_silently_on_empty_input(self):
        # Should not raise
        simba.tailor.hook.process_hook("")

    def test_exits_silently_on_invalid_json(self):
        simba.tailor.hook.process_hook("{invalid}")

    def test_exits_silently_on_missing_transcript(self, tmp_path: pathlib.Path):
        hook_input = json.dumps(
            {"transcript_path": str(tmp_path / "nonexistent.jsonl")}
        )
        simba.tailor.hook.process_hook(hook_input)

    def test_exits_silently_on_no_errors(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ):
        db_path = tmp_path / ".simba" / "simba.db"
        monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)

        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text(
            json.dumps({"toolUseResult": "All tests passed"}) + "\n"
        )
        hook_input = json.dumps(
            {"transcript_path": str(transcript_path), "cwd": str(tmp_path)}
        )
        simba.tailor.hook.process_hook(hook_input)

        # DB may or may not exist; if it does, reflections table should be empty
        if db_path.exists():
            with simba.db.get_db(tmp_path) as conn:
                rows = conn.execute("SELECT * FROM reflections").fetchall()
            assert len(rows) == 0
        # If DB doesn't exist, that's fine too

    def test_handles_write_errors_silently(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ):
        """DB write to problematic path should not raise."""
        # Point get_db_path to an unwritable location
        monkeypatch.setattr(
            simba.db,
            "get_db_path",
            lambda cwd=None: pathlib.Path("/nonexistent/path/.simba/simba.db"),
        )

        transcript_content = json.dumps({"toolUseResult": "Error: something failed"})
        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text(transcript_content + "\n")

        hook_input = json.dumps(
            {
                "transcript_path": str(transcript_path),
                "cwd": str(tmp_path),
            }
        )
        simba.tailor.hook.process_hook(hook_input)  # Should not raise
