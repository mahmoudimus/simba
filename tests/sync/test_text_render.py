"""Tests for sync text_render module -- row-to-text and row-to-markdown conversion."""

from __future__ import annotations

import simba.sync.text_render


class TestIndexableTables:
    def test_indexable_tables_list(self) -> None:
        expected = [
            "reflections",
            "sessions",
            "knowledge",
            "facts",
            "proven_facts",
            "activities",
            "agent_runs",
        ]
        assert expected == simba.sync.text_render.INDEXABLE_TABLES

    def test_indexable_tables_is_list_of_strings(self) -> None:
        assert all(isinstance(t, str) for t in simba.sync.text_render.INDEXABLE_TABLES)


class TestRenderRow:
    def test_render_row_reflections(self) -> None:
        row = {
            "error_type": "TypeError",
            "snippet": "x is not a function",
            "signature": "abc123",
        }
        result = simba.sync.text_render.render_row("reflections", row)
        assert result == "Error [TypeError]: x is not a function (sig: abc123)"

    def test_render_row_sessions(self) -> None:
        row = {
            "summary": "Refactored auth module",
            "files_touched": "auth.py, config.py",
            "tools_used": "Read, Edit",
            "topics": "auth, refactor",
        }
        result = simba.sync.text_render.render_row("sessions", row)
        assert result == (
            "Session: Refactored auth module. "
            "Files: auth.py, config.py. "
            "Tools: Read, Edit. "
            "Topics: auth, refactor"
        )

    def test_render_row_knowledge(self) -> None:
        row = {
            "area": "testing",
            "summary": "Use pytest fixtures for setup",
            "patterns": "fixture, parametrize",
        }
        result = simba.sync.text_render.render_row("knowledge", row)
        assert result == (
            "Knowledge [testing]: Use pytest fixtures for setup. "
            "Patterns: fixture, parametrize"
        )

    def test_render_row_facts(self) -> None:
        row = {"category": "build", "fact": "Project uses hatchling"}
        result = simba.sync.text_render.render_row("facts", row)
        assert result == "Project fact [build]: Project uses hatchling"

    def test_render_row_proven_facts(self) -> None:
        row = {
            "subject": "simba.db",
            "predicate": "uses",
            "object": "sqlite3",
            "proof": "direct import in db.py",
        }
        result = simba.sync.text_render.render_row("proven_facts", row)
        assert result == (
            "Proven: simba.db uses sqlite3 (proof: direct import in db.py)"
        )

    def test_render_row_activities(self) -> None:
        row = {"tool_name": "Edit", "detail": "Modified server.py line 42"}
        result = simba.sync.text_render.render_row("activities", row)
        assert result == "Activity [Edit]: Modified server.py line 42"

    def test_render_row_agent_runs(self) -> None:
        row = {
            "agent": "neuron",
            "ticket_id": "TICKET-99",
            "result": "All tests passed",
        }
        result = simba.sync.text_render.render_row("agent_runs", row)
        assert result == "Agent [neuron] TICKET-99: All tests passed"

    def test_render_row_unknown_table(self) -> None:
        row = {"key": "value"}
        result = simba.sync.text_render.render_row("nonexistent_table", row)
        assert result == ""

    def test_render_row_missing_fields(self) -> None:
        result = simba.sync.text_render.render_row("reflections", {})
        assert result == "Error []:  (sig: )"

    def test_render_row_none_values_treated_as_missing(self) -> None:
        row = {"error_type": None, "snippet": None, "signature": None}
        result = simba.sync.text_render.render_row("reflections", row)
        assert result == "Error []:  (sig: )"


class TestRenderRowMarkdown:
    def test_render_row_markdown_reflections(self) -> None:
        row = {
            "error_type": "ValueError",
            "snippet": "invalid literal for int()",
            "signature": "sig456",
        }
        result = simba.sync.text_render.render_row_markdown("reflections", row)
        assert result == (
            "## Reflection: ValueError\n\n"
            "invalid literal for int()\n\n"
            "Signature: sig456"
        )

    def test_render_row_markdown_sessions(self) -> None:
        row = {
            "summary": "Added caching layer",
            "files_touched": "cache.py",
            "tools_used": "Write",
            "topics": "performance",
        }
        result = simba.sync.text_render.render_row_markdown("sessions", row)
        assert result == (
            "## Session\n\n"
            "Added caching layer\n\n"
            "- Files: cache.py\n"
            "- Tools: Write\n"
            "- Topics: performance"
        )

    def test_render_row_markdown_knowledge(self) -> None:
        row = {
            "area": "deployment",
            "summary": "Use Docker for CI",
            "patterns": "container, CI/CD",
        }
        result = simba.sync.text_render.render_row_markdown("knowledge", row)
        assert result == (
            "## Knowledge: deployment\n\n"
            "Use Docker for CI\n\n"
            "Patterns: container, CI/CD"
        )

    def test_render_row_markdown_facts(self) -> None:
        row = {"category": "language", "fact": "Python 3.11+"}
        result = simba.sync.text_render.render_row_markdown("facts", row)
        assert result == "## Fact (language)\n\nPython 3.11+"

    def test_render_row_markdown_proven_facts(self) -> None:
        row = {
            "subject": "ruff",
            "predicate": "replaces",
            "object": "flake8",
            "proof": "pyproject.toml config",
        }
        result = simba.sync.text_render.render_row_markdown("proven_facts", row)
        assert result == (
            "## Proven Fact\n\nruff replaces flake8\n\nProof: pyproject.toml config"
        )

    def test_render_row_markdown_activities(self) -> None:
        row = {
            "tool_name": "Bash",
            "detail": "Ran pytest suite",
            "timestamp": "2025-01-15T10:30:00",
        }
        result = simba.sync.text_render.render_row_markdown("activities", row)
        assert result == (
            "## Activity: Bash\n\nRan pytest suite\n\n_2025-01-15T10:30:00_"
        )

    def test_render_row_markdown_agent_runs(self) -> None:
        row = {
            "agent": "implementer",
            "ticket_id": "TASK-7",
            "result": "Feature complete",
            "status": "completed",
        }
        result = simba.sync.text_render.render_row_markdown("agent_runs", row)
        assert result == (
            "## Agent Run: implementer\n\n"
            "Ticket: TASK-7\n"
            "Status: completed\n\n"
            "Feature complete"
        )

    def test_render_row_markdown_unknown_table(self) -> None:
        row = {"key": "value"}
        result = simba.sync.text_render.render_row_markdown("nonexistent_table", row)
        assert result == ""

    def test_render_row_markdown_missing_fields(self) -> None:
        result = simba.sync.text_render.render_row_markdown("sessions", {})
        assert result == ("## Session\n\n\n\n- Files: \n- Tools: \n- Topics: ")
