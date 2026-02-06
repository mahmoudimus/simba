"""Tests for simba.neuron.agents — dispatch, status, DB, and logging."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import simba.neuron.config
from simba.neuron.agents import (
    VALID_AGENTS,
    _check_process_alive,
    agent_status_check,
    agent_status_update,
    dispatch_agent,
    get_agent_db,
)


@pytest.fixture(autouse=True)
def _isolate_agent_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point AGENT_DB_PATH to a temp directory for every test."""
    db_path = tmp_path / "agents.db"
    monkeypatch.setattr(simba.neuron.config, "AGENT_DB_PATH", db_path)

    # Reset the global logger so it re-initialises with the temp path.
    import simba.neuron.agents as _mod

    monkeypatch.setattr(_mod, "_agent_logger", None)


# ---- 1. get_agent_db creates schema ----------------------------------------


class TestGetAgentDb:
    def test_creates_all_tables(self, tmp_path: Path):
        """Verify the four expected tables exist after opening the DB."""
        with get_agent_db() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = sorted(row[0] for row in cursor.fetchall())

        assert "agent_logs" in tables
        assert "agent_runs" in tables
        assert "log_levels" in tables
        assert "status_types" in tables

    def test_status_types_populated(self):
        """status_types should contain all Status enum members."""
        with get_agent_db() as conn:
            rows = conn.execute("SELECT id, name FROM status_types").fetchall()

        names = {name for _, name in rows}
        for status in simba.neuron.config.Status:
            assert status.name.lower() in names

    def test_log_levels_populated(self):
        """log_levels should contain all LogLevel enum members."""
        with get_agent_db() as conn:
            rows = conn.execute("SELECT id, name FROM log_levels").fetchall()

        names = {name for _, name in rows}
        for level in simba.neuron.config.LogLevel:
            assert level.name.lower() in names

    def test_idempotent(self):
        """Opening the DB twice must not raise or duplicate rows."""
        with get_agent_db() as conn1:
            count1 = conn1.execute("SELECT count(*) FROM status_types").fetchone()[0]

        with get_agent_db() as conn2:
            count2 = conn2.execute("SELECT count(*) FROM status_types").fetchone()[0]

        assert count1 == count2


# ---- 2. agent_status_update ------------------------------------------------


class TestAgentStatusUpdate:
    def _insert_run(self, ticket_id: str = "tkt-001", agent: str = "analyst"):
        """Helper: insert a minimal agent_runs row."""
        with get_agent_db() as conn:
            conn.execute(
                """INSERT INTO agent_runs
                   (ticket_id, agent, pid, status_id, created_at_utc)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    ticket_id,
                    agent,
                    12345,
                    simba.neuron.config.Status.PENDING,
                    simba.neuron.config.utc_now(),
                ),
            )
            conn.commit()

    def test_update_status(self):
        """Insert a run then update its status; verify it changes."""
        self._insert_run("tkt-up1")
        result = agent_status_update("tkt-up1", "running")
        assert "running" in result

        with get_agent_db() as conn:
            row = conn.execute(
                "SELECT status_id FROM agent_runs WHERE ticket_id=?", ("tkt-up1",)
            ).fetchone()

        assert row is not None
        assert row[0] == simba.neuron.config.Status.RUNNING

    def test_update_to_completed_sets_timestamp(self):
        """Completing a run should populate completed_at_utc."""
        self._insert_run("tkt-cmp")
        agent_status_update("tkt-cmp", "completed")

        with get_agent_db() as conn:
            row = conn.execute(
                "SELECT completed_at_utc FROM agent_runs WHERE ticket_id=?",
                ("tkt-cmp",),
            ).fetchone()

        assert row is not None
        assert row[0] is not None

    def test_update_to_failed_stores_error(self):
        """Failing a run should store the error message."""
        self._insert_run("tkt-fail")
        agent_status_update("tkt-fail", "failed", message="something broke")

        with get_agent_db() as conn:
            row = conn.execute(
                "SELECT error FROM agent_runs WHERE ticket_id=?", ("tkt-fail",)
            ).fetchone()

        assert row is not None
        assert row[0] == "something broke"

    def test_invalid_status_returns_error(self):
        """An invalid status string should return an error message."""
        result = agent_status_update("tkt-x", "bogus_status")
        assert result.startswith("Error:")
        assert "bogus_status" in result


# ---- 3. agent_status_check with no runs ------------------------------------


class TestAgentStatusCheckNoRuns:
    def test_no_active_agents(self):
        """With an empty DB, checking all active agents returns 'No active agents.'"""
        # Ensure DB is initialized.
        with get_agent_db():
            pass

        result = agent_status_check()
        assert "No active agents" in result

    def test_specific_ticket_not_found(self):
        """Querying a nonexistent ticket returns 'No status for ...'."""
        with get_agent_db():
            pass

        result = agent_status_check(ticket_id="nonexistent-ticket")
        assert "No status" in result
        assert "nonexistent-ticket" in result


# ---- 4. agent_status_check with a run --------------------------------------


class TestAgentStatusCheckWithRun:
    def test_returns_formatted_status(self):
        """A pending run should appear in the status output."""
        with get_agent_db() as conn:
            conn.execute(
                """INSERT INTO agent_runs
                   (ticket_id, agent, pid, status_id, created_at_utc)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    "tkt-fmt",
                    "researcher",
                    9999,
                    simba.neuron.config.Status.PENDING,
                    simba.neuron.config.utc_now(),
                ),
            )
            conn.commit()

        result = agent_status_check(ticket_id="tkt-fmt")
        assert "tkt-fmt" in result
        assert "researcher" in result
        assert "9999" in result


# ---- 5. _check_process_alive -----------------------------------------------


class TestCheckProcessAlive:
    def test_none_pid(self):
        """pid=None should return (False, False)."""
        assert _check_process_alive(None) == (False, False)

    def test_alive_process(self):
        """When os.kill succeeds and ps reports a normal status, alive=True."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "S"

        with (
            patch("simba.neuron.agents.os.kill") as mock_kill,
            patch("simba.neuron.agents.subprocess.run", return_value=mock_result),
        ):
            mock_kill.return_value = None  # signal 0 success
            is_alive, is_zombie = _check_process_alive(42)

        assert is_alive is True
        assert is_zombie is False

    def test_dead_process(self):
        """When os.kill raises ProcessLookupError, process is dead."""
        with patch("simba.neuron.agents.os.kill", side_effect=ProcessLookupError):
            is_alive, is_zombie = _check_process_alive(42)

        assert is_alive is False
        assert is_zombie is False

    def test_zombie_process(self):
        """When ps reports 'Z+' status, report as zombie."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Z+"

        with (
            patch("simba.neuron.agents.os.kill") as mock_kill,
            patch("simba.neuron.agents.subprocess.run", return_value=mock_result),
        ):
            mock_kill.return_value = None
            is_alive, is_zombie = _check_process_alive(42)

        assert is_alive is False
        assert is_zombie is True


# ---- 6. dispatch_agent with mocked subprocess.Popen ------------------------


class TestDispatchAgent:
    def test_creates_db_entry_and_starts_process(self, tmp_path: Path):
        """dispatch_agent should insert a row and return PID info."""
        mock_proc = MagicMock()
        mock_proc.pid = 54321

        with patch("simba.neuron.agents.subprocess.Popen", return_value=mock_proc):
            result = dispatch_agent("analyst", "tkt-d1", "Do analysis")

        assert "54321" in result
        assert "analyst" in result
        assert "tkt-d1" in result

        # Verify DB row was created.
        with get_agent_db() as conn:
            row = conn.execute(
                "SELECT agent, pid, status_id FROM agent_runs WHERE ticket_id=?",
                ("tkt-d1",),
            ).fetchone()

        assert row is not None
        assert row[0] == "analyst"
        assert row[1] == 54321
        assert row[2] == simba.neuron.config.Status.STARTED

    def test_popen_called_with_correct_args(self, tmp_path: Path):
        """Verify subprocess.Popen is invoked with expected command structure."""
        mock_proc = MagicMock()
        mock_proc.pid = 11111

        with patch(
            "simba.neuron.agents.subprocess.Popen", return_value=mock_proc
        ) as mock_popen:
            dispatch_agent("researcher", "tkt-d2", "Research topic X")

        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        cmd = call_args[0][0] if call_args[0] else call_args[1]["cmd"]
        assert cmd[0] == "claude"
        assert "--print" in cmd
        assert "--output-format" in cmd

    # ---- 7. dispatch_agent with invalid agent name --------------------------

    def test_invalid_agent_name_returns_error(self):
        """An unrecognized agent name should return an error immediately."""
        result = dispatch_agent("nonexistent-agent", "tkt-bad", "Do something")
        assert result.startswith("Error:")
        assert "nonexistent-agent" in result

    def test_valid_agents_list(self):
        """Sanity-check that VALID_AGENTS is a non-empty list of strings."""
        assert len(VALID_AGENTS) > 0
        assert all(isinstance(a, str) for a in VALID_AGENTS)
