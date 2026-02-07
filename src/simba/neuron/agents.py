"""Agent orchestration — dispatch, status, DB, and logging.

Manages async Claude subagents with process monitoring, output capture,
and structured SQLite logging.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import simba.db
import simba.neuron.config

if TYPE_CHECKING:
    import sqlite3

# ---------------------------------------------------------------------------
# Global logger (initialized lazily)
# ---------------------------------------------------------------------------

_agent_logger: logging.Logger | None = None


# ---------------------------------------------------------------------------
# Schema registration
# ---------------------------------------------------------------------------


def _init_agent_db_schema(conn: sqlite3.Connection) -> None:
    """Initialize agent database schema with enum tables and main tables."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS status_types (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL
        )"""
    )
    for status in simba.neuron.config.Status:
        conn.execute(
            "INSERT OR IGNORE INTO status_types (id, name) VALUES (?, ?)",
            (status.value, status.name.lower()),
        )

    conn.execute(
        """CREATE TABLE IF NOT EXISTS log_levels (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL
        )"""
    )
    for level in simba.neuron.config.LogLevel:
        conn.execute(
            "INSERT OR IGNORE INTO log_levels (id, name) VALUES (?, ?)",
            (level.value, level.name.lower()),
        )

    conn.execute(
        """CREATE TABLE IF NOT EXISTS agent_runs (
            ticket_id TEXT PRIMARY KEY,
            agent TEXT NOT NULL,
            pid INTEGER,
            status_id INTEGER REFERENCES status_types(id),
            command TEXT,
            working_dir TEXT,
            output_format TEXT,
            created_at_utc INTEGER NOT NULL,
            started_at_utc INTEGER,
            completed_at_utc INTEGER,
            result TEXT,
            error TEXT,
            stdout TEXT,
            stderr TEXT
        )"""
    )

    conn.execute(
        """CREATE TABLE IF NOT EXISTS agent_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id TEXT,
            level_id INTEGER REFERENCES log_levels(id),
            event TEXT NOT NULL,
            func TEXT,
            data TEXT,
            timestamp_utc INTEGER NOT NULL
        )"""
    )

    conn.commit()


simba.db.register_schema(_init_agent_db_schema)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _safe_read_file(file_path: Path) -> str:
    """Safely read file with UTF-8 fallback to replace mode."""
    if not file_path.exists():
        return ""
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"[Error reading file: {exc}]"
    except PermissionError as exc:
        return f"[Permission denied: {exc}]"
    except Exception as exc:
        return f"[Unexpected error: {exc}]"


def _extract_result(stdout: str, stderr: str, output_format: str = "text") -> str:
    """Extract result from Claude Code output, parsing stream-json if needed."""
    if output_format == "stream-json":
        try:
            lines = [line for line in stdout.strip().split("\n") if line.strip()]
            for line in lines:
                try:
                    obj = json.loads(line)
                    if obj.get("type") == "result":
                        return obj.get("result", "")
                except json.JSONDecodeError:
                    continue
            return stdout.strip()
        except Exception:
            return stdout.strip()

    if output_format == "json":
        try:
            data = json.loads(stdout.strip())
            if isinstance(data, dict):
                return json.dumps(data, indent=2, ensure_ascii=False)
            return stdout.strip()
        except json.JSONDecodeError:
            pass

    result = stdout.strip()
    if not result and stderr:
        result = stderr.strip()
    return result or "No output from Claude Code"


# ---------------------------------------------------------------------------
# SIGCHLD handler for zombie reaping
# ---------------------------------------------------------------------------


def _sigchld_handler(signum: int, frame: object) -> None:
    """Reap terminated child processes to prevent zombie accumulation."""
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                break
            exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
            logger = _get_logger()
            if logger:
                logger.debug(
                    "Zombie process reaped",
                    extra={
                        "ticket_id": None,
                        "event": "zombie_reaped",
                        "data": {"pid": pid, "exit_code": exit_code},
                    },
                )
        except ChildProcessError:
            break
        except Exception:
            break


def register_sigchld_handler() -> None:
    """Register the SIGCHLD handler for zombie process reaping."""
    signal.signal(signal.SIGCHLD, _sigchld_handler)


# ---------------------------------------------------------------------------
# SQLite logging infrastructure
# ---------------------------------------------------------------------------


class SQLiteLogHandler(logging.Handler):
    """Custom logging handler that writes structured logs to SQLite."""

    def __init__(self) -> None:
        super().__init__()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            ticket_id = getattr(record, "ticket_id", None)
            event = getattr(record, "event", "log")
            data = getattr(record, "data", {})
            level_id = simba.neuron.config.LOGGING_LEVEL_MAP.get(
                record.levelno, simba.neuron.config.LogLevel.INFO
            )
            data_json = json.dumps(data) if data else "{}"

            with simba.db.get_db() as conn:
                conn.execute(
                    """INSERT INTO agent_logs
                       (ticket_id, level_id, event, func, data, timestamp_utc)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        ticket_id,
                        int(level_id),
                        event,
                        record.funcName,
                        data_json,
                        simba.neuron.config.utc_now(),
                    ),
                )
                conn.commit()
        except Exception:
            self.handleError(record)


class StructuredFormatter(logging.Formatter):
    """Formatter that ensures data field is JSON-serializable."""

    def format(self, record: logging.LogRecord) -> str:
        if hasattr(record, "data") and isinstance(record.data, dict):
            record.data_json = json.dumps(record.data)  # type: ignore[attr-defined]
        else:
            record.data_json = "{}"  # type: ignore[attr-defined]
        return super().format(record)


def _setup_logger() -> logging.Logger:
    """Set up the agent logger with SQLite handler."""
    logger = logging.getLogger("neuron.agents")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    handler = SQLiteLogHandler()
    handler.setFormatter(StructuredFormatter())
    logger.addHandler(handler)
    return logger


def _get_logger() -> logging.Logger | None:
    """Get the agent logger (lazy initialization)."""
    global _agent_logger
    if _agent_logger is None:
        try:
            _agent_logger = _setup_logger()
        except Exception:
            return None
    return _agent_logger


# ---------------------------------------------------------------------------
# Process management helpers
# ---------------------------------------------------------------------------


def _check_process_alive(pid: int | None) -> tuple[bool, bool]:
    """Check if a process is still running (and not a zombie).

    Returns (is_alive, is_zombie).
    """
    if pid is None:
        return False, False

    try:
        os.kill(pid, 0)
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "stat="],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=1,
            )
            if result.returncode == 0:
                stat = result.stdout.strip()
                is_zombie = stat.startswith("Z")
                if is_zombie:
                    logger = _get_logger()
                    if logger:
                        logger.debug(
                            "Zombie process detected",
                            extra={
                                "ticket_id": None,
                                "event": "process_checked",
                                "data": {
                                    "pid": pid,
                                    "alive": True,
                                    "is_zombie": True,
                                },
                            },
                        )
                return not is_zombie, is_zombie
            return False, False
        except Exception:
            return True, False
    except ProcessLookupError:
        return False, False
    except PermissionError:
        return True, False
    except Exception:
        return False, False


def _capture_and_cleanup(
    ticket_id: str,
    stdout_path: Path,
    stderr_path: Path,
    output_format: str,
) -> tuple[str, str, str]:
    """Read output files, parse result, store in DB, delete files.

    Returns (stdout, stderr, result).
    """
    stdout = _safe_read_file(stdout_path)
    stderr = _safe_read_file(stderr_path)
    result = _extract_result(stdout, stderr, output_format)

    with simba.db.get_db() as conn:
        conn.execute(
            """UPDATE agent_runs
               SET stdout=?, stderr=?, result=?, completed_at_utc=?,
                   status_id=?
               WHERE ticket_id=?""",
            (
                stdout,
                stderr,
                result,
                simba.neuron.config.utc_now(),
                simba.neuron.config.Status.COMPLETED,
                ticket_id,
            ),
        )
        conn.commit()

    logger = _get_logger()
    if logger:
        logger.info(
            "Output captured and cleaned up",
            extra={
                "ticket_id": ticket_id,
                "event": "output_captured",
                "data": {
                    "stdout_len": len(stdout),
                    "stderr_len": len(stderr),
                    "result_len": len(result),
                },
            },
        )

    stdout_path.unlink(missing_ok=True)
    stderr_path.unlink(missing_ok=True)

    return stdout, stderr, result


# ---------------------------------------------------------------------------
# MCP tool functions
# ---------------------------------------------------------------------------

VALID_AGENTS = [
    "analyst",
    "implementer",
    "log-analyst",
    "logician",
    "performance-optimizer",
    "researcher",
    "test-specialist",
    "verifier",
]


def agent_status_update(ticket_id: str, status: str, message: str = "") -> str:
    """Update the status of an async agent task.

    Args:
        ticket_id: The ticket ID this agent is working on.
        status: One of: pending, started, running, completed, failed.
        message: Optional status message or error details.
    """
    status_lower = status.lower()
    if status_lower not in simba.neuron.config.STATUS_NAME_MAP:
        valid = list(simba.neuron.config.STATUS_NAME_MAP.keys())
        return f"Error: Invalid status '{status}'. Valid: {valid}"

    status_id = simba.neuron.config.STATUS_NAME_MAP[status_lower]

    with simba.db.get_db() as conn:
        cursor = conn.cursor()
        row = cursor.execute(
            "SELECT status_id FROM agent_runs WHERE ticket_id=?",
            (ticket_id,),
        ).fetchone()
        old_status_id = row[0] if row else None

        if status_id in (
            simba.neuron.config.Status.COMPLETED,
            simba.neuron.config.Status.FAILED,
        ):
            conn.execute(
                """UPDATE agent_runs
                   SET status_id=?, error=?, completed_at_utc=?
                   WHERE ticket_id=?""",
                (
                    status_id,
                    message if status_id == simba.neuron.config.Status.FAILED else None,
                    simba.neuron.config.utc_now(),
                    ticket_id,
                ),
            )
        else:
            conn.execute(
                "UPDATE agent_runs SET status_id=? WHERE ticket_id=?",
                (status_id, ticket_id),
            )
        conn.commit()

        logger = _get_logger()
        if logger:
            old_name = (
                simba.neuron.config.Status(old_status_id).name.lower()
                if old_status_id
                else "unknown"
            )
            logger.info(
                "Status updated",
                extra={
                    "ticket_id": ticket_id,
                    "event": "status_changed",
                    "data": {
                        "old_status": old_name,
                        "new_status": status_lower,
                        "message": message,
                    },
                },
            )

        return f"Status updated: {ticket_id} -> {status_lower}"


def agent_status_check(ticket_id: str | None = None) -> str:
    """Check the status of async agent tasks.

    Auto-detects completion via process state.

    Args:
        ticket_id: Optional specific ticket to check. If None, returns all
                 active agents.
    """
    with simba.db.get_db() as conn:
        cursor = conn.cursor()

        base_query = """
            SELECT ar.ticket_id, ar.agent, ar.pid, ar.status_id, st.name,
                   ar.output_format, ar.result, ar.error, ar.created_at_utc
            FROM agent_runs ar
            LEFT JOIN status_types st ON ar.status_id = st.id
        """

        if ticket_id:
            rows = cursor.execute(
                base_query + " WHERE ar.ticket_id=?", (ticket_id,)
            ).fetchall()
        else:
            rows = cursor.execute(
                base_query + " WHERE ar.status_id NOT IN (?, ?)",
                (
                    simba.neuron.config.Status.COMPLETED,
                    simba.neuron.config.Status.FAILED,
                ),
            ).fetchall()

        if not rows:
            if not ticket_id:
                return "No active agents."
            return f"No status for {ticket_id}"

        lines: list[str] = []
        for row in rows:
            (
                bid,
                agent,
                pid,
                status_id,
                status_name,
                output_format,
                result,
                error,
                created_at,
            ) = row

            # Auto-detect completion
            if status_id in (
                simba.neuron.config.Status.STARTED,
                simba.neuron.config.Status.RUNNING,
            ):
                is_alive, _is_zombie = _check_process_alive(pid)
                if not is_alive:
                    stdout_path = Path(tempfile.gettempdir()) / f"neuron_{bid}.stdout"
                    stderr_path = Path(tempfile.gettempdir()) / f"neuron_{bid}.stderr"

                    if stdout_path.exists() or stderr_path.exists():
                        _stdout, _stderr, result = _capture_and_cleanup(
                            bid,
                            stdout_path,
                            stderr_path,
                            output_format or "text",
                        )
                        status_name = "completed"

                        logger = _get_logger()
                        if logger:
                            logger.info(
                                "Agent completed",
                                extra={
                                    "ticket_id": bid,
                                    "event": "status_changed",
                                    "data": {
                                        "old_status": "running",
                                        "new_status": "completed",
                                        "reason": "process_exited",
                                    },
                                },
                            )
                    else:
                        status_name = "finished (no output files)"

            elapsed = simba.neuron.config.utc_now() - created_at if created_at else 0
            status_line = f"{bid} ({agent}, PID {pid}): {status_name}"
            if elapsed:
                status_line += f" [{elapsed}s]"
            if result:
                preview = result[:100] + "..." if len(result) > 100 else result
                status_line += f"\n   Result: {preview}"
            if error:
                status_line += f"\n   Error: {error}"

            lines.append(status_line)

        return "\n".join(lines)


def dispatch_agent(agent_name: str, ticket_id: str, instructions: str) -> str:
    """Launch an async subagent. Non-blocking, returns immediately with PID.

    Args:
        agent_name: Agent name (see .claude/agents/ for available agents).
        ticket_id: The Ticket ID this agent is working on.
        instructions: High-level goal for the agent.
    """
    if agent_name not in VALID_AGENTS:
        return f"Error: Unknown agent '{agent_name}'. Valid: {VALID_AGENTS}"

    project_root = Path.cwd()

    agent_prompt = (
        f"You are working on Ticket {ticket_id}.\n"
        f"Project: {project_root.name}\n"
        f"Root: {project_root}\n\n"
        f"INSTRUCTIONS: {instructions}\n\n"
        f"When finished, output a clear summary of what you accomplished.\n"
    )

    stdout_path = Path(tempfile.gettempdir()) / f"neuron_{ticket_id}.stdout"
    stderr_path = Path(tempfile.gettempdir()) / f"neuron_{ticket_id}.stderr"

    output_format = "stream-json"
    cmd = [
        "claude",
        "--print",
        "-p",
        agent_prompt,
        "--output-format",
        output_format,
        "--verbose",
        "--dangerously-skip-permissions",
    ]

    try:
        with (
            open(stdout_path, "w") as stdout_f,
            open(stderr_path, "w") as stderr_f,
        ):
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=stdout_f,
                stderr=stderr_f,
                cwd=project_root,
                start_new_session=True,
            )

        now = simba.neuron.config.utc_now()
        cmd_str = " ".join(cmd)

        with simba.db.get_db() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO agent_runs
                   (ticket_id, agent, pid, status_id, command,
                    working_dir, output_format,
                    created_at_utc, started_at_utc)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ticket_id,
                    agent_name,
                    proc.pid,
                    simba.neuron.config.Status.STARTED,
                    cmd_str,
                    str(project_root),
                    output_format,
                    now,
                    now,
                ),
            )
            conn.commit()

        logger = _get_logger()
        if logger:
            logger.info(
                "Agent dispatched",
                extra={
                    "ticket_id": ticket_id,
                    "event": "agent_dispatched",
                    "data": {
                        "pid": proc.pid,
                        "agent": agent_name,
                        "command": cmd_str,
                        "working_dir": str(project_root),
                    },
                },
            )

        return (
            f"Subagent '{agent_name}' dispatched (PID: {proc.pid}).\n"
            f"   - Task: {ticket_id}\n"
            f"   - Check: agent_status_check('{ticket_id}')"
        )
    except Exception as exc:
        logger = _get_logger()
        if logger:
            logger.error(
                "Failed to dispatch agent",
                extra={
                    "ticket_id": ticket_id,
                    "event": "error",
                    "data": {
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                },
            )
        return f"Failed to dispatch agent: {exc}"
