"""Per-turn preflight flag (spec 28) — reuses the spec-25 signal-flag plumbing.

The mandated-preflight pattern converts "consult before acting" into a
precondition: ``simba preflight`` sets a per-turn flag, and ``PreToolUse`` blocks
a mutating tool that runs without it. The flag is the same tiny session-scoped
JSON tempfile pattern as ``guardian/signal_flag.py`` (and the PreToolUse
dedup/context-low caches) so it needs no daemon/DB.

Lifecycle: ``set_preflight`` (on a ``simba preflight`` invocation) → the gate sees
it this turn → ``reset_preflight`` on the next ``UserPromptSubmit`` (the turn
boundary). ``preflight_ran`` fail-soft returns False on any read error (the gate
is only armed by config; a missing flag means "no preflight yet", which is the
gate's whole point).
"""

from __future__ import annotations

import contextlib
import json
import pathlib
import tempfile
import time

# Module-level so tests can monkeypatch it to a tmp_path (mirrors signal_flag).
_TMP_DIR = pathlib.Path(tempfile.gettempdir())
_PREFIX = "claude-preflight-"


def flag_path(session_id: str) -> pathlib.Path:
    """Return the flag file path for ``session_id`` (stable, session-scoped)."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
    return _TMP_DIR / f"{_PREFIX}{safe}.json"


def set_preflight(session_id: str, *, task: str = "") -> None:
    """Record that a ``simba preflight`` fired this turn for ``session_id``.

    Fail-soft: a write error is swallowed (the gate fail-... is config-gated, and
    a missing flag simply reads as "no preflight").
    """
    if not session_id:
        return
    with contextlib.suppress(OSError):
        flag_path(session_id).write_text(
            json.dumps({"preflight": True, "task": task, "ts": time.time()})
        )


def preflight_ran(session_id: str) -> bool:
    """Return True iff a preflight is recorded for ``session_id`` this turn.

    Missing / corrupt / unreadable flag → False (no preflight yet).
    """
    if not session_id:
        return False
    try:
        data = json.loads(flag_path(session_id).read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return bool(data.get("preflight")) if isinstance(data, dict) else False


def reset_preflight(session_id: str) -> None:
    """Delete the flag (idempotent) at the turn boundary (UserPromptSubmit)."""
    if not session_id:
        return
    with contextlib.suppress(OSError):
        flag_path(session_id).unlink(missing_ok=True)


# ── Risk-tier mandate arming (preflight_mandate_risk_only) ──────────────────
# When the mandate is risk_only (the default over-fire guard), the PreToolUse
# gate is armed only AFTER a risk-tier doctrine was primed this turn. That arming
# is recorded as a sibling per-turn flag, set by UserPromptSubmit when intent
# priming matches a risk-tier doctrine, and cleared at the next turn boundary.

_ARM_PREFIX = "claude-mandate-armed-"


def _arm_path(session_id: str) -> pathlib.Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
    return _TMP_DIR / f"{_ARM_PREFIX}{safe}.json"


def arm_mandate(session_id: str) -> None:
    """Record that a risk-tier doctrine was primed this turn (arms the gate)."""
    if not session_id:
        return
    with contextlib.suppress(OSError):
        _arm_path(session_id).write_text(json.dumps({"armed": True, "ts": time.time()}))


def mandate_armed(session_id: str) -> bool:
    """Return True iff a risk-tier prime armed the mandate this turn."""
    if not session_id:
        return False
    try:
        data = json.loads(_arm_path(session_id).read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return bool(data.get("armed")) if isinstance(data, dict) else False


def reset_mandate(session_id: str) -> None:
    """Clear the arming flag (idempotent) at the turn boundary."""
    if not session_id:
        return
    with contextlib.suppress(OSError):
        _arm_path(session_id).unlink(missing_ok=True)
