"""Per-turn engagement record (spec 27) — reuses the spec-25 signal-flag plumbing.

Records, per session, the ``🦁☑`` ledger simba EMITTED this turn (and that simba
therefore surfaced activity). ``Stop`` reads it to verify the agent echoed the
marker: a turn where simba acted but the response lacks the marker is flagged
(the same observe-and-nudge shape as the ``[✓ rules]`` signal check).

Same tiny session-scoped JSON tempfile pattern as ``guardian/signal_flag.py`` and
``guardian/preflight_flag.py`` (and the PreToolUse dedup/context-low caches) — no
daemon/DB. ``UserPromptSubmit`` writes the prompt ledger; ``PreToolUse`` rewrites
it with the appended gate action; ``Stop`` reads it then it ages out next turn.
"""

from __future__ import annotations

import contextlib
import json
import pathlib
import tempfile

# Module-level so tests can monkeypatch it to a tmp_path (mirrors signal_flag).
_TMP_DIR = pathlib.Path(tempfile.gettempdir())
_PREFIX = "claude-engagement-"


def flag_path(session_id: str) -> pathlib.Path:
    """Return the flag file path for ``session_id`` (stable, session-scoped)."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
    return _TMP_DIR / f"{_PREFIX}{safe}.json"


def record_engagement(session_id: str, *, ledger: str) -> None:
    """Record that simba emitted ``ledger`` (surfaced activity) this turn.

    Fail-soft: a write error is swallowed (Stop's verify is advisory).
    """
    if not session_id:
        return
    with contextlib.suppress(OSError):
        flag_path(session_id).write_text(
            json.dumps({"engaged": True, "ledger": ledger})
        )


def engaged(session_id: str) -> bool:
    """Return True iff simba recorded surfaced activity for ``session_id`` this turn.

    Missing / corrupt / unreadable flag → False (no recorded engagement).
    """
    if not session_id:
        return False
    try:
        data = json.loads(flag_path(session_id).read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return bool(data.get("engaged")) if isinstance(data, dict) else False


def last_ledger(session_id: str) -> str:
    """Return the ledger line simba emitted this turn (``""`` if none recorded)."""
    if not session_id:
        return ""
    try:
        data = json.loads(flag_path(session_id).read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return ""
    if not isinstance(data, dict):
        return ""
    ledger = data.get("ledger", "")
    return ledger if isinstance(ledger, str) else ""


def reset_engagement(session_id: str) -> None:
    """Delete the flag (idempotent) at the turn boundary (UserPromptSubmit)."""
    if not session_id:
        return
    with contextlib.suppress(OSError):
        flag_path(session_id).unlink(missing_ok=True)
