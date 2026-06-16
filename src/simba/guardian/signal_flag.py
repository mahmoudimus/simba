"""Per-session ``[✓ rules]`` signal flag — the Proposal-A plumbing (spec 25).

The CLAUDE.md guardian re-injects the ``SIMBA:core`` block on every prompt.
That is ~2k tokens of per-turn overhead even when the model still has the rules.
This module records, per session, whether the model's *previous* response carried
the ``[✓ rules]`` signal, so ``user_prompt_submit`` can SKIP the re-injection when
the rules are still present and only re-inject when they've decayed.

The flag is a tiny JSON file under the system temp dir, keyed by session id —
the same ``/tmp/claude-*.json`` flag-file pattern the PreToolUse hook already uses
for its dedup/context-low/tool-rule-count caches.

Decision contract (``should_inject``): **fail-open**. The block is the safety
layer, so any uncertainty — no flag yet (first prompt / post-compaction), a
missing-signal flag (decayed), an empty session id, or any read error — returns
True (inject). Only an explicit "signal was present last turn" returns False
(skip).
"""

from __future__ import annotations

import contextlib
import json
import pathlib
import tempfile

import simba.guardian.check_signal

# Module-level so tests can monkeypatch it to a tmp_path.
_TMP_DIR = pathlib.Path(tempfile.gettempdir())
_PREFIX = "claude-rules-signal-"


def flag_path(session_id: str) -> pathlib.Path:
    """Return the flag file path for ``session_id`` (stable, session-scoped)."""
    # Sanitize so an exotic session id can't escape the temp dir or collide.
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
    return _TMP_DIR / f"{_PREFIX}{safe}.json"


def record_signal(session_id: str, *, present: bool) -> None:
    """Record whether ``session_id``'s last response carried ``[✓ rules]``.

    Fail-soft: a write error is swallowed (the reader fail-opens to inject).
    """
    if not session_id:
        return
    with contextlib.suppress(OSError):
        flag_path(session_id).write_text(json.dumps({"signal": bool(present)}))


def signal_present(session_id: str) -> bool:
    """Return True iff the recorded flag says the last response had the signal.

    Missing / corrupt / unreadable flag → False (no recorded presence).
    """
    try:
        data = json.loads(flag_path(session_id).read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return bool(data.get("signal")) if isinstance(data, dict) else False


def reset_signal(session_id: str) -> None:
    """Delete the flag (idempotent) so the next prompt re-injects the block."""
    if not session_id:
        return
    with contextlib.suppress(OSError):
        flag_path(session_id).unlink(missing_ok=True)


def should_inject(session_id: str) -> bool:
    """Return True when the CORE block should be (re)injected this prompt.

    Fail-open: inject on the first prompt (no flag), after a decayed signal
    (flag says absent), with no session id, or on any error. Skip ONLY when the
    flag explicitly records the signal was present last turn.
    """
    if not session_id:
        return True
    try:
        return not signal_present(session_id)
    except Exception:
        return True


def signal_in_response(response: str) -> bool:
    """Thin re-export of the detection logic (single source of truth)."""
    return simba.guardian.check_signal.check_signal(response)
