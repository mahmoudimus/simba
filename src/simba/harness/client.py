"""Client identity for outbound memory-daemon requests.

Every runtime that talks to the daemon (Claude Code hooks, Codex hooks, the pi
extension, raw ``simba`` CLI) stamps an ``X-Simba-Client`` header so the daemon
can attribute traffic instead of guessing from transport artifacts. This module
resolves the *local* process's client name; the daemon just logs whatever it
receives.

Resolution precedence (first match wins):

1. an ``explicit`` argument (e.g. a ``--client`` flag baked into generated hooks)
2. the ``SIMBA_CLIENT`` environment variable (a runtime override, like
   ``SIMBA_DAEMON_URL``)
3. runtime env markers — ``CLAUDECODE`` / ``CLAUDE_CODE_ENTRYPOINT`` (Claude
   Code), ``CODEX_SANDBOX`` (Codex executing under its sandbox)
4. the caller-supplied ``default``
"""

from __future__ import annotations

import contextvars
import os

CLIENT_HEADER = "X-Simba-Client"

# Known client names. Free-form by design — the daemon logs the raw value, so a
# new runtime needs no daemon change, only its own header.
CLAUDE_CODE = "claude-code"
CODEX = "codex"
PI = "pi"
CLI = "cli"
DAEMON = "daemon"
UNKNOWN = "unknown"

# Origin of the request currently being served by the daemon. Set from the
# inbound X-Simba-Client header in run_hook; read here so a loopback /recall
# (run_hook → dispatch → recall) nests as ``<origin>.daemon`` instead of a flat
# "daemon" that loses where the recall came from. Contextvars propagate into the
# threadpool run_in_threadpool uses, so the sync dispatch path sees it.
_ORIGIN_CLIENT: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "simba_origin_client", default=None
)


def set_origin_client(name: str | None) -> contextvars.Token:
    """Set the per-request origin client; returns a token for resetting."""
    return _ORIGIN_CLIENT.set(name)


def get_origin_client() -> str | None:
    """Return the origin client for the request being served, or ``None``."""
    return _ORIGIN_CLIENT.get()


def reset_origin_client(token: contextvars.Token) -> None:
    """Restore the origin client to its prior value (pairs with set)."""
    _ORIGIN_CLIENT.reset(token)


def detect_client(explicit: str | None = None, *, default: str = CLI) -> str:
    """Resolve the client name for outbound daemon requests.

    See the module docstring for the precedence order. ``CODEX_HOME`` is
    deliberately NOT sniffed: a Claude Code machine may export it globally for
    the ``codex`` CLI, which would mis-tag Claude Code hooks. Codex hooks carry
    an explicit ``--client codex`` (and run under ``CODEX_SANDBOX``).
    """
    base = _detect_base(explicit, default=default)
    # The daemon-loopback hop: a recall issued from inside the daemon while
    # serving a hook nests under its origin → "claude-code.daemon". Collapse if
    # the origin is already a ".daemon" value (never double-append).
    if base == DAEMON:
        origin = get_origin_client()
        if origin:
            return origin if origin.endswith(f".{DAEMON}") else f"{origin}.{DAEMON}"
    return base


def _detect_base(explicit: str | None, *, default: str) -> str:
    """Resolve the un-nested client name (precedence per the module docstring)."""
    if explicit:
        return explicit
    env = os.environ.get("SIMBA_CLIENT")
    if env:
        return env
    if os.environ.get("CLAUDECODE") or os.environ.get("CLAUDE_CODE_ENTRYPOINT"):
        return CLAUDE_CODE
    if os.environ.get("CODEX_SANDBOX"):
        return CODEX
    return default


def client_headers(
    explicit: str | None = None, *, default: str = CLI
) -> dict[str, str]:
    """Return the outbound header dict carrying the resolved client name."""
    return {CLIENT_HEADER: detect_client(explicit, default=default)}
