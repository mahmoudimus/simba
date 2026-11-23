"""Billion Context adapter: notice its folds and keep simba's rules through them.

Billion Context (bili) compresses the live context itself, so the host's
compaction hook rarely or never fires while it runs (pi does not compact;
``bili claude`` sets ``DISABLE_AUTO_COMPACT=1``; Codex's compaction is served by
bili's own summaries). simba's PreCompact-driven behaviour — exporting the
transcript and re-injecting CORE rules — therefore has to key on bili's folds.

What this module does, from the Stop hook (``on_turn_end``):

1. Finds bili for the session. The host process's ``BILLION_CONTEXT_PROXY``
   (set by bili's launchers and by its pi-native bootstrap once the in-process
   proxy is up) reaches simba as the hook payload's ``bili_origin``; the daemon's
   own environment and bili's ``proxy-origin`` state file are fallbacks.
2. Detects a fold: the set of active blocks for the conversation changed. It
   asks ``GET /__bili/plugin/status`` (documented plugin protocol) and falls
   back to reading bili's SQLite store read-only when no proxy is reachable.
   For Codex it also counts bili's ``[bili] context summary after compaction:``
   marker in the rollout tail. The first observation of a conversation only
   records a baseline.
3. On a fold: resets the rules-signal flag (CORE is re-injected on the next
   prompt) and hands the transcript to the PreCompact export in a detached
   process (trigger ``bili_fold``).
4. Keeps CORE rules as bili rules: when the proxy advertises ``acp_rule``
   (``compress.rules: true``), each CORE rule is recorded once per conversation
   as a ``[simba] ``-prefixed rule, which bili protects from folds. Re-synced
   when the CORE text changes.

Everything is fail-soft and bounded: sub-second HTTP timeouts, a read-only
SQLite query on two small indexed tables, a 256 KiB rollout tail, and a small
JSON state file per conversation under ``~/.cache/simba/bili``.
See docs/adr/2026-11-23-billion-context-adapter.md.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pathlib
import re
import sqlite3
import subprocess
import sys
import time
import typing

HTTP_TIMEOUT = 2.0
RULE_PREFIX = "[simba] "
RULE_MAX_CHARS = 300  # bili's per-rule limit
MAX_RULES = 40  # bili allows 50; leave room for the user's own rules
CODEX_FOLD_MARKER = b"[bili] context summary after compaction:"
TAIL_BYTES = 256 * 1024
_LIST_MARKER = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
ACTIVE_SECONDS = 6 * 3600


def state_dir() -> pathlib.Path:
    return pathlib.Path.home() / ".cache" / "simba" / "bili"


def bili_state_dir() -> pathlib.Path:
    base = os.environ.get("XDG_STATE_HOME") or str(
        pathlib.Path.home() / ".local" / "state"
    )
    return pathlib.Path(base) / "billion-context"


def bili_db_path() -> pathlib.Path:
    base = os.environ.get("XDG_DATA_HOME") or str(
        pathlib.Path.home() / ".local" / "share"
    )
    return pathlib.Path(base) / "billion-context" / "sessions" / "index.db"


# ---------------------------------------------------------------------------
# Discovery and HTTP
# ---------------------------------------------------------------------------


def proxy_origin(payload: dict[str, typing.Any] | None = None) -> str | None:
    """bili's proxy origin for this session, or ``None``."""
    for raw in (
        (payload or {}).get("bili_origin"),
        os.environ.get("BILLION_CONTEXT_PROXY"),
    ):
        if isinstance(raw, str) and raw.strip().startswith(("http://", "https://")):
            return raw.strip().rstrip("/")
    try:
        text = (bili_state_dir() / "proxy-origin").read_text().strip()
    except OSError:
        return None
    with contextlib.suppress(ValueError):
        text = str(json.loads(text).get("origin", ""))
    return text.rstrip("/") if text.startswith(("http://", "https://")) else None


_client: typing.Any = None


def _http_client() -> typing.Any:
    """One reused client: a fresh one costs ~0.6 s of setup per call. bili's
    proxy is always local, so environment proxies are ignored."""
    global _client
    if _client is None:
        import httpx

        _client = httpx.Client(timeout=HTTP_TIMEOUT, trust_env=False)
    return _client


def _http(
    method: str, origin: str, path: str, *, params=None, body=None
) -> dict[str, typing.Any] | None:
    import httpx

    try:
        resp = _http_client().request(
            method, f"{origin}{path}", params=params, json=body
        )
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    with contextlib.suppress(ValueError):
        data = resp.json()
        return data if isinstance(data, dict) else None
    return None


def status(origin: str, conversation_id: str) -> dict[str, typing.Any] | None:
    return _http(
        "GET",
        origin,
        "/__bili/plugin/status",
        params={"conversationId": conversation_id},
    )


def manifest(origin: str) -> dict[str, typing.Any] | None:
    return _http("GET", origin, "/__bili/plugin/manifest")


def call_tool(
    origin: str, conversation_id: str, tool: str, args: dict[str, typing.Any]
) -> dict[str, typing.Any] | None:
    return _http(
        "POST",
        origin,
        "/__bili/plugin/tool",
        body={"conversationId": conversation_id, "tool": tool, "args": args},
    )


# ---------------------------------------------------------------------------
# Fold signatures
# ---------------------------------------------------------------------------


def http_signature(st: dict[str, typing.Any]) -> str | None:
    """Active block ids from a status response, as a stable signature."""
    blocks = st.get("blocks")
    if not isinstance(blocks, list):
        return None
    active = sorted(
        str(b.get("id"))
        for b in blocks
        if isinstance(b, dict) and b.get("active", True)
    )
    return ",".join(active)


def db_signature(conversation_id: str, db: pathlib.Path | None = None) -> str | None:
    """Active block ids from bili's SQLite store (read-only), or ``None``."""
    path = db or bili_db_path()
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=0.5)
    except sqlite3.Error:
        return None
    try:
        conn.execute("PRAGMA query_only = 1")
        session_ids = [conversation_id]
        with contextlib.suppress(sqlite3.Error):
            session_ids += [
                row[0]
                for row in conn.execute(
                    "SELECT session_id FROM conversations WHERE conversation_id = ?"
                    " LIMIT 4",
                    (conversation_id,),
                )
            ]
        for session_id in dict.fromkeys(session_ids):
            rows = conn.execute(
                "SELECT block_id FROM blocks WHERE session_id = ? AND active = 1"
                " ORDER BY block_id LIMIT 5000",
                (session_id,),
            ).fetchall()
            if rows:
                # Same shape as http_signature, so a turn that falls back from
                # HTTP to the store does not look like a fold.
                return ",".join(sorted(str(r[0]) for r in rows))
        return None
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def codex_marker_count(transcript: pathlib.Path) -> int | None:
    """How many bili compaction markers sit in the rollout's last 256 KiB."""
    try:
        with open(transcript, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - TAIL_BYTES))
            return handle.read().count(CODEX_FOLD_MARKER)
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Per-conversation state
# ---------------------------------------------------------------------------


def _state_path(conversation_id: str) -> pathlib.Path:
    digest = hashlib.sha1(conversation_id.encode()).hexdigest()[:20]
    return state_dir() / f"{digest}.json"


def _load(conversation_id: str) -> dict[str, typing.Any]:
    try:
        data = json.loads(_state_path(conversation_id).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(conversation_id: str, data: dict[str, typing.Any]) -> None:
    path = _state_path(conversation_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def is_active(conversation_id: str) -> bool:
    """Whether bili was seen managing this conversation in the last 6 hours."""
    if not conversation_id:
        return False
    seen = _load(conversation_id).get("seen_at", 0)
    return isinstance(seen, (int, float)) and time.time() - seen < ACTIVE_SECONDS


# ---------------------------------------------------------------------------
# CORE rules as bili rules
# ---------------------------------------------------------------------------


def core_rules(cwd: pathlib.Path) -> list[str]:
    """CORE rule lines for *cwd*, each at most ``RULE_MAX_CHARS`` with prefix."""
    import simba.guardian.extract_core as core

    try:
        cfg = core._cfg(cwd)
        blocks = core._read_core_blocks(cwd, cfg.core_filename)
    except Exception:
        return []
    rules: list[str] = []
    for block in blocks:
        for raw in block.content.splitlines():
            line = _LIST_MARKER.sub("", raw.strip()).strip()
            if len(line) < 12 or line.startswith(("#", "<", "```", "|")):
                continue
            text = RULE_PREFIX + core._compact_rule(
                line, RULE_MAX_CHARS - len(RULE_PREFIX)
            )
            if text not in rules:
                rules.append(text)
            if len(rules) >= MAX_RULES:
                return rules
    return rules


def sync_rules(
    origin: str, conversation_id: str, rules: list[str]
) -> tuple[int, str | None]:
    """Record missing ``[simba]`` rules; return (added, reason-if-skipped)."""
    info = manifest(origin)
    if info is None:
        return 0, "proxy unreachable"
    if "acp_rule" not in (info.get("toolNames") or []):
        return 0, "acp_rule disabled (compress.rules is off)"
    listing = call_tool(origin, conversation_id, "acp_rule", {})
    if listing is None:
        return 0, "conversation unknown to bili"
    existing = str(listing.get("result", ""))
    added = 0
    for rule in rules:
        if rule in existing:
            continue
        resp = call_tool(origin, conversation_id, "acp_rule", {"rule": rule})
        if resp is None or resp.get("outcome") == "failure":
            break
        added += 1
    return added, None


# ---------------------------------------------------------------------------
# Hook entry point
# ---------------------------------------------------------------------------


def _spawn_export(hook_input: dict[str, typing.Any]) -> None:
    payload = {
        "session_id": hook_input.get("session_id", ""),
        "transcript_path": hook_input.get("transcript_path", ""),
        "cwd": hook_input.get("cwd", ""),
        "trigger": "bili_fold",
        "source": hook_input.get("source", ""),
    }
    cmd = [sys.executable, "-m", "simba", "hook-canonical", "pre_compact"]
    child = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    with contextlib.suppress(OSError):
        assert child.stdin is not None
        child.stdin.write(json.dumps(payload).encode())
        child.stdin.close()


def on_turn_end(hook_input: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Stop-hook step. Returns what happened (for logs and tests)."""
    conversation_id = str(hook_input.get("session_id") or "")
    if not conversation_id:
        return {"bili": False}
    origin = proxy_origin(hook_input)
    signature: str | None = None
    st = status(origin, conversation_id) if origin else None
    if st is not None:
        signature = http_signature(st)
    if signature is None:
        signature = db_signature(conversation_id)
    transcript = pathlib.Path(str(hook_input.get("transcript_path") or ""))
    markers = codex_marker_count(transcript) if transcript.name else None
    if signature is None and not markers:
        return {"bili": False}

    state = _load(conversation_id)
    folded = False
    if signature is not None and "signature" in state:
        folded = signature != state["signature"]
    if markers:
        # A marker newer than the last count is a fold. An unseen conversation
        # whose tail already holds one counts too (at worst one extra export
        # after a resume), so the first Codex fold is never missed.
        folded = folded or markers > state.get("markers", 0)
    if signature is not None:
        state["signature"] = signature
    if markers is not None:
        state["markers"] = markers
    state["seen_at"] = time.time()

    if folded:
        if not hook_input.get("source"):
            with contextlib.suppress(Exception):
                import simba.hooks.pre_compact as pre_compact

                hook_input = {**hook_input, "source": pre_compact._export_source()}
        with contextlib.suppress(Exception):
            import simba.guardian.signal_flag as signal_flag

            signal_flag.reset_signal(conversation_id)
        with contextlib.suppress(Exception):
            _spawn_export(hook_input)

    rules_added = 0
    recently_skipped = time.time() - state.get("rules_checked_at", 0) < 3600
    if origin and hook_input.get("cwd") and not recently_skipped:
        rules = core_rules(pathlib.Path(str(hook_input["cwd"])))
        digest = hashlib.sha1("\n".join(rules).encode()).hexdigest()
        if rules and state.get("rules_digest") != digest:
            rules_added, skipped = sync_rules(origin, conversation_id, rules)
            if skipped is None:
                state["rules_digest"] = digest
                state.pop("rules_skipped", None)
                state.pop("rules_checked_at", None)
            else:  # re-check at most hourly, e.g. until compress.rules is on
                state["rules_skipped"] = skipped
                state["rules_checked_at"] = time.time()
    _save(conversation_id, state)
    return {"bili": True, "folded": folded, "rules_added": rules_added}
