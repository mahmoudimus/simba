"""Per-turn / per-session usage-signal capture (spec 33 Phase 1).

The audit found the ledger's ``use``/``noise`` counters had NO writer — the one
signal decay needs to mean anything. This module supplies the hook-side half:
``UserPromptSubmit`` records what was injected this turn (id + the memory's
distinctive terms), ``Stop`` checks the response for term overlap (citation ⇒
``use``) and sweeps repeat-injected-never-used ids (⇒ ``noise``), and a fired
``PreToolUse`` rule gate counts as a use directly.

Two session-scoped tempfiles (the guardian-flag pattern — no daemon/DB):
  ``claude-usage-turn-<sid>.json``    ``[{"id", "terms"}]``       (per turn)
  ``claude-usage-session-<sid>.json`` ``{"counts","used","noised"}`` (session)

Citation detection is DETERMINISTIC (no LLM): distinctive terms come from the
entropy-terms machinery (identifier shape + general-English rarity), matched
whole-token in the response — evidence the memory was consumed, not merely
on-topic. Everything fail-softs; callers gate on
``hooks.usage_signals_enabled`` (default off → byte-identical to today).
"""

from __future__ import annotations

import contextlib
import json
import pathlib
import tempfile

import simba.memory.entropy_terms as entropy_terms

# Module-level so tests can monkeypatch to a tmp_path (mirrors signal_flag).
_TMP_DIR = pathlib.Path(tempfile.gettempdir())
_TURN_PREFIX = "claude-usage-turn-"
_SESSION_PREFIX = "claude-usage-session-"
_MAX_TERMS = 6


def _safe(session_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)


def turn_path(session_id: str) -> pathlib.Path:
    """Per-turn record path (overwritten each prompt, consumed at Stop)."""
    return _TMP_DIR / f"{_TURN_PREFIX}{_safe(session_id)}.json"


def session_path(session_id: str) -> pathlib.Path:
    """Per-session accumulator path (inject counts, used/noised sets)."""
    return _TMP_DIR / f"{_SESSION_PREFIX}{_safe(session_id)}.json"


def distinctive_terms(text: str, *, limit: int = _MAX_TERMS) -> list[str]:
    """High-information tokens of a memory's content — identifier-shaped or
    rare-English, per the entropy gate. Their whole-token presence in a
    response is evidence the memory was actually consumed."""
    try:
        return entropy_terms.high_entropy_terms(text or "")[:limit]
    except Exception:
        return []


def _read_json(path: pathlib.Path, default):
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return default
    return data if isinstance(data, type(default)) else default


def _read_session(session_id: str) -> dict:
    data = _read_json(session_path(session_id), {})
    counts = data.get("counts") if isinstance(data.get("counts"), dict) else {}
    used = data.get("used") if isinstance(data.get("used"), list) else []
    noised = data.get("noised") if isinstance(data.get("noised"), list) else []
    return {"counts": counts, "used": used, "noised": noised}


def _write_session(session_id: str, data: dict) -> None:
    with contextlib.suppress(OSError, TypeError):
        session_path(session_id).write_text(json.dumps(data))


def record_turn_injections(session_id: str, memories: list[dict]) -> None:
    """Record this turn's injected memories (id + terms); bump session counts."""
    if not session_id or not memories:
        return
    turn: list[dict] = []
    for m in memories:
        mid = m.get("id")
        if not mid:
            continue
        turn.append({"id": mid, "terms": distinctive_terms(m.get("content", ""))})
    if not turn:
        return
    with contextlib.suppress(OSError, TypeError):
        turn_path(session_id).write_text(json.dumps(turn))
    sess = _read_session(session_id)
    for entry in turn:
        mid = entry["id"]
        sess["counts"][mid] = int(sess["counts"].get(mid, 0)) + 1
    _write_session(session_id, sess)


def read_turn(session_id: str) -> list[dict]:
    """This turn's injected ``[{"id", "terms"}]`` (``[]`` when none/unreadable)."""
    if not session_id:
        return []
    return _read_json(turn_path(session_id), [])


def reset_turn(session_id: str) -> None:
    """Consume the per-turn record (idempotent)."""
    if not session_id:
        return
    with contextlib.suppress(OSError):
        turn_path(session_id).unlink(missing_ok=True)


def detect_used(
    response: str, turn_memories: list[dict], *, min_overlap: int = 2
) -> list[str]:
    """Ids whose distinctive terms appear (whole-token) in ``response``.

    A memory with fewer than ``min_overlap`` terms needs ALL of them (and at
    least one); zero-term memories are never citation-detected (no signal —
    prose-only memories don't get spurious credit).
    """
    if not response:
        return []
    used: list[str] = []
    seen: set[str] = set()
    for m in turn_memories:
        mid = m.get("id")
        terms = [t for t in m.get("terms", []) if t]
        if not mid or mid in seen or not terms:
            continue
        needed = max(1, min(min_overlap, len(terms)))
        hits = sum(1 for t in terms if entropy_terms.contains_whole(response, t))
        if hits >= needed:
            seen.add(mid)
            used.append(mid)
    return used


def mark_used(session_id: str, memory_ids: list[str]) -> None:
    """Record ids as used this session (exempts them from the noise sweep)."""
    if not session_id or not memory_ids:
        return
    sess = _read_session(session_id)
    sess["used"] = sorted(set(sess["used"]) | {m for m in memory_ids if m})
    _write_session(session_id, sess)


def sweep_noise(session_id: str, *, min_injects: int = 2) -> list[str]:
    """Ids injected ≥ ``min_injects`` times this session, never used, not yet
    noised. Marks them noised — each id is flagged at most once per session
    (the weak, asymmetric half of the feedback loop)."""
    if not session_id:
        return []
    sess = _read_session(session_id)
    used = set(sess["used"])
    noised = set(sess["noised"])
    out = sorted(
        mid
        for mid, n in sess["counts"].items()
        if int(n) >= min_injects and mid not in used and mid not in noised
    )
    if out:
        sess["noised"] = sorted(noised | set(out))
        _write_session(session_id, sess)
    return out
