"""PreCompact hook — export transcript as JSONL + markdown.

Reads stdin JSON with session_id and transcript_path, parses the
transcript into markdown, writes to ~/.claude/transcripts/{sessionId}/,
and creates metadata for later learning extraction.

2026-07-17 incident: the daemon runs this hook IN-PROCESS (POST
/hook/pre_compact), and the export used to ``read_text()`` the whole
transcript. Claude Code transcripts run 10-25MB+, several sessions compact
concurrently and repeatedly, and ``read_text()`` decodes the whole file into
one string WHILE HOLDING THE GIL, then ``.strip()`` copies it again, then
``.split("\\n")`` explodes it into tens of thousands of line strings —
several transient full-file copies per firing, several firings in flight at
once. RSS ballooned to 13GB peak within 30-60s of every boot, the RSS
watchdog restarted the daemon, sessions re-fired their hooks, repeat (a
~5-minute crash loop). The export now (1) streams line-by-line instead of
slurping, (2) skips re-exporting an unchanged transcript, (3) skips
transcripts over ``hooks.pre_compact_max_transcript_mb``, and (4) single
flights concurrent exports for the same session.
"""

from __future__ import annotations

import contextlib
import json
import logging
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import typing

import simba.config
import simba.guardian.signal_flag
import simba.hooks._memory_client
from simba.harness.core import CanonicalResult

logger = logging.getLogger("simba.hooks.pre_compact")

# Single-flight guard (part 4): two concurrent POST /hook/pre_compact for the
# SAME session_id must not both run the export — the daemon dispatches each
# hook call on a threadpool thread (see memory/routes.py::run_hook), so this
# is a plain in-process lock, not an asyncio primitive. Different sessions
# still export concurrently; that's fine (they touch different session_dirs).
_INFLIGHT_LOCK = threading.Lock()
_INFLIGHT_SESSIONS: set[str] = set()


def _try_acquire_export_slot(session_id: str) -> bool:
    with _INFLIGHT_LOCK:
        if session_id in _INFLIGHT_SESSIONS:
            return False
        _INFLIGHT_SESSIONS.add(session_id)
        return True


def _release_export_slot(session_id: str) -> None:
    with _INFLIGHT_LOCK:
        _INFLIGHT_SESSIONS.discard(session_id)


def _hooks_cfg():
    import simba.hooks.config  # registers the "hooks" section

    return simba.config.load("hooks")


class _TranscriptScan:
    """Mutable accumulator for session_id/cwd/msg_count during a transcript
    scan — shared state between ``_iter_message_blocks`` and its two
    callers (the legacy list-based parser and the streaming writer)."""

    __slots__ = ("cwd", "msg_count", "session_id")

    def __init__(self) -> None:
        self.session_id = ""
        self.cwd = ""
        self.msg_count = 0


def _entry_to_block(entry: dict) -> str | None:
    """Build one markdown message block for a decoded transcript entry, or
    None if the entry doesn't contribute a message (tool results, unknown
    roles, blank content, ...). Shared by the legacy list-based parser and
    the streaming writer so both produce byte-identical output.
    """
    message = entry.get("message", {})
    if not isinstance(message, dict):
        # Check for top-level content/text
        if "toolUseResult" in entry:
            return None
        text = entry.get("text") or entry.get("content")
        if text and isinstance(text, str):
            return f"<user>\n{text}\n</user>"
        return None

    role = message.get("role", "")
    content = message.get("content", "")

    if role == "user":
        if isinstance(content, str):
            user_text = content
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    val = item.get("text", "") or item.get("content", "")
                    if isinstance(val, str):
                        parts.append(val)
                elif isinstance(item, str):
                    parts.append(item)
            user_text = "\n".join(p for p in parts if p)
        else:
            return None
        if user_text:
            return f"<user>\n{user_text}\n</user>"
        return None

    if role == "assistant":
        thinking_parts: list[str] = []
        response_parts: list[str] = []

        if isinstance(content, str):
            response_parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "thinking":
                    thinking_parts.append(item.get("thinking", ""))
                elif item.get("type") == "text":
                    response_parts.append(item.get("text", ""))

        # Also check top-level thinking field
        if entry.get("thinking"):
            thinking_parts.append(entry["thinking"])

        assistant_block = "<assistant>\n"
        if thinking_parts:
            thinking_text = "\n".join(t for t in thinking_parts if t)
            assistant_block += f"<thinking>\n{thinking_text}\n</thinking>\n"
        response_text = "\n".join(r for r in response_parts if r)
        if response_text:
            assistant_block += f"<response>\n{response_text}\n</response>\n"
        assistant_block += "</assistant>"
        return assistant_block

    return None


def _iter_message_blocks(
    lines: typing.Iterable[str], scan: _TranscriptScan
) -> typing.Iterator[str]:
    """Yield one markdown block per transcript entry, updating *scan* in
    place. Single pass over *lines* — works identically whether *lines* is
    an in-memory list (the legacy ``_parse_transcript_to_markdown`` callers)
    or a line-by-line file iterator (the streaming export path), since both
    are plain iterables of str.
    """
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue

        if not scan.session_id:
            scan.session_id = entry.get("session_id", "")
        if not scan.cwd:
            scan.cwd = entry.get("cwd", "")

        block = _entry_to_block(entry)
        if block is None:
            continue
        scan.msg_count += 1
        yield block


def _parse_transcript_to_markdown(lines: list[str]) -> tuple[str, int]:
    """Parse JSONL transcript lines into markdown. Returns (md, msg_count).

    Kept for callers that already hold the transcript in memory (e.g. unit
    tests exercising a handful of lines). The production export path uses
    ``_stream_transcript_to_markdown`` instead, which shares
    ``_iter_message_blocks``/``_entry_to_block`` with this function so the
    two produce byte-identical output without ever materializing the whole
    file — or the whole resulting markdown — in memory.
    """
    scan = _TranscriptScan()
    messages = list(_iter_message_blocks(lines, scan))

    # Build final markdown
    md_parts = [
        "<session-transcript>",
        "<metadata>",
        f"  <session-id>{scan.session_id}</session-id>",
        f"  <project-path>{scan.cwd}</project-path>",
        "</metadata>",
        "",
    ]
    md_parts.extend(messages)
    md_parts.append("</session-transcript>")
    return "\n\n".join(md_parts), scan.msg_count


def _stream_transcript_to_markdown(
    transcript_path: pathlib.Path, dest_md: pathlib.Path
) -> tuple[str, str, int]:
    """Stream *transcript_path* straight into *dest_md* as markdown.

    Single pass, line-by-line: each message block is written to a scratch
    file as soon as it's parsed, so peak memory is bounded by one
    line/block, not the transcript or the resulting markdown. The header
    (session-id/project-path) is only known once the pass completes, so it's
    written to *dest_md* first, followed by a streamed copy of the scratch
    file's content — never a full in-memory join of the whole document.
    Shares ``_iter_message_blocks``/``_entry_to_block`` with
    ``_parse_transcript_to_markdown`` (byte-for-byte identical output).

    Returns (session_id, cwd, msg_count) as parsed from the transcript.
    """
    scan = _TranscriptScan()
    tail = "</session-transcript>"

    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as scratch:
        first = True
        with transcript_path.open(encoding="utf-8", errors="replace") as fh:
            for block in _iter_message_blocks(fh, scan):
                if not first:
                    scratch.write("\n\n")
                scratch.write(block)
                first = False
        if not first:
            scratch.write("\n\n")
        scratch.write(tail)
        scratch.seek(0)

        header = (
            "<session-transcript>\n\n"
            "<metadata>\n\n"
            f"  <session-id>{scan.session_id}</session-id>\n\n"
            f"  <project-path>{scan.cwd}</project-path>\n\n"
            "</metadata>\n\n"
        )
        with dest_md.open("w", encoding="utf-8") as out:
            out.write(header)
            out.write("\n\n")
            shutil.copyfileobj(scratch, out)

    return scan.session_id, scan.cwd, scan.msg_count


def _maybe_dispatch_rlm_digest(session_id: str, cwd: str, msg_count: int) -> None:
    """Opt-in: dispatch the autonomous RLM engine to digest this transcript.

    No-op unless rlm.engine != 'claude'. Rate-limited by msg_count and deduped
    via rlm_jobs. The engine runs detached, so this never blocks the hook.
    """
    import simba.config
    import simba.rlm.config  # registers the "rlm" section
    import simba.rlm.engine
    import simba.rlm.jobs

    cfg = simba.config.load("rlm")
    engine = simba.rlm.engine.get_engine(cfg)
    if engine is None:
        return
    if msg_count < cfg.engine_min_new_exchanges:
        return
    project = cwd or str(pathlib.Path.cwd())
    if not simba.rlm.jobs.claim(session_id, project, cfg.engine):
        return
    engine.digest(session_id, "", cwd=project)


def _maybe_consolidate_episodes(cwd: str) -> None:
    """Opt-in: roll eligible past sessions into EPISODE memories (detached).

    No-op unless episodes.auto_on_precompact and an RLM engine is configured.
    Consolidates *eligible* sessions (>= min_memories, no EPISODE yet) for this
    project — naturally deferring the just-ended session until its memories land.
    """
    import simba.config
    import simba.episodes.config  # registers the "episodes" section
    import simba.episodes.consolidate

    ecfg = simba.config.load("episodes")
    if not ecfg.enabled or not ecfg.auto_on_precompact:
        return
    simba.episodes.consolidate.consolidate_eligible(
        cwd or str(pathlib.Path.cwd()), ecfg=ecfg
    )


def _spawn_distiller_detached(
    session_id: str, transcript_path: pathlib.Path, cwd_str: str
) -> bool:
    """Spawn ``simba transcript distill`` DETACHED for a transcript over
    ``hooks.pre_compact_max_transcript_mb`` -- a bounded replacement for the
    blind skip in ``_export_transcript`` (see ``hooks.pre_compact_distill_enabled``
    in ``hooks/config.py`` and ``transcripts/distill.py``'s module docstring
    for the full incident/rationale).

    This runs IN the daemon process (same GIL as every other request), so it
    does strictly: a marker check (cheap -- one small JSON file) and a
    ``Popen``. It must NEVER parse the transcript itself -- that single-pass
    scan happens entirely inside the detached subprocess. stdout/stderr go to
    an append-mode ``distill.log`` under the session export dir (never
    ``DEVNULL`` -- mirrors ``session_start.py``'s daemon-log rationale: a
    crashed distiller must leave a trace).

    Returns True iff the subprocess was actually spawned (marker-match skip,
    an I/O setup failure, or a Popen failure all return False) -- callers use
    this to decide whether a compact-relay ``systemMessage`` is warranted.
    """
    import simba.transcripts
    import simba.transcripts.distill as distill

    session_dir = simba.transcripts.default_transcripts_dir() / session_id
    try:
        src_size = transcript_path.stat().st_size
    except OSError:
        return False
    if distill.marker_matches(session_dir, transcript_path, src_size):
        logger.debug(
            "[pre-compact] distill marker already matches for %s; skipping spawn",
            transcript_path,
        )
        return False

    try:
        session_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(session_dir / "distill.log", "a")  # noqa: SIM115 -- closed below
    except OSError:
        return False
    spawned = False
    try:
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "simba",
                "transcript",
                "distill",
                str(transcript_path),
                "--session-id",
                session_id,
                "--out",
                str(session_dir),
                "--project-path",
                cwd_str,
            ],
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
        spawned = True
    except OSError:
        logger.warning(
            "[pre-compact] failed to spawn distiller for %s", transcript_path
        )
    finally:
        log_file.close()
    return spawned


def _export_transcript(
    session_id: str, transcript_path: pathlib.Path, cwd_str: str
) -> tuple[int | None, str]:
    """Export transcript.jsonl/.md/metadata.json for one PreCompact firing.

    Returns ``(msg_count, system_message)``. ``msg_count`` is None if the
    export was skipped — the transcript is unchanged since the last export
    (idempotent re-compaction), over ``hooks.pre_compact_max_transcript_mb``,
    or an I/O error occurred. Fail-soft either way: callers still return
    their normal hook result. ``system_message`` is a terse one-line note for
    the human (compact relay leg A) -- "" unless something actually happened
    (a real export, or an over-cap distiller spawn); a skipped/idempotent/
    errored export never gets one.
    """
    cfg = _hooks_cfg()

    try:
        src_size = transcript_path.stat().st_size
    except OSError:
        return None, ""

    cap_mb = cfg.pre_compact_max_transcript_mb
    if cap_mb and src_size > cap_mb * 1_000_000:
        logger.warning(
            "[pre-compact] transcript %s is %.1fMB, over "
            "hooks.pre_compact_max_transcript_mb=%.1fMB; skipping export",
            transcript_path,
            src_size / 1_000_000,
            cap_mb,
        )
        system_message = ""
        if cfg.pre_compact_distill_enabled:
            with contextlib.suppress(Exception):
                if _spawn_distiller_detached(session_id, transcript_path, cwd_str):
                    system_message = (
                        f"simba: transcript {src_size / 1_000_000:.1f}MB over cap "
                        "-> distiller spawned"
                    )
        return None, system_message

    transcripts_dir = pathlib.Path.home() / ".claude" / "transcripts"
    session_dir = transcripts_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    dest_jsonl = session_dir / "transcript.jsonl"
    dest_md = session_dir / "transcript.md"

    if dest_jsonl.exists():
        try:
            unchanged = dest_jsonl.stat().st_size == src_size
        except OSError:
            unchanged = False
        if unchanged:
            logger.debug(
                "[pre-compact] transcript %s unchanged since last export "
                "(%d bytes); skipping re-export",
                transcript_path,
                src_size,
            )
            return None, ""

    # 1. Copy original JSONL (kernel copy, no decode — safe at any size).
    with contextlib.suppress(OSError):
        shutil.copy2(transcript_path, dest_jsonl)

    # 2. Stream-parse into markdown (bounded memory — see
    #    _stream_transcript_to_markdown's docstring for why).
    try:
        _session_id_in_transcript, _cwd_in_transcript, msg_count = (
            _stream_transcript_to_markdown(transcript_path, dest_md)
        )
    except OSError:
        return None, ""

    # 3. Write metadata
    metadata = {
        "session_id": session_id,
        "project_path": cwd_str,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "transcript_path": str(dest_md),
        "daemon_url": simba.hooks._memory_client.daemon_url(),
        "status": "pending_extraction",
    }
    metadata_path = session_dir / "metadata.json"
    with contextlib.suppress(OSError):
        metadata_path.write_text(json.dumps(metadata, indent=2))

    # 4. Symlink latest
    latest = transcripts_dir / "latest.json"
    try:
        latest.unlink(missing_ok=True)
        latest.symlink_to(metadata_path)
    except OSError:
        pass

    # Log to stderr
    print(f"[pre-compact] Exported transcript ({msg_count} messages)", file=sys.stderr)
    print(f"[pre-compact] Transcript: {dest_md}", file=sys.stderr)

    system_message = f"simba: exported {msg_count} messages -> {session_dir}"
    return msg_count, system_message


def run(hook_input: dict) -> CanonicalResult:
    """Run the PreCompact hook pipeline. Returns a CanonicalResult."""
    session_id = hook_input.get("session_id") or hook_input.get("sessionId") or ""
    transcript_path_str = (
        hook_input.get("transcript_path") or hook_input.get("transcriptPath") or ""
    )
    cwd_str = hook_input.get("cwd", "")

    # Reset the rules-signal flag (spec 25): the model loses context across a
    # compaction, so the next prompt must re-inject the CORE block. Fail-soft.
    if session_id:
        with contextlib.suppress(Exception):
            simba.guardian.signal_flag.reset_signal(session_id)

    if not session_id or not transcript_path_str:
        return CanonicalResult(suppress_output=True)

    transcript_path = pathlib.Path(transcript_path_str)
    if not transcript_path.exists():
        return CanonicalResult(suppress_output=True)

    if not _try_acquire_export_slot(session_id):
        # Another thread is already exporting this same session (the daemon
        # dispatches concurrent /hook/pre_compact calls on a threadpool) —
        # never run two exports for one session_id at once.
        logger.debug(
            "[pre-compact] export already in flight for session=%s; skipping",
            session_id,
        )
        return CanonicalResult(suppress_output=True)
    try:
        msg_count, system_message = _export_transcript(
            session_id, transcript_path, cwd_str
        )
    finally:
        _release_export_slot(session_id)

    # 5/6. Opt-in helpers — only when the export actually ran (msg_count is
    #      not None) and the payload carried a cwd. A skipped export (cap,
    #      idempotent unchanged-size, single-flight, or I/O error) has no
    #      fresh msg_count to rate-limit on, and — for the idempotent/
    #      single-flight cases — there's nothing new to digest/consolidate
    #      anyway (or another in-flight call already will). Both helpers
    #      fall back to str(Path.cwd()) on an empty cwd, which inside the
    #      daemon would digest/consolidate against the wrong project.
    if msg_count is not None and cwd_str:
        # 5. Autonomous RLM digest (opt-in via rlm.engine; detached, never blocks)
        with contextlib.suppress(Exception):
            _maybe_dispatch_rlm_digest(session_id, cwd_str, msg_count)

        # 6. Episodic consolidation (opt-in via episodes.auto_on_precompact; detached)
        with contextlib.suppress(Exception):
            _maybe_consolidate_episodes(cwd_str)

    return CanonicalResult(suppress_output=True, system_message=system_message)


def main(hook_input: dict) -> str:
    """Run the PreCompact hook and render the Claude/Codex envelope."""
    import simba.harness.adapters.claude as claude

    return claude.render("PreCompact", run(hook_input))
