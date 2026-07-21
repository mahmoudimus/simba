"""Bounded, single-pass transcript distiller.

``hooks/pre_compact.py`` streams transcript exports but SKIPS transcripts over
``hooks.pre_compact_max_transcript_mb`` (default 256MB) entirely -- see its
module docstring for the 2026-07-17 RSS-storm incident that put that cap
there. Observed live 2026-07-20: a 2153.4MB Codex rollout produced "skipping
export", so nothing was learned from that session at all. These rollouts are
~99% tool-output/telemetry noise; the learning signal (user intent, assistant
decisions, and especially FAILED tool calls with their eventual fixes) is
single-digit MB.

This module is a bounded replacement for that blind skip: a single
line-by-line pass over the transcript that
  1. never slurps the source file (``open(...).__iter__``, not
     ``Path.read_text()`` -- peak memory is one line + the bounded output
     state, not the file);
  2. cheaply prefilters lines that are unambiguously zero-signal telemetry
     (Codex ``token_count``/``world_state``/``turn_context``/encrypted
     ``reasoning`` blocks, ...) via a raw-substring probe BEFORE ``json.loads``;
  3. keeps user/assistant text, tool calls, and tool outputs at small
     role-aware budgets (module-level constants below -- these are NOT
     ``simba config`` knobs, just internal truncation limits);
  4. tracks failure -> fix arcs as the top-priority signal: a failed tool
     call held in a small look-ahead window, resolved by a later success of
     the same tool, or left as an UNRESOLVED dead-end marker if the window
     closes first. Repeats of the same (tool, normalized-error-signature)
     collapse into one arc with a growing ``repeat_count`` instead of N
     separate arcs;
  5. bounds total output to ``hooks.distill_max_output_mb`` (a
     ``simba config`` knob -- resolved by the CLI layer, passed in here as
     a plain float so this module stays config-agnostic and easily
     testable) via tiered degradation: tool-output head/tails go first, then
     tool-call heads, then assistant text, then user text -- failure arcs are
     written from a separately-reserved section and are never subject to
     this body-tier eviction, so they survive even a worst-case budget squeeze.

Two transcript shapes are understood (see ``_classify_codex``/
``_classify_claude`` below for the exact record shapes each was reverse
engineered from):

* **Claude Code** (``~/.claude/projects/*/[session].jsonl``): entries with a
  ``message.role``/``message.content`` shape. Text extraction reuses
  ``hooks.pre_compact._entry_to_block`` (the existing, tested classifier) for
  plain user/assistant text so that logic isn't reinvented here; tool calls
  (``content[].type == "tool_use"``) and tool results
  (``content[].type == "tool_result"``, ``is_error`` bool) are handled here
  since ``_entry_to_block`` doesn't look at either.
* **Codex** (``~/.codex/sessions/**/rollout-*.jsonl``): entries with a
  top-level ``type`` in {``session_meta``, ``response_item``, ``event_msg``,
  ...} and a nested ``payload``. Handled: ``response_item``/``message`` (user/
  assistant text), ``response_item``/``function_call``+``function_call_output``
  and ``response_item``/``custom_tool_call``+``custom_tool_call_output`` (tool
  calls/results, correlated by ``call_id``), and ``event_msg``/
  ``patch_apply_end`` (an explicit ``success`` bool -- the most reliable
  failure signal available). Failure detection for
  function_call_output/custom_tool_call_output primarily keys off the
  observed "Script completed"/"Script failed" wrapper Codex prepends to tool
  output; a generic error-marker regex (reused from ``simba.tailor.hook``)
  is the fallback when that wrapper isn't present.

Error-signature normalization reuses ``simba.tailor.hook.normalize_snippet``
(line numbers -> ``:LINE:COL``, absolute paths -> ``/PATH/``, hex addresses ->
``0xADDR``, long numbers -> ``NUM``) rather than inventing a second token
convention -- it's already the project's single source of truth for
clustering error text.

NO DATABASE ACCESS happens in this module -- the scan is a single pass over
the source file and writes only ``transcript.md`` + ``distill-meta.json``
under ``out_dir``. Persisting the returned ``DistillResult.arcs`` to the
``failure_arc`` sidecar table (``transcripts/arcs.py``) is the CLI layer's
job, done once per distinct arc (a handful of rows), not part of the scan.
"""

from __future__ import annotations

import collections
import dataclasses
import difflib
import json
import pathlib
import re
import time
import typing

import simba.hooks.pre_compact as _pre_compact
import simba.tailor.hook as _tailor_hook
import simba.transcripts.focus as _focus

# ── role-aware budgets (module-level constants, NOT simba config knobs) ────
TEXT_MESSAGE_BUDGET = 4096  # user/assistant message text
TOOL_CALL_ARGS_HEAD_BUDGET = 512  # tool call: name + args head
TOOL_OUTPUT_HEAD_BUDGET = 500  # tool output head (chars)
TOOL_OUTPUT_TAIL_BUDGET = 500  # tool output tail (chars)
THINKING_HEAD_BUDGET = 256  # thinking/reasoning head-only
ARC_ERROR_HEAD_BUDGET = 1024  # failure arc error head
ARC_ARGS_HEAD_BUDGET = 512  # failure arc failed/fix args head

# Failure -> fix arc tracking
ARC_LOOKAHEAD_WINDOW = 25  # tool-call events a failure stays eligible for resolution
ARC_MAX_TRACKED = 2000  # safety valve: distinct (tool, signature) arcs held in memory

# Bounded-output tiers, in DROP ORDER (index 0 dropped first when squeezed).
_TIER_TOOL_OUTPUT = 0
_TIER_TOOL_CALL = 1
_TIER_ASSISTANT = 2
_TIER_USER = 3
_TIER_NAMES = ("tool_output", "tool_call", "assistant", "user")

# Streaming eviction ceiling: the body buffer is trimmed to the true budget
# only ONCE at the end (cheap -- the buffer itself is already bounded), but
# during the scan it's kept from growing without bound relative to the
# source file size by evicting the oldest lowest-tier block whenever this
# multiple of the configured budget is exceeded. The floor keeps a
# near-zero/zero configured budget from starving the pass entirely (arcs and
# a handful of blocks always have room to be considered for the final trim).
_BODY_SOFT_CAP_MULTIPLIER = 2.0
_BODY_SOFT_CAP_FLOOR_BYTES = 500_000

# Final backstop when even an all-arcs, zero-body output still needs
# trimming within a tier: keep the first/last N blocks, drop the middle.
_FIRST_LAST_WINDOW_BLOCKS = 20

_TRUNCATION_MARK = "...[truncated]"


def _head(text: str, n: int) -> str:
    text = text.strip()
    if len(text) <= n:
        return text
    return text[:n].rstrip() + _TRUNCATION_MARK


def _head_tail(text: str, head_n: int, tail_n: int) -> str:
    text = text.strip()
    if len(text) <= head_n + tail_n + 32:
        return text
    omitted = len(text) - head_n - tail_n
    return (
        text[:head_n].rstrip()
        + f"\n...[{omitted} chars omitted]...\n"
        + text[-tail_n:].lstrip()
    )


def normalize_error(text: str) -> str:
    """Normalize error text for clustering -- delegates to the project's
    existing normalizer (``simba.tailor.hook.normalize_snippet``, the
    ``:LINE:COL``/``/PATH/``/``0xADDR``/``NUM`` convention) instead of
    inventing a second one."""
    return _tailor_hook.normalize_snippet(text)


def _looks_like_failure(text: str) -> bool:
    """Generic fallback failure heuristic (reuses tailor's ERROR_PATTERNS)
    for tool outputs that carry no harness-specific success/failure signal."""
    return _tailor_hook.detect_error(text)


def _args_diff_hint(before: str, after: str, *, max_len: int = 200) -> str:
    """A short, bounded hint at what changed between failed and fix args."""
    if before == after:
        return ""
    diff = difflib.unified_diff(
        before.splitlines(), after.splitlines(), lineterm="", n=0
    )
    hint = " ".join(
        line for line in diff if line[:1] in "+-" and line[:3] not in ("+++", "---")
    )
    return _head(hint, max_len) if hint else _head(f"{before!r} -> {after!r}", max_len)


# ── failure -> fix arc tracking ─────────────────────────────────────────────


@dataclasses.dataclass
class ArcRecord:
    tool: str
    signature: str
    failed_args_head: str
    error_head: str
    fix_args_head: str | None = None
    resolved: bool = False
    repeat_count: int = 1
    diff_hint: str = ""


class ArcTracker:
    """Holds a small look-ahead buffer of unresolved failures per tool.

    A later success of the SAME tool within ``window`` tool-call events
    resolves the most recently failed (tool, signature) pending for that
    tool (LIFO -- the most recent failure is the one a following retry is
    presumably fixing). Repeats of an identical (tool, signature) collapse
    into the same ``ArcRecord``, incrementing ``repeat_count``, regardless
    of window expiry (expiry only governs whether a LATER success can still
    attribute itself as the fix -- it never stops counting repeats).
    """

    def __init__(self, window: int = ARC_LOOKAHEAD_WINDOW) -> None:
        self.window = window
        self.arcs: dict[tuple[str, str], ArcRecord] = {}
        self._pending: collections.deque[tuple[str, str, int]] = collections.deque()
        self._call_index = 0

    def _expire(self) -> None:
        while self._pending and self._call_index - self._pending[0][2] > self.window:
            self._pending.popleft()

    def record_failure(self, tool: str, error_text: str, args_head: str) -> None:
        self._call_index += 1
        self._expire()
        signature = _head(normalize_error(error_text), 300)
        key = (tool, signature)
        arc = self.arcs.get(key)
        if arc is None:
            if len(self.arcs) >= ARC_MAX_TRACKED:
                return  # safety valve -- drop new distinct signatures past the cap
            arc = ArcRecord(
                tool=tool,
                signature=signature,
                failed_args_head=_head(args_head, ARC_ARGS_HEAD_BUDGET),
                error_head=_head(error_text, ARC_ERROR_HEAD_BUDGET),
            )
            self.arcs[key] = arc
        else:
            arc.repeat_count += 1
        self._pending.append((tool, signature, self._call_index))

    def record_success(self, tool: str, args_head: str) -> None:
        self._call_index += 1
        self._expire()
        for i in range(len(self._pending) - 1, -1, -1):
            pending_tool, signature, _idx = self._pending[i]
            if pending_tool != tool:
                continue
            arc = self.arcs.get((tool, signature))
            if arc is not None and not arc.resolved:
                arc.resolved = True
                arc.fix_args_head = _head(args_head, ARC_ARGS_HEAD_BUDGET)
                arc.diff_hint = _args_diff_hint(arc.failed_args_head, arc.fix_args_head)
            del self._pending[i]
            return

    def finalize(self) -> list[ArcRecord]:
        return list(self.arcs.values())


# ── bounded, tiered body buffer ─────────────────────────────────────────────


@dataclasses.dataclass
class _Block:
    tier: int
    text: str
    order: int


class BodyBuffer:
    """Bounded accumulator for the non-arc message body.

    Blocks are pre-truncated to their role budget before being appended, so
    each block is already small. To keep total memory genuinely independent
    of source file size (not just "small per record"), a soft cap evicts the
    OLDEST block from the lowest surviving tier whenever total bytes exceed
    ``max(soft_cap_floor, budget_bytes * soft_cap_multiplier)`` -- so a
    pathological transcript with millions of tiny records still can't grow
    the buffer past a small multiple of the configured output budget.
    """

    def __init__(self, budget_bytes: float) -> None:
        self._tiers: list[collections.deque[_Block]] = [
            collections.deque() for _ in _TIER_NAMES
        ]
        self._total_bytes = 0
        self._order = 0
        self._soft_cap = max(
            _BODY_SOFT_CAP_FLOOR_BYTES, budget_bytes * _BODY_SOFT_CAP_MULTIPLIER
        )

    def add(self, tier: int, text: str) -> None:
        block = _Block(tier=tier, text=text, order=self._order)
        self._order += 1
        self._tiers[tier].append(block)
        self._total_bytes += len(text)
        while self._total_bytes > self._soft_cap:
            if not self._evict_one():
                break

    def _evict_one(self) -> bool:
        for tier_blocks in self._tiers:
            if tier_blocks:
                dropped = tier_blocks.popleft()
                self._total_bytes -= len(dropped.text)
                return True
        return False

    def trim_to_budget(self, budget_bytes: float) -> list[str]:
        """Return the surviving blocks' text, in original stream order,
        trimmed to fit ``budget_bytes``: drop whole tiers (lowest first),
        then apply a first-N/last-N window backstop within what remains."""
        blocks = [b for tier_blocks in self._tiers for b in tier_blocks]
        blocks.sort(key=lambda b: b.order)

        def total(bs: list[_Block]) -> int:
            return sum(len(b.text) for b in bs)

        for tier in range(len(_TIER_NAMES)):
            if total(blocks) <= budget_bytes:
                break
            blocks = [b for b in blocks if b.tier != tier]

        if total(blocks) > budget_bytes and len(blocks) > 2 * _FIRST_LAST_WINDOW_BLOCKS:
            head = blocks[:_FIRST_LAST_WINDOW_BLOCKS]
            tail = blocks[-_FIRST_LAST_WINDOW_BLOCKS:]
            omitted = len(blocks) - len(head) - len(tail)
            blocks = [
                *head,
                _Block(
                    tier=-1, text=f"<!-- {omitted} blocks omitted (budget) -->", order=0
                ),
                *tail,
            ]

        # Absolute backstop: even the window can overflow a tiny budget --
        # truncate the joined text itself rather than exceed it unbounded.
        out = [b.text for b in blocks]
        joined = "\n\n".join(out)
        if len(joined) > budget_bytes > 0:
            joined = joined[: int(budget_bytes)] + _TRUNCATION_MARK
        return [joined] if joined else []


# ── harness detection + prefilter ───────────────────────────────────────────

_CODEX_TOP_TYPES = frozenset(
    {
        "session_meta",
        "response_item",
        "event_msg",
        "turn_context",
        "compacted",
        "world_state",
        "inter_agent_communication_metadata",
    }
)

# Raw-substring markers of Codex records that carry zero learning signal:
# pure telemetry (token accounting, UI/thread bookkeeping) or content that's
# never plaintext (reasoning summaries are usually `encrypted_content`).
# Checked BEFORE `json.loads` -- a plain `in` scan of the raw line, so lines
# that will be thrown away never pay for parsing. Safe across both harnesses:
# these are exact `"key":"value"` JSON tokens, which (per JSON string
# escaping) can only appear literally at a real key/value boundary, never
# inside escaped user-authored text.
_PREFILTER_SKIP_MARKERS = (
    '"token_count"',
    '"world_state"',
    '"turn_context"',
    '"thread_settings_applied"',
    '"task_started"',
    '"task_complete"',
    '"inter_agent_communication_metadata"',
    '"context_compacted"',
    '"compacted"',
    '"reasoning"',
    '"sub_agent_activity"',
    '"mcp_tool_call_end"',
)


def _prefilter_skip(line: str) -> bool:
    return any(marker in line for marker in _PREFILTER_SKIP_MARKERS)


def _detect_harness(entry: dict) -> str:
    if isinstance(entry, dict) and entry.get("type") in _CODEX_TOP_TYPES:
        return "codex"
    return "claude-code"


# ── event model shared by both harness classifiers ──────────────────────────


@dataclasses.dataclass
class _Event:
    kind: str  # "user" | "assistant" | "thinking" | "tool_call" | "tool_result"
    name: str = ""  # role (user/assistant) or tool name
    text: str = ""  # message text / tool args text / tool output text
    call_id: str = ""
    is_failure: bool | None = None  # only meaningful for "tool_result"
    args_hint: str = ""  # fallback args-head for a tool_result with no call_id
    # to correlate against (e.g. Codex's self-contained patch_apply_end)


# ── Codex classifier ─────────────────────────────────────────────────────────

_SCRIPT_FAILED_RE = re.compile(r"^\s*Script failed\b", re.IGNORECASE)
_SCRIPT_OK_RE = re.compile(r"^\s*Script completed\b", re.IGNORECASE)


def _codex_output_text_and_failure(output: typing.Any) -> tuple[str, bool | None]:
    if isinstance(output, str):
        return output, None
    if isinstance(output, list):
        parts = [item.get("text", "") for item in output if isinstance(item, dict)]
        joined = "\n".join(p for p in parts if p)
        if parts:
            first = parts[0]
            if _SCRIPT_FAILED_RE.match(first):
                return joined, True
            if _SCRIPT_OK_RE.match(first):
                return joined, False
        return joined, None
    return "", None


def _classify_codex(entry: dict) -> tuple[list[_Event], str, str]:
    """Return (events, session_id_hint, project_path_hint)."""
    etype = entry.get("type")
    payload = entry.get("payload")
    if not isinstance(payload, dict):
        return [], "", ""

    if etype == "session_meta":
        sid = payload.get("id") or payload.get("session_id") or ""
        cwd = payload.get("cwd") or ""
        return [], str(sid), str(cwd)

    if etype == "event_msg" and payload.get("type") == "patch_apply_end":
        success = bool(payload.get("success", True))
        stdout = str(payload.get("stdout") or "")
        stderr = str(payload.get("stderr") or "")
        changes = payload.get("changes")
        paths = sorted(changes.keys()) if isinstance(changes, dict) else []
        args_head = "; ".join(paths) or "apply_patch"
        text = (stderr or stdout).strip()
        return (
            [
                _Event(
                    kind="tool_result",
                    name="apply_patch",
                    text=text or ("ok" if success else "patch failed"),
                    is_failure=not success,
                    args_hint=args_head,
                )
            ],
            "",
            "",
        )

    if etype != "response_item":
        return [], "", ""

    ptype = payload.get("type")

    if ptype == "message":
        role = str(payload.get("role") or "")
        content = payload.get("content")
        parts: list[str] = []
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    val = item.get("text")
                    if isinstance(val, str) and val:
                        parts.append(val)
        text = "\n".join(parts)
        if not text:
            return [], "", ""
        kind = "assistant" if role == "assistant" else "user"
        return [_Event(kind=kind, name=role, text=text)], "", ""

    if ptype == "function_call":
        name = str(payload.get("name") or "")
        call_id = str(payload.get("call_id") or "")
        args = payload.get("arguments")
        args_text = args if isinstance(args, str) else json.dumps(args)
        return (
            [_Event(kind="tool_call", name=name, text=args_text, call_id=call_id)],
            "",
            "",
        )

    if ptype == "custom_tool_call":
        name = str(payload.get("name") or "")
        call_id = str(payload.get("call_id") or "")
        inp = payload.get("input")
        args_text = inp if isinstance(inp, str) else json.dumps(inp)
        return (
            [_Event(kind="tool_call", name=name, text=args_text, call_id=call_id)],
            "",
            "",
        )

    if ptype in ("function_call_output", "custom_tool_call_output"):
        call_id = str(payload.get("call_id") or "")
        text, is_failure = _codex_output_text_and_failure(payload.get("output"))
        if is_failure is None and text:
            is_failure = _looks_like_failure(text)
        return (
            [
                _Event(
                    kind="tool_result",
                    name="",
                    text=text,
                    call_id=call_id,
                    is_failure=is_failure,
                )
            ],
            "",
            "",
        )

    return [], "", ""


# ── Claude Code classifier ───────────────────────────────────────────────────


def _claude_tool_events(entry: dict) -> list[_Event]:
    """Tool call/result events _entry_to_block doesn't handle (it only looks
    at "thinking"/"text" content items, never "tool_use"/"tool_result")."""
    message = entry.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if not isinstance(content, list):
        return []

    events: list[_Event] = []
    tool_result_ids = {
        item.get("tool_use_id")
        for item in content
        if isinstance(item, dict) and item.get("type") == "tool_result"
    }
    top_level_result = entry.get("toolUseResult")

    for item in content:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "tool_use":
            name = str(item.get("name") or "")
            call_id = str(item.get("id") or "")
            args_text = json.dumps(item.get("input") or {})
            events.append(
                _Event(kind="tool_call", name=name, text=args_text, call_id=call_id)
            )
        elif itype == "tool_result":
            call_id = str(item.get("tool_use_id") or "")
            is_error = bool(item.get("is_error", False))
            raw_content = item.get("content")
            text = (
                raw_content if isinstance(raw_content, str) else json.dumps(raw_content)
            )
            # Prefer the richer stdout/stderr sibling payload when this
            # entry's own tool_result is the only one referencing it.
            if (
                is_error
                and isinstance(top_level_result, dict)
                and len(tool_result_ids) == 1
            ):
                stderr = top_level_result.get("stderr")
                stdout = top_level_result.get("stdout")
                richer = (stderr or "") + ("\n" + stdout if stdout else "")
                if richer.strip():
                    text = richer
            events.append(
                _Event(
                    kind="tool_result",
                    name="",
                    text=text or "",
                    call_id=call_id,
                    is_failure=is_error,
                )
            )
    return events


def _classify_claude(entry: dict) -> list[_Event]:
    """Text via the existing, tested ``_entry_to_block`` classifier (reused
    rather than reinvented); tool_use/tool_result via ``_claude_tool_events``
    since ``_entry_to_block`` never looks at either.
    """
    events: list[_Event] = []
    message = entry.get("message") or {}
    content = message.get("content") if isinstance(message, dict) else None
    # A user entry whose content is ENTIRELY tool_result item(s) is a tool
    # reply, not a real user message -- _entry_to_block doesn't distinguish
    # a tool_result's "content" field from a genuine text field (it just
    # extracts any text/content string generically), so calling it here
    # would double-count/mislabel the same text as a fake <user> block on
    # top of the proper tool-output block _claude_tool_events emits below.
    is_pure_tool_result = (
        bool(content)
        and isinstance(content, list)
        and all(
            isinstance(item, dict) and item.get("type") == "tool_result"
            for item in content
        )
    )
    block = None if is_pure_tool_result else _pre_compact._entry_to_block(entry)
    if block:
        role = message.get("role") if isinstance(message, dict) else ""
        if role == "assistant":
            # _entry_to_block wraps thinking+response together; split back out
            # so thinking gets its own (shorter) budget.
            think_match = re.search(r"<thinking>\n(.*?)\n</thinking>", block, re.S)
            resp_match = re.search(r"<response>\n(.*?)\n</response>", block, re.S)
            if think_match:
                events.append(_Event(kind="thinking", text=think_match.group(1)))
            if resp_match:
                events.append(
                    _Event(kind="assistant", name="assistant", text=resp_match.group(1))
                )
            if not think_match and not resp_match:
                events.append(_Event(kind="assistant", name="assistant", text=block))
        else:
            # role == "user", or an unrecognized/fallback shape
            # _entry_to_block still turned into a <user>...</user> block.
            text_match = re.search(r"<user>\n(.*?)\n</user>", block, re.S)
            events.append(
                _Event(
                    kind="user",
                    name="user",
                    text=text_match.group(1) if text_match else block,
                )
            )
    events.extend(_claude_tool_events(entry))
    return events


# ── stats + result ───────────────────────────────────────────────────────────


@dataclasses.dataclass
class DistillStats:
    source_path: str
    source_bytes: int
    harness: str = ""
    session_id: str = ""
    project_path: str = ""
    prefiltered_lines: int = 0
    prefiltered_bytes: int = 0
    unparsed_lines: int = 0
    kept_by_class: dict[str, int] = dataclasses.field(default_factory=dict)
    dropped_by_class: dict[str, int] = dataclasses.field(default_factory=dict)
    arc_resolved_count: int = 0
    arc_unresolved_count: int = 0
    arc_total_repeat_count: int = 0
    output_bytes: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class DistillResult:
    md_path: pathlib.Path
    meta_path: pathlib.Path
    arcs: list[ArcRecord]
    stats: DistillStats
    skipped: bool = False


def _meta_path(out_dir: pathlib.Path) -> pathlib.Path:
    return out_dir / "distill-meta.json"


def marker_matches(
    out_dir: pathlib.Path, source_path: pathlib.Path, source_bytes: int
) -> bool:
    """True when ``out_dir/distill-meta.json`` already reflects a completed
    distill of this exact (source path, source size) -- the idempotence
    check both the pre_compact spawn gate and a direct CLI re-run use so
    re-distilling the same untouched session is a cheap no-op."""
    meta_file = _meta_path(out_dir)
    try:
        meta = json.loads(meta_file.read_text())
    except (OSError, ValueError):
        return False
    return (
        meta.get("source_path") == str(source_path)
        and meta.get("source_bytes") == source_bytes
    )


def _arc_section(arcs: list[ArcRecord]) -> str:
    if not arcs:
        return '<failure-arcs count="0">\n</failure-arcs>'
    parts = [f'<failure-arcs count="{len(arcs)}">']
    for arc in arcs:
        parts.append(
            f'<arc tool="{arc.tool}" resolved="{"true" if arc.resolved else "false"}" '
            f'repeat_count="{arc.repeat_count}">'
        )
        parts.append(f"<signature>{arc.signature}</signature>")
        parts.append(f"<failed-args>{arc.failed_args_head}</failed-args>")
        parts.append(f"<error>{arc.error_head}</error>")
        if arc.resolved:
            parts.append(f"<fix-args>{arc.fix_args_head or ''}</fix-args>")
            if arc.diff_hint:
                parts.append(f"<diff-hint>{arc.diff_hint}</diff-hint>")
        else:
            parts.append("<status>UNRESOLVED (dead end)</status>")
        parts.append("</arc>")
    parts.append("</failure-arcs>")
    return "\n".join(parts)


def distill_transcript(
    source_path: pathlib.Path,
    *,
    out_dir: pathlib.Path,
    session_id: str = "",
    project_path: str = "",
    max_output_mb: float = 12.0,
    force: bool = False,
    focus: str = "",
) -> DistillResult:
    """Single-pass, bounded distillation of ``source_path`` into
    ``out_dir/transcript.md`` (+ ``distill-meta.json``).

    Idempotent: if ``out_dir/distill-meta.json`` already matches this exact
    (source path, source size) and ``force`` is False, returns immediately
    with ``skipped=True`` and no re-scan.

    *focus* is an optional ``/compact`` focus string (``hooks/pre_compact.py``'s
    over-cap spawn forwards it as ``--focus``). It affects ONLY the ordering of
    ``result.arcs`` (and therefore the ``<failure-arcs>`` section of the output
    document): focus-matching arcs are listed first, via the same deterministic
    token-overlap scoring ``hooks/session_start.py`` uses for the compact-relay
    ranking (``transcripts/focus.py`` -- no LLM, no embeddings). It never
    changes the ``failure_arc`` sidecar schema or what gets stored -- the CLI
    layer's ``upsert_arc`` loop over ``result.arcs`` is order-independent. ""
    (the default) -- identical output to before this parameter existed.
    """
    start = time.monotonic()
    source_path = pathlib.Path(source_path)
    out_dir = pathlib.Path(out_dir)
    source_bytes = source_path.stat().st_size

    meta_path = _meta_path(out_dir)
    md_path = out_dir / "transcript.md"
    if not force and marker_matches(out_dir, source_path, source_bytes):
        stats = DistillStats(source_path=str(source_path), source_bytes=source_bytes)
        return DistillResult(
            md_path=md_path, meta_path=meta_path, arcs=[], stats=stats, skipped=True
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    budget_bytes = max(0.0, max_output_mb) * 1_000_000
    body = BodyBuffer(budget_bytes)
    arc_tracker = ArcTracker()
    pending_calls: dict[str, tuple[str, str]] = {}

    harness = ""
    resolved_session_id = session_id
    resolved_project_path = project_path

    kept_by_class: collections.Counter[str] = collections.Counter()
    dropped_by_class: collections.Counter[str] = collections.Counter()
    prefiltered_lines = 0
    prefiltered_bytes = 0
    unparsed_lines = 0

    def _handle_event(ev: _Event) -> None:
        nonlocal harness
        if ev.kind == "user":
            kept_by_class["user"] += 1
            body.add(
                _TIER_USER, f"<user>\n{_head(ev.text, TEXT_MESSAGE_BUDGET)}\n</user>"
            )
        elif ev.kind == "assistant":
            kept_by_class["assistant"] += 1
            body.add(
                _TIER_ASSISTANT,
                f"<assistant>\n{_head(ev.text, TEXT_MESSAGE_BUDGET)}\n</assistant>",
            )
        elif ev.kind == "thinking":
            kept_by_class["thinking"] += 1
            head = _head(ev.text, THINKING_HEAD_BUDGET)
            if head:
                body.add(_TIER_ASSISTANT, f"<thinking>\n{head}\n</thinking>")
        elif ev.kind == "tool_call":
            kept_by_class["tool_call"] += 1
            args_head = _head(ev.text, TOOL_CALL_ARGS_HEAD_BUDGET)
            body.add(
                _TIER_TOOL_CALL,
                f'<tool-call name="{ev.name}">\n{args_head}\n</tool-call>',
            )
            if ev.call_id:
                pending_calls[ev.call_id] = (ev.name, args_head)
        elif ev.kind == "tool_result":
            kept_by_class["tool_result"] += 1
            tool_name, args_head = pending_calls.pop(
                ev.call_id, (ev.name, ev.args_hint)
            )
            out_text = _head_tail(
                ev.text, TOOL_OUTPUT_HEAD_BUDGET, TOOL_OUTPUT_TAIL_BUDGET
            )
            body.add(_TIER_TOOL_OUTPUT, f"<tool-output>\n{out_text}\n</tool-output>")
            if ev.is_failure is True:
                arc_tracker.record_failure(tool_name or "unknown", ev.text, args_head)
            elif ev.is_failure is False:
                arc_tracker.record_success(tool_name or "unknown", args_head)

    with source_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if _prefilter_skip(stripped):
                prefiltered_lines += 1
                prefiltered_bytes += len(stripped)
                continue
            try:
                entry = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                unparsed_lines += 1
                continue
            if not isinstance(entry, dict):
                unparsed_lines += 1
                continue

            if not harness:
                harness = _detect_harness(entry)

            if harness == "codex":
                events, sid_hint, cwd_hint = _classify_codex(entry)
                if not resolved_session_id and sid_hint:
                    resolved_session_id = sid_hint
                if not resolved_project_path and cwd_hint:
                    resolved_project_path = cwd_hint
            else:
                events = _classify_claude(entry)

            if not events:
                dropped_by_class["skipped"] += 1
                continue
            for ev in events:
                _handle_event(ev)

    arcs = arc_tracker.finalize()
    focus_tokens = set(_focus.tokenize(focus)) if focus else set()
    if focus_tokens:
        arcs.sort(
            key=lambda a: (
                -_focus.score_overlap(
                    focus_tokens, f"{a.signature} {a.fix_args_head or ''}"
                ),
                not a.resolved,
                -a.repeat_count,
            )
        )
    else:
        arcs.sort(key=lambda a: (not a.resolved, -a.repeat_count))
    arc_section = _arc_section(arcs)
    remaining = max(0.0, budget_bytes - len(arc_section))
    body_parts = body.trim_to_budget(remaining)

    resolved_count = sum(1 for a in arcs if a.resolved)
    unresolved_count = len(arcs) - resolved_count
    total_repeats = sum(a.repeat_count for a in arcs)

    size_mb = source_bytes / 1_000_000
    header = f"<!-- distilled: source {size_mb:.1f}MB -->\n"
    doc = [
        header,
        "<session-transcript-distilled>",
        "<metadata>",
        f"  <session-id>{resolved_session_id}</session-id>",
        f"  <project-path>{resolved_project_path}</project-path>",
        f"  <harness>{harness}</harness>",
        f"  <source-bytes>{source_bytes}</source-bytes>",
        "</metadata>",
        "",
        arc_section,
        "",
        "<messages>",
        *body_parts,
        "</messages>",
        "</session-transcript-distilled>",
    ]
    content = "\n".join(doc)
    md_path.write_text(content)

    elapsed = time.monotonic() - start
    stats = DistillStats(
        source_path=str(source_path),
        source_bytes=source_bytes,
        harness=harness,
        session_id=resolved_session_id,
        project_path=resolved_project_path,
        prefiltered_lines=prefiltered_lines,
        prefiltered_bytes=prefiltered_bytes,
        unparsed_lines=unparsed_lines,
        kept_by_class=dict(kept_by_class),
        dropped_by_class=dict(dropped_by_class),
        arc_resolved_count=resolved_count,
        arc_unresolved_count=unresolved_count,
        arc_total_repeat_count=total_repeats,
        output_bytes=len(content),
        elapsed_seconds=elapsed,
    )
    meta = stats.to_dict()
    meta_path.write_text(json.dumps(meta, indent=2))

    return DistillResult(md_path=md_path, meta_path=meta_path, arcs=arcs, stats=stats)
