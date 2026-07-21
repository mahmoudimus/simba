"""PostToolBatch hook -- batch-scoped memory recall (default-OFF).

``PostToolBatch`` is a new Claude Code hook event: it fires once per
tool-call round (every ``tool_use`` block the model emitted in one turn)
before the next model call, carrying ``tool_calls``: a list of
``{tool_name, tool_input, tool_use_id, tool_response}``, where
``tool_response`` is the SERIALIZED result the model saw and can be large.
Codex has no equivalent event and this module is never registered for it.

Off by default (UNMEASURED lever, ``hooks.post_tool_batch_enabled``): ``run``
returns immediately, no daemon calls, near-zero cost -- byte-identical to the
event not existing at all.

When enabled: the (mandatory, unconditional) payload trim runs first --
``tool_response``/``tool_input`` can be arbitrarily large and the daemon must
never see an unbounded batch -- then a compact recall query is built from the
trimmed batch (tool names, short input heads, error-looking response
fragments) and recalled memories are formatted and returned as
``additional_context``, the same machinery ``pre_tool_use.py`` uses for its
thinking-block recall.
"""

from __future__ import annotations

import json
import re

import simba.config
import simba.hooks._io
import simba.hooks._memory_client
from simba.harness.core import CanonicalResult

# Marks a field that was shortened by the payload trim -- lets a test (or a
# human reading a captured payload) tell "genuinely short" apart from
# "truncated to fit the cap" at a glance.
_TRUNCATION_MARKER = "...[simba: truncated]"

# Cheap "this response looks like a failure" scan for the recall-query
# builder -- deliberately narrower than post_tool_use.py's full
# _ERROR_PATTERNS (this only needs to decide "worth a short excerpt in the
# query", not "worth learning a TOOL_RULE from").
_ERROR_LOOKING_RE = re.compile(r"error|exception|traceback|failed|fatal", re.IGNORECASE)

# Metadata (tool_name, tool_use_id, and similar small fields) is assumed to
# cost roughly this many bytes of JSON overhead per item (keys, braces,
# quoting) -- an approximation, not a re-serialize-and-measure loop, so the
# trim stays O(n) over the batch instead of O(n) reserializations.
_METADATA_OVERHEAD_BYTES = 256

# Recall query cap -- keeps the daemon call itself cheap regardless of how
# many tool calls are in the round.
_MAX_QUERY_CHARS = 500


def _hooks_cfg():
    import simba.hooks.config

    _ = simba.hooks.config  # side-effect: registers "hooks" section
    return simba.config.load("hooks")


def _as_text(value) -> str:
    """Return ``value`` as text -- pass strings through, JSON-encode anything else.

    ``tool_response`` is documented as already-serialized text, but a caller
    (or a test) may hand this a dict/list instead; encoding it here keeps the
    trim/query-builder logic string-only without trusting the input shape.
    """
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def _bounded_field(value, budget_bytes: int) -> tuple[object, bool]:
    """Return ``value`` unchanged if it already fits ``budget_bytes``, else
    a truncated-and-marked text version.

    A field within budget is returned in its ORIGINAL shape (e.g. a dict
    ``tool_input`` stays a dict) -- only a field that actually needs
    shortening gets serialized to text, so an under-cap batch is untouched
    byte-for-byte, structure included.
    """
    text = _as_text(value)
    encoded = text.encode("utf-8", errors="ignore")
    if len(encoded) <= budget_bytes:
        return value, False
    marker = _TRUNCATION_MARKER.encode("utf-8")
    keep = max(0, budget_bytes - len(marker))
    truncated = encoded[:keep] + marker
    return truncated.decode("utf-8", errors="ignore"), True


def _trim_batch_payload(
    tool_calls: list[dict], max_kb: float
) -> tuple[list[dict], bool]:
    """Trim a PostToolBatch's ``tool_calls`` to fit ``max_kb`` serialized.

    Runs client-side, unconditionally, BEFORE any query is built or daemon
    call is made -- ``tool_response`` is the serialized result the model saw
    and can be arbitrarily large (this repo spent a week killing unbounded
    reads; this lane does not get to rebuild that balloon out of payloads).

    ``tool_name`` / ``tool_use_id`` / permission-shaped fields are kept
    intact (small, and needed downstream for the recall query + attribution).
    ``tool_input`` and ``tool_response`` are each truncated to a fair
    per-item share of the budget, so one huge response cannot starve every
    other item's share. Returns ``(trimmed_calls, trimmed)`` where ``trimmed``
    is True iff any item's ``tool_input``/``tool_response`` was shortened.
    """
    if not tool_calls:
        return [], False

    max_bytes = max(1, int(max_kb * 1024))
    n = len(tool_calls)
    budget_per_item = max(64, (max_bytes // n) - _METADATA_OVERHEAD_BYTES)
    # tool_response is typically the bulkier field (tool output/errors);
    # tool_input is usually a short command/path -- split unevenly so a
    # legitimately long response keeps more of its content.
    input_budget = max(32, budget_per_item // 4)
    response_budget = max(32, budget_per_item - input_budget)

    trimmed_any = False
    out: list[dict] = []
    for call in tool_calls:
        item = dict(call)

        new_input, input_trimmed = _bounded_field(
            call.get("tool_input", ""), input_budget
        )
        item["tool_input"] = new_input

        new_response, response_trimmed = _bounded_field(
            call.get("tool_response", ""), response_budget
        )
        item["tool_response"] = new_response

        trimmed_any = trimmed_any or input_trimmed or response_trimmed
        out.append(item)

    return out, trimmed_any


def _build_recall_query(tool_calls: list[dict]) -> str:
    """Compact recall query from an (already-trimmed) batch.

    Tool names + short input heads always contribute; an error-looking
    fragment from the response (if any) contributes a short window around
    the first match -- enough to steer recall toward "what just went wrong",
    without embedding the whole (already-bounded) response text.
    """
    parts: list[str] = []
    for call in tool_calls:
        name = call.get("tool_name", "")
        if name:
            parts.append(str(name))

        input_text = call.get("tool_input", "")
        if not isinstance(input_text, str):
            input_text = _as_text(input_text)
        if input_text:
            parts.append(input_text[:80])

        response_text = call.get("tool_response", "")
        if not isinstance(response_text, str):
            response_text = _as_text(response_text)
        match = _ERROR_LOOKING_RE.search(response_text) if response_text else None
        if match:
            start = max(0, match.start() - 20)
            parts.append(response_text[start : start + 120])

    query = " ".join(p for p in parts if p)
    return query[:_MAX_QUERY_CHARS]


def run(hook_input: dict) -> CanonicalResult:
    """Run the PostToolBatch pipeline. Returns a CanonicalResult.

    Disabled (default): returns immediately -- no trim, no query, no daemon
    call. Enabled: trim -> build query -> recall -> format, mirroring
    ``pre_tool_use.py``'s thinking-block recall machinery.
    """
    cfg = _hooks_cfg()
    if not getattr(cfg, "post_tool_batch_enabled", False):
        return CanonicalResult()

    tool_calls = hook_input.get("tool_calls", [])
    if not isinstance(tool_calls, list) or not tool_calls:
        return CanonicalResult()

    max_kb = getattr(cfg, "post_tool_batch_max_payload_kb", 256.0)
    trimmed_calls, _ = _trim_batch_payload(tool_calls, max_kb)

    query = _build_recall_query(trimmed_calls)
    if not query:
        return CanonicalResult()

    project_path = hook_input.get("cwd")
    memories = simba.hooks._memory_client.recall_memories(
        query, project_path=project_path
    )
    formatted = simba.hooks._memory_client.format_memories(
        memories, source="post-tool-batch", query=query
    )
    if not formatted:
        return CanonicalResult()

    return CanonicalResult(additional_context=formatted, memory_count=len(memories))


def main(hook_input: dict) -> str:
    """Run the PostToolBatch hook and render the Claude envelope.

    Codex has no PostToolBatch equivalent; this is Claude-only (see
    ``.claude-plugin/hooks.json`` -- not registered in Codex's hooks.json).
    """
    import simba.harness.adapters.claude as claude

    return claude.render("PostToolBatch", run(hook_input))
