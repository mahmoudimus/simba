"""Reflection synthesis prompt builder (Phase 5, Task A.3).

Pure function: turns a list of memory dicts into the LLM prompt that asks for
cross-session insight synthesis. No LLM dependency, testable in isolation.
"""

from __future__ import annotations

_REFLECTION_PROMPT = """\
You are a memory consolidator reviewing {n} memories captured across multiple \
coding sessions for project '{project}'.

Your task: identify CROSS-SESSION patterns, recurring friction points, evolving \
decisions, or durable insights that are NOT captured in any single session \
summary (EPISODE). These must be non-obvious and high-value — skip anything \
already stated verbatim in a memory.

For each insight you identify (maximum {max_reflections}), store it with:
  simba memory store --type REFLECTION \\
    --content "<≤200-char cross-session insight>" \\
    --context "<evidence: list 2-4 memory ids that support this>" \\
    --confidence <0.0-1.0> \\
    --project-path '{project}'

Rules:
- content MUST be ≤200 characters
- confidence ≥ {importance_threshold} (discard weaker candidates)
- Do NOT paraphrase a single memory — that is an EPISODE, not a REFLECTION
- Do NOT store a reflection if an existing REFLECTION memory already captures it
- Prefer actionable insights over observations

Existing REFLECTIONs (do not duplicate):
{existing_reflections}

Memories to analyse:
{memory_lines}
"""


def _format_lines(memories: list[dict]) -> str:
    lines = []
    for m in memories:
        content = (m.get("content") or "").strip()
        ctx = (m.get("context") or "").strip()
        line = f"- [{m.get('type', '?')}] {content}"
        if ctx:
            line += f" ({ctx[:120]})"
        mid = m.get("id", "")
        if mid:
            line += f"  <{mid}>"
        lines.append(line)
    return "\n".join(lines)


def build_reflection_prompt(
    memories: list[dict],
    *,
    project: str,
    existing_reflections: list[dict],
    max_source_memories: int = 100,
    max_reflections: int = 3,
    importance_threshold: float = 0.6,
) -> str:
    """Return the reflection synthesis prompt for the LLM."""
    capped = memories[:max_source_memories]
    return _REFLECTION_PROMPT.format(
        n=len(capped),
        project=project,
        max_reflections=max_reflections,
        importance_threshold=importance_threshold,
        existing_reflections=_format_lines(existing_reflections) or "(none)",
        memory_lines=_format_lines(capped) or "(none)",
    )
