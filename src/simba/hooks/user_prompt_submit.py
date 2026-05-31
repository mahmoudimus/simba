"""UserPromptSubmit hook — CORE extraction + memory recall.

Reads stdin JSON with user prompt, extracts CORE blocks from CLAUDE.md,
queries memory daemon for relevant memories, outputs combined context.
"""

from __future__ import annotations

import contextlib
import pathlib
import sys

import simba.guardian.extract_core
import simba.hooks._io
import simba.hooks._memory_client
import simba.search.rag_context


def _rlm_pointer_context(memories: list[dict], cwd_str: str | None) -> str:
    """Return an <rlm-pointers> block when rlm.inject_pointers is enabled.

    Reuses the memories already recalled this turn (no second recall) and
    surfaces navigable transcripts so the agent knows it can rlm_grep/rlm_peek
    them for lossless detail. Never raises into the hook.
    """
    import simba.config
    import simba.rlm.config  # registers the "rlm" section
    import simba.rlm.recall

    if not simba.config.load("rlm").inject_pointers:
        return ""
    pointers = simba.rlm.recall.pointers_from_memories(memories, cwd_str)
    nav = [p for p in pointers if p.available]
    if not nav:
        return ""
    lines = [
        "<rlm-pointers>",
        "Lossless transcripts available — call rlm_grep/rlm_peek on these ids "
        "if the recalled snippets aren't enough:",
    ]
    lines += [f"  - {p.transcript_id} :: {p.snippet[:70]}" for p in nav[:3]]
    lines.append("</rlm-pointers>")
    return "\n".join(lines)

_MIN_PROMPT_LENGTH = 10
_MIN_SIMILARITY = 0.45


def main(hook_input: dict) -> str:
    """Run the UserPromptSubmit hook pipeline. Returns JSON output string."""

    prompt = hook_input.get("prompt", "")
    cwd_str = hook_input.get("cwd")
    cwd = pathlib.Path(cwd_str) if cwd_str else pathlib.Path.cwd()

    parts: list[str] = []

    # 1. Guardian: extract CORE blocks from CLAUDE.md
    core_blocks = simba.guardian.extract_core.main(cwd=cwd)
    if core_blocks:
        parts.append(core_blocks)

    # 2. Memory: recall relevant memories using prompt
    memories: list[dict] = []
    if prompt and len(prompt) >= _MIN_PROMPT_LENGTH:
        project_path = str(cwd) if cwd_str else None
        memories = simba.hooks._memory_client.recall_memories(
            prompt, project_path=project_path, min_similarity=_MIN_SIMILARITY
        )
        formatted = simba.hooks._memory_client.format_memories(
            memories, source="user-prompt"
        )
        if formatted:
            parts.append(formatted)

    # 3. Search: project memory + QMD context
    if prompt and len(prompt) >= _MIN_PROMPT_LENGTH:
        try:
            search_ctx = simba.search.rag_context.build_context(prompt, cwd)
            if search_ctx:
                parts.append(search_ctx)
        except Exception:
            pass

    # 4. RLM: surface navigable transcript pointers (opt-in via rlm.inject_pointers)
    if memories:
        with contextlib.suppress(Exception):
            rlm_ctx = _rlm_pointer_context(memories, cwd_str)
            if rlm_ctx:
                parts.append(rlm_ctx)

    combined = "\n\n".join(parts)
    if combined:
        tokens = len(combined) // 4
        tags = f"~{tokens} tokens"
        if core_blocks:
            tags += " | \u2713 rules"
        combined += f"\n[simba: {tags}]"
        print(f"[simba: {tags}]", file=sys.stderr)
    return simba.hooks._io.context("UserPromptSubmit", combined)
