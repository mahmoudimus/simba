"""UserPromptSubmit hook — CORE extraction + memory recall.

Reads stdin JSON with user prompt, extracts CORE blocks from CLAUDE.md,
queries memory daemon for relevant memories, outputs combined context.
"""

from __future__ import annotations

import json
import pathlib

import simba.guardian.extract_core
import simba.hooks._memory_client
import simba.search.rag_context

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

    combined = "\n\n".join(parts)
    if combined:
        tokens = len(combined) // 4
        tags = f"~{tokens} tokens"
        if core_blocks:
            tags += " | \u2713 rules"
        combined += f"\n[simba: {tags}]"
    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": combined,
        }
    }
    return json.dumps(output)
