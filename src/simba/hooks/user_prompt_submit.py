"""UserPromptSubmit hook — CORE extraction + memory recall.

Reads stdin JSON with user prompt, extracts CORE blocks from CLAUDE.md,
queries memory daemon for relevant memories, outputs combined context.
"""

from __future__ import annotations

import contextlib
import pathlib
import sys

import simba.guardian.extract_core
import simba.hooks._memory_client
import simba.search.rag_context
from simba.harness.core import CanonicalResult


def _rlm_pointer_context(memories: list[dict], cwd_str: str | None) -> str:
    """Return an <rlm-pointers> block when rlm.inject_pointers is enabled.

    Reuses the memories already recalled this turn (no second recall) and
    surfaces navigable transcripts so the agent knows it can rlm_grep/rlm_peek
    them for lossless detail. Never raises into the hook.

    TODO(rlm): reusing the turn's recall (top-N at the hook's higher similarity
    bar) makes the nudge sparse — it only fires when a top hit is navigable. To
    surface pointers more reliably, do a dedicated wider route() here (top-5 at
    ~0.35) at the cost of one extra recall per prompt. Deferred.
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


def _cfg():
    """Load the hooks config section (registers it on first access)."""
    import simba.config
    import simba.hooks.config

    _ = simba.hooks.config  # ensure "hooks" section is registered
    return simba.config.load("hooks")


def run(hook_input: dict) -> CanonicalResult:
    """Run the UserPromptSubmit hook pipeline. Returns a CanonicalResult."""
    prompt = hook_input.get("prompt", "")
    cwd_str = hook_input.get("cwd")
    # Path derives from payload only \u2014 dispatch may run in the daemon process
    # whose own cwd differs from the agent's.
    cwd = pathlib.Path(cwd_str) if cwd_str else None

    cfg = _cfg()
    parts: list[str] = []

    # 1. Guardian: extract CORE blocks from CLAUDE.md
    core_blocks = simba.guardian.extract_core.main(cwd=cwd)
    if core_blocks:
        parts.append(core_blocks)

    # 2. Memory: recall relevant memories using prompt
    memories: list[dict] = []
    if prompt and len(prompt) >= cfg.prompt_min_length:
        project_path = str(cwd) if cwd_str else None
        memories = simba.hooks._memory_client.recall_memories(
            prompt, project_path=project_path, min_similarity=cfg.prompt_min_similarity
        )
        formatted = simba.hooks._memory_client.format_memories(
            memories, source="user-prompt"
        )
        if formatted:
            parts.append(formatted)

    # 3. Search: project memory + QMD context
    if cwd is not None and prompt and len(prompt) >= cfg.prompt_min_length:
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
    return CanonicalResult(additional_context=combined)


def main(hook_input: dict) -> str:
    """Run the UserPromptSubmit hook and render the Claude/Codex envelope."""
    import simba.harness.adapters.claude as claude

    return claude.render("UserPromptSubmit", run(hook_input))
