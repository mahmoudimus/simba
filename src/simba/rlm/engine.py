"""RLM autonomous engine — pluggable, cheap-by-default execution.

Phase 1 ships ClaudeCliEngine (tool-driven): spawn a detached, cheap
`claude -p` that navigates the transcript via the rlm_* tools and stores
memories itself. `get_engine` returns None for engine="claude" (the
default agent-driven path) and for engines not yet implemented.
"""

from __future__ import annotations

import os
import subprocess
import typing


class RlmEngine(typing.Protocol):
    def run(self, prompt: str, *, cwd: str) -> None:
        """Dispatch a detached, cheap agent with ``prompt``. Fire-and-forget —
        the agent does the work itself and never blocks the caller."""

    def digest(self, transcript_id: str, query: str, *, cwd: str) -> None:
        """Extract lossless memories from a transcript and store them.
        Always invoked detached/background — never blocks a hook."""


_DIGEST_PROMPT = (
    "A coding session just ended; its full transcript is loaded as "
    "transcript_id '{tid}'. Use rlm_grep / rlm_peek / rlm_window to read the "
    "key decisions, gotchas, working solutions, and failures from it — read "
    "the actual regions, be lossless, do not guess. Store each as a memory:\n"
    "  simba memory store --type <TYPE> --content <<=200 chars> "
    "--context <details> --project-path '{cwd}' --session-source '{tid}'\n"
    "TYPE is one of WORKING_SOLUTION, GOTCHA, PATTERN, DECISION, FAILURE, "
    "PREFERENCE. Store 5-15 high-value, specific learnings; skip generic "
    "knowledge. When finished, run: simba rlm complete '{tid}' --stored <count>."
)


def _build_digest_prompt(transcript_id: str, cwd: str) -> str:
    return _DIGEST_PROMPT.format(tid=transcript_id, cwd=cwd)


class ClaudeCliEngine:
    def __init__(self, cfg) -> None:
        self._cfg = cfg

    def _argv(self, prompt: str) -> list[str]:
        return [
            "claude", "-p", prompt,
            "--model", self._cfg.engine_model,
            "--allowedTools", self._cfg.engine_allowed_tools,
            "--permission-mode", "acceptEdits",
            "--max-turns", str(self._cfg.engine_max_turns),
            "--output-format", "json",
        ]

    def _env(self) -> dict:
        env = os.environ.copy()
        if self._cfg.engine_base_url:
            env["ANTHROPIC_BASE_URL"] = self._cfg.engine_base_url
            env["ANTHROPIC_AUTH_TOKEN"] = os.environ.get(
                self._cfg.engine_api_key_env, ""
            )
        return env

    def run(self, prompt: str, *, cwd: str) -> None:
        """Spawn a detached, cheap ``claude -p`` with ``prompt`` (no blocking)."""
        subprocess.Popen(
            self._argv(prompt),
            env=self._env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            cwd=cwd,
        )

    def digest(self, transcript_id: str, query: str, *, cwd: str) -> None:
        self.run(_build_digest_prompt(transcript_id, cwd), cwd=cwd)


def get_engine(cfg) -> RlmEngine | None:
    """Return the configured engine, or None for the agent-driven default."""
    if cfg.engine == "claude-cli":
        return ClaudeCliEngine(cfg)
    return None  # "claude" (default) + api/local-gguf (later phases)
