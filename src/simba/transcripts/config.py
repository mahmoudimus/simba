"""Configuration for transcript extraction (``simba transcript extract``)."""

from __future__ import annotations

import dataclasses

import simba.config


@simba.config.configurable("transcripts")
@dataclasses.dataclass
class TranscriptsConfig:
    # Extract every new export in the background, one worker at a time. See
    # transcripts/llm_extract.py and
    # docs/adr/2026-11-23-headless-transcript-extraction.md.
    auto_extract: bool = True
    # Each host's transcripts go to that host's own CLI and provider.
    model_pi: str = "deepseek/deepseek-flash"  # run with `pi -p`
    model_claude: str = "claude-haiku-5-5"  # run with `claude -p`
    # Codex's smallest current model (gpt-5.6-luna is "Legacy" in Codex's model
    # cache); supports reasoning up to "max".
    model_codex: str = "gpt-6-luna"  # run with `codex exec`
    codex_reasoning_effort: str = "max"
    # Characters per model call. A chunk is the largest piece held in memory.
    chunk_chars: int = 240_000
    # Concurrent model calls for a manual run. The background worker always
    # runs one at a time.
    manual_jobs: int = 2
    # Heap cap for the Node-based CLIs (pi, claude); codex is a native binary.
    node_max_old_space_mb: int = 1024
    # Seconds before one model call is abandoned (the transcript stays pending).
    call_timeout: float = 900.0
