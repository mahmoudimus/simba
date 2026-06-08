"""Configuration for the synchronous LLM client."""

from __future__ import annotations

import dataclasses
import typing

import simba.config


@simba.config.configurable("llm")
@dataclasses.dataclass
class LlmConfig:
    # Backend. "none" disables all LLM features (they degrade gracefully).
    # Cloud CLIs: claude-cli, llm-cli. 100%-local CLIs (reload per call — slow):
    # llama-cli (llama.cpp), mlx-lm (MLX) — set ``model_path``. HTTP/persistent
    # (OpenAI-compatible, set ``base_url``; model loaded once): mlx-server &
    # llama-server (auto-spawned locally) and openai-http (a server you run, local
    # or remote GPU box). See docs/eval-remote-gpu.md.
    # claude-cli|llm-cli|llama-cli|mlx-lm|mlx-server|llama-server|openai-http|none
    provider: str = "claude-cli"
    # Model name as the chosen CLI expects it (claude aliases: haiku/sonnet/opus;
    # llm: whatever `llm models` lists, e.g. a deepseek alias).
    model: str = "haiku"
    # Local model/GGUF path (or HF repo) for llama-cli / mlx-lm. Falls back to
    # ``model`` when empty.
    model_path: str = ""
    # Reasoning-effort hint, best-effort + provider-specific (e.g. xhigh). Passed
    # to llm-cli as `-o reasoning_effort <thinking>` when set; ignored otherwise.
    thinking: str = ""
    # Point claude-cli at an Anthropic-compatible endpoint (e.g. a DeepSeek proxy).
    base_url: str = ""
    api_key_env: str = "ANTHROPIC_API_KEY"  # env var holding the key for base_url
    # Extra CLI args (shell-split) appended to the chosen provider's argv — an
    # escape hatch for provider-specific flags (e.g. --n-gpu-layers, --temp).
    extra_args: str = ""
    # Auto-spawn command template for mlx-server / llama-server (or a custom one,
    # e.g. vLLM). Empty -> the provider's preset in llm/local_server.py.
    # {model}/{host}/{port} are substituted; see docs/eval-remote-gpu.md.
    serve_cmd: str = ""
    timeout_seconds: float = 60.0
    max_tokens: int = 2048


def load_config(**overrides: typing.Any) -> LlmConfig:
    """Load config from TOML files, then apply CLI/keyword overrides."""
    base = simba.config.load("llm")
    valid = {f.name for f in dataclasses.fields(LlmConfig)}
    filtered = {k: v for k, v in overrides.items() if v is not None and k in valid}
    if not filtered:
        return base
    merged = dataclasses.asdict(base)
    merged.update(filtered)
    return LlmConfig(**merged)
