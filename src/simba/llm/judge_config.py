"""Configuration for the LLM *judge* (grader), separate from the answerer.

The judge config mirrors ``LlmConfig`` field-for-field so ``get_client(cfg)``
accepts either without modification. Keeping the grader on a different
provider/model than the answerer breaks the self-grading loop where one model
both answers and grades its own answer.

CAVEAT: the judge is still a local model and may differ from GPT-4 calibration.
Scores are more trustworthy than answerer==judge but not directly comparable to
published frontier-judge numbers.
"""

from __future__ import annotations

import dataclasses
import typing

import simba.config
import simba.llm.client


@simba.config.configurable("judge")
@dataclasses.dataclass
class JudgeConfig:
    # Backend for the grader; default differs from ``llm.provider`` so the answerer
    # and judge are not the same model. Same options as LlmConfig.provider, incl.
    # mlx-server / llama-server / openai-http (HTTP, set ``base_url``); see
    # docs/eval-remote-gpu.md.
    provider: str = "llm-cli"  # …| mlx-server | llama-server | openai-http | none
    # Model name as the judge CLI expects it (a local reasoning model by default).
    model: str = "deepseek-r1"
    # Local model/GGUF path (or HF repo); falls back to ``model`` when empty.
    model_path: str = ""
    # Reasoning-effort hint, best-effort (llm-cli -o reasoning_effort).
    thinking: str = ""
    # Anthropic-compatible proxy endpoint (e.g. a DeepSeek proxy).
    base_url: str = ""
    api_key_env: str = "ANTHROPIC_API_KEY"  # env var holding the key for base_url
    # Extra CLI args (shell-split) appended to the chosen provider's argv.
    extra_args: str = ""
    # Auto-spawn command template for mlx-server / llama-server (empty -> preset);
    # mirrors LlmConfig.serve_cmd. See docs/eval-remote-gpu.md.
    serve_cmd: str = ""
    # Grading may need more time than answering.
    timeout_seconds: float = 90.0
    # The judge needs only short verdicts.
    max_tokens: int = 512


def load_judge_config(**overrides: typing.Any) -> JudgeConfig:
    """Load judge config from TOML files then apply keyword overrides."""
    base = simba.config.load("judge")
    valid = {f.name for f in dataclasses.fields(JudgeConfig)}
    filtered = {k: v for k, v in overrides.items() if v is not None and k in valid}
    if not filtered:
        return base
    merged = dataclasses.asdict(base)
    merged.update(filtered)
    return JudgeConfig(**merged)


def get_judge_client(cfg: typing.Any | None = None) -> simba.llm.client.LlmClient:
    """Return an ``LlmClient`` built from the judge section (or loaded default)."""
    if cfg is None:
        import simba.config
        import simba.llm.judge_config  # registers the "judge" section

        _ = simba.llm.judge_config
        cfg = simba.config.load("judge")
    return simba.llm.client.LlmClient(cfg)
