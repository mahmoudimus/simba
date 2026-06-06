"""Configuration for the eval harness."""

from __future__ import annotations

import dataclasses
import typing

import simba.config

_DEFAULT_KS = (1, 3, 5, 10)


@simba.config.configurable("eval")
@dataclasses.dataclass
class EvalConfig:
    # Cutoffs for recall@k / precision@k / hit@k / ndcg@k, comma-separated so it
    # round-trips through `simba config set eval.ks "1,3,5"`.
    ks: str = "1,3,5,10"
    # Path to a dataset JSON. Empty ⇒ the bundled seed dataset.
    dataset: str = ""
    # Directory (relative to project root) where append-only baseline result
    # histories are written by ``baseline_store``.
    baseline_dir: str = ".simba/eval/baselines"
    # Comma-separated refusal phrases for abstention scoring; split on load. A
    # predicted answer that contains any phrase (case-insensitive) counts as a
    # correct refusal for ``_abs`` questions.
    abstention_phrases: str = (
        "don't know,do not know,no information,cannot find,"
        "not in my memories,i have no record"
    )
    # IRCoT (answer-time multi-hop QA). When enabled, run_qa routes cases with
    # intent == "multi-hop" through the interleaved retrieve-and-reason loop
    # (ircot_answer) instead of the single-pass score_case. False = current
    # behavior for every case.
    ircot_enabled: bool = False
    # Max retrieve-reason iterations before forcing the final answer.
    ircot_max_steps: int = 4
    # Memories retrieved per IRCoT step (smaller than QA k to stay in context).
    ircot_k_per_step: int = 3
    # Max accumulated evidence items passed to the final answer prompt.
    ircot_k_final: int = 10

    def ks_tuple(self) -> tuple[int, ...]:
        """Parse ``ks`` into a tuple of ints; fall back to the default."""
        out = []
        for part in self.ks.split(","):
            part = part.strip()
            if part.isdigit():
                out.append(int(part))
        return tuple(out) if out else _DEFAULT_KS


def load_config(**overrides: typing.Any) -> EvalConfig:
    """Load config from TOML files, then apply CLI/keyword overrides."""
    base = simba.config.load("eval")
    valid = {f.name for f in dataclasses.fields(EvalConfig)}
    filtered = {k: v for k, v in overrides.items() if v is not None and k in valid}
    if not filtered:
        return base
    merged = dataclasses.asdict(base)
    merged.update(filtered)
    return EvalConfig(**merged)
