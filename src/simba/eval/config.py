"""Configuration for the eval harness."""

from __future__ import annotations

import dataclasses

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

    def ks_tuple(self) -> tuple[int, ...]:
        """Parse ``ks`` into a tuple of ints; fall back to the default."""
        out = []
        for part in self.ks.split(","):
            part = part.strip()
            if part.isdigit():
                out.append(int(part))
        return tuple(out) if out else _DEFAULT_KS
