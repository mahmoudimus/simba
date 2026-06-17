"""Intent-priming injection builder (spec 28 Phase B).

Front-loads the right approach from the user's stated intent: classify the prompt
against the project's doctrine triggers (cheap embedding match, no LLM) and build
an ``<intent-priming>`` block listing the matched doctrine + the applicable
TOOL_RULEs / redirects. Pre-emptive guidance, not generic recall.

Reports whether a RISK-TIER doctrine matched, so ``UserPromptSubmit`` can arm the
preflight mandate for the turn. Pure + fail-open: any error yields an empty prime
(advisory — never crash the hook).
"""

from __future__ import annotations

import dataclasses
import typing

import simba.doctrine.intent
import simba.doctrine.store

if typing.TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclasses.dataclass
class PrimeResult:
    """The injection text + whether a risk-tier doctrine was primed this turn."""

    text: str = ""
    risk_primed: bool = False


def prime(
    prompt: str,
    *,
    doctrines: Sequence[simba.doctrine.store.Doctrine],
    embed_fn: Callable[[str], list[float]],
    min_similarity: float = 0.55,
    max_doctrines: int = 3,
) -> PrimeResult:
    """Classify ``prompt`` and build the intent-priming injection.

    Returns an empty ``PrimeResult`` when nothing matches (so the caller appends
    nothing — byte-identical to no-priming). ``risk_primed`` is True iff any
    matched doctrine is risk-tier (arms the preflight mandate).
    """
    try:
        matches = simba.doctrine.intent.classify(
            prompt, doctrines, embed_fn=embed_fn, min_similarity=min_similarity
        )
    except Exception:
        return PrimeResult()
    if not matches:
        return PrimeResult()

    top = matches[: max(1, max_doctrines)]
    risk_primed = any(m.doctrine.risk_tier for m in top)

    lines = ["<intent-priming>", "The stated task matches known doctrine:"]
    for m in top:
        tag = " [risk]" if m.doctrine.risk_tier else ""
        lines.append(f"  - {m.doctrine.doctrine}{tag}")
        for rule in m.doctrine.applicable_rules:
            lines.append(f"      applies: {rule}")
    lines.append("</intent-priming>")
    return PrimeResult(text="\n".join(lines), risk_primed=risk_primed)
