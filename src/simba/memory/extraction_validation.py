"""Write-time extraction validation gate (Eywa-style, arXiv 2605.30771).

When an LLM extracts a claim/atom from a conversation, it can drift from the source
(hallucinate a number, flip a polarity, drift to an unsupported subject). This is a
pure, deterministic check that an extracted ``claim`` is grounded in its ``source``
*supporting span* before the claim is promoted to a stored belief — no LLM, so it is
a free safety net on the (default-off) extraction path. ``source`` is meant to be the
short supporting span the claim was extracted from, not a whole transcript.

Three checks (each independently toggleable):
  * **hard_value** — every number/date in the claim must appear in the source.
    Catches the highest-stakes drift (a hallucinated amount/count/date), the exact
    failure mode that breaks counting/temporal answers.
  * **support**   — the claim's content tokens must overlap the source above a ratio.
  * **polarity**  — the claim must not invent (or drop) a negation vs the source.

``validate_extraction`` returns a :class:`ValidationResult`; a caller drops the claim
when ``not result.ok``. Off-switch (``enabled=False``) passes everything (fail-open).
"""

from __future__ import annotations

import dataclasses
import re

# Content-word filter: tiny stoplist so support overlap reflects substance, not glue.
_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "is",
        "am",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "and",
        "or",
        "but",
        "with",
        "as",
        "by",
        "from",
        "into",
    ]
)
_NEG = frozenset(["not", "no", "never", "none", "without", "cannot", "cant", "nor"])
_WORD = re.compile(r"[a-z0-9][a-z0-9'-]*")
# Digit-bearing tokens: amounts ($3,750 / 12000 / 0.5), dates (2023-05-16), years.
_NUM = re.compile(r"\d[\d,./:-]*\d|\d")


def _tokens(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def _stem(tok: str) -> str:
    """Conservative singular/3rd-person fold: drop a trailing 's' (likes -> like)."""
    return tok[:-1] if len(tok) >= 4 and tok.endswith("s") else tok


def _content(text: str) -> set[str]:
    return {_stem(t) for t in _tokens(text) if t not in _STOP and t not in _NEG}


def _numbers(text: str) -> set[str]:
    """Normalized digit tokens (drop thousands commas / surrounding punctuation)."""
    out = set()
    for m in _NUM.findall(text or ""):
        norm = m.replace(",", "").strip(".:-/")
        if norm:
            out.add(norm)
    return out


def _has_negation(text: str) -> bool:
    toks = set(_tokens(text))
    return bool(toks & _NEG) or "n't" in (text or "").lower()


@dataclasses.dataclass
class ValidationResult:
    ok: bool
    failed: list[str]
    support: float


def validate_extraction(
    claim: str,
    source: str,
    *,
    enabled: bool = True,
    min_support: float = 0.5,
    check_values: bool = True,
    check_polarity: bool = True,
) -> ValidationResult:
    """Validate an extracted ``claim`` against its ``source`` supporting span."""
    if not enabled:
        return ValidationResult(ok=True, failed=[], support=1.0)

    claim_content = _content(claim)
    src_content = _content(source)
    support = (
        len(claim_content & src_content) / len(claim_content) if claim_content else 1.0
    )

    failed: list[str] = []
    if check_values:
        src_nums = _numbers(source)
        if any(n not in src_nums for n in _numbers(claim)):
            failed.append("hard_value")
    if support < min_support:
        failed.append("support")
    if check_polarity and _has_negation(claim) != _has_negation(source):
        failed.append("polarity")

    return ValidationResult(ok=not failed, failed=failed, support=round(support, 3))
