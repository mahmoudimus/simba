"""LLM HyDE: generate a short hypothetical answer to seed the 2nd vector arm.

In ``keyword`` mode the second recall arm embeds a focus-term string. In ``llm``
mode it instead embeds a one-sentence *hypothetical answer* generated here, which
carries genuine semantic content rather than a bag of keywords. Every failure
mode (no provider, timeout, empty reply, exception) collapses to ``""`` so the
caller transparently falls back to the keyword arm — this function NEVER raises.
"""

from __future__ import annotations

import contextlib
import typing

_MAX_CHARS = 300

_PROMPT_TEMPLATE = (
    "Write one short, factual sentence that would directly ANSWER the question "
    "below, as if it were a stored memory. Do not hedge, do not explain, output "
    "only the sentence.\n\nQuestion: {query}\nAnswer:"
)


def build_hyde_prompt(query: str) -> str:
    """Return the instruction string that asks the LLM for a hypothetical answer."""
    return _PROMPT_TEMPLATE.format(query=query)


def hypothetical_answer(query: str, llm: typing.Any) -> str:
    """Generate a hypothetical answer for ``query`` via ``llm.complete``.

    Returns ``""`` on any failure (unavailable client, timeout, empty reply,
    exception). The returned string is passed directly to ``embed_query`` and is
    sliced to ``_MAX_CHARS``. Never raises.
    """
    with contextlib.suppress(Exception):
        available = getattr(llm, "available", None)
        if available is not None and not available():
            return ""
        text = llm.complete(build_hyde_prompt(query))
        if isinstance(text, str) and text.strip():
            return text.strip()[:_MAX_CHARS]
    return ""
