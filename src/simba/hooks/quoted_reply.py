"""Construct recall queries for Pi replies to Markdown blockquotes."""

from __future__ import annotations

import re

_QUOTE = re.compile(r"^ {0,3}>[ \t]?")
# Markdown allows wrapped lines inside a quote to omit the leading ">".
_QUOTE_BLOCK = re.compile(r"(^ {0,3}>[^\r\n]*(?:\r?\n[ \t]*\S[^\r\n]*)*)", re.MULTILINE)
_FENCE = re.compile(r"^ {0,3}(?:`{3,}|~{3,})", re.MULTILINE)
_QUESTION = re.compile(r"\?(?=\s|$)")
_REFERENCE = re.compile(r"\b(?:it|this|that|these|those|they|them|same)\b", re.I)
_IDENTIFIER = re.compile(
    r"\b(?:[\w-]+(?:\.[\w-]+)+|[A-Za-z]\w*_[A-Za-z0-9_]+|"
    r"[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+)\b"
)


def recall_queries(prompt: str, *, min_length: int, limit: int) -> list[str]:
    """Use one query per question-bearing reply block, up to the recall budget."""
    if limit <= 0 or _FENCE.search(prompt):
        return [prompt]

    parts = _QUOTE_BLOCK.split(prompt)
    if len(parts) == 1:
        return [prompt]

    queries: list[str] = []
    for quote_block, reply_block in zip(["", *parts[1::2]], parts[::2], strict=True):
        reply = reply_block.strip()
        if len(reply) < min_length or not _QUESTION.search(reply):
            continue
        quote = _QUOTE.sub("", quote_block.partition("\n")[0]).strip()
        if quote and _REFERENCE.search(reply) and not _IDENTIFIER.search(reply):
            name = _IDENTIFIER.search(quote)
            subject = name.group() if name else " ".join(quote.split()[:8])
            reply = f"{subject} {reply}"
        queries.append(reply)

    # Keep the original query for ordinary prompts, quoted code, and replies
    # whose intent cannot be identified from the question-bearing blocks.
    return list(dict.fromkeys(queries))[:limit] or [prompt]
