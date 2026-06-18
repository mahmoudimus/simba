"""Cheap retrieval triage for UserPromptSubmit.

The classifier is deliberately conservative: false negatives are expensive, so
only narrow self-contained prompts return ``skip``. Everything else retrieves.
"""

from __future__ import annotations

import dataclasses
import re


@dataclasses.dataclass(frozen=True)
class RecallTriage:
    decision: str
    reason: str

    @property
    def should_retrieve(self) -> bool:
        return self.decision != "skip"


_ACK_RE = re.compile(
    r"^(ok|okay|k|thanks|thank you|ty|cool|great|nice|sounds good|got it|"
    r"makes sense|perfect|excellent|awesome|👍|yes|no)[.!?\s]*$",
    re.IGNORECASE,
)
_TIME_RE = re.compile(
    r"\b(what('| i)?s|what is|tell me)\s+(the\s+)?(current\s+)?"
    r"(time|date)\b|\bcurrent\s+(time|date)\b",
    re.IGNORECASE,
)
_SELF_CONTAINED_RE = re.compile(
    r"\b(translate|rewrite|rephrase|summari[sz]e|format)\b",
    re.IGNORECASE,
)
_MEMORY_NEEDED_RE = re.compile(
    r"\b(previous|earlier|last time|remember|memory|recall|continue|handoff|"
    r"roadmap|borrow|next|repo|worktree|commit|push|run|test|fix|implement|"
    r"change|file|path|bug|error|failure|why|how)\b",
    re.IGNORECASE,
)


def classify(prompt: str) -> RecallTriage:
    """Return a conservative retrieval decision for a user prompt."""
    text = " ".join(prompt.strip().split())
    if not text:
        return RecallTriage("skip", "empty_prompt")

    if _MEMORY_NEEDED_RE.search(text):
        return RecallTriage("recall", "memory_or_repo_cue")

    if _ACK_RE.fullmatch(text):
        return RecallTriage("skip", "acknowledgement")

    if _TIME_RE.search(text):
        return RecallTriage("skip", "current_time_or_date")

    if _SELF_CONTAINED_RE.search(text) and len(text) < 400:
        return RecallTriage("skip", "self_contained_text_task")

    return RecallTriage("uncertain", "no_safe_skip_rule")


def render(triage: RecallTriage) -> str:
    """Render a tiny diagnostics block for opt-in observability."""
    return (
        "<recall-triage>\n"
        f"decision: {triage.decision}\n"
        f"reason: {triage.reason}\n"
        "</recall-triage>"
    )
