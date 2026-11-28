"""Strip harness and simba boilerplate from text before it becomes a recall query.

Prompts and responses carry text that no agent or user wrote for meaning:
``<system-reminder>`` blocks, the ``<task-notification>`` envelope Claude Code
delivers as a prompt when a background task finishes, simba's own ``🦁☑``
engagement line, the ``[✓ rules]`` signal and ``[simba: ~N tokens]`` footers.
Embedded as part of the query, that text pulls in memories that share the
boilerplate rather than the topic. Measured on the live daemon log
(2026-11-28): of 15,070 recall queries, 385 started with the ``🦁☑`` line
(Stop-hook reasoning verify embeds the whole reply) and 610 contained a
``task-notification`` envelope (prompt recall).

Pattern from memkraft da06678 (``_strip_host_notes``). Gated by
``hooks.recall_query_clean_enabled`` (default OFF until A/B'd).
"""

from __future__ import annotations

import re

# Whole blocks the harness injects; none of their text is the user's.
_DROP_BLOCKS = re.compile(
    r"<(system-reminder)\b[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
# The task-notification envelope: keep the task's summary and result text,
# drop the ids, paths, status, usage and harness note around them.
_NOTIFICATION = re.compile(
    r"<task-notification\b[^>]*>(.*?)(?:</task-notification>|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_NOTIFICATION_KEEP = re.compile(
    r"<(summary|result)\b[^>]*>(.*?)(?:</\1>|\Z)", re.DOTALL | re.IGNORECASE
)
# simba's own markers: the engagement ledger line, the rules signal, and the
# token footer the hooks append.
_ENGAGEMENT_LINE = re.compile(r"^[ \t]*🦁☑[^\n]*$", re.MULTILINE)
_SIGNAL = re.compile(r"\[✓ rules\]")
_FOOTER = re.compile(r"\[simba:[^\]\n]*\]")
_SPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def _keep_notification_text(match: re.Match[str]) -> str:
    body = match.group(1)
    kept = [m.group(2).strip() for m in _NOTIFICATION_KEEP.finditer(body)]
    return "\n".join(k for k in kept if k)


def clean(text: str) -> str:
    """Return ``text`` with harness/simba boilerplate removed.

    Pure and idempotent. Text with none of the patterns comes back stripped of
    surrounding whitespace and otherwise unchanged in content.
    """
    if not text:
        return ""
    out = _DROP_BLOCKS.sub("", text)
    out = _NOTIFICATION.sub(_keep_notification_text, out)
    out = _ENGAGEMENT_LINE.sub("", out)
    out = _SIGNAL.sub("", out)
    out = _FOOTER.sub("", out)
    out = _SPACE.sub(" ", out)
    out = "\n".join(line.strip() for line in out.splitlines())
    return _BLANK_LINES.sub("\n\n", out).strip()


def recall_query(text: str, cfg) -> str:
    """The query to embed for ``text``: cleaned when the lever is on."""
    if getattr(cfg, "recall_query_clean_enabled", False):
        return clean(text)
    return text
