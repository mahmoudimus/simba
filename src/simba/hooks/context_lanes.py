"""Context lane allocator for UserPromptSubmit.

The hook protocol still receives one ``additionalContext`` string. Lanes give the
assembly step bounded, named buckets so verbose recall/RAG/RLM context cannot
crowd out protected guardian/rules text.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class ContextLane:
    name: str
    text: str
    max_chars: int
    protected: bool = False


@dataclasses.dataclass(frozen=True)
class LaneRender:
    text: str
    stats: dict[str, dict[str, int | bool]]


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    marker = "\n[simba lane truncated]"
    keep = max(0, max_chars - len(marker))
    return text[:keep].rstrip() + marker, True


def render(lanes: list[ContextLane], *, enabled: bool = True) -> LaneRender:
    """Deduplicate and budget lanes, preserving input order."""
    parts: list[str] = []
    seen: set[str] = set()
    stats: dict[str, dict[str, int | bool]] = {}
    for lane in lanes:
        text = lane.text.strip()
        if not text:
            continue
        if not enabled:
            parts.append(text)
            stats[lane.name] = {
                "chars": len(text),
                "original_chars": len(text),
                "truncated": False,
                "protected": lane.protected,
            }
            continue
        if text in seen:
            continue
        seen.add(text)
        budget = 0 if lane.protected else lane.max_chars
        rendered, truncated = _truncate(text, budget)
        parts.append(rendered)
        stats[lane.name] = {
            "chars": len(rendered),
            "original_chars": len(text),
            "truncated": truncated,
            "protected": lane.protected,
        }
    return LaneRender(text="\n\n".join(parts), stats=stats)
