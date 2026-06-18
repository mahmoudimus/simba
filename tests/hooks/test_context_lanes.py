"""Tests for UserPromptSubmit context lane budgeting."""

from __future__ import annotations

import simba.hooks.context_lanes as lanes


def test_render_truncates_unprotected_lane() -> None:
    rendered = lanes.render(
        [
            lanes.ContextLane("guardian", "RULES", 10, protected=True),
            lanes.ContextLane("recall", "x" * 50, 20),
        ]
    )
    assert "RULES" in rendered.text
    assert "x" * 50 not in rendered.text
    assert "[simba lane truncated]" in rendered.text
    assert rendered.stats["recall"]["truncated"] is True


def test_disabled_preserves_duplicate_context() -> None:
    rendered = lanes.render(
        [
            lanes.ContextLane("a", "same", 1),
            lanes.ContextLane("b", "same", 1),
        ],
        enabled=False,
    )
    assert rendered.text == "same\n\nsame"


def test_enabled_dedupes_exact_duplicate_context() -> None:
    rendered = lanes.render(
        [
            lanes.ContextLane("a", "same", 100),
            lanes.ContextLane("b", "same", 100),
        ]
    )
    assert rendered.text == "same"
