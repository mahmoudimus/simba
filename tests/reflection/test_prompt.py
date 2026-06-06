"""Reflection prompt builder (Phase 5, Task A.3)."""

from __future__ import annotations


def test_prompt_includes_project_and_memories() -> None:
    from simba.reflection.prompt import build_reflection_prompt

    mems = [{"id": "m1", "type": "GOTCHA", "content": "rg has no -r", "context": ""}]
    prompt = build_reflection_prompt(
        mems, project="/myproj", existing_reflections=[], max_reflections=3
    )
    assert "/myproj" in prompt
    assert "rg has no -r" in prompt
    assert "m1" in prompt
    assert "REFLECTION" in prompt


def test_prompt_caps_source_memories() -> None:
    from simba.reflection.prompt import build_reflection_prompt

    mems = [
        {"id": f"m{i}", "type": "PATTERN", "content": f"c{i}", "context": ""}
        for i in range(200)
    ]
    prompt = build_reflection_prompt(
        mems, project="/p", existing_reflections=[], max_source_memories=10
    )
    # Only m0..m9 should appear, m10+ must not
    assert "m9" in prompt
    assert "m10" not in prompt


def test_prompt_lists_existing_reflections() -> None:
    from simba.reflection.prompt import build_reflection_prompt

    existing = [
        {"id": "r1", "type": "REFLECTION", "content": "old insight", "context": ""}
    ]
    prompt = build_reflection_prompt([], project="/p", existing_reflections=existing)
    assert "old insight" in prompt
