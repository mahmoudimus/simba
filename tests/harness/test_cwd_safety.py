"""cwd-absent safety: dispatch must not leak to the process cwd.

Real Claude/Codex/pi payloads always carry ``cwd``. The only behavior change
allowed is that a *missing* ``cwd`` becomes a safe no-op instead of resolving to
``pathlib.Path.cwd()`` — which, inside the daemon, is the wrong project.
"""

from __future__ import annotations

import simba.harness.core as core

# A recognizable CORE block we can grep for in the rendered context. Uses the
# SIMBA:core markers that simba.guardian.extract_core detects.
_CORE_MARKER = "FIXTURE-CWD-SAFETY-SENTINEL-RULE"
_CLAUDE_MD = (
    "# Test project\n\n"
    "<!-- BEGIN SIMBA:core -->\n"
    f"- {_CORE_MARKER}\n"
    "<!-- END SIMBA:core -->\n"
)


def test_prompt_submit_with_cwd_injects_core_block(tmp_path):
    (tmp_path / "CLAUDE.md").write_text(_CLAUDE_MD)
    result = core.dispatch(
        "prompt_submit",
        {"prompt": "a" * 60, "cwd": str(tmp_path)},
    )
    assert _CORE_MARKER in result.additional_context


def test_prompt_submit_without_cwd_skips_core_block(tmp_path):
    # Same fixture exists on disk, but with no cwd in the payload the hook must
    # not scan ANY filesystem location — neither tmp_path nor the daemon's own
    # checkout (whose CLAUDE.md / .claude/rules carry real CORE rules).
    (tmp_path / "CLAUDE.md").write_text(_CLAUDE_MD)
    result = core.dispatch("prompt_submit", {"prompt": "a" * 60})
    # Our fixture marker must not appear (no scan of tmp_path) ...
    assert _CORE_MARKER not in result.additional_context
    # ... and neither must any of the simba repo's own CORE rules (no leak to the
    # process cwd). These strings live in .claude/rules/CORE_INSTRUCTIONS.md.
    assert "Pure Python" not in result.additional_context
    assert "Append-only storage" not in result.additional_context
    assert "Hook I/O protocol" not in result.additional_context
