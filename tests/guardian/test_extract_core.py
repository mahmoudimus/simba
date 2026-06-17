"""Tests for guardian extract_core module."""

from __future__ import annotations

import pathlib
import types

import simba.guardian.extract_core


class TestExtractCoreBlocks:
    def test_extracts_single_core_block(self):
        content = """\
# Rules
<!-- BEGIN SIMBA:core -->
- Rule one
- Rule two
<!-- END SIMBA:core -->
Other content.
"""
        blocks = simba.guardian.extract_core.extract_core_blocks(content)
        assert len(blocks) == 1
        assert "- Rule one" in blocks[0]
        assert "- Rule two" in blocks[0]

    def test_extracts_multiple_core_blocks(self):
        content = """\
## Section A
<!-- BEGIN SIMBA:core -->
- Block A rule
<!-- END SIMBA:core -->

## Section B
<!-- BEGIN SIMBA:core -->
- Block B rule
<!-- END SIMBA:core -->
"""
        blocks = simba.guardian.extract_core.extract_core_blocks(content)
        assert len(blocks) == 2
        assert "Block A rule" in blocks[0]
        assert "Block B rule" in blocks[1]

    def test_no_core_blocks_returns_empty(self):
        content = "# No core tags\nJust regular markdown.\n"
        blocks = simba.guardian.extract_core.extract_core_blocks(content)
        assert blocks == []

    def test_empty_content_returns_empty(self):
        blocks = simba.guardian.extract_core.extract_core_blocks("")
        assert blocks == []

    def test_excludes_core_tags_from_output(self):
        content = """\
<!-- BEGIN SIMBA:core -->
- Important rule
<!-- END SIMBA:core -->
"""
        blocks = simba.guardian.extract_core.extract_core_blocks(content)
        assert len(blocks) == 1
        assert "<!-- BEGIN SIMBA:core -->" not in blocks[0]
        assert "<!-- END SIMBA:core -->" not in blocks[0]

    def test_preserves_inner_content_formatting(self):
        content = """\
<!-- BEGIN SIMBA:core -->
- Rule with **bold**
- Rule with `code`
  - Nested item
<!-- END SIMBA:core -->
"""
        blocks = simba.guardian.extract_core.extract_core_blocks(content)
        assert "**bold**" in blocks[0]
        assert "`code`" in blocks[0]
        assert "  - Nested item" in blocks[0]

    def test_handles_empty_core_block(self):
        content = """\
<!-- BEGIN SIMBA:core -->
<!-- END SIMBA:core -->
"""
        blocks = simba.guardian.extract_core.extract_core_blocks(content)
        assert len(blocks) == 1
        assert blocks[0].strip() == ""

    def test_handles_core_tags_with_extra_spaces(self):
        content = """\
<!--  BEGIN  SIMBA:core  -->
- Spaced rule
<!--  END  SIMBA:core  -->
"""
        blocks = simba.guardian.extract_core.extract_core_blocks(content)
        assert len(blocks) == 1
        assert "Spaced rule" in blocks[0]

    def test_ignores_marker_examples_inside_fenced_code(self):
        content = """\
```markdown
<!-- BEGIN SIMBA:core -->
- Example only
<!-- END SIMBA:core -->
```

<!-- BEGIN SIMBA:core -->
- Live rule
<!-- END SIMBA:core -->
"""
        blocks = simba.guardian.extract_core.extract_core_blocks(content)
        assert len(blocks) == 1
        assert "Live rule" in blocks[0]
        assert "Example only" not in blocks[0]

    def test_malformed_closing_tag_no_match(self):
        content = """\
<!-- BEGIN SIMBA:core -->
- Unclosed rule
"""
        blocks = simba.guardian.extract_core.extract_core_blocks(content)
        assert blocks == []

    def test_malformed_opening_tag_no_match(self):
        content = """\
- Not a core block
<!-- END SIMBA:core -->
"""
        blocks = simba.guardian.extract_core.extract_core_blocks(content)
        assert blocks == []


class TestMain:
    def test_main_reads_claude_md(self, claude_md_with_core):
        cwd = claude_md_with_core.parent
        result = simba.guardian.extract_core.main(cwd=cwd)
        assert "Never delete files" in result
        assert "Always run tests" in result
        assert "descriptive variable names" in result

    def test_main_no_claude_md_returns_empty(self, tmp_path: pathlib.Path):
        result = simba.guardian.extract_core.main(cwd=tmp_path)
        assert result == ""

    def test_main_claude_md_without_core_returns_empty(
        self, claude_md_no_core: pathlib.Path
    ):
        result = simba.guardian.extract_core.main(cwd=claude_md_no_core.parent)
        assert result == ""

    def test_main_reads_configured_rules_file(self, tmp_path: pathlib.Path):
        rules = tmp_path / ".claude" / "rules"
        rules.mkdir(parents=True)
        (rules / "CORE_INSTRUCTIONS.md").write_text(
            "<!-- BEGIN SIMBA:core -->\n"
            "- Real rules-file constraint\n"
            "<!-- END SIMBA:core -->\n"
        )
        result = simba.guardian.extract_core.main(cwd=tmp_path)
        assert "Real rules-file constraint" in result

    def test_main_ignores_claude_skills_and_worktrees(
        self, tmp_path: pathlib.Path
    ):
        rules = tmp_path / ".claude" / "rules"
        rules.mkdir(parents=True)
        (rules / "CORE_INSTRUCTIONS.md").write_text(
            "<!-- BEGIN SIMBA:core -->\n"
            "- Real guardian rule\n"
            "<!-- END SIMBA:core -->\n"
        )
        skill = tmp_path / ".claude" / "skills" / "demo"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "<!-- BEGIN SIMBA:core -->\n"
            "- Skill example must not inject\n"
            "<!-- END SIMBA:core -->\n"
        )
        worktree = tmp_path / ".claude" / "worktrees" / "agent"
        worktree.mkdir(parents=True)
        (worktree / "README.md").write_text(
            "<!-- BEGIN SIMBA:core -->\n"
            "- Worktree stale rule must not inject\n"
            "<!-- END SIMBA:core -->\n"
        )
        result = simba.guardian.extract_core.main(cwd=tmp_path)
        assert "Real guardian rule" in result
        assert "Skill example" not in result
        assert "Worktree stale" not in result

    def test_capsule_mode_compacts_core(self, tmp_path: pathlib.Path, monkeypatch):
        rules = tmp_path / ".claude" / "rules"
        rules.mkdir(parents=True)
        (rules / "CORE_INSTRUCTIONS.md").write_text(
            "<!-- BEGIN SIMBA:core -->\n"
            "- **Very long rule**: keep this first clause. "
            "This extra explanatory detail should stay in the source file only.\n"
            "- **Second rule**: another clause with trailing detail that is not "
            "hot-path critical.\n"
            "**Signal**: End every response with `[✓ rules]`.\n"
            "<!-- END SIMBA:core -->\n"
        )
        monkeypatch.setattr(
            simba.guardian.extract_core,
            "_cfg",
            lambda _cwd: types.SimpleNamespace(
                core_filename="CORE_INSTRUCTIONS.md",
                core_injection_mode="capsule",
                core_capsule_max_chars=420,
                core_capsule_rule_chars=70,
            ),
        )
        result = simba.guardian.extract_core.main(cwd=tmp_path)
        assert "<simba-core-capsule>" in result
        assert "Source: .claude/rules/CORE_INSTRUCTIONS.md" in result
        assert "- Very long rule: keep this first clause" in result
        assert "extra explanatory detail" not in result
        assert "Signal: end every response with `[✓ rules]`." in result
        assert len(result) <= 420
