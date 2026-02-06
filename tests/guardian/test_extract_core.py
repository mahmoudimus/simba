"""Tests for guardian extract_core module."""

from __future__ import annotations

import pathlib

import simba.guardian.extract_core


class TestExtractCoreBlocks:
    def test_extracts_single_core_block(self):
        content = """\
# Rules
<!-- CORE -->
- Rule one
- Rule two
<!-- /CORE -->
Other content.
"""
        blocks = simba.guardian.extract_core.extract_core_blocks(content)
        assert len(blocks) == 1
        assert "- Rule one" in blocks[0]
        assert "- Rule two" in blocks[0]

    def test_extracts_multiple_core_blocks(self):
        content = """\
## Section A
<!-- CORE -->
- Block A rule
<!-- /CORE -->

## Section B
<!-- CORE -->
- Block B rule
<!-- /CORE -->
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
<!-- CORE -->
- Important rule
<!-- /CORE -->
"""
        blocks = simba.guardian.extract_core.extract_core_blocks(content)
        assert len(blocks) == 1
        assert "<!-- CORE -->" not in blocks[0]
        assert "<!-- /CORE -->" not in blocks[0]

    def test_preserves_inner_content_formatting(self):
        content = """\
<!-- CORE -->
- Rule with **bold**
- Rule with `code`
  - Nested item
<!-- /CORE -->
"""
        blocks = simba.guardian.extract_core.extract_core_blocks(content)
        assert "**bold**" in blocks[0]
        assert "`code`" in blocks[0]
        assert "  - Nested item" in blocks[0]

    def test_handles_empty_core_block(self):
        content = """\
<!-- CORE -->
<!-- /CORE -->
"""
        blocks = simba.guardian.extract_core.extract_core_blocks(content)
        assert len(blocks) == 1
        assert blocks[0].strip() == ""

    def test_handles_core_tags_with_extra_spaces(self):
        content = """\
<!--  CORE  -->
- Spaced rule
<!--  /CORE  -->
"""
        blocks = simba.guardian.extract_core.extract_core_blocks(content)
        assert len(blocks) == 1
        assert "Spaced rule" in blocks[0]

    def test_malformed_closing_tag_no_match(self):
        content = """\
<!-- CORE -->
- Unclosed rule
"""
        blocks = simba.guardian.extract_core.extract_core_blocks(content)
        assert blocks == []

    def test_malformed_opening_tag_no_match(self):
        content = """\
- Not a core block
<!-- /CORE -->
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
