"""Validation for bundled skill definitions."""

from __future__ import annotations

import pathlib

_REPO = pathlib.Path(__file__).resolve().parents[1]


def _frontmatter(path: pathlib.Path) -> tuple[dict[str, str], str]:
    text = path.read_text()
    assert text.startswith("---"), "SKILL.md must start with YAML frontmatter"
    block = text.split("---", 2)[1]
    data: dict[str, str] = {}
    for line in block.strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            data[key.strip()] = value.strip()
    return data, text


class TestRecallVerifySkill:
    SKILL = _REPO / "skills" / "memories-recall-verify" / "SKILL.md"

    def test_skill_file_exists(self) -> None:
        assert self.SKILL.is_file()

    def test_frontmatter_name_and_description(self) -> None:
        fm, _ = _frontmatter(self.SKILL)
        assert fm["name"] == "memories-recall-verify"
        assert fm["description"]

    def test_body_uses_recall_cli(self) -> None:
        _, text = _frontmatter(self.SKILL)
        # The self-correcting loop re-queries memory via the CLI.
        assert "simba memory recall" in text
