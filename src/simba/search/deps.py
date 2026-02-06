"""External tool dependency checker for search utilities.

Checks availability and versions of CLI tools required by the search package.
"""

from __future__ import annotations

import platform
import shutil
import subprocess


def check_dependency(name: str) -> tuple[bool, str]:
    """Check whether *name* is installed and return its version string.

    Returns ``(True, version)`` when the tool is found, or
    ``(False, "not found")`` otherwise.
    """
    if shutil.which(name) is None:
        return (False, "not found")
    try:
        result = subprocess.run(
            [name, "--version"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        version = result.stdout.strip() or result.stderr.strip()
        return (True, version or "unknown")
    except (subprocess.SubprocessError, OSError):
        return (True, "unknown")


def check_all() -> dict[str, tuple[bool, str]]:
    """Check availability of all required external tools."""
    return {name: check_dependency(name) for name in ("rg", "fzf", "jq", "qmd")}


def get_install_instructions(name: str) -> str:
    """Return platform-specific install instructions for *name*."""
    system = platform.system()
    if name == "qmd":
        if system == "Darwin":
            return "bun install -g https://github.com/tobi/qmd"
        return "bun install -g https://github.com/tobi/qmd"
    if system == "Darwin":
        return f"brew install {name}"
    if system == "Linux":
        return f"apt install {name}  (or)  pacman -S {name}"
    return f"Install {name} via your system package manager"
