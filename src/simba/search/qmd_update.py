"""Install or upgrade the qmd CLI (``@tobilu/qmd``) for simba's project search.

qmd moved from a GitHub install (``qmd`` 1.x) to the npm package
``@tobilu/qmd``. ``simba update-qmd`` installs the latest release globally with
bun (npm when bun is missing) and reports the version before and after.

qmd migrates its index on first open after a major upgrade (1.x to 2.x
repartitioned the vectors and doubled the file), so a major-version change
first copies ``index.sqlite`` and its WAL next to the original as
``index.sqlite.bak-qmd-<old version>``.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import re
import shutil
import subprocess

PACKAGE = "@tobilu/qmd"
_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)")


@dataclasses.dataclass
class UpdateResult:
    before: str | None
    after: str | None
    latest: str | None
    installer: str | None
    backup: pathlib.Path | None
    ok: bool
    message: str = ""


def _run(cmd: list[str], timeout: float = 600.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def installed_version() -> str | None:
    """The installed qmd version, or ``None`` when qmd is missing."""
    if not shutil.which("qmd"):
        return None
    try:
        out = _run(["qmd", "--version"], timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    match = _VERSION.search(out.stdout + out.stderr)
    return match.group(0) if match else None


def latest_version() -> str | None:
    """The newest published version, or ``None`` when it cannot be looked up."""
    if not shutil.which("npm"):
        return None
    try:
        out = _run(["npm", "view", PACKAGE, "version"], timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    match = _VERSION.search(out.stdout)
    return match.group(0) if out.returncode == 0 and match else None


def index_path() -> pathlib.Path:
    cache = os.environ.get("XDG_CACHE_HOME") or str(pathlib.Path.home() / ".cache")
    return pathlib.Path(cache) / "qmd" / "index.sqlite"


def _major(version: str | None) -> int | None:
    match = _VERSION.search(version or "")
    return int(match.group(1)) if match else None


def backup_index(
    version: str, index: pathlib.Path | None = None
) -> pathlib.Path | None:
    """Copy the index and its WAL aside; return the backup path or ``None``."""
    index = index or index_path()
    if not index.exists():
        return None
    target = index.with_name(f"{index.name}.bak-qmd-{version}")
    shutil.copy2(index, target)
    wal = index.with_name(index.name + "-wal")
    if wal.exists():
        shutil.copy2(wal, target.with_name(target.name + "-wal"))
    return target


def installer() -> list[str] | None:
    if shutil.which("bun"):
        return ["bun", "install", "-g", f"{PACKAGE}@latest"]
    if shutil.which("npm"):
        return ["npm", "install", "-g", f"{PACKAGE}@latest"]
    return None


def update(*, dry_run: bool = False, force_backup: bool = False) -> UpdateResult:
    before = installed_version()
    latest = latest_version()
    cmd = installer()
    result = UpdateResult(before, before, latest, cmd[0] if cmd else None, None, False)
    if cmd is None:
        result.message = "neither bun nor npm is installed"
        return result
    if before and latest and before == latest and not force_backup:
        result.ok = True
        result.message = f"qmd {before} is already the latest"
        return result
    major_change = before is not None and _major(before) != _major(latest)
    if dry_run:
        result.ok = True
        result.message = f"would run: {' '.join(cmd)}" + (
            " (after backing up the index)" if major_change or force_backup else ""
        )
        return result
    if before and (major_change or force_backup):
        result.backup = backup_index(before)
    try:
        proc = _run(cmd)
    except (OSError, subprocess.SubprocessError) as exc:
        result.message = f"{' '.join(cmd)} failed: {exc}"
        return result
    result.after = installed_version()
    # A failed native build step (e.g. better-sqlite3 without node-gyp) can make
    # the installer exit non-zero while qmd itself installed and runs.
    result.ok = result.after is not None and (latest is None or result.after == latest)
    if not result.ok:
        tail = (proc.stderr or proc.stdout).strip()[-500:]
        result.message = f"{' '.join(cmd)} exited {proc.returncode}: {tail}"
    elif proc.returncode != 0:
        result.message = (
            f"installer exited {proc.returncode}, but qmd {result.after} runs"
        )
    return result
