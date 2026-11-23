"""Lazily create a qmd collection for a project that has none.

``rag_context`` only searches the collection that indexes the current project.
When there is none, this schedules ``qmd collection add`` for the project's git
root in a detached, low-priority process, so later prompts get project-scoped
results without the user running ``simba index``. Only BM25 indexing runs here;
vector embedding (``qmd embed``) stays an explicit step.

Guards: the directory must be inside a git work tree, must not be the home
directory or one of its ancestors, and must not live on a network filesystem
(walking an SMB share stalls the machine). Each root is attempted at most once
per ``_RETRY_SECONDS``, recorded in a marker file under ``~/.cache/simba``.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

import simba.search.qmd

_RETRY_SECONDS = 24 * 3600
_NETWORK_FS = frozenset(
    {"smbfs", "nfs", "nfs4", "afpfs", "cifs", "smb3", "webdav", "sshfs", "fuse.sshfs"}
)
_MOUNT_LINE = re.compile(r"^.+? on (?P<point>/.*?) \((?P<fstype>[^,\s)]+)")


def state_dir() -> pathlib.Path:
    return pathlib.Path.home() / ".cache" / "simba" / "qmd-autoindex"


def project_root(cwd: pathlib.Path) -> pathlib.Path | None:
    """Return the git work-tree root containing *cwd*, or ``None``."""
    try:
        out = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    top = out.stdout.strip()
    if out.returncode != 0 or not top:
        return None
    return pathlib.Path(top).resolve()


def mounts() -> list[tuple[str, str]]:
    """Return ``(mount point, filesystem type)`` pairs. Raises ``OSError``."""
    if sys.platform.startswith("linux"):
        pairs = []
        for line in pathlib.Path("/proc/mounts").read_text().splitlines():
            fields = line.split()
            if len(fields) >= 3:
                pairs.append((fields[1].replace("\\040", " "), fields[2]))
        return pairs
    mount = shutil.which("mount") or "/sbin/mount"
    try:
        text = subprocess.run(
            [mount], capture_output=True, text=True, timeout=3, check=True
        ).stdout
    except subprocess.SubprocessError as exc:
        raise OSError(str(exc)) from exc
    pairs = []
    for line in text.splitlines():
        match = _MOUNT_LINE.match(line)
        if match:
            pairs.append((match["point"], match["fstype"]))
    return pairs


def is_network_path(path: pathlib.Path) -> bool:
    """Whether *path* lives on a network filesystem. Unknown counts as network."""
    try:
        table = mounts()
    except OSError:
        return True
    best_point, best_type = "", ""
    text = str(path)
    for point, fstype in table:
        inside = text == point or text.startswith(point.rstrip("/") + "/")
        if inside and len(point) > len(best_point):
            best_point, best_type = point, fstype
    return best_type.lower() in _NETWORK_FS


def eligible(root: pathlib.Path) -> bool:
    home = pathlib.Path.home().resolve()
    if root == home or root in home.parents or len(root.parts) < 3:
        return False
    return not is_network_path(root)


def collection_name(root: pathlib.Path, existing: dict[str, str]) -> str:
    """A collection name for *root* that no other path already uses."""
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", root.name).strip("-") or "project"
    owner = existing.get(base)
    if owner is None or pathlib.Path(owner).expanduser().resolve() == root:
        return base
    digest = hashlib.sha1(str(root).encode()).hexdigest()[:6]
    return f"{base}-{digest}"


def _claim(root: pathlib.Path) -> pathlib.Path | None:
    """Create this root's attempt marker; ``None`` if attempted recently."""
    directory = state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / (hashlib.sha1(str(root).encode()).hexdigest()[:16] + ".json")
    try:
        if time.time() - marker.stat().st_mtime < _RETRY_SECONDS:
            return None
        marker.unlink()
    except FileNotFoundError:
        pass
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return None
    with os.fdopen(fd, "w") as handle:
        json.dump({"root": str(root), "attempted_at": time.time()}, handle)
    return marker


def schedule_index(cwd: pathlib.Path, config: pathlib.Path | None = None) -> bool:
    """Start background indexing for *cwd*'s project. Returns whether it started."""
    if not simba.search.qmd.is_available():
        return False
    root = project_root(cwd)
    if root is None or not eligible(root):
        return False
    existing = simba.search.qmd.collection_paths(config)
    if simba.search.qmd.collection_for(root, config) is not None:
        return False
    if _claim(root) is None:
        return False
    name = collection_name(root, existing)
    cmd = ["qmd", "collection", "add", str(root), "--name", name, "--mask", "**/*.md"]
    if shutil.which("nice"):
        cmd = ["nice", "-n", "10", *cmd]
    _spawn(cmd, root, state_dir() / f"{name}.log")
    return True


def _spawn(cmd: list[str], cwd: pathlib.Path, log: pathlib.Path) -> None:
    """Run *cmd* detached from the hook, appending its output to *log*."""
    with open(log, "ab") as out:
        subprocess.Popen(
            cmd,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
