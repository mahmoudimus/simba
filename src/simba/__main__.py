"""Simba CLI — unified Claude Code plugin.

Usage:
    simba install          Register hooks in current project
    simba install --global Register hooks globally (~/.claude/settings.json)
    simba install --remove Remove hooks (add --global for global)
    simba codex-install    Install bundled skills for Codex (~/.codex/skills)
    simba codex-install --remove
                           Remove bundled Codex skills
    simba codex-status     Check daemon health + pending transcript extraction
    simba codex-extract    Show extraction prompt for pending transcript
    simba codex-recall     Query semantic memory (/recall) for a text query
    simba codex-finalize   Run end-of-task signal/error checks
    simba codex-automation Print suggested Codex automation directive
    simba server [opts]    Start the memory daemon
    simba memory store     Store a memory (--type, --content, --context, --confidence)
    simba memory recall    Recall memories for a query text
    simba memory list      List all memories (optional --type filter)
    simba memory delete    Delete a memory by ID
    simba memory update    Update memory metadata (--project-path, --session-source)
    simba memory reindex   Rebuild the hybrid-recall BM25 keyword mirror
    simba search <cmd>     Project memory operations
    simba sync <cmd>       Sync SQLite, LanceDB, and QMD
    simba stats            Show token economics and project statistics
    simba eval <cmd>       Recall eval harness (run | build from real corpus)
    simba neuron <cmd>     Neuro-symbolic logic server (MCP)
    simba orchestration <cmd> Agent orchestration server (MCP)
    simba config <cmd>     Unified configuration (get/set/list/show)
    simba markers <cmd>    Discover, audit, and update SIMBA markers
    simba rule <cmd>       Manage tool rules (auto-learned + manual)
    simba rlm <cmd>        RLM autonomous engine commands (digest)
    simba episodes <cmd>   Episodic consolidation control (complete)
    simba db <subcmd>      Inspect or migrate the shared database
    simba hook <event>     Run a hook (called by Claude Code, not users)
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import re
import sys
from typing import Any

_HOOK_EVENTS = {
    "SessionStart": "simba.hooks.session_start",
    "UserPromptSubmit": "simba.hooks.user_prompt_submit",
    "PreToolUse": "simba.hooks.pre_tool_use",
    "PostToolUse": "simba.hooks.post_tool_use",
    "PreCompact": "simba.hooks.pre_compact",
    "Stop": "simba.hooks.stop",
    # Codex-only: emitted just before Codex prompts for approval.
    # Claude Code never invokes this event, so registering it here is
    # harmless for Claude installs.
    "PermissionRequest": "simba.hooks.permission_request",
}

_HOOK_TIMEOUTS = {
    "SessionStart": 15000,
    "UserPromptSubmit": 5000,
    "PreToolUse": 5000,
    "PostToolUse": 5000,
    "PreCompact": 5000,
    "Stop": 5000,
    "PermissionRequest": 3000,
}

# Subset of _HOOK_EVENTS that Claude Code understands.  PermissionRequest
# is Codex-only and must not appear in Claude's settings.json.
_CLAUDE_HOOK_EVENTS = (
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PreCompact",
    "Stop",
)

# Subset that Codex understands.  Codex has no PreCompact event.
_CODEX_HOOK_EVENTS = (
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "Stop",
    "PermissionRequest",
)

_GLOBAL_SETTINGS = pathlib.Path.home() / ".claude" / "settings.json"


_CODEX_HOOKS_FLAG = "hooks"
_CODEX_HOOKS_FLAG_LEGACY = "codex_hooks"


def _apply_codex_feature_flag(
    config_path: pathlib.Path,
    *,
    remove: bool = False,
    create_if_missing: bool = True,
) -> str:
    """Toggle ``[features] hooks`` in a Codex ``config.toml``.

    Returns one of:
      - ``"added"`` (newly set to true)
      - ``"already-set"`` (was already true; no write)
      - ``"migrated"`` (renamed legacy ``codex_hooks`` to ``hooks``)
      - ``"removed"`` (flag deleted)
      - ``"not-present"`` (remove called or file missing and not creating)

    When ``create_if_missing`` is false and the file doesn't exist, this
    returns ``"not-present"`` without creating anything — used for the
    project-local config which only needs migration, not creation.
    """
    import tomllib

    import tomli_w

    data: dict = {}
    if config_path.exists():
        try:
            data = tomllib.loads(config_path.read_text())
        except (OSError, tomllib.TOMLDecodeError):
            data = {}
    elif not create_if_missing:
        return "not-present"

    features = data.setdefault("features", {})
    had_legacy = features.pop(_CODEX_HOOKS_FLAG_LEGACY, None) is not None

    if remove:
        current = features.pop(_CODEX_HOOKS_FLAG, None)
        if current is None and not had_legacy:
            return "not-present"
        if not features:
            data.pop("features", None)
        status = "removed"
    else:
        if features.get(_CODEX_HOOKS_FLAG) is True and not had_legacy:
            return "already-set"
        features[_CODEX_HOOKS_FLAG] = True
        status = "migrated" if had_legacy else "added"

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_bytes(tomli_w.dumps(data).encode())
    return status


def _ensure_codex_feature_flag(*, remove: bool = False) -> str:
    """Toggle ``[features] hooks`` in ``<CODEX_HOME>/config.toml``.

    Codex deprecated ``codex_hooks`` in favor of ``hooks``; if the old
    key exists it is removed in both install and ``--remove`` modes.
    """
    return _apply_codex_feature_flag(_codex_home() / "config.toml", remove=remove)


def _migrate_project_codex_features(project_dir: pathlib.Path) -> str:
    """Rename legacy ``codex_hooks`` to ``hooks`` in ``<project>/.codex/config.toml``.

    No-op if the file doesn't exist — we never create a project-local
    config.toml, only migrate one if the user already has it.
    """
    return _apply_codex_feature_flag(
        project_dir / ".codex" / "config.toml", create_if_missing=False
    )


def _codex_home() -> pathlib.Path:
    """Return CODEX_HOME (or ~/.codex)."""
    env_home = os.environ.get("CODEX_HOME")
    if env_home:
        return pathlib.Path(env_home).expanduser()
    return pathlib.Path.home() / ".codex"


def _build_hooks_config() -> dict:
    """Build the hooks section for Claude Code's settings.json."""
    hooks: dict = {}
    for event in _CLAUDE_HOOK_EVENTS:
        timeout = _HOOK_TIMEOUTS[event]
        hooks[event] = [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": f"simba hook {event}",
                        "timeout": timeout,
                    }
                ]
            }
        ]
    return hooks


# Matchers used in per-project .codex/hooks.json
_CODEX_TOOL_MATCHER = "Bash|apply_patch|Edit|Write"
_CODEX_SESSION_MATCHER = "startup|resume|clear"


def _build_codex_hooks_config() -> dict:
    """Build the hooks dict for a per-project .codex/hooks.json."""
    hooks: dict = {}
    for event in _CODEX_HOOK_EVENTS:
        timeout_ms = _HOOK_TIMEOUTS[event]
        timeout_s = timeout_ms // 1000
        entry: dict = {
            "hooks": [
                {
                    "type": "command",
                    "command": f"simba hook {event}",
                    "timeout": timeout_s,
                }
            ]
        }
        if event == "SessionStart":
            entry["matcher"] = _CODEX_SESSION_MATCHER
        elif event in ("PreToolUse", "PostToolUse", "PermissionRequest"):
            entry["matcher"] = _CODEX_TOOL_MATCHER
        # UserPromptSubmit and Stop: no matcher (Codex ignores it anyway)
        hooks[event] = [entry]
    return {"hooks": hooks}


def _write_codex_project_hooks(
    project_dir: pathlib.Path, *, remove: bool = False
) -> bool:
    """Write or remove .codex/hooks.json in ``project_dir``.

    Returns True if a change was made.
    """
    hooks_path = project_dir / ".codex" / "hooks.json"
    if remove:
        if hooks_path.exists():
            hooks_path.unlink()
            return True
        return False
    hooks_path.parent.mkdir(parents=True, exist_ok=True)
    hooks_path.write_text(json.dumps(_build_codex_hooks_config(), indent=2) + "\n")
    return True


def _latest_codex_transcript_metadata() -> dict[str, Any] | None:
    """Build transcript metadata from the newest Codex session JSONL."""
    sessions_dir = _codex_home() / "sessions"
    if not sessions_dir.exists():
        return None

    candidates: list[pathlib.Path] = []
    try:
        for path in sessions_dir.rglob("*.jsonl"):
            if path.is_file():
                candidates.append(path)
    except OSError:
        return None

    if not candidates:
        return None

    def _session_sort_key(path: pathlib.Path) -> tuple[int, str, float]:
        # Prefer Codex rollout timestamp from filename when available.
        # Example: rollout-2026-02-20T08-33-13-<id>.jsonl
        m = re.search(
            r"rollout-(\d{4})-(\d{2})-(\d{2})T(\d{2})-(\d{2})-(\d{2})-",
            path.name,
        )
        ts_key = ""
        has_ts = 0
        if m:
            has_ts = 1
            ts_key = "".join(m.groups())  # yyyymmddHHMMSS
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (has_ts, ts_key, mtime)

    try:
        latest = max(candidates, key=_session_sort_key)
    except ValueError:
        return None

    session_id = latest.stem
    project_path = str(pathlib.Path.cwd())
    try:
        with latest.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(entry, dict):
                    continue
                if entry.get("type") != "session_meta":
                    continue
                payload = entry.get("payload", {})
                if isinstance(payload, dict):
                    sid = payload.get("id")
                    cwd = payload.get("cwd")
                    if isinstance(sid, str) and sid:
                        session_id = sid
                    if isinstance(cwd, str) and cwd:
                        project_path = cwd
                break
    except OSError:
        return None

    return {
        "session_id": session_id,
        "project_path": project_path,
        "transcript_path": str(latest),
        "status": "pending_extraction",
        "source": "codex",
    }


def _latest_claude_transcript_metadata() -> dict[str, Any] | None:
    """Load latest transcript metadata from ~/.claude/transcripts/latest.json."""
    latest = pathlib.Path.home() / ".claude" / "transcripts" / "latest.json"
    if not latest.exists():
        return None
    try:
        data = json.loads(latest.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    target = latest.resolve() if latest.is_symlink() else latest
    data["_metadata_path"] = str(target)
    data.setdefault("source", "claude")
    return data


def _latest_transcript_metadata() -> dict[str, Any] | None:
    """Load latest transcript metadata, preferring Codex sessions."""
    codex_meta = _latest_codex_transcript_metadata()
    if codex_meta is not None:
        return codex_meta
    return _latest_claude_transcript_metadata()


def _extract_transcript_text(path: pathlib.Path) -> str:
    """Extract plain text from markdown or JSONL transcript."""
    if not path.exists():
        return ""
    try:
        raw = path.read_text()
    except OSError:
        return ""

    # JSONL transcript: parse message/tool fields.
    if path.suffix == ".jsonl":
        parts: list[str] = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            msg = entry.get("message", {})
            if isinstance(msg, dict):
                content = msg.get("content", [])
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            txt = (
                                item.get("text")
                                or item.get("content")
                                or item.get("thinking")
                            )
                            if isinstance(txt, str) and txt.strip():
                                parts.append(txt.strip())
            for key in ("toolUseResult", "text", "content"):
                val = entry.get(key)
                if isinstance(val, str) and val.strip():
                    parts.append(val.strip())
        return "\n".join(parts)

    # Markdown transcript: drop tags and use remaining text.
    return re.sub(r"<[^>]+>", " ", raw)


def _classify_learning(sentence: str) -> tuple[str, float] | None:
    """Classify a sentence into a memory type with confidence."""
    s = sentence.lower()
    if re.search(r"\b(prefer|prefers|always use|always prefer|likes?)\b", s):
        return ("PREFERENCE", 0.90)
    if re.search(r"\b(fail|fails|failed|broke|broken|error|exception)\b", s):
        return ("FAILURE", 0.88)
    if re.search(r"\b(chose|decided|selected|picked)\b", s):
        return ("DECISION", 0.90)
    if re.search(r"\b(watch out|beware|careful|avoid|don't|never)\b", s):
        return ("GOTCHA", 0.88)
    if re.search(r"\b(pattern|convention|workflow|approach)\b", s):
        return ("PATTERN", 0.85)
    if re.search(r"\b(use|run|fix|resolve|works|worked|solves?)\b", s):
        return ("WORKING_SOLUTION", 0.86)
    return None


def _extract_learnings(
    transcript_text: str,
    *,
    max_items: int = 15,
    max_content_length: int | None = None,
) -> list[dict[str, Any]]:
    """Extract candidate learnings from transcript text heuristically."""
    if max_content_length is None:
        max_content_length = _memory_max_content_length()
    # Split into sentence-like units and normalize whitespace.
    chunks = re.split(r"[.\n]+", transcript_text)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []

    for raw in chunks:
        sentence = " ".join(raw.strip().split())
        if len(sentence) < 24:
            continue
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)

        tagged = _classify_learning(sentence)
        if tagged is None:
            continue

        mtype, conf = tagged
        out.append(
            {
                "type": mtype,
                "content": sentence[:max_content_length],
                "context": "extracted from transcript",
                "confidence": conf,
            }
        )
        if len(out) >= max_items:
            break

    return out


def _bundled_skill_names() -> list[str]:
    """Return names of all bundled skills."""
    import importlib.resources

    skills_pkg = importlib.resources.files("simba") / "skills"
    if not skills_pkg.is_dir():
        return []
    return [
        d.name
        for d in skills_pkg.iterdir()
        if d.is_dir() and (d / "skill.md").is_file()
    ]


def _install_skills(skills_dir: pathlib.Path) -> int:
    """Copy bundled skills into *skills_dir*."""
    import importlib.resources

    skills_pkg = importlib.resources.files("simba") / "skills"
    if not skills_pkg.is_dir():
        return 0

    copied = 0
    for skill_dir in skills_pkg.iterdir():
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "skill.md"
        if not skill_md.is_file():
            continue
        dest_dir = skills_dir / skill_dir.name
        dest_file = dest_dir / "skill.md"
        if dest_file.exists():
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file.write_text(skill_md.read_text())
        print(f"  + skill: /{skill_dir.name}")
        copied += 1
    return copied


def _remove_skills(skills_dir: pathlib.Path) -> int:
    """Remove bundled skills from *skills_dir*."""
    import shutil

    removed = 0
    for name in _bundled_skill_names():
        dest_dir = skills_dir / name
        if dest_dir.is_dir():
            shutil.rmtree(dest_dir)
            print(f"  - skill: /{name}")
            removed += 1
    return removed


def _bundled_codex_skill_names() -> list[str]:
    """Return names of bundled Codex skills (SKILL.md)."""
    import importlib.resources

    skills_pkg = importlib.resources.files("simba") / "codex_skills"
    if not skills_pkg.is_dir():
        return []
    return [
        d.name
        for d in skills_pkg.iterdir()
        if d.is_dir() and (d / "SKILL.md").is_file()
    ]


def _install_codex_skills(skills_dir: pathlib.Path) -> int:
    """Copy bundled Codex skills (SKILL.md + agents metadata)."""
    import importlib.resources

    skills_pkg = importlib.resources.files("simba") / "codex_skills"
    if not skills_pkg.is_dir():
        return 0

    installed = 0
    for skill_dir in skills_pkg.iterdir():
        if not skill_dir.is_dir():
            continue
        src_skill = skill_dir / "SKILL.md"
        if not src_skill.is_file():
            continue

        dest_dir = skills_dir / skill_dir.name
        dest_skill = dest_dir / "SKILL.md"
        dest_dir.mkdir(parents=True, exist_ok=True)
        if not dest_skill.exists():
            dest_skill.write_text(src_skill.read_text())
            print(f"  + codex skill: {skill_dir.name}")
            installed += 1

        src_agents = skill_dir / "agents"
        if src_agents.is_dir():
            for src_file in src_agents.rglob("*"):
                if not src_file.is_file():
                    continue
                rel_path = src_file.relative_to(skill_dir)
                dst_file = dest_dir / rel_path
                if dst_file.exists():
                    continue
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                dst_file.write_text(src_file.read_text())
    return installed


def _remove_codex_skills(skills_dir: pathlib.Path) -> int:
    """Remove bundled Codex skills from CODEX_HOME."""
    import shutil

    removed = 0
    for name in _bundled_codex_skill_names():
        dest_dir = skills_dir / name
        if dest_dir.is_dir():
            shutil.rmtree(dest_dir)
            print(f"  - codex skill: {name}")
            removed += 1
    return removed


def _cmd_install(args: list[str]) -> int:
    """Register or remove simba hooks.

    By default writes to .claude/settings.local.json in the current
    project.  Use ``--global`` to write to ~/.claude/settings.json
    instead.
    """
    remove = "--remove" in args
    is_global = "--global" in args

    if is_global:
        settings_path = _GLOBAL_SETTINGS
    else:
        settings_path = pathlib.Path.cwd() / ".claude" / "settings.local.json"

    if not settings_path.parent.exists():
        settings_path.parent.mkdir(parents=True, exist_ok=True)

    settings: dict = {}
    if settings_path.exists():
        settings = json.loads(settings_path.read_text())

    if is_global:
        skills_dir = pathlib.Path.home() / ".claude" / "skills"
    else:
        skills_dir = pathlib.Path.cwd() / ".claude" / "skills"

    simba_permission = "Bash(simba:*)"

    project_dir = pathlib.Path.cwd()

    if remove:
        if "hooks" in settings:
            del settings["hooks"]
        perms = settings.get("permissions", {})
        allow = perms.get("allow", [])
        if simba_permission in allow:
            allow.remove(simba_permission)
            perms["allow"] = allow
            settings["permissions"] = perms
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")
        print("Simba hooks removed from", settings_path)
        removed = _remove_skills(skills_dir)
        if removed:
            print(f"  {removed} skill(s) removed")
        if not is_global and _write_codex_project_hooks(project_dir, remove=True):
            print(f"  Codex hooks removed from {project_dir / '.codex' / 'hooks.json'}")
        return 0

    settings["hooks"] = _build_hooks_config()
    perms = settings.setdefault("permissions", {})
    allow = perms.setdefault("allow", [])
    if simba_permission not in allow:
        allow.append(simba_permission)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    scope = "global" if is_global else "project"
    print(f"Simba hooks registered ({scope}) in {settings_path}")
    hooks_list = ", ".join(_CLAUDE_HOOK_EVENTS)
    print(f"  {len(_CLAUDE_HOOK_EVENTS)} Claude hooks: {hooks_list}")
    print(f"  permission granted: {simba_permission}")

    skill_count = _install_skills(skills_dir)
    if skill_count:
        print(f"  {skill_count} skill(s) installed")

    if not is_global:
        _write_codex_project_hooks(project_dir)
        codex_hooks_path = project_dir / ".codex" / "hooks.json"
        print(f"  {len(_CODEX_HOOK_EVENTS)} Codex hooks written to {codex_hooks_path}")
        project_flag = _migrate_project_codex_features(project_dir)
        if project_flag == "migrated":
            local_cfg = project_dir / ".codex" / "config.toml"
            print(f"  [features] codex_hooks -> hooks migrated in {local_cfg}")

    return 0


def _cmd_codex_install(args: list[str]) -> int:
    """Install or remove bundled skills + hook feature flag for Codex."""
    remove = "--remove" in args
    skills_dir = _codex_home() / "skills"
    config_path = _codex_home() / "config.toml"

    if remove:
        removed = _remove_codex_skills(skills_dir)
        print(f"Codex skills removed from {skills_dir}")
        if removed:
            print(f"  {removed} skill(s) removed")
        flag_status = _ensure_codex_feature_flag(remove=True)
        if flag_status == "removed":
            print(f"  [features] hooks removed from {config_path}")
        return 0

    skills_dir.mkdir(parents=True, exist_ok=True)
    installed = _install_codex_skills(skills_dir)
    print(f"Codex skills installed in {skills_dir}")
    if installed:
        print(f"  {installed} skill(s) installed")
    flag_status = _ensure_codex_feature_flag()
    if flag_status == "added":
        print(f"  [features] hooks = true set in {config_path}")
    elif flag_status == "migrated":
        print(f"  [features] codex_hooks -> hooks migrated in {config_path}")
    elif flag_status == "already-set":
        print(f"  [features] hooks = true already set in {config_path}")
    return 0


def _cmd_codex_status(args: list[str]) -> int:
    """Check Codex-oriented Simba status: daemon + pending extraction."""
    del args
    import httpx

    import simba.hooks._memory_client

    url = simba.hooks._memory_client.daemon_url()
    print(f"[codex] daemon: {url}")

    health_ok = False
    try:
        resp = httpx.get(f"{url}/health", timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            health_ok = True
            print(
                "[codex] memory: up "
                f"(count={data.get('memoryCount', 0)}, "
                f"model={data.get('embeddingModel', 'unknown')})"
            )
            # Mirror Claude SessionStart behavior: trigger one sync cycle.
            with contextlib.suppress(httpx.HTTPError, ValueError):
                httpx.post(f"{url}/sync", timeout=1.0)
                print("[codex] sync: triggered")
    except (httpx.HTTPError, ValueError):
        pass

    if not health_ok:
        print("[codex] memory: down (start with `simba server`)")

    meta = _latest_codex_transcript_metadata()
    if not meta:
        print("[codex] extraction: no latest Codex transcript metadata found")
        return 0

    status = meta.get("status", "unknown")
    transcript = meta.get("transcript_path", "")
    session_id = meta.get("session_id", "")
    source = meta.get("source", "unknown")
    print(f"[codex] transcript source: {source}")
    print(f"[codex] latest transcript: {transcript or 'unknown'}")
    print(f"[codex] latest session: {session_id or 'unknown'}")
    print(f"[codex] extraction status: {status}")
    if status == "pending_extraction":
        print("[codex] next: run `simba codex-extract`")
    return 0


def _cmd_codex_extract(args: list[str]) -> int:
    """Print a ready-to-run extraction prompt and optionally mark it done."""
    import httpx

    mark_done = "--mark-done" in args
    run_mode = "--run" in args

    meta = _latest_codex_transcript_metadata()
    if not meta:
        print("No transcript metadata found in Codex sessions (~/.codex/sessions).")
        return 1

    transcript = meta.get("transcript_path", "")
    session_id = meta.get("session_id", "")
    project_path = meta.get("project_path", str(pathlib.Path.cwd()))
    status = meta.get("status", "")

    if not transcript:
        print("Latest transcript metadata is missing transcript_path")
        return 1

    if status and status != "pending_extraction":
        print(f"Extraction status is '{status}' (not pending).")

    if run_mode:
        transcript_path = pathlib.Path(transcript)
        text = _extract_transcript_text(transcript_path)
        if not text.strip():
            print(f"No readable transcript content found in {transcript_path}")
            return 1

        learnings = _extract_learnings(text, max_items=15)
        if not learnings:
            print("No candidate learnings found heuristically.")
            print(
                "Fallback: run `simba codex-extract` without --run for manual prompt."
            )
            return 1

        daemon = "http://localhost:8741"
        stored = 0
        duplicates = 0
        errors = 0

        for mem in learnings:
            payload = {
                "type": mem["type"],
                "content": mem["content"],
                "context": mem["context"],
                "confidence": mem["confidence"],
                "sessionSource": session_id,
                "projectPath": project_path,
            }
            try:
                resp = httpx.post(f"{daemon}/store", json=payload, timeout=10.0)
                resp.raise_for_status()
                body = resp.json()
                if body.get("status") == "stored":
                    stored += 1
                elif body.get("status") == "duplicate":
                    duplicates += 1
                else:
                    errors += 1
            except (httpx.HTTPError, ValueError):
                errors += 1

        print(
            f"[codex] extract run complete: candidates={len(learnings)} "
            f"stored={stored} duplicate={duplicates} errors={errors}"
        )
    else:
        print("Use this prompt with Codex (or the `memories-learn` skill):")
        print("---")
        print(
            f"Read transcript `{transcript}` and extract as many "
            "high-value learnings. As you go through each learning, "
            "semantically cluster them and see if similar learnings "
            "can be coalesced into just one learning. "
            "Store each learning to semantic memory using:"
        )
        print(
            'simba memory store --type <TYPE> --content "<LEARNING>" '
            '--context "<CONTEXT>" --confidence <SCORE> '
            f'--session-source "{session_id}" --project-path "{project_path}"'
        )
        print(
            "Types: WORKING_SOLUTION, GOTCHA, PATTERN, DECISION, FAILURE, PREFERENCE."
        )
        print("---")

    if mark_done:
        meta_path = meta.get("_metadata_path")
        if isinstance(meta_path, str) and meta_path:
            meta["status"] = "extracted"
            to_write = {k: v for k, v in meta.items() if not k.startswith("_")}
            target = pathlib.Path(meta_path)
            target.write_text(json.dumps(to_write, indent=2))
            print(f"Updated extraction status to 'extracted' in {target}")
        else:
            print(
                "Mark-done not persisted for Codex session JSONL metadata "
                "(no writable latest.json)."
            )

    return 0


def _cmd_codex_recall(args: list[str]) -> int:
    """Recall memories for a query via the memory daemon."""
    if not args:
        print("Usage: simba codex-recall <query text>", file=sys.stderr)
        return 1

    query = " ".join(args).strip()
    if not query:
        print("Usage: simba codex-recall <query text>", file=sys.stderr)
        return 1

    import simba.hooks._memory_client

    memories = simba.hooks._memory_client.recall_memories(
        query,
        project_path=str(pathlib.Path.cwd()),
    )
    if not memories:
        print("[codex] recall: no memories")
        return 0

    print(f"[codex] recall: {len(memories)} memories")
    for m in memories:
        mtype = m.get("type", "UNKNOWN")
        sim = m.get("similarity", 0.0)
        content = str(m.get("content", "")).strip()
        print(f"- [{mtype}] ({sim:.2f}) {content}")
    return 0


def _parse_opt_value(args: list[str], key: str) -> str | None:
    """Parse `--key value` from args."""
    if key not in args:
        return None
    idx = args.index(key)
    if idx + 1 >= len(args):
        return None
    return args[idx + 1]


def _cmd_codex_finalize(args: list[str]) -> int:
    """Run end-of-task checks equivalent to the Stop hook."""
    response = _parse_opt_value(args, "--response") or ""
    response_file = _parse_opt_value(args, "--response-file")
    transcript = _parse_opt_value(args, "--transcript")

    if response_file:
        try:
            response = pathlib.Path(response_file).read_text()
        except OSError as exc:
            print(f"Failed to read --response-file: {exc}", file=sys.stderr)
            return 1

    if not transcript:
        meta = _latest_codex_transcript_metadata()
        if meta:
            transcript = meta.get("transcript_path", "")

    import simba.guardian.check_signal
    import simba.tailor.hook

    if response:
        signal_result = simba.guardian.check_signal.main(
            response=response, cwd=pathlib.Path.cwd()
        )
        if signal_result:
            print(signal_result)
        else:
            print("[codex] signal check: ok ([✓ rules] present)")
    else:
        print("[codex] signal check: skipped (no response provided)")

    if transcript:
        simba.tailor.hook.process_hook(
            json.dumps(
                {
                    "transcript_path": transcript,
                    "cwd": str(pathlib.Path.cwd()),
                }
            )
        )
        print(f"[codex] reflection capture: processed {transcript}")
    else:
        print("[codex] reflection capture: skipped (no transcript found)")

    return 0


def _cmd_codex_automation(args: list[str]) -> int:
    """Print a suggested Codex automation directive for Simba checks."""
    del args
    cwd = str(pathlib.Path.cwd())
    print(
        '::automation-update{mode="suggested create" '
        'name="Simba Codex Health" '
        'prompt="Run simba codex-status and report whether extraction is pending '
        "or memory daemon is down. If pending extraction exists, include the exact "
        'simba codex-extract command in the result." '
        'rrule="FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=9;BYMINUTE=0" '
        f'cwds="{cwd}" status="ACTIVE"}}'
    )
    return 0


def _cmd_hook(args: list[str]) -> int:
    """Dispatch a hook event. Called by Claude Code, not users."""
    if not args:
        print("Usage: simba hook <event>", file=sys.stderr)
        print(f"Events: {', '.join(_HOOK_EVENTS)}", file=sys.stderr)
        return 1

    event = args[0]
    module_name = _HOOK_EVENTS.get(event)
    if module_name is None:
        print(f"Unknown hook event: {event}", file=sys.stderr)
        return 1

    import importlib

    module = importlib.import_module(module_name)

    hook_data: dict = {}
    try:
        raw = sys.stdin.read()
        if raw:
            hook_data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    print(module.main(hook_data))
    return 0


_MEMORY_USAGE = """\
Usage: simba memory <subcommand> [options]

Subcommands:
    store    Store a learning in semantic memory
    recall   Recall memories for a query
    list     List all memories
    delete   Delete a memory by ID
    prune    Bulk-delete memories by age / confidence / type
    update   Update memory metadata
    reindex  Rebuild the hybrid-recall FTS keyword mirror from LanceDB
    reembed  Re-embed the whole corpus with the current model (after a swap)
    consolidate  Roll a session's memories into one EPISODE (engine-gated)

store options:
    --type TYPE            Memory type: WORKING_SOLUTION, GOTCHA, PATTERN,
                           DECISION, FAILURE, PREFERENCE
    --content TEXT         Learning text (max memory.max_content_length;
                           default 1000 chars)
    --context TEXT         Additional context / details
    --confidence FLOAT     Confidence score 0.0-1.0 (default: 0.85)
    --session-source ID    Session ID this came from
    --project-path PATH    Project path for scoping (default: cwd)

recall options:
    --limit N              Max results to return (default: 5)
    --project-path PATH    Project path for scoping (default: cwd)

list options:
    --type TYPE            Filter by memory type
    --limit N              Max results (default: all)

delete:
    simba memory delete <memory_id>

prune options (at least one filter required):
    --type TYPE            Only prune this memory type (e.g. TOOL_RULE)
    --older-than DURATION  Prune entries older than 14d / 48h / 2w / 30m
    --max-confidence FLOAT Only prune entries at or below this confidence
    --dry-run              Show what would be pruned without deleting

update:
    simba memory update <memory_id> [--project-path PATH] [--session-source ID]
"""

_VALID_MEMORY_TYPES = {
    "WORKING_SOLUTION",
    "GOTCHA",
    "PATTERN",
    "DECISION",
    "FAILURE",
    "PREFERENCE",
    "EPISODE",
}


def _memory_max_content_length() -> int:
    """Return configured memory content length cap (default 200)."""
    try:
        import simba.config
        import simba.memory.config

        _ = simba.memory.config  # ensure section registration
        cfg = simba.config.load("memory")
        max_len = int(getattr(cfg, "max_content_length", 200))
        if max_len <= 0:
            return 200
        return max_len
    except Exception:
        return 200


def _cmd_memory(args: list[str]) -> int:
    """Store or recall memories via the daemon."""
    if not args:
        print(_MEMORY_USAGE)
        return 1

    subcmd = args[0]
    rest = args[1:]

    if subcmd == "store":
        return _memory_store(rest)
    elif subcmd == "recall":
        return _memory_recall(rest)
    elif subcmd == "list":
        return _memory_list(rest)
    elif subcmd == "delete":
        return _memory_delete(rest)
    elif subcmd == "prune":
        return _memory_prune(rest)
    elif subcmd == "update":
        return _memory_update(rest)
    elif subcmd == "reindex":
        return _memory_reindex(rest)
    elif subcmd == "reembed":
        return _memory_reembed(rest)
    elif subcmd == "consolidate":
        return _memory_consolidate(rest)
    else:
        print(f"Unknown memory subcommand: {subcmd}")
        print(_MEMORY_USAGE)
        return 1


def _memory_reindex(args: list[str]) -> int:
    """Force a rebuild of the hybrid-recall FTS keyword mirror from LanceDB."""
    import httpx

    import simba.hooks._memory_client

    url = simba.hooks._memory_client.daemon_url()
    try:
        resp = httpx.post(f"{url}/reindex", timeout=60.0)
        resp.raise_for_status()
        body = resp.json()
    except httpx.HTTPError as exc:
        print(f"Error: daemon request failed: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: invalid daemon response: {exc}", file=sys.stderr)
        return 1

    if body.get("status") == "no_mirror":
        print("no FTS mirror configured (hybrid recall disabled?)")
        return 0
    print(f"reindexed: {body.get('indexed', 0)} memories")
    return 0


def _memory_reembed(args: list[str]) -> int:
    """Re-embed the whole corpus with the current model (after an embedder swap)."""
    import httpx

    import simba.hooks._memory_client

    url = simba.hooks._memory_client.daemon_url()
    print("re-embedding the corpus (this can take a while)…", file=sys.stderr)
    try:
        resp = httpx.post(f"{url}/reembed", timeout=3600.0)
        resp.raise_for_status()
        body = resp.json()
    except httpx.HTTPError as exc:
        print(f"Error: daemon request failed: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: invalid daemon response: {exc}", file=sys.stderr)
        return 1

    if body.get("status") == "not_ready":
        print("daemon not ready (no table/embedder)")
        return 1
    print(f"reembedded: {body.get('count', 0)} memories")
    return 0


def _memory_store(args: list[str]) -> int:
    """Store a single memory in the daemon."""
    import httpx

    import simba.hooks._memory_client

    mtype = _parse_opt_value(args, "--type")
    content = _parse_opt_value(args, "--content")
    context = _parse_opt_value(args, "--context") or ""
    confidence_raw = _parse_opt_value(args, "--confidence")
    session_source = _parse_opt_value(args, "--session-source") or ""
    project_path = _parse_opt_value(args, "--project-path") or str(pathlib.Path.cwd())

    if not mtype:
        print("Error: --type is required", file=sys.stderr)
        print(f"Valid types: {', '.join(sorted(_VALID_MEMORY_TYPES))}", file=sys.stderr)
        return 1
    if mtype not in _VALID_MEMORY_TYPES:
        print(f"Error: unknown type '{mtype}'", file=sys.stderr)
        print(f"Valid types: {', '.join(sorted(_VALID_MEMORY_TYPES))}", file=sys.stderr)
        return 1
    if not content:
        print("Error: --content is required", file=sys.stderr)
        return 1
    max_len = _memory_max_content_length()
    if len(content) > max_len:
        print(
            f"Error: --content exceeds {max_len} chars ({len(content)})",
            file=sys.stderr,
        )
        return 1

    try:
        confidence = float(confidence_raw) if confidence_raw else 0.85
    except ValueError:
        print(
            f"Error: --confidence must be a float, got '{confidence_raw}'",
            file=sys.stderr,
        )
        return 1

    payload: dict = {
        "type": mtype,
        "content": content,
        "context": context,
        "confidence": confidence,
        "projectPath": project_path,
    }
    if session_source:
        payload["sessionSource"] = session_source

    url = simba.hooks._memory_client.daemon_url()
    try:
        resp = httpx.post(f"{url}/store", json=payload, timeout=10.0)
        resp.raise_for_status()
        body = resp.json()
    except httpx.HTTPError as exc:
        print(f"Error: daemon request failed: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: invalid daemon response: {exc}", file=sys.stderr)
        return 1

    status = body.get("status", "unknown")
    if status == "stored":
        print(f"stored: {body.get('id', '?')}")
    elif status == "duplicate":
        print(
            f"duplicate: existing={body.get('existing_id', '?')} "
            f"similarity={body.get('similarity', 0):.2f}"
        )
    else:
        print(f"status: {status}")
    return 0


def _memory_recall(args: list[str]) -> int:
    """Recall memories for a query."""
    limit_raw = _parse_opt_value(args, "--limit")
    project_path = _parse_opt_value(args, "--project-path") or str(pathlib.Path.cwd())

    # Query is everything that isn't a --flag or its value
    skip_next = False
    query_parts: list[str] = []
    for tok in args:
        if skip_next:
            skip_next = False
            continue
        if tok.startswith("--"):
            skip_next = True
            continue
        query_parts.append(tok)
    query = " ".join(query_parts).strip()

    if not query:
        print(
            "Usage: simba memory recall [--limit N] [--project-path P] <query text>",
            file=sys.stderr,
        )
        return 1

    try:
        limit = int(limit_raw) if limit_raw else 5
    except ValueError:
        print(f"Error: --limit must be an integer, got '{limit_raw}'", file=sys.stderr)
        return 1

    import simba.hooks._memory_client

    memories = simba.hooks._memory_client.recall_memories(
        query, project_path=project_path, max_results=limit
    )
    if not memories:
        print("no memories found")
        return 0

    print(f"{len(memories)} memories:")
    for m in memories:
        mid = m.get("id", "?")
        mtype = m.get("type", "UNKNOWN")
        sim = m.get("similarity", 0.0)
        content = str(m.get("content", "")).strip()
        print(f"  {mid} [{mtype}] ({sim:.2f}) {content}")
    return 0


def _memory_list(args: list[str]) -> int:
    """List all memories from the daemon."""
    import httpx

    import simba.hooks._memory_client

    mtype = _parse_opt_value(args, "--type")
    limit_raw = _parse_opt_value(args, "--limit")

    url = simba.hooks._memory_client.daemon_url()
    try:
        resp = httpx.get(f"{url}/list", timeout=10.0)
        resp.raise_for_status()
        body = resp.json()
    except httpx.HTTPError as exc:
        print(f"Error: daemon request failed: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: invalid daemon response: {exc}", file=sys.stderr)
        return 1

    memories = body.get("memories", [])

    if mtype:
        memories = [m for m in memories if m.get("type") == mtype]

    if limit_raw:
        try:
            memories = memories[: int(limit_raw)]
        except ValueError:
            print(
                f"Error: --limit must be an integer, got '{limit_raw}'",
                file=sys.stderr,
            )
            return 1

    if not memories:
        print("no memories found")
        return 0

    print(f"{len(memories)} memories:")
    for m in memories:
        mid = m.get("id", "?")
        mt = m.get("type", "UNKNOWN")
        content = str(m.get("content", "")).strip()
        confidence = m.get("confidence", 0)
        print(f"  {mid} [{mt}] (conf={confidence}) {content}")
    return 0


def _memory_delete(args: list[str]) -> int:
    """Delete a memory by ID."""
    import httpx

    import simba.hooks._memory_client

    if not args or args[0].startswith("--"):
        print("Usage: simba memory delete <memory_id>", file=sys.stderr)
        return 1

    memory_id = args[0]
    url = simba.hooks._memory_client.daemon_url()
    try:
        resp = httpx.delete(f"{url}/memory/{memory_id}", timeout=10.0)
        resp.raise_for_status()
        body = resp.json()
    except httpx.HTTPError as exc:
        print(f"Error: daemon request failed: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: invalid daemon response: {exc}", file=sys.stderr)
        return 1

    print(f"deleted: {body.get('id', memory_id)}")
    return 0


def _parse_duration_seconds(raw: str) -> int | None:
    """Parse a duration like ``14d``, ``48h``, ``2w``, ``30m`` (bare int = days)."""
    raw = raw.strip().lower()
    if not raw:
        return None
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    if raw[-1] in units:
        try:
            value = float(raw[:-1])
        except ValueError:
            return None
        return int(value * units[raw[-1]])
    try:
        return int(float(raw) * 86400)  # bare number = days
    except ValueError:
        return None


def _memory_age_seconds(created_at: str | None, now: float) -> float | None:
    """Age in seconds of an ISO ``...Z`` timestamp, or None if unparseable."""
    if not created_at:
        return None
    import calendar
    import time

    try:
        parsed = time.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, TypeError):
        return None
    return now - calendar.timegm(parsed)


def _memory_prune(args: list[str]) -> int:
    """Prune memories matching age / confidence / type filters."""
    import time

    import httpx

    import simba.hooks._memory_client

    mtype = _parse_opt_value(args, "--type")
    older_than = _parse_opt_value(args, "--older-than")
    max_conf_raw = _parse_opt_value(args, "--max-confidence")
    dry_run = "--dry-run" in args

    if not mtype and older_than is None and max_conf_raw is None:
        print(
            "Error: prune requires at least one filter "
            "(--type, --older-than, or --max-confidence)",
            file=sys.stderr,
        )
        return 1

    max_age_seconds = None
    if older_than is not None:
        max_age_seconds = _parse_duration_seconds(older_than)
        if max_age_seconds is None:
            print(
                f"Error: invalid --older-than '{older_than}' "
                "(use e.g. 14d, 48h, 2w, 30m)",
                file=sys.stderr,
            )
            return 1

    max_conf = None
    if max_conf_raw is not None:
        try:
            max_conf = float(max_conf_raw)
        except ValueError:
            print(
                f"Error: --max-confidence must be a number, got '{max_conf_raw}'",
                file=sys.stderr,
            )
            return 1

    url = simba.hooks._memory_client.daemon_url()
    params: dict = {"limit": 1_000_000}
    if mtype:
        params["type"] = mtype
    try:
        resp = httpx.get(f"{url}/list", params=params, timeout=30.0)
        resp.raise_for_status()
        body = resp.json()
    except httpx.HTTPError as exc:
        print(f"Error: daemon request failed: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: invalid daemon response: {exc}", file=sys.stderr)
        return 1

    now = time.time()
    matched = []
    for m in body.get("memories", []):
        if max_age_seconds is not None:
            age = _memory_age_seconds(m.get("createdAt"), now)
            if age is None or age < max_age_seconds:
                continue
        if max_conf is not None and m.get("confidence", 0) > max_conf:
            continue
        matched.append(m)

    if not matched:
        print("no memories matched prune criteria")
        return 0

    deleted = 0
    for m in matched:
        mid = m.get("id", "?")
        mt = m.get("type", "?")
        content = str(m.get("content", "")).strip()[:80]
        if dry_run:
            print(f"  [dry-run] {mid} [{mt}] {content}")
            continue
        try:
            dresp = httpx.delete(f"{url}/memory/{mid}", timeout=10.0)
            dresp.raise_for_status()
            deleted += 1
            print(f"  deleted {mid} [{mt}] {content}")
        except httpx.HTTPError as exc:
            print(f"  failed {mid}: {exc}", file=sys.stderr)

    if dry_run:
        print(f"dry-run: {len(matched)} memories would be pruned (no changes made)")
    else:
        print(f"pruned {deleted}/{len(matched)} memories")
    return 0


def _memory_update(args: list[str]) -> int:
    """Update memory metadata by ID."""
    import httpx

    import simba.hooks._memory_client

    if not args or args[0].startswith("--"):
        print(
            "Usage: simba memory update <memory_id> "
            "[--project-path PATH] [--session-source ID]",
            file=sys.stderr,
        )
        return 1

    memory_id = args[0]
    rest = args[1:]
    project_path = _parse_opt_value(rest, "--project-path")
    session_source = _parse_opt_value(rest, "--session-source")

    if project_path is None and session_source is None:
        print(
            "Error: at least one of --project-path or --session-source is required",
            file=sys.stderr,
        )
        return 1

    payload: dict = {}
    if project_path is not None:
        payload["projectPath"] = project_path
    if session_source is not None:
        payload["sessionSource"] = session_source

    url = simba.hooks._memory_client.daemon_url()
    try:
        resp = httpx.patch(f"{url}/memory/{memory_id}", json=payload, timeout=10.0)
        resp.raise_for_status()
        body = resp.json()
    except httpx.HTTPError as exc:
        print(f"Error: daemon request failed: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: invalid daemon response: {exc}", file=sys.stderr)
        return 1

    fields = body.get("fields", [])
    print(f"updated: {body.get('id', memory_id)} fields={','.join(fields)}")
    return 0


def _cmd_server(args: list[str]) -> int:
    """Start the memory daemon."""
    # Rewrite sys.argv so argparse in server.main() sees the right args
    sys.argv = ["simba server", *args]
    import simba.memory.server

    simba.memory.server.main()
    return 0


def _cmd_search(args: list[str]) -> int:
    """Project memory operations."""
    sys.argv = ["simba search", *args]
    import simba.search.__main__

    return simba.search.__main__.main()


def _cmd_stats() -> int:
    """Show token economics and project statistics."""
    import simba.stats

    print(simba.stats.run_stats(pathlib.Path.cwd()))
    return 0


def _cmd_sync(args: list[str]) -> int:
    """Sync SQLite, LanceDB, and QMD."""
    sys.argv = ["simba sync", *args]
    import simba.sync.__main__

    return simba.sync.__main__.main()


def _cmd_neuron(args: list[str]) -> int:
    """Neuro-symbolic logic server (MCP)."""
    sys.argv = ["simba neuron", *args]
    import simba.neuron.__main__

    return simba.neuron.__main__.main()


def _cmd_orchestration(args: list[str]) -> int:
    """Agent orchestration server (MCP)."""
    sys.argv = ["simba orchestration", *args]
    import simba.orchestration.__main__

    return simba.orchestration.__main__.main(args)


_DB_USAGE = """\
Usage: simba db <subcommand> [options]

Subcommands:
    stats                  Row counts for all tables
    reflections [options]  Show error reflections
    activities [options]   Show tool activity log
    facts                  Show proven facts (neuron)
    agents [options]       Show agent runs
    sessions [options]     Show project memory sessions
    migrate                Migrate data from old per-module databases

Options:
    --limit N              Max rows to display (default: 20)
    --type TYPE            Filter reflections by error type
    --status STATUS        Filter agents by status
"""


def _parse_db_opts(args: list[str]) -> dict[str, str]:
    """Parse --key value pairs from args."""
    opts: dict[str, str] = {}
    i = 0
    while i < len(args):
        if args[i].startswith("--") and i + 1 < len(args):
            opts[args[i][2:]] = args[i + 1]
            i += 2
        else:
            i += 1
    return opts


def _cmd_db(args: list[str]) -> int:
    """Inspect or migrate the shared simba.db database."""
    if not args:
        print(_DB_USAGE)
        return 1

    import simba.db

    # Ensure all schemas are registered by importing modules
    import simba.episodes.jobs
    import simba.kg.store
    import simba.orchestration.agents
    import simba.redirect.store
    import simba.rlm.jobs
    import simba.search.activity_tracker
    import simba.search.project_memory
    import simba.tailor.hook

    _use = (
        simba.episodes.jobs,
        simba.redirect.store,
        simba.orchestration.agents,
        simba.kg.store,
        simba.search.activity_tracker,
        simba.search.project_memory,
        simba.tailor.hook,
    )
    del _use

    subcmd = args[0]
    opts = _parse_db_opts(args[1:])
    limit = int(opts.get("limit", "20"))
    cwd = pathlib.Path.cwd()

    if subcmd == "stats":
        return _db_stats(cwd)
    elif subcmd == "reflections":
        return _db_reflections(cwd, limit, opts.get("type"))
    elif subcmd == "activities":
        return _db_activities(cwd, limit)
    elif subcmd == "facts":
        return _db_facts(cwd, limit)
    elif subcmd == "agents":
        return _db_agents(cwd, limit, opts.get("status"))
    elif subcmd == "sessions":
        return _db_sessions(cwd, limit)
    elif subcmd == "migrate":
        return _db_migrate(cwd)
    else:
        print(f"Unknown db subcommand: {subcmd}")
        print(_DB_USAGE)
        return 1


def _db_stats(cwd: pathlib.Path) -> int:
    """Print row counts for all tables."""
    import simba.db

    if not simba.db.get_db_path(cwd).exists():
        print("Database not found. Run a simba command first to initialize it.")
        return 1

    # Table introspection (sqlite_master + dynamic COUNT) is inherently raw;
    # run it through the peewee connection.
    with simba.db.connect(cwd) as db:
        conn = db.connection()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()

        print(f"Database: {simba.db.get_db_path(cwd)}")
        print()
        total = 0
        for (name,) in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM [{name}]").fetchone()[0]
            total += count
            print(f"  {name:<20s} {count:>6d} rows")
        print(f"  {'─' * 28}")
        print(f"  {'total':<20s} {total:>6d} rows")
    return 0


def _db_reflections(cwd: pathlib.Path, limit: int, error_type: str | None) -> int:
    """Print recent reflections."""
    import simba.db
    import simba.tailor.hook as tailor_hook

    if not simba.db.get_db_path(cwd).exists():
        print("Database not found.")
        return 1

    with simba.db.connect(cwd):
        q = tailor_hook.Reflection.select()
        if error_type:
            q = q.where(tailor_hook.Reflection.error_type == error_type)
        q = q.order_by(tailor_hook.Reflection.ts.desc()).limit(limit)
        rows = list(q)

    if not rows:
        print("No reflections found.")
        return 0

    for row in rows:
        s = row.snippet
        snippet = s[:80] + "..." if len(s) > 80 else s
        print(f"[{row.ts}] {row.error_type} — {row.signature}")
        if snippet:
            print(f"  {snippet}")
        print()
    return 0


def _db_activities(cwd: pathlib.Path, limit: int) -> int:
    """Print recent activities."""
    import simba.db
    import simba.search.activity_tracker as activity_tracker

    if not simba.db.get_db_path(cwd).exists():
        print("Database not found.")
        return 1

    with simba.db.connect(cwd):
        rows = list(
            activity_tracker.Activity.select()
            .order_by(activity_tracker.Activity.id.desc())
            .limit(limit)
        )

    if not rows:
        print("No activities logged.")
        return 0

    for row in rows:
        d = row.detail
        detail = d[:60] + "..." if len(d) > 60 else d
        print(f"[{row.timestamp}] {row.tool_name:<12s} {detail}")
    return 0


def _db_facts(cwd: pathlib.Path, limit: int) -> int:
    """Print currently-valid knowledge-graph facts."""
    import simba.db
    import simba.kg.store as kg_store

    if not simba.db.get_db_path(cwd).exists():
        print("Database not found.")
        return 1

    with simba.db.connect(cwd):
        rows = list(
            kg_store.KgEdge.select()
            .where(kg_store.KgEdge.valid_to.is_null())
            .limit(limit)
        )

    if not rows:
        print("No facts recorded.")
        return 0

    for row in rows:
        p = row.proof or ""
        proof = p[:40] + "..." if len(p) > 40 else p
        print(f"  {row.subject} {row.predicate} {row.object}")
        if row.occurred_at:
            print(f"    occurred: {row.occurred_at}")
        print(f"    proof: {proof}")
    return 0


def _db_agents(cwd: pathlib.Path, limit: int, status: str | None) -> int:
    """Print agent runs."""
    import simba.db
    import simba.orchestration.agents as agents
    import simba.orchestration.config as orch_config

    if not simba.db.get_db_path(cwd).exists():
        print("Database not found.")
        return 1

    with simba.db.connect(cwd):
        q = agents.AgentRun.select()
        if status:
            sid = orch_config.STATUS_NAME_MAP.get(status.lower())
            q = q.where(
                agents.AgentRun.status_id == (int(sid) if sid is not None else -1)
            )
        q = q.order_by(agents.AgentRun.created_at_utc.desc()).limit(limit)
        rows = list(q)

    if not rows:
        print("No agent runs found.")
        return 0

    for row in rows:
        status_name = agents._status_name(row.status_id) or "unknown"
        elapsed = ""
        if row.completed_at_utc and row.created_at_utc:
            elapsed = f" [{row.completed_at_utc - row.created_at_utc}s]"
        result_preview = ""
        if row.result:
            r = row.result
            result_preview = f"\n    Result: {r[:80]}{'...' if len(r) > 80 else ''}"
        error = f"\n    Error: {row.error}" if row.error else ""
        print(
            f"  {row.ticket_id} ({row.agent}, PID {row.pid}): "
            f"{status_name}{elapsed}{result_preview}{error}"
        )
    return 0


def _db_sessions(cwd: pathlib.Path, limit: int) -> int:
    """Print project memory sessions (legacy session_id schema)."""
    import sqlite3

    import simba.db

    if not simba.db.get_db_path(cwd).exists():
        print("Database not found.")
        return 1

    # Legacy schema (session_id/started_at) — not modelled; query raw via the
    # peewee connection and degrade gracefully when the columns are absent.
    with simba.db.connect(cwd) as db:
        conn = db.connection()
        try:
            rows = conn.execute(
                "SELECT session_id, started_at, summary FROM sessions "
                "ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []

    if not rows:
        print("No sessions recorded.")
        return 0

    for session_id, started_at, summary in rows:
        summary = summary or ""
        if len(summary) > 60:
            summary = summary[:60] + "..."
        print(f"[{started_at}] {session_id}")
        if summary:
            print(f"  {summary}")
    return 0


def _db_migrate(cwd: pathlib.Path) -> int:
    """Migrate data from old per-module databases into simba.db."""
    import sqlite3

    import simba.db

    base = simba.db.find_repo_root(cwd)
    if base is None:
        base = cwd
    simba_dir = base / ".simba"

    # Ensure the target DB exists with all schemas
    with simba.db.get_db(cwd):
        pass

    migrated: dict[str, int] = {}

    # 1. neuron/truth.db → kg_edges (open edges)
    truth_db = simba_dir / "neuron" / "truth.db"
    if truth_db.exists():
        import simba.kg.store

        project_path = simba.db.resolve_project_id(cwd)
        src = sqlite3.connect(str(truth_db))
        try:
            rows = src.execute(
                "SELECT subject, predicate, object, proof FROM facts"
            ).fetchall()
            for subject, predicate, obj, proof in rows:
                simba.kg.store.kg_add(
                    subject,
                    predicate,
                    obj,
                    proof,
                    project_path=project_path,
                )
            if rows:
                migrated["kg_edges (from neuron/truth.db)"] = len(rows)
        except sqlite3.OperationalError:
            pass
        finally:
            src.close()

    # 2. neuron/agents.db → agent_runs, agent_logs
    agents_db = simba_dir / "neuron" / "agents.db"
    if agents_db.exists():
        src = sqlite3.connect(str(agents_db))
        try:
            runs = src.execute("SELECT * FROM agent_runs").fetchall()
            desc = src.execute("SELECT * FROM agent_runs LIMIT 0").description
            run_cols = [d[0] for d in desc]
            if runs:
                placeholders = ", ".join("?" * len(run_cols))
                cols = ", ".join(run_cols)
                with simba.db.get_db(cwd) as conn:
                    q = (
                        f"INSERT OR IGNORE INTO agent_runs "
                        f"({cols}) VALUES ({placeholders})"
                    )
                    conn.executemany(
                        q,
                        runs,
                    )
                    conn.commit()
                migrated["agent_runs (from neuron/agents.db)"] = len(runs)

            logs = src.execute("SELECT * FROM agent_logs").fetchall()
            desc = src.execute("SELECT * FROM agent_logs LIMIT 0").description
            log_cols = [d[0] for d in desc]
            if logs:
                # Skip the auto-increment id column
                non_id_cols = [c for c in log_cols if c != "id"]
                non_id_idx = [i for i, c in enumerate(log_cols) if c != "id"]
                placeholders = ", ".join("?" * len(non_id_cols))
                cols = ", ".join(non_id_cols)
                filtered_logs = [tuple(row[i] for i in non_id_idx) for row in logs]
                with simba.db.get_db(cwd) as conn:
                    conn.executemany(
                        f"INSERT INTO agent_logs ({cols}) VALUES ({placeholders})",
                        filtered_logs,
                    )
                    conn.commit()
                migrated["agent_logs (from neuron/agents.db)"] = len(logs)
        except sqlite3.OperationalError:
            pass
        finally:
            src.close()

    # 3. search/memory.db → sessions, knowledge, facts
    memory_db = simba_dir / "search" / "memory.db"
    if memory_db.exists():
        src = sqlite3.connect(str(memory_db))
        try:
            for table in ("sessions", "knowledge", "facts"):
                try:
                    rows = src.execute(f"SELECT * FROM {table}").fetchall()
                    desc = src.execute(f"SELECT * FROM {table} LIMIT 0").description
                    cols = [d[0] for d in desc]
                except sqlite3.OperationalError:
                    continue
                if rows:
                    placeholders = ", ".join("?" * len(cols))
                    col_str = ", ".join(cols)
                    with simba.db.get_db(cwd) as conn:
                        q = (
                            f"INSERT OR IGNORE INTO {table} "
                            f"({col_str}) VALUES ({placeholders})"
                        )
                        conn.executemany(
                            q,
                            rows,
                        )
                        conn.commit()
                    migrated[f"{table} (from search/memory.db)"] = len(rows)
        except sqlite3.OperationalError:
            pass
        finally:
            src.close()

    # 4. search/activity.log → activities
    activity_log = simba_dir / "search" / "activity.log"
    if activity_log.exists():
        try:
            lines = activity_log.read_text().strip().splitlines()
            count = 0
            with simba.db.get_db(cwd) as conn:
                for line in lines:
                    parts = line.split("|", 2)
                    if len(parts) >= 2:
                        ts = parts[0].strip()
                        tool = parts[1].strip()
                        detail = parts[2].strip() if len(parts) > 2 else ""
                        conn.execute(
                            "INSERT INTO activities (timestamp, tool_name, detail) "
                            "VALUES (?, ?, ?)",
                            (ts, tool, detail),
                        )
                        count += 1
                conn.commit()
            if count:
                migrated["activities (from search/activity.log)"] = count
        except OSError:
            pass

    # 5. tailor/reflections.jsonl → reflections
    reflections_jsonl = simba_dir / "tailor" / "reflections.jsonl"
    if reflections_jsonl.exists():
        try:
            lines = reflections_jsonl.read_text().strip().splitlines()
            count = 0
            with simba.db.get_db(cwd) as conn:
                for line in lines:
                    try:
                        entry = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    conn.execute(
                        "INSERT OR IGNORE INTO reflections "
                        "(id, ts, error_type, snippet, context, signature) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            entry.get("id", ""),
                            entry.get("ts", ""),
                            entry.get("error_type", ""),
                            entry.get("snippet", ""),
                            json.dumps(entry.get("context", {})),
                            entry.get("signature", ""),
                        ),
                    )
                    count += 1
                conn.commit()
            if count:
                migrated["reflections (from tailor/reflections.jsonl)"] = count
        except OSError:
            pass

    # Report
    if not migrated:
        print("No old data files found to migrate.")
        print(f"Looked in: {simba_dir}")
        return 0

    print(f"Migration complete → {simba.db.get_db_path(cwd)}")
    print()
    for source, count in migrated.items():
        print(f"  {source}: {count} rows")
    print()
    print("Old files were NOT deleted. Remove manually when satisfied:")
    print(
        f"  rm -rf {simba_dir}/neuron/ {simba_dir}/search/ "
        f"{simba_dir}/tailor/reflections.jsonl"
    )
    return 0


def _cmd_config(args: list[str]) -> int:
    """Unified configuration."""
    import simba.config_cli

    return simba.config_cli.main(args)


def _cmd_markers(args: list[str]) -> int:
    """Discover, audit, and update SIMBA markers."""
    import simba.markers_cli

    return simba.markers_cli.main(args)


def _cmd_rule(args: list[str]) -> int:
    """Manage tool rules (auto-learned + manual) and tool-call redirects."""
    if args and args[0] == "redirect":
        return _cmd_rule_redirect(args[1:])

    import simba.rules_cli

    return simba.rules_cli.main(args)


def _cmd_rule_redirect(args: list[str]) -> int:
    """Manage tool-call redirect rules (cargo->soldr, python->uv run, ...)."""
    import simba.db
    import simba.redirect.store as store

    usage = (
        "Usage:\n"
        "  simba rule redirect add <program> <replacement> [--reason TEXT]\n"
        "  simba rule redirect list\n"
        "  simba rule redirect rm <program>"
    )
    if not args:
        print(usage, file=sys.stderr)
        return 1

    cwd = pathlib.Path.cwd()
    project_id = simba.db.resolve_project_id(cwd)
    sub = args[0]

    if sub == "add":
        if len(args) < 3:
            print(usage, file=sys.stderr)
            return 1
        program, replacement = args[1], args[2]
        reason = ""
        if "--reason" in args:
            i = args.index("--reason")
            reason = args[i + 1] if i + 1 < len(args) else ""
        store.add(program, replacement, reason=reason, project_path=project_id, cwd=cwd)
        print(f"redirect added: {program} -> {replacement}")
        return 0

    if sub == "list":
        rules = store.load_rules(cwd, project_path=project_id)
        if not rules:
            print("no redirect rules (store or .simba/redirects.toml)")
            return 0
        for r in rules:
            extra = f"  # {r.reason}" if r.reason else ""
            print(f"[{r.source}] {r.program} -> {r.replacement}{extra}")
        return 0

    if sub == "rm":
        if len(args) < 2:
            print(usage, file=sys.stderr)
            return 1
        n = store.remove(args[1], project_path=project_id, cwd=cwd)
        print(f"removed {n} rule(s) for {args[1]}")
        return 0

    print(usage, file=sys.stderr)
    return 1


def _cmd_eval(args: list[str]) -> int:
    """Run the recall eval harness against a benchmark dataset."""
    import json as _json
    import tempfile

    import simba.config
    import simba.eval.config  # registers the "eval" section
    import simba.eval.report as report
    import simba.eval.run as run
    import simba.memory.config

    if args and args[0] == "build":
        return _eval_build(args[1:])
    if args and args[0] == "run":
        args = args[1:]

    dataset_arg = ""
    ks_arg = ""
    split_arg = ""
    as_json = False
    i = 0
    while i < len(args):
        if args[i] == "--dataset" and i + 1 < len(args):
            dataset_arg = args[i + 1]
            i += 2
        elif args[i] == "--ks" and i + 1 < len(args):
            ks_arg = args[i + 1]
            i += 2
        elif args[i] == "--split" and i + 1 < len(args):
            split_arg = args[i + 1]
            i += 2
        elif args[i] == "--json":
            as_json = True
            i += 1
        else:
            print(f"Unknown eval option: {args[i]}", file=sys.stderr)
            print(
                "Usage: simba eval run [--dataset NAME|PATH] [--ks 1,3,5] "
                "[--split dev|test] [--json]"
            )
            return 1

    ecfg = simba.config.load("eval")
    dataset_ref = dataset_arg or ecfg.dataset
    if dataset_ref:
        try:
            dataset_path = str(report.resolve_dataset(dataset_ref))
        except FileNotFoundError as exc:
            print(f"eval: {exc}", file=sys.stderr)
            return 1
    else:
        dataset_path = str(report.default_dataset_path())
    ks = (
        simba.eval.config.EvalConfig(ks=ks_arg).ks_tuple()
        if ks_arg
        else ecfg.ks_tuple()
    )

    mcfg = simba.config.load("memory")
    try:
        embed_doc, embed_query = run.sync_embedders(mcfg)
    except Exception as exc:  # model download/load failure
        print(f"eval: could not load the embedding model: {exc}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="simba-eval-") as td:
        rep = run.run_dataset(
            dataset_path,
            ks=ks,
            data_dir=td,
            embed_doc=embed_doc,
            embed_query=embed_query,
            cfg=mcfg,
            split=split_arg or None,
        )

    if as_json:
        print(_json.dumps(rep.to_dict(), indent=2))
    else:
        print(report.format_report(rep, top_n_worst=5))
    return 0


def _eval_build(args: list[str]) -> int:
    """Build an eval dataset from the real memory corpus (LLM-generated queries)."""
    import json as _json

    import httpx

    import simba.eval.build as build
    import simba.hooks._memory_client
    import simba.llm.client

    n = 50
    out = ""
    project = ""
    i = 0
    while i < len(args):
        if args[i] == "--n" and i + 1 < len(args):
            n = int(args[i + 1])
            i += 2
        elif args[i] == "--out" and i + 1 < len(args):
            out = args[i + 1]
            i += 2
        elif args[i] == "--project-path" and i + 1 < len(args):
            project = args[i + 1]
            i += 2
        else:
            print(f"Unknown build option: {args[i]}", file=sys.stderr)
            print("Usage: simba eval build --out PATH [--n 50] [--project-path P]")
            return 1
    if not out:
        print("eval build: --out PATH is required", file=sys.stderr)
        return 1

    url = simba.hooks._memory_client.daemon_url()
    try:
        resp = httpx.get(f"{url}/list", params={"limit": max(n * 3, 200)}, timeout=30)
        resp.raise_for_status()
        mems = resp.json().get("memories", [])
    except (httpx.HTTPError, ValueError) as exc:
        print(f"eval build: daemon list failed: {exc}", file=sys.stderr)
        return 1

    mems = [m for m in mems if m.get("type") != "SYSTEM"]
    if project:
        mems = [m for m in mems if m.get("projectPath") == project]
    if not mems:
        print("eval build: no memories to build from", file=sys.stderr)
        return 1

    client = simba.llm.client.get_client()
    if not client.available():
        print(
            "eval build needs an llm provider (simba config set llm.provider)",
            file=sys.stderr,
        )
        return 1

    print(
        f"generating queries for up to {n} of {len(mems)} memories…", file=sys.stderr
    )
    dataset = build.build_from_memories(
        mems, client=client, name=f"real-corpus-{n}", max_cases=n
    )
    pathlib.Path(out).write_text(_json.dumps(dataset.to_dict(), indent=2))
    print(f"wrote {out}: {len(dataset.corpus)} memories, {len(dataset.cases)} cases")
    return 0


def _cmd_episodes(args: list[str]) -> int:
    """Episodic consolidation control commands (job close, called by the agent)."""
    if not args or args[0] != "complete" or len(args) < 2:
        print("Usage: simba episodes complete <session_id>", file=sys.stderr)
        return 1
    sid = args[1]
    import simba.episodes.jobs

    cwd = pathlib.Path.cwd()
    simba.episodes.jobs.complete(sid, str(cwd), cwd=cwd)
    print(f"episode complete: {sid}")
    return 0


def _memory_consolidate(args: list[str]) -> int:
    """Dispatch episodic consolidation (engine-gated, fire-and-forget)."""
    import simba.episodes.consolidate as ec

    cwd = str(pathlib.Path.cwd())
    if "--session" in args:
        i = args.index("--session")
        if i + 1 >= len(args):
            print("Usage: simba memory consolidate --session <id>", file=sys.stderr)
            return 1
        sid = args[i + 1]
        print(f"{sid}: {ec.consolidate_session(sid, cwd=cwd)}")
        return 0
    result = ec.consolidate_eligible(cwd, all_projects=("--all" in args))
    if result.get("no_engine"):
        print("no RLM engine configured (set rlm.engine=claude-cli) — skipped")
        return 0
    print(f"dispatched {len(result['dispatched'])}, skipped {result['skipped']}")
    return 0


def _cmd_rlm(args: list[str]) -> int:
    """RLM autonomous engine commands."""
    if not args or args[0] not in ("digest", "complete"):
        print(
            "Usage: simba rlm digest <transcript_id|--latest>\n"
            "       simba rlm complete <transcript_id> [--stored N]",
            file=sys.stderr,
        )
        return 1

    if args[0] == "complete":
        crest = args[1:]
        if not crest:
            print(
                "Usage: simba rlm complete <transcript_id> [--stored N]",
                file=sys.stderr,
            )
            return 1
        tid = crest[0]
        n_stored = 0
        if "--stored" in crest:
            i = crest.index("--stored")
            if i + 1 < len(crest):
                try:
                    n_stored = int(crest[i + 1])
                except ValueError:
                    n_stored = 0
        import simba.rlm.jobs

        simba.rlm.jobs.complete(tid, str(pathlib.Path.cwd()), n_stored)
        print(f"marked complete: {tid} (stored {n_stored})")
        return 0

    rest = args[1:]
    transcript_id = rest[0] if rest else ""
    if transcript_id in ("", "--latest"):
        transcripts = pathlib.Path.home() / ".claude" / "transcripts"
        dirs = (
            [d for d in transcripts.iterdir() if d.is_dir()]
            if transcripts.is_dir()
            else []
        )
        if not dirs:
            print("no transcripts found", file=sys.stderr)
            return 1
        transcript_id = max(dirs, key=lambda d: d.stat().st_mtime).name

    import simba.config
    import simba.rlm.config  # registers "rlm"
    import simba.rlm.engine
    import simba.rlm.jobs

    cfg = simba.config.load("rlm")
    engine = simba.rlm.engine.get_engine(cfg)
    if engine is None:
        print(
            f"rlm.engine='{cfg.engine}' has no autonomous engine; "
            "set it to claude-cli (simba config set rlm.engine claude-cli)"
        )
        return 1

    project = str(pathlib.Path.cwd())
    if not simba.rlm.jobs.claim(transcript_id, project, cfg.engine):
        print(f"already digested/running: {transcript_id}")
        return 0

    engine.digest(transcript_id, "", cwd=project)
    print(f"digest dispatched for {transcript_id} via {cfg.engine}")
    return 0


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    cmd = args[0]
    rest = args[1:]

    if cmd == "install":
        sys.exit(_cmd_install(rest))
    elif cmd == "codex-install":
        sys.exit(_cmd_codex_install(rest))
    elif cmd == "codex-status":
        sys.exit(_cmd_codex_status(rest))
    elif cmd == "codex-extract":
        sys.exit(_cmd_codex_extract(rest))
    elif cmd == "codex-recall":
        sys.exit(_cmd_codex_recall(rest))
    elif cmd == "codex-finalize":
        sys.exit(_cmd_codex_finalize(rest))
    elif cmd == "codex-automation":
        sys.exit(_cmd_codex_automation(rest))
    elif cmd == "hook":
        sys.exit(_cmd_hook(rest))
    elif cmd == "memory":
        sys.exit(_cmd_memory(rest))
    elif cmd == "server":
        sys.exit(_cmd_server(rest))
    elif cmd == "search":
        sys.exit(_cmd_search(rest))
    elif cmd == "stats":
        sys.exit(_cmd_stats())
    elif cmd == "sync":
        sys.exit(_cmd_sync(rest))
    elif cmd == "neuron":
        sys.exit(_cmd_neuron(rest))
    elif cmd == "orchestration":
        sys.exit(_cmd_orchestration(rest))
    elif cmd == "config":
        sys.exit(_cmd_config(rest))
    elif cmd == "markers":
        sys.exit(_cmd_markers(rest))
    elif cmd == "rule":
        sys.exit(_cmd_rule(rest))
    elif cmd == "rlm":
        sys.exit(_cmd_rlm(rest))
    elif cmd == "eval":
        sys.exit(_cmd_eval(rest))
    elif cmd == "episodes":
        sys.exit(_cmd_episodes(rest))
    elif cmd == "db":
        sys.exit(_cmd_db(rest))
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
