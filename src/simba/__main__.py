"""Simba CLI — unified Claude Code plugin.

Usage:
    simba install          Register hooks in current project
    simba install --global Register hooks globally (~/.claude/settings.json)
    simba install --remove Remove hooks (add --global for global)
    simba codex-install    Install bundled skills for Codex (~/.codex/skills)
    simba codex-install --remove
                           Remove bundled Codex skills
    simba codex-status     Check daemon health + transcript extraction status
    simba codex-extract    Show manual extraction prompt for pending transcript
    simba codex-extract --run --trace
                           Run extraction and write an analysis trace artifact
    simba codex-curate     Summarize extraction traces into review reports
    simba codex-recall     Query semantic memory (/recall) for a text query
    simba codex-finalize   Run end-of-task signal/error checks
    simba codex-automation Print suggested Codex automation directive
    simba pi-install       Install bundled bridge extension for pi (~/.pi/agent)
    simba pi-install --remove
                           Remove the pi bridge extension
    simba server [opts]    Start the memory daemon
    simba memory store     Store a memory (--type, --content, --context, --confidence)
    simba memory recall    Recall memories for a query text
    simba memory list      List all memories (optional --type filter)
    simba memory delete    Delete a memory by ID
    simba memory update    Update memory metadata (--project-path, --session-source)
    simba memory reindex   Rebuild the hybrid-recall BM25 keyword mirror
    simba memory maintain  Run one decay+hygiene pass (shadow unless --apply)
    simba memory restart   Restart the daemon in place (os.execv self-exec)
    simba memory normalize-scopes
                           Fold worktree scopes onto repo roots (--run applies)
    simba search <cmd>     Project memory operations
    simba sync <cmd>       Sync SQLite, LanceDB, and QMD
    simba stats            Show token economics and project statistics
    simba eval <cmd>       Recall eval harness (run | build from real corpus)
    simba eval bench DATASET [opts]
                           Run recall@k (+ QA) on locomo/longmemeval benchmarks
    simba eval ambiguity   Run ambiguity-preserving executable smoke cases
    simba eval leaderboard [--no-write]
                           Render BENCHMARKS.md from results log
    simba neuron <cmd>     Neuro-symbolic logic server (MCP)
    simba orchestration <cmd> Agent orchestration server (MCP)
    simba config <cmd>     Unified configuration (get/set/list/show)
    simba markers <cmd>    Discover, audit, and update SIMBA markers
    simba rule <cmd>       Manage tool rules (auto-learned + manual)
    simba rlm <cmd>        RLM autonomous engine commands (digest)
    simba episodes <cmd>   Episodic consolidation control (complete)
    simba sessions <cmd>   Index/search raw transcript messages
    simba task <cmd>       Active task snapshot operations
    simba db <subcmd>      Inspect or migrate the shared database
    simba hook <event>     Run a hook (called by Claude Code, not users)
    simba hook-canonical <event>
                           Run a canonical hook, print CanonicalResult JSON
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import re
import sys
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import simba.harness.core

_HOOK_EVENTS = {
    "SessionStart": "simba.hooks.session_start",
    "UserPromptSubmit": "simba.hooks.user_prompt_submit",
    "PreToolUse": "simba.hooks.pre_tool_use",
    "PostToolUse": "simba.hooks.post_tool_use",
    "PreCompact": "simba.hooks.pre_compact",
    "Stop": "simba.hooks.stop",
    # SubagentStop: verify-before-report doctrine check (spec 27). Observe-only /
    # empty unless hooks.reasoning_verify_enabled; harmless to register otherwise.
    "SubagentStop": "simba.hooks.subagent_stop",
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
    "SubagentStop": 5000,
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
    "SubagentStop",
)

# Subset that Codex understands.  Codex also has PermissionRequest, which
# Claude Code does not expose.
_CODEX_HOOK_EVENTS = (
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PreCompact",
    "Stop",
    "SubagentStop",
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
_CODEX_SESSION_MATCHER = "startup|resume|clear|compact"
_CODEX_COMPACT_MATCHER = "manual|auto"


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
                    # --client codex so the daemon attributes Codex traffic
                    # deterministically (no reliable Codex env marker at hook time).
                    "command": f"simba hook {event} --client codex",
                    "timeout": timeout_s,
                }
            ]
        }
        if event == "SessionStart":
            entry["matcher"] = _CODEX_SESSION_MATCHER
        elif event == "PreCompact":
            entry["matcher"] = _CODEX_COMPACT_MATCHER
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


def _norm_path(p: str) -> str:
    """Normalize a path for cwd-vs-stored comparison (mirrors transcripts._norm)."""
    return str(pathlib.PurePath(p)) if p else p


def _read_codex_session_meta(path: pathlib.Path) -> tuple[str, str]:
    """Return ``(session_id, cwd)`` from a Codex session JSONL's session_meta.

    Either field is ``""`` when absent/unreadable; the caller supplies defaults.
    """
    try:
        with path.open() as fh:
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
                    return (
                        sid if isinstance(sid, str) else "",
                        cwd if isinstance(cwd, str) else "",
                    )
                break
    except OSError:
        return "", ""
    return "", ""


def _latest_codex_transcript_metadata(
    project_path: str | None = None,
) -> dict[str, Any] | None:
    """Build transcript metadata from the newest Codex session JSONL.

    When ``project_path`` is given, only sessions whose recorded ``cwd`` matches
    it are considered — the Codex analog of ``transcripts.find_pending``'s
    project scoping, which prevents cross-wiring (surfacing or extracting another
    project's session, e.g. running ``codex-extract`` in project A and getting
    B's newer session). ``None`` keeps the legacy global-newest behavior for
    callers without a project context (diagnostics only).
    """
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

    target = _norm_path(project_path) if project_path is not None else None

    import simba.codex.ledger as codex_ledger

    # Newest-first; with a target, return the newest whose cwd matches it.
    for latest in sorted(candidates, key=_session_sort_key, reverse=True):
        sid, cwd = _read_codex_session_meta(latest)
        session_id = sid or latest.stem
        proj = cwd or str(pathlib.Path.cwd())
        if target is not None and _norm_path(proj) != target:
            continue

        fingerprint = codex_ledger.transcript_fingerprint(latest)
        status = codex_ledger.status_for(
            codex_home=_codex_home(),
            transcript_path=str(latest),
            session_id=session_id,
            project_path=proj,
            fingerprint=fingerprint,
        )
        return {
            "session_id": session_id,
            "project_path": proj,
            "transcript_path": str(latest),
            "status": status,
            "source": "codex",
            "_fingerprint": fingerprint,
        }
    return None


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
    """Extract plain text from markdown or JSONL transcript.

    Size-capped by ``hooks.pre_compact_max_transcript_mb`` (same semantic as
    the export path): slurping a multi-GB rollout here reproduces the
    2026-07-20 RSS incident in the CLI process instead of the daemon.
    """
    if not path.exists():
        return ""
    try:
        import simba.hooks.config  # registers the "hooks" section

        _ = simba.hooks.config
        cap_mb = float(simba.config.load("hooks").pre_compact_max_transcript_mb)
        if cap_mb > 0 and path.stat().st_size > cap_mb * 1024 * 1024:
            print(
                f"transcript {path} is over hooks.pre_compact_max_transcript_mb="
                f"{cap_mb:.0f}MB; run `simba transcript distill` and extract from "
                "the distilled export instead",
                file=sys.stderr,
            )
            return ""
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
    # Split into sentence-like units and preserve source spans for trace output.
    chunks = re.finditer(r"[^.\n]+", transcript_text)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []

    for raw in chunks:
        sentence = " ".join(raw.group(0).strip().split())
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
        evidence = sentence[:2000]
        if len(sentence) > len(evidence):
            evidence += "..."
        out.append(
            {
                "type": mtype,
                "content": sentence[:max_content_length],
                "context": "extracted from transcript",
                "confidence": conf,
                "score": conf,
                "reason": f"matched {mtype.lower()} transcript heuristic",
                "evidence": evidence,
                "source_span": [raw.start(), raw.end()],
            }
        )
        if len(out) >= max_items:
            break

    return out


def _bundled_skill_names() -> list[str]:
    """Return names of all bundled skills."""
    import importlib.resources

    import simba.skill_install as si

    skills_pkg = importlib.resources.files("simba") / "skills"
    if not skills_pkg.is_dir():
        return []
    return [
        d.name
        for d in skills_pkg.iterdir()
        if d.is_dir() and si.find_skill_md(d) is not None
    ]


def _install_skills(skills_dir: pathlib.Path) -> int:
    """Install/refresh bundled skills into *skills_dir* (updates changed ones)."""
    import importlib.resources

    import simba.skill_install as si

    src = importlib.resources.files("simba") / "skills"
    installed, updated = si.sync_skills(src, skills_dir)
    if installed:
        print(f"  + {installed} skill(s) installed")
    if updated:
        print(f"  ~ {updated} skill(s) updated")
    return installed + updated


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

    import simba.skill_install as si

    skills_pkg = importlib.resources.files("simba") / "codex_skills"
    if not skills_pkg.is_dir():
        return []
    return [
        d.name
        for d in skills_pkg.iterdir()
        if d.is_dir() and si.find_skill_md(d) is not None
    ]


def _install_codex_skills(skills_dir: pathlib.Path) -> int:
    """Install/refresh bundled Codex skills (whole dir incl. agents metadata)."""
    import importlib.resources

    import simba.skill_install as si

    src = importlib.resources.files("simba") / "codex_skills"
    installed, updated = si.sync_skills(src, skills_dir)
    if installed:
        print(f"  + {installed} codex skill(s) installed")
    if updated:
        print(f"  ~ {updated} codex skill(s) updated")
    return installed + updated


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


def _run_codex_extraction(
    meta: dict[str, Any],
    *,
    max_items: int,
    force: bool = False,
    trace_enabled: bool = False,
    trace_dir: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """Extract and store learnings for one Codex transcript metadata record."""
    import httpx

    import simba.codex.analysis_runs as analysis_runs
    import simba.codex.ledger as codex_ledger
    import simba.hooks._memory_client

    transcript = str(meta.get("transcript_path", ""))
    session_id = str(meta.get("session_id", ""))
    project_path = str(meta.get("project_path", pathlib.Path.cwd()))
    status = str(meta.get("status", ""))

    if not transcript:
        return {
            "status": "error",
            "message": "latest transcript metadata is missing transcript_path",
        }

    if status and status != codex_ledger.PENDING and not force:
        return {
            "status": "already_extracted",
            "candidates": 0,
            "stored": 0,
            "duplicates": 0,
            "errors": 0,
            "message": f"extraction status is '{status}'",
        }

    transcript_path = pathlib.Path(transcript)
    fingerprint = meta.get("_fingerprint")
    if not isinstance(fingerprint, dict):
        fingerprint = codex_ledger.transcript_fingerprint(transcript_path)
    if not fingerprint:
        return {
            "status": "error",
            "message": f"could not fingerprint transcript {transcript_path}",
        }

    if not force and codex_ledger.is_extracted(
        codex_home=_codex_home(),
        transcript_path=transcript,
        session_id=session_id,
        project_path=project_path,
        fingerprint=fingerprint,
    ):
        return {
            "status": "already_extracted",
            "candidates": 0,
            "stored": 0,
            "duplicates": 0,
            "errors": 0,
            "message": "transcript fingerprint already extracted",
        }

    trace_run: analysis_runs.AnalysisRun | None = None
    if trace_enabled:
        root = pathlib.Path(trace_dir).expanduser() if trace_dir else None
        trace_run = analysis_runs.start_run(
            session_id=session_id,
            project_path=project_path,
            transcript_path=transcript,
            root=root,
            cwd=pathlib.Path(project_path),
        )

    def _trace(event: str, payload: dict[str, Any]) -> None:
        if trace_run is not None:
            analysis_runs.append_event(trace_run, event, payload)

    def _with_trace(result: dict[str, Any]) -> dict[str, Any]:
        if trace_run is not None:
            result["trace_path"] = str(trace_run.trace_path)
        return result

    text = _extract_transcript_text(transcript_path)
    _trace(
        "transcript_loaded",
        {
            "text_chars": len(text),
            "fingerprint": fingerprint,
        },
    )
    if not text.strip():
        result = _with_trace(
            {
                "status": "no_content",
                "candidates": 0,
                "stored": 0,
                "duplicates": 0,
                "errors": 0,
                "message": f"no readable transcript content found in {transcript_path}",
            }
        )
        _trace("run_completed", result)
        return result

    learnings = _extract_learnings(text, max_items=max_items)
    if not learnings:
        result = _with_trace(
            {
                "status": "no_candidates",
                "candidates": 0,
                "stored": 0,
                "duplicates": 0,
                "errors": 0,
                "message": "no candidate learnings found heuristically",
            }
        )
        _trace("run_completed", result)
        return result

    daemon = simba.hooks._memory_client.daemon_url()
    stored = 0
    duplicates = 0
    errors = 0

    for index, mem in enumerate(learnings):
        candidate_payload = {
            "index": index,
            "type": mem["type"],
            "content": mem["content"],
            "context": mem["context"],
            "confidence": mem["confidence"],
            "score": mem.get("score", mem["confidence"]),
            "reason": mem.get("reason", ""),
            "evidence": mem.get("evidence", mem["content"]),
            "source_span": mem.get("source_span"),
        }
        _trace("candidate", candidate_payload)
        _trace(
            "curator_decision",
            {
                "index": index,
                "decision": "keep",
                "reason": "conservative heuristic candidate",
                "score": mem.get("score", mem["confidence"]),
            },
        )
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
            store_status = body.get("status")
            if store_status in {"stored", "superseded"}:
                stored += 1
            elif store_status == "duplicate":
                duplicates += 1
            else:
                errors += 1
                _trace(
                    "negative_lesson",
                    {
                        "index": index,
                        "reason": "store_status_unaccepted",
                        "status": store_status or "unknown",
                    },
                )
            _trace(
                "store_result",
                {
                    "index": index,
                    "status": store_status or "unknown",
                    "memory_id": body.get("id") or body.get("memoryId"),
                    "superseded_id": body.get("supersededId"),
                },
            )
        except (httpx.HTTPError, ValueError) as exc:
            errors += 1
            _trace(
                "store_error",
                {
                    "index": index,
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:500],
                },
            )
            _trace(
                "negative_lesson",
                {
                    "index": index,
                    "reason": "store_exception",
                    "error_type": type(exc).__name__,
                },
            )

    result = {
        "status": "stored" if errors == 0 else "store_errors",
        "candidates": len(learnings),
        "stored": stored,
        "duplicates": duplicates,
        "errors": errors,
    }
    if errors == 0 and stored + duplicates == len(learnings):
        ledger = codex_ledger.append_extracted(
            codex_home=_codex_home(),
            transcript_path=transcript,
            session_id=session_id,
            project_path=project_path,
            fingerprint=fingerprint,
            candidates=len(learnings),
            stored=stored,
            duplicates=duplicates,
        )
        result["ledger_path"] = str(ledger)
    result = _with_trace(result)
    _trace("run_completed", result)
    return result


def _cmd_codex_status(args: list[str]) -> int:
    """Check Codex-oriented Simba status: daemon + transcript extraction."""
    import httpx

    import simba.codex.config  # registers "codex"
    import simba.codex.ledger as codex_ledger
    import simba.config
    import simba.hooks._memory_client

    cfg = simba.config.load("codex")
    auto_extract = (cfg.auto_extract_on_status or "--auto-extract" in args) and (
        "--no-auto-extract" not in args
    )

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
            components = data.get("components") or {}
            if "ready" in data or components:
                print(
                    "[codex] readiness: "
                    f"status={data.get('status', 'unknown')} "
                    f"ready={data.get('ready', 'unknown')} "
                    f"degraded={data.get('degraded', 'unknown')}"
                )
            vector = components.get("vector") or {}
            fts = components.get("fts") or {}
            if vector or fts:
                print(
                    "[codex] storage: "
                    f"db={data.get('dbPath') or vector.get('path') or 'unknown'} "
                    f"table={vector.get('table') or 'unknown'} "
                    f"fts={data.get('ftsPath') or fts.get('path') or 'unknown'}"
                )
            embedder = components.get("embedder") or {}
            reranker = components.get("reranker") or {}
            if embedder or reranker:
                embedding_dims = data.get("embeddingDims") or embedder.get(
                    "dims", "unknown"
                )
                print(
                    "[codex] retrieval: "
                    f"embedding_dims={embedding_dims} "
                    f"provider={embedder.get('provider', 'unknown')} "
                    f"reranker={reranker.get('mode', 'unknown')}"
                )
            if data.get("lastError"):
                err = data["lastError"]
                print(
                    "[codex] last error: "
                    f"{err.get('type', 'unknown')} "
                    f"endpoint={err.get('endpoint', 'unknown')} "
                    f"request={err.get('request_id', data.get('requestId', 'unknown'))}"
                )
            # Mirror Claude SessionStart behavior: trigger one sync cycle.
            with contextlib.suppress(httpx.HTTPError, ValueError):
                httpx.post(f"{url}/sync", timeout=1.0)
                print("[codex] sync: triggered")
    except (httpx.HTTPError, ValueError):
        pass

    if not health_ok:
        print("[codex] memory: down (start with `simba server`)")

    meta = _latest_codex_transcript_metadata(str(pathlib.Path.cwd()))
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
    if status == codex_ledger.PENDING:
        if auto_extract and health_ok:
            result = _run_codex_extraction(
                meta,
                max_items=max(1, int(cfg.auto_extract_max_items)),
                trace_enabled=bool(cfg.extraction_trace_enabled),
                trace_dir=cfg.extraction_trace_dir,
            )
            r_status = result.get("status", "unknown")
            print(
                "[codex] auto-extract: "
                f"status={r_status} candidates={result.get('candidates', 0)} "
                f"stored={result.get('stored', 0)} "
                f"duplicate={result.get('duplicates', 0)} "
                f"errors={result.get('errors', 0)}"
            )
            if result.get("ledger_path"):
                print(f"[codex] extraction ledger: {result['ledger_path']}")
            if result.get("trace_path"):
                print(f"[codex] analysis trace: {result['trace_path']}")
            if r_status != "stored":
                if result.get("message"):
                    print(f"[codex] auto-extract detail: {result['message']}")
                print("[codex] next: run `simba codex-extract` for the manual prompt")
        else:
            if auto_extract and not health_ok:
                reason = "skipped because memory daemon is down"
            elif "--no-auto-extract" in args:
                reason = "disabled for this run"
            else:
                reason = "disabled by codex.auto_extract_on_status"
            print(f"[codex] auto-extract: {reason}")
            print("[codex] next: run `simba codex-extract`")
    return 0


def _cmd_codex_extract(args: list[str]) -> int:
    """Print a ready-to-run extraction prompt or run extraction explicitly."""
    import simba.codex.config  # registers "codex"
    import simba.config

    cfg = simba.config.load("codex")
    mark_done = "--mark-done" in args
    run_mode = "--run" in args
    force = "--force" in args
    raw_trace_dir = _parse_opt_value(args, "--trace-dir")
    trace_enabled = bool(
        cfg.extraction_trace_enabled or "--trace" in args or raw_trace_dir
    )
    trace_dir = raw_trace_dir or cfg.extraction_trace_dir
    max_items = max(1, int(cfg.auto_extract_max_items))
    raw_max_items = _parse_opt_value(args, "--max-items")
    if raw_max_items:
        try:
            max_items = max(1, int(raw_max_items))
        except ValueError:
            print(f"Error: --max-items must be an integer, got {raw_max_items!r}")
            return 1

    meta = _latest_codex_transcript_metadata(str(pathlib.Path.cwd()))
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
        result = _run_codex_extraction(
            meta,
            max_items=max_items,
            force=force,
            trace_enabled=trace_enabled,
            trace_dir=trace_dir,
        )
        print(
            f"[codex] extract run complete: status={result.get('status', 'unknown')} "
            f"candidates={result.get('candidates', 0)} "
            f"stored={result.get('stored', 0)} "
            f"duplicate={result.get('duplicates', 0)} "
            f"errors={result.get('errors', 0)}"
        )
        if result.get("ledger_path"):
            print(f"[codex] extraction ledger: {result['ledger_path']}")
        if result.get("trace_path"):
            print(f"[codex] analysis trace: {result['trace_path']}")
        if result.get("message"):
            print(f"[codex] extract detail: {result['message']}")
        if result.get("status") == "no_candidates":
            print(
                "Fallback: run `simba codex-extract` without --run for manual prompt."
            )
        if result.get("status") in {"stored", "already_extracted"}:
            return 0
        return 1
    else:
        if mark_done:
            print(
                "[codex] --mark-done is ignored without --run; "
                "Codex sessions are marked extracted only after successful storage."
            )
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

    return 0


def _cmd_codex_curate(args: list[str]) -> int:
    """Summarize Codex analysis traces into reviewable curator reports."""
    import simba.codex.config  # registers "codex"
    import simba.codex.curator as curator
    import simba.config

    if "--help" in args or "-h" in args:
        print(
            "Usage: simba codex-curate [--latest | --trace PATH] "
            "[--out PATH] [--json]\n"
            "       simba codex-curate review REPORT "
            "[--accept N] [--reject N] [--duplicate N] [--noisy N] "
            "[--needs-more-evidence N]"
        )
        return 0
    if args and args[0] == "review":
        return _cmd_codex_curate_review(args[1:])

    cfg = simba.config.load("codex")
    raw_trace = _parse_opt_value(args, "--trace")
    raw_trace_dir = _parse_opt_value(args, "--trace-dir")
    raw_out = _parse_opt_value(args, "--out")
    use_json = "--json" in args or (
        "--markdown" not in args and cfg.curator_default_format == "json"
    )

    if raw_trace:
        trace_path = pathlib.Path(raw_trace).expanduser()
    else:
        trace_dir = curator.resolve_trace_dir(
            raw_trace_dir or cfg.extraction_trace_dir,
            pathlib.Path.cwd(),
        )
        trace_path = curator.find_latest_trace(trace_dir)
        if trace_path is None:
            print(f"No analysis traces found in {trace_dir}", file=sys.stderr)
            return 1

    if not trace_path.exists():
        print(f"Trace not found: {trace_path}", file=sys.stderr)
        return 1

    trace = curator.load_trace(trace_path)
    report = curator.summarize_trace(trace)
    if cfg.curator_min_candidate_score:
        report = curator.filter_report(
            report,
            min_score=float(cfg.curator_min_candidate_score),
        )
    out_path = curator.resolve_report_path(
        trace_path=trace_path,
        raw_out=raw_out,
        raw_report_dir=cfg.curator_report_dir,
        as_json=use_json,
        cwd=pathlib.Path.cwd(),
    )
    if use_json:
        written = curator.write_json(report, out_path)
    else:
        written = curator.write_markdown(report, out_path)
    print(
        "[codex] curator report: "
        f"{written} candidates={report.metrics.get('candidate_count', 0)} "
        f"stored={report.metrics.get('stored_count', 0)} "
        f"duplicates={report.metrics.get('duplicate_count', 0)} "
        f"errors={report.metrics.get('store_error_count', 0)}"
    )
    return 0


def _cmd_codex_curate_review(args: list[str]) -> int:
    """Append review labels for a curator report or trace."""
    import simba.codex.curator as curator

    if not args or args[0].startswith("--"):
        print(
            "Usage: simba codex-curate review REPORT "
            "[--accept N] [--reject N] [--duplicate N] [--noisy N] "
            "[--needs-more-evidence N]",
            file=sys.stderr,
        )
        return 1

    subject = pathlib.Path(args[0]).expanduser()
    if not subject.exists():
        print(f"Report or trace not found: {subject}", file=sys.stderr)
        return 1
    rest = args[1:]
    try:
        decisions = _curator_review_decisions(rest)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if not decisions:
        print("Error: provide at least one review label", file=sys.stderr)
        return 1

    try:
        report = curator.load_report_or_trace(subject)
        review_path = pathlib.Path(
            _parse_opt_value(rest, "--review-out") or curator.review_path_for(subject)
        )
        written = curator.append_review_decisions(report, decisions, review_path)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    commands = curator.memory_store_commands(report, decisions)
    if "--commands-only" not in rest:
        print(f"[codex] curator review: {written} decisions={len(decisions)}")
        print(f"[codex] accepted promotion commands: {len(commands)} (not executed)")
    for command in commands:
        print(command)
    return 0


def _curator_review_decisions(args: list[str]) -> tuple[Any, ...]:
    """Parse curator review label flags into ReviewDecision records."""
    import simba.codex.curator as curator

    reason = _parse_opt_value(args, "--reason") or ""
    reviewer = _parse_opt_value(args, "--reviewer") or ""
    labels = {
        "--accept": "accepted",
        "--reject": "rejected",
        "--duplicate": "duplicate",
        "--noisy": "noisy",
        "--needs-more-evidence": "needs_more_evidence",
    }
    decisions = []
    for flag, label in labels.items():
        for raw in _split_values(_values_for(args, flag)):
            try:
                index = int(raw)
            except ValueError as exc:
                msg = f"{flag} expects integer indexes, got {raw!r}"
                raise ValueError(msg) from exc
            decisions.append(
                curator.ReviewDecision(
                    candidate_index=index,
                    label=label,
                    reason=reason,
                    reviewer=reviewer,
                )
            )
    return tuple(decisions)


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
        client="codex",
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
        meta = _latest_codex_transcript_metadata(str(pathlib.Path.cwd()))
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
        'prompt="Run simba codex-status --auto-extract and report whether memory '
        "is down, whether auto-extraction ran, and whether extraction is still "
        "pending. "
        'If pending remains, include the exact simba codex-extract command." '
        'rrule="FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=9;BYMINUTE=0" '
        f'cwds="{cwd}" status="ACTIVE"}}'
    )
    return 0


def _hook_via_daemon(
    event: str, payload: dict
) -> simba.harness.core.CanonicalResult | None:
    """Try the daemon's /hook/{event}; return None on any failure.

    Callers fall back to running the hook inline when this returns None.
    """
    import httpx

    import simba.harness.client
    import simba.harness.core
    import simba.hooks._memory_client

    url = f"{simba.hooks._memory_client.daemon_url()}/hook/{event}"
    try:
        resp = httpx.post(
            url,
            json=payload,
            timeout=3.0,
            headers=simba.harness.client.client_headers(),
        )
        if resp.status_code == 200:
            b = resp.json()
            return simba.harness.core.CanonicalResult(
                additional_context=b.get("additional_context", ""),
                suppress_output=b.get("suppress_output", False),
                memory_count=b.get("memory_count", 0),
                block_reason=b.get("block_reason"),
                transform=b.get("transform"),
                escalated_block=b.get("escalated_block"),
            )
    except (httpx.HTTPError, ValueError):
        pass
    return None


def _dispatch_canonical(
    event: str, payload: dict
) -> simba.harness.core.CanonicalResult:
    """Daemon-first, inline fallback. Honors hooks.dispatch_via_daemon.

    Injects the process cwd into ``payload`` when it's absent so the inline and
    daemon paths are equivalent: this CLI runs in the agent's project directory,
    so its cwd is the correct one and the daemon never falls back to its own.
    """
    import simba.config
    import simba.harness.core
    import simba.hooks.config

    _ = simba.hooks.config  # ensure the "hooks" section is registered
    payload.setdefault("cwd", os.getcwd())
    if simba.config.load("hooks").dispatch_via_daemon:
        result = _hook_via_daemon(event, payload)
        if result is not None:
            return result
    return simba.harness.core.dispatch(event, payload)


def _cmd_hook_canonical(args: list[str]) -> int:
    """Run a canonical hook and print its CanonicalResult as JSON."""
    if not args:
        print("Usage: simba hook-canonical <canonical_event>", file=sys.stderr)
        return 1
    event = args[0]
    payload: dict = {}
    try:
        raw = sys.stdin.read()
        if raw:
            payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        result = _dispatch_canonical(event, payload)
    except KeyError:
        print(f"Unknown canonical event: {event}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "additional_context": result.additional_context,
                "suppress_output": result.suppress_output,
                "memory_count": result.memory_count,
                "block_reason": result.block_reason,
                "transform": result.transform,
                "escalated_block": result.escalated_block,
            }
        )
    )
    return 0


def _pi_agent_home() -> pathlib.Path:
    """Return pi's agent home.

    PI_CODING_AGENT_DIR if set, else the configured pi.agent_home.
    """
    env = os.environ.get("PI_CODING_AGENT_DIR")
    if env:
        return pathlib.Path(env).expanduser()
    import simba.config
    import simba.pi  # registers the "pi" section

    _ = simba.pi
    return pathlib.Path(simba.config.load("pi").agent_home).expanduser()


def _cmd_pi_install(args: list[str]) -> int:
    """Install or remove the bundled pi bridge extension."""
    import importlib.resources

    remove = "--remove" in args
    home = _pi_agent_home()
    ext_dir = home / "extensions"
    ext_path = ext_dir / "simba.ts"
    settings_path = home / "settings.json"

    settings: dict = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except (json.JSONDecodeError, OSError):
            settings = {}
    extensions = settings.setdefault("extensions", [])

    if remove:
        if ext_path.exists():
            ext_path.unlink()
        if str(ext_path) in extensions:
            extensions.remove(str(ext_path))
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")
        print(f"pi extension removed from {settings_path}")
        return 0

    ext_dir.mkdir(parents=True, exist_ok=True)
    src = importlib.resources.files("simba") / "pi" / "extension" / "simba.ts"
    ext_path.write_text(src.read_text())
    if str(ext_path) not in extensions:
        extensions.append(str(ext_path))
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"pi extension installed: {ext_path}")
    print(f"  registered in {settings_path}")
    print("  daemon URL: $SIMBA_DAEMON_URL or http://localhost:8741")
    return 0


def _resolve_hook_client(args: list[str]) -> tuple[list[str], str]:
    """Split a ``--client NAME`` (or ``--client=NAME``) flag out of ``args``.

    Returns ``(positional_args_without_flag, resolved_client_name)``. The client
    name follows ``simba.harness.client.detect_client`` precedence, defaulting to
    ``claude-code`` (the primary caller of ``simba hook``). Codex's generated
    hooks pass ``--client codex`` so they self-identify deterministically.
    """
    import simba.harness.client

    client_flag: str | None = None
    positional: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--client":
            client_flag = args[i + 1] if i + 1 < len(args) else None
            i += 2
            continue
        if a.startswith("--client="):
            client_flag = a.split("=", 1)[1]
            i += 1
            continue
        positional.append(a)
        i += 1
    resolved = simba.harness.client.detect_client(
        client_flag, default=simba.harness.client.CLAUDE_CODE
    )
    return positional, resolved


def _cmd_hook(args: list[str]) -> int:
    """Dispatch a hook event. Called by Claude Code / Codex, not users."""
    # Resolve client identity once, then export it so BOTH the daemon path
    # (X-Simba-Client header) and the inline fallback (recall via _memory_client)
    # attribute this process consistently.
    args, client = _resolve_hook_client(args)
    os.environ["SIMBA_CLIENT"] = client

    if not args:
        print("Usage: simba hook <event>", file=sys.stderr)
        print(f"Events: {', '.join(_HOOK_EVENTS)}", file=sys.stderr)
        return 1

    event = args[0]

    import simba.harness.adapters.claude as claude

    # Canonicalized (MVP) events: daemon-first, inline fallback, render to envelope.
    canonical = claude.NATIVE_TO_CANONICAL.get(event)
    if canonical is not None:
        payload: dict = {}
        try:
            raw = sys.stdin.read()
            if raw:
                payload = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            pass
        result = _dispatch_canonical(canonical, payload)
        print(claude.render(event, result))
        return 0

    # Legacy path for not-yet-canonicalized events
    # (PostToolUse/PermissionRequest).
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
    compact  Inspect/compact LanceDB fragments and retained versions
    reembed  Re-embed the whole corpus with the current model (after a swap)
    consolidate  Roll a session's memories into one EPISODE (engine-gated)
    feedback Mark a recalled memory as good or bad (feeds decay ranking)
    maintain Run one decay+hygiene maintenance pass (shadow unless --apply)
    restart  Restart the daemon in place (os.execv self-exec)
    normalize-scopes Fold worktree scopes onto repo roots (dry-run unless --run)
    promote  List usage-triggered promotion candidates (spec 33)
    import-curated Mirror curated markdown memories into the daemon
    gaps     List knowledge gaps — queries asked often, answered poorly
    supersession Inspect append-only supersession lineage for one memory

store options:
    --type TYPE            Memory type: WORKING_SOLUTION, GOTCHA, PATTERN,
                           DECISION, FAILURE, PREFERENCE
    --content TEXT         Learning text (max memory.max_content_length;
                           default 1000 chars)
    --context TEXT         Additional context / details
    --confidence FLOAT     Confidence score 0.0-1.0 (default: 0.85)
    --session-source ID    Session ID this came from
    --project-path PATH    Project path for scoping (default: cwd)
    --occurred-at TEXT     Event/belief time for provenance metadata
    --observed-at TEXT     Observation time for provenance metadata
    --source-file PATH     Source file for provenance metadata
    --source-span TEXT     Source line/span for provenance metadata
    --source-url URL       Source URL for provenance metadata
    --extraction-agent ID  Extractor/agent name for provenance metadata
    --extraction-version V Extractor/agent version for provenance metadata
    --anticipated-query Q  Future query phrasing for this memory (repeatable)
    --trust-source SOURCE  user_stated, agreed_upon, agent_suggested,
                           llm_extracted, hook_auto_learned, external
    --capture-origin ID    store origin for trust scoring (cli, hook, etc.)

recall options:
    --limit N              Max results to return (default: 5)
    --project-path PATH    Project path for scoping (default: cwd)

list options:
    --type TYPE            Filter by memory type
    --limit N              Max results (default: all)

delete:
    simba memory delete <memory_id>

compact:
    simba memory compact
    simba memory compact --run [--older-than 24h] [--delete-unverified]

prune options (at least one filter required):
    --type TYPE            Only prune this memory type (e.g. TOOL_RULE)
    --older-than DURATION  Prune entries older than 14d / 48h / 2w / 30m
    --max-confidence FLOAT Only prune entries at or below this confidence
    --dry-run              Show what would be pruned without deleting

update:
    simba memory update <memory_id> [--project-path PATH] [--session-source ID]

feedback:
    simba memory feedback <memory_id> good|bad [--weight 0.3]

supersession:
    simba memory supersession <memory_id>
    simba memory supersession confirm|reject <audit_id>
"""

_VALID_MEMORY_TYPES = {
    "WORKING_SOLUTION",
    "GOTCHA",
    "PATTERN",
    "DECISION",
    "FAILURE",
    "PREFERENCE",
    "EPISODE",
    "REFLECTION",  # cross-session synthesized insight (Phase 5)
}


def _memory_max_content_length(root: pathlib.Path | None = None) -> int:
    """Return the configured memory content length cap (default 200).

    Thin wrapper over the single source of truth in ``simba.memory.config`` so
    the CLI store path and the daemon agree on the cap. *root* resolves the
    cap for that project's layered config (local ``.simba/config.toml``
    overrides global); ``None`` resolves the cap with no project root pinned
    (falls back through global config to the dataclass default).
    """
    import simba.memory.config

    return simba.memory.config.resolve_max_content_length(root)


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
    elif subcmd == "compact":
        return _memory_compact(rest)
    elif subcmd == "reembed":
        return _memory_reembed(rest)
    elif subcmd == "consolidate":
        return _memory_consolidate(rest)
    elif subcmd == "feedback":
        return _memory_feedback(rest)
    elif subcmd == "maintain":
        return _memory_maintain(rest)
    elif subcmd == "restart":
        return _memory_restart(rest)
    elif subcmd == "normalize-scopes":
        return _memory_normalize_scopes(rest)
    elif subcmd == "promote":
        return _memory_promote(rest)
    elif subcmd == "import-curated":
        return _memory_import_curated(rest)
    elif subcmd == "gaps":
        return _memory_gaps(rest)
    elif subcmd == "supersession":
        return _memory_supersession(rest)
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


def _memory_compact(args: list[str]) -> int:
    """Inspect or compact LanceDB fragments and retained versions."""
    import httpx

    import simba.hooks._memory_client

    run = "--run" in args
    older_than_raw = _parse_opt_value(args, "--older-than") or "24h"
    older_than_seconds = _parse_duration_seconds(older_than_raw)
    if older_than_seconds is None:
        print(
            f"Error: invalid --older-than '{older_than_raw}' "
            "(use e.g. 24h, 7d, 2w, 30m)",
            file=sys.stderr,
        )
        return 1

    url = simba.hooks._memory_client.daemon_url()
    params = {
        "dry_run": str(not run).lower(),
        "older_than_seconds": older_than_seconds,
        "delete_unverified": str("--delete-unverified" in args).lower(),
    }
    try:
        resp = httpx.post(
            f"{url}/compact",
            params=params,
            timeout=3600.0 if run else 120.0,
        )
        resp.raise_for_status()
        body = resp.json()
    except httpx.HTTPError as exc:
        print(f"Error: daemon request failed: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: invalid daemon response: {exc}", file=sys.stderr)
        return 1

    status = body.get("status", "unknown")
    if status == "not_ready":
        print("daemon not ready (no LanceDB table)")
        return 1

    before = body.get("before") or {}
    if status == "dry_run":
        print(f"dry-run: {_memory_compact_snapshot(before)}")
        print(
            "retention: old LanceDB versions older than "
            f"{older_than_raw} would be pruned"
        )
        print("run with: simba memory compact --run --older-than " + older_than_raw)
        return 0

    after = body.get("after") or {}
    print(f"{status}: {_memory_compact_snapshot(before)} ->")
    print(f"           {_memory_compact_snapshot(after)}")
    return 0 if status == "compacted" else 1


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
    import simba.memory.vector_db

    mtype = _parse_opt_value(args, "--type")
    content = _parse_opt_value(args, "--content")
    context = _parse_opt_value(args, "--context") or ""
    confidence_raw = _parse_opt_value(args, "--confidence")
    session_source = _parse_opt_value(args, "--session-source") or ""
    occurred_at = _parse_opt_value(args, "--occurred-at") or ""
    observed_at = _parse_opt_value(args, "--observed-at") or ""
    source_file = _parse_opt_value(args, "--source-file") or ""
    source_span = _parse_opt_value(args, "--source-span") or ""
    source_url = _parse_opt_value(args, "--source-url") or ""
    extraction_agent = _parse_opt_value(args, "--extraction-agent") or ""
    extraction_version = _parse_opt_value(args, "--extraction-version") or ""
    trust_source = _parse_opt_value(args, "--trust-source") or ""
    capture_origin = _parse_opt_value(args, "--capture-origin") or ""
    anticipated_queries = _split_values(_values_for(args, "--anticipated-query"))
    anticipated_queries += _split_values(_values_for(args, "--anticipated-queries"))
    # Normalize the scope path to an absolute, symlink-resolved path (spec 26) so
    # it matches the client's resolved ancestor chain on recall (the daemon also
    # normalizes on store; doing it here keeps the CLI path consistent).
    project_path = simba.memory.vector_db.normalize_project_path(
        _parse_opt_value(args, "--project-path") or str(pathlib.Path.cwd())
    )

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
    # Per-project cap (spec: layered config) -- an empty/blank project_path
    # must resolve as None, NOT Path("") (== Path(".") == cwd of whatever
    # happens to be running this); project_path is normalized above and
    # defaults to cwd itself when --project-path is omitted, so this is
    # almost always a concrete root, but stay defensive here too.
    project_root = pathlib.Path(project_path) if project_path.strip() else None
    max_len = _memory_max_content_length(project_root)
    if len(content) > max_len:
        got_len = len(content)
        print(
            f"Error: --content exceeds {max_len} chars ({got_len})",
            file=sys.stderr,
        )
        # Path A (recommended): content stays atomic; context is unbounded,
        # so pushing detail there never needs a config change.
        print(
            f"Keep --content <= {max_len} chars and move detail into "
            "--context (recommended)",
            file=sys.stderr,
        )
        # Path B: raise the cap outright, pre-filled with the actual length
        # so this same content would be admitted. Bare `config set` is
        # project-local (scoped to the cwd it's run from); --global instead
        # raises it for every project AND loosens the "keep content under N
        # chars" guidance given to every extraction/digest/episode/
        # reflection prompt corpus-wide, since resolve_max_content_length()
        # is the same single source of truth those prompts hydrate from.
        print(
            "Or raise the cap: simba config set memory.max_content_length "
            f"{got_len} (run from this project to scope it there; add "
            "--global to raise it everywhere, which also loosens "
            "auto-extraction terseness corpus-wide)",
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
    if occurred_at:
        payload["occurredAt"] = occurred_at
    if observed_at:
        payload["observedAt"] = observed_at
    if source_file:
        payload["sourceFile"] = source_file
    if source_span:
        payload["sourceSpan"] = source_span
    if source_url:
        payload["sourceUrl"] = source_url
    if extraction_agent:
        payload["extractionAgent"] = extraction_agent
    if extraction_version:
        payload["extractionVersion"] = extraction_version
    if trust_source:
        payload["trustSource"] = trust_source
    if capture_origin:
        payload["captureOrigin"] = capture_origin
    if anticipated_queries:
        payload["anticipatedQueries"] = anticipated_queries

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
    elif status == "superseded":
        print(
            f"superseded: old={body.get('supersededId', '?')} new={body.get('id', '?')}"
        )
    elif status == "pending_confirmation":
        print(
            f"pending supersession: audit={body.get('pendingSupersessionId', '?')} "
            f"old={body.get('supersededCandidateId', '?')} "
            f"new={body.get('id', '?')}"
        )
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


def _memory_maintain(args: list[str]) -> int:
    """Run one maintenance pass (decay + hygiene) via the daemon (spec 33).

    Usage: simba memory maintain [--apply]

    Shadow (the config default) counts every would-be strength change /
    dormancy transition / rule expiry without persisting; ``--apply``
    persists this pass regardless of ``memory.maintenance_apply``.
    """
    import httpx

    import simba.hooks._memory_client

    payload: dict[str, object] = {}
    if "--apply" in args:
        payload["apply"] = True
        args = [a for a in args if a != "--apply"]
    if args:
        print("Usage: simba memory maintain [--apply]", file=sys.stderr)
        return 1

    url = simba.hooks._memory_client.daemon_url()
    try:
        resp = httpx.post(f"{url}/maintenance/run", json=payload, timeout=120.0)
        resp.raise_for_status()
        body = resp.json()
    except httpx.HTTPError as exc:
        print(f"Error: daemon request failed: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: invalid daemon response: {exc}", file=sys.stderr)
        return 1

    mode = "apply" if body.get("apply") else "shadow (dry-run)"
    print(f"maintenance @ {body.get('at', '?')} — {mode}")
    decay = body.get("decay") or {}
    if decay.get("skipped"):
        print("  decay: skipped")
    else:
        print(
            "  decay: processed={} updated={} newly_dormant={} revived={}".format(
                decay.get("processed", 0),
                decay.get("updated", 0),
                decay.get("newly_dormant", 0),
                decay.get("revived", 0),
            )
        )
    hygiene = body.get("hygiene") or {}
    if hygiene.get("skipped"):
        print("  hygiene: skipped")
    else:
        print(
            f"  hygiene: expired={hygiene.get('expired_count', 0)} "
            f"checked={hygiene.get('checked_count', 0)}"
        )
    return 0


def _memory_restart(args: list[str]) -> int:
    """Restart the daemon in place via POST /restart (os.execv self-exec).

    Usage: simba memory restart

    The daemon responds immediately with the pre-restart pid, then replaces
    its own process image with a fresh interpreter running the current
    on-disk code --- same PID, same terminal, stdout/stderr piping intact.
    Fails with a clear error (not a hang) when the daemon has no boot argv on
    record (503, e.g. it wasn't started via `python -m simba.memory.server`
    / `simba server`) or on a non-POSIX platform (501).
    """
    import httpx

    import simba.harness.client
    import simba.hooks._memory_client

    if args:
        print("Usage: simba memory restart", file=sys.stderr)
        return 1

    url = simba.hooks._memory_client.daemon_url()
    try:
        resp = httpx.post(
            f"{url}/restart",
            timeout=10.0,
            headers=simba.harness.client.client_headers(),
        )
        body = resp.json()
    except httpx.HTTPError as exc:
        print(f"Error: daemon request failed: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: invalid daemon response: {exc}", file=sys.stderr)
        return 1

    if resp.status_code != 202:
        print(f"Error: {body.get('error', resp.text)}", file=sys.stderr)
        return 1

    print(f"restart requested (old pid={body.get('pid', '?')})")
    print("watch GET /health (uptimeSeconds resets) to confirm the new image is up")
    return 0


def _memory_promote(args: list[str]) -> int:
    """List usage-triggered promotion candidates (spec 33 Phase 5).

    Usage: simba memory promote [--limit N]

    Candidates = use_count >= memory.promotion_min_uses with noise/use below
    memory.promotion_max_noise_ratio and not dormant. The promotion itself
    stays human: turn a candidate into a TOOL_RULE / CLAUDE.md bullet / skill.
    """
    import httpx

    import simba.hooks._memory_client

    limit_raw = _parse_opt_value(args, "--limit") or "20"
    try:
        limit = int(limit_raw)
    except ValueError:
        print(f"Error: invalid --limit: {limit_raw}", file=sys.stderr)
        return 1

    url = simba.hooks._memory_client.daemon_url()
    try:
        resp = httpx.get(
            f"{url}/promotions/candidates", params={"limit": limit}, timeout=30.0
        )
        resp.raise_for_status()
        body = resp.json()
    except httpx.HTTPError as exc:
        print(f"Error: daemon request failed: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: invalid daemon response: {exc}", file=sys.stderr)
        return 1

    candidates = body.get("candidates") or []
    if not candidates:
        print("no promotion candidates (yet — usage signals feed this)")
        return 0
    print(
        f"{body.get('total', len(candidates))} promotion candidate(s) "
        f"(use>={body.get('minUses', '?')}, "
        f"noise/use<{body.get('maxNoiseRatio', '?')}):"
    )
    for c in candidates:
        content = (c.get("content") or "")[:80]
        print(
            f"  {c.get('id', '?')} [{c.get('type', '?')}] "
            f"use={c.get('useCount', 0)} noise={c.get('noiseCount', 0)} — {content}"
        )
    return 0


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse a curated memory file's ``---`` frontmatter (tolerant, no YAML dep).

    Returns ``(meta, body)`` where meta carries flat ``key: value`` lines plus
    the nested ``metadata.type``. Files without frontmatter → ({}, full text).
    """
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text
    meta: dict[str, str] = {}
    body_start = len(lines)
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body_start = i + 1
            break
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        # YAML-style quoted scalars keep their quotes through the tolerant
        # parse; strip a MATCHING pair so content never carries stray quotes.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1].strip()
        if key == "type" and value:  # the nested metadata.type line
            meta["type"] = value
        elif value:
            meta.setdefault(key, value)
    return meta, "\n".join(lines[body_start:]).strip()


# Curated-layer types → daemon memory types (spec 33 Phase 6 bridge).
_CURATED_TYPE_MAP = {
    "user": "PREFERENCE",
    "feedback": "PREFERENCE",
    "project": "DECISION",
    "reference": "PATTERN",
}


def _memory_gaps(args: list[str]) -> int:
    """List the corpus's known unknowns (spec 33 v2, yantrikdb borrow).

    Usage: simba memory gaps [--min-asks N] [--max-best F] [--limit N]

    Queries recalled repeatedly whose best hit never cleared the bar — the
    demand-side signal for what memory SHOULD exist. Requires
    memory.demand_log_enabled to have been collecting.
    """
    import httpx

    import simba.hooks._memory_client

    params: dict[str, object] = {}
    for flag, key, cast in (
        ("--min-asks", "minAsks", int),
        ("--max-best", "maxBest", float),
        ("--limit", "limit", int),
    ):
        raw = _parse_opt_value(args, flag)
        if raw is not None:
            try:
                params[key] = cast(raw)
            except ValueError:
                print(f"Error: invalid {flag}: {raw}", file=sys.stderr)
                return 1

    url = simba.hooks._memory_client.daemon_url()
    try:
        resp = httpx.get(f"{url}/demand/gaps", params=params, timeout=30.0)
        resp.raise_for_status()
        body = resp.json()
    except httpx.HTTPError as exc:
        print(f"Error: daemon request failed: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: invalid daemon response: {exc}", file=sys.stderr)
        return 1

    gaps = body.get("gaps") or []
    if not gaps:
        print("no knowledge gaps recorded (memory.demand_log_enabled collects them)")
        return 0
    print(
        f"{len(gaps)} knowledge gap(s) "
        f"(asked>={body.get('minAsks', '?')}, best<{body.get('maxBest', '?')}):"
    )
    for gap in gaps:
        print(
            f"  asked={gap.get('askCount', 0)} zero={gap.get('zeroCount', 0)} "
            f"best={gap.get('bestScoreMax', 0.0):.2f} — {gap.get('query', '')[:90]}"
        )
    return 0


def _memory_import_curated(args: list[str]) -> int:
    """Mirror curated markdown memories into the daemon (spec 33 Phase 6).

    Usage: simba memory import-curated --dir DIR (--project-path P | --global)
                                       [--run]

    The audit's disjoint-brain fix: the curated layer (one fact per .md with
    name/description frontmatter) is the sharpest knowledge in the harness
    and was unsearchable from every other runtime. One-way mirror, dry-run by
    default, idempotent via the daemon's duplicate detection. MEMORY.md (the
    index) is skipped.

    Hardened per the hippo-memory exploit report (spec 33 v2): every file
    runs the secret veto before entering the plan, and scope is an EXPLICIT
    choice — ``--project-path`` or ``--global`` — never a silent
    share-everywhere default.
    """
    import httpx

    import simba.hooks._memory_client
    import simba.memory.config as memory_config
    import simba.memory.secrets

    usage_line = (
        "Usage: simba memory import-curated --dir DIR "
        "(--project-path P | --global) [--run]"
    )
    directory = _parse_opt_value(args, "--dir")
    project_path = _parse_opt_value(args, "--project-path") or ""
    scope_global = "--global" in args
    run = "--run" in args
    if not directory or (not project_path and not scope_global):
        print(usage_line, file=sys.stderr)
        return 1
    root = pathlib.Path(directory).expanduser()
    if not root.is_dir():
        print(f"Error: not a directory: {root}", file=sys.stderr)
        return 1

    max_len = memory_config.resolve_max_content_length()
    plan: list[dict] = []
    vetoed: list[tuple[str, str]] = []
    for path in sorted(root.glob("*.md")):
        if path.name == "MEMORY.md":
            continue
        try:
            text = path.read_text()
        except OSError:
            continue
        secret_kind = simba.memory.secrets.detect_secret(text)
        if secret_kind:
            vetoed.append((path.name, secret_kind))
            continue
        meta, body = _parse_frontmatter(text)
        description = (meta.get("description") or "").strip()
        if not description:
            # Fall back to the first non-empty body line.
            description = next(
                (ln.strip() for ln in body.split("\n") if ln.strip()), ""
            )
        if not description:
            continue
        mtype = _CURATED_TYPE_MAP.get((meta.get("type") or "").lower(), "PATTERN")
        plan.append(
            {
                "file": path.name,
                "type": mtype,
                "content": description[:max_len],
                "context": body[:400],
            }
        )

    if vetoed:
        print(f"secret veto: vetoed {len(vetoed)} file(s), never stored:")
        for name, kind in vetoed:
            print(f"  VETOED ({kind}) {name}")

    if not plan:
        print("no curated memories found")
        return 0

    if not run:
        print(f"import-curated (dry-run): {len(plan)} memories from {root}")
        for item in plan:
            print(f"  {item['type']:<11} {item['file']} — {item['content'][:70]}")
        print("re-run with --run to store")
        return 0

    url = simba.hooks._memory_client.daemon_url()
    stored = 0
    duplicates = 0
    errors = 0
    for item in plan:
        payload = {
            "type": item["type"],
            "content": item["content"],
            "context": item["context"],
            "confidence": 0.95,
            "projectPath": project_path,
            "trustSource": "user_confirmed",
            "captureOrigin": "curated_import",
        }
        try:
            resp = httpx.post(f"{url}/store", json=payload, timeout=30.0)
            resp.raise_for_status()
            body_json = resp.json()
        except (httpx.HTTPError, ValueError):
            errors += 1
            continue
        if body_json.get("status") == "duplicate":
            duplicates += 1
        else:
            stored += 1
    print(
        f"import-curated: stored {stored}, duplicates {duplicates}, "
        f"errors {errors} (of {len(plan)})"
    )

    # Re-import cadence marker (spec 33 R4): on a fully clean --run, stamp
    # dir + time so SessionStart can nudge a re-import once the curated
    # MEMORY.md changes again — the bridge stays fresh without a human
    # remembering it. A cadence marker, not a memory store: safe (and
    # intended) to overwrite every clean run. Skipped when any item errored
    # (daemon down, etc.) so the nudge keeps firing until a run truly lands.
    # Best-effort — a marker write failure must never flip the exit code.
    if errors == 0:
        with contextlib.suppress(Exception):
            target_root = (
                pathlib.Path(project_path).expanduser()
                if project_path
                else pathlib.Path.cwd()
            )
            marker_dir = target_root / ".simba"
            marker_dir.mkdir(parents=True, exist_ok=True)
            marker = {"dir": str(root.resolve()), "last_import_at": time.time()}
            (marker_dir / "curated-import.json").write_text(json.dumps(marker))

    return 0 if errors == 0 else 1


def _memory_normalize_scopes(args: list[str]) -> int:
    """Fold linked-worktree scopes onto their main repo root (spec 33).

    Usage: simba memory normalize-scopes [--run]

    Dry-run by default: prints the fold plan without touching anything.
    """
    import httpx

    import simba.hooks._memory_client

    payload: dict[str, object] = {}
    if "--run" in args:
        payload["run"] = True
        args = [a for a in args if a != "--run"]
    if args:
        print("Usage: simba memory normalize-scopes [--run]", file=sys.stderr)
        return 1

    url = simba.hooks._memory_client.daemon_url()
    try:
        resp = httpx.post(f"{url}/scopes/normalize", json=payload, timeout=120.0)
        resp.raise_for_status()
        body = resp.json()
    except httpx.HTTPError as exc:
        print(f"Error: daemon request failed: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: invalid daemon response: {exc}", file=sys.stderr)
        return 1

    mode = "applied" if body.get("run") else "dry-run"
    folds = body.get("folds") or []
    if not folds:
        print(f"scope normalize ({mode}): nothing to fold")
        return 0
    print(f"scope normalize ({mode}): {body.get('changed', 0)} memories")
    for fold in folds:
        print(
            f"  {fold.get('from', '?')} -> {fold.get('to', '?')} "
            f"({fold.get('count', 0)} memories)"
        )
    return 0


def _memory_feedback(args: list[str]) -> int:
    """Mark a memory as good or bad.

    Usage: simba memory feedback <id> good|bad [--weight 0.3]
    """
    import httpx

    import simba.hooks._memory_client

    usage = "Usage: simba memory feedback <memory_id> good|bad [--weight 0.3]"
    if len(args) < 2 or args[0].startswith("--"):
        print(usage, file=sys.stderr)
        return 1

    memory_id = args[0]
    signal = args[1]
    if signal not in ("good", "bad"):
        print(usage, file=sys.stderr)
        return 1

    weight: float | None = None
    rest = args[2:]
    i = 0
    while i < len(rest):
        if rest[i] == "--weight" and i + 1 < len(rest):
            try:
                weight = float(rest[i + 1])
            except ValueError:
                print(f"Error: invalid --weight: {rest[i + 1]}", file=sys.stderr)
                return 1
            i += 2
        else:
            print(f"Error: unknown option: {rest[i]}", file=sys.stderr)
            return 1

    payload: dict[str, object] = {"signal": signal}
    if weight is not None:
        payload["weight"] = weight

    url = simba.hooks._memory_client.daemon_url()
    try:
        resp = httpx.post(
            f"{url}/memory/{memory_id}/feedback", json=payload, timeout=10.0
        )
        resp.raise_for_status()
        body = resp.json()
    except httpx.HTTPError as exc:
        print(f"Error: daemon request failed: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: invalid daemon response: {exc}", file=sys.stderr)
        return 1

    print(
        f"feedback recorded: {body.get('id', memory_id)} "
        f"-> feedback_score={body.get('feedback_score', 0.0)}"
    )
    return 0


def _memory_supersession(args: list[str]) -> int:
    """Inspect the append-only supersession chain for a memory."""
    if not args or args[0].startswith("--"):
        print(
            "Usage: simba memory supersession <memory_id>|confirm|reject <audit_id>",
            file=sys.stderr,
        )
        return 1

    import time

    import simba.db
    import simba.memory.supersession

    if args[0] in {"confirm", "reject"}:
        if len(args) != 2:
            print(
                f"Usage: simba memory supersession {args[0]} <audit_id>",
                file=sys.stderr,
            )
            return 1
        try:
            audit_id = int(args[1])
        except ValueError:
            print(
                f"Error: audit_id must be an integer, got {args[1]!r}",
                file=sys.stderr,
            )
            return 1
        try:
            with simba.db.connect(pathlib.Path.cwd()):
                if args[0] == "confirm":
                    row = simba.memory.supersession.confirm(audit_id, now=time.time())
                else:
                    row = simba.memory.supersession.reject(audit_id, now=time.time())
        except (KeyError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(
            f"{args[0]}ed supersession: audit={audit_id} "
            f"decision={row.id} status={row.status} old={row.old_id} new={row.new_id}"
        )
        return 0

    if len(args) != 1:
        print(
            "Usage: simba memory supersession <memory_id>|confirm|reject <audit_id>",
            file=sys.stderr,
        )
        return 1

    memory_id = args[0]
    with simba.db.connect(pathlib.Path.cwd()):
        rows = simba.memory.supersession.chain(memory_id)
        events = simba.memory.supersession.events_for(memory_id)
    if not rows and not events:
        print(f"no supersession chain for {memory_id}")
        return 0

    if rows:
        print(f"active supersession chain for {memory_id}:")
        current = memory_id
        for row in rows:
            print(
                f"  {current} -> {row.new_id} "
                f"[{row.memory_type}] sim={row.similarity:.3f} "
                f"reason={row.reason} at={row.created_at_iso}"
            )
            current = row.new_id
    if events:
        print(f"supersession events for {memory_id}:")
        for row in events:
            print(
                f"  audit={row.id} status={row.status} "
                f"{row.old_id} -> {row.new_id} "
                f"oldTrust={row.old_trust_score:.3f} "
                f"newTrust={row.new_trust_score:.3f} "
                f"reason={row.reason} at={row.created_at_iso}"
            )
    return 0


def _format_bytes(value: object) -> str:
    """Format optional byte counts for CLI output."""
    if not isinstance(value, (int, float)):
        return "unknown"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            if unit == "B":
                return f"{int(amount)} {unit}"
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TiB"


def _memory_compact_snapshot(snapshot: dict) -> str:
    """Render a LanceDB storage snapshot in one line."""
    rows = snapshot.get("rows", "?")
    versions = snapshot.get("versions", "?")
    fragments = snapshot.get("fragments", "?")
    live = _format_bytes(snapshot.get("liveBytes"))
    disk = _format_bytes(snapshot.get("onDiskBytes"))
    return (
        f"rows={rows} live={live} disk={disk} versions={versions} fragments={fragments}"
    )


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
    reconcile [--run]      Audit drift across LanceDB/FTS/usage stores
                           (dry-run by default; --run repairs missing FTS rows)

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
    elif subcmd == "reconcile":
        return _db_reconcile(cwd, run="--run" in args)
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


def _db_reconcile(cwd: pathlib.Path, *, run: bool) -> int:
    """Audit (and, with --run, repair) drift across LanceDB/FTS/usage stores.

    Dry-run by default: only prints the 3-way drift report. ``--run`` repairs
    ONLY the safe direction (re-upserting Lance rows missing from the FTS
    mirror) — it never deletes Lance rows, never deletes usage rows, and never
    touches vectors. Ghost FTS rows and orphaned usage rows are always
    report-only.
    """
    import asyncio

    import simba.memory.fts
    import simba.memory.reconcile

    data_dir = simba.memory.reconcile.resolve_data_dir(cwd)
    lance_path = data_dir / "memories.lance"
    if not lance_path.exists():
        print(f"Error: no LanceDB store found at {lance_path}", file=sys.stderr)
        return 1

    async def _run() -> simba.memory.reconcile.ReconcileReport:
        import lancedb

        db = await lancedb.connect_async(str(lance_path))
        table = await db.open_table("memories")
        fts_path = data_dir / simba.memory.fts.FTS_FILENAME
        return await simba.memory.reconcile.reconcile(table, fts_path, cwd, apply=run)

    try:
        report = asyncio.run(_run())
    except Exception as exc:
        print(f"Error: failed to open memory stores: {exc}", file=sys.stderr)
        return 1

    print(simba.memory.reconcile.format_report(report))
    print()
    if run:
        print(f"mode: --run (repaired {len(report.repaired_ids)} missing-fts row(s))")
    elif report.missing_fts_ids:
        print("mode: dry-run (pass --run to upsert missing-fts rows from LanceDB)")
    else:
        print("mode: dry-run (no repairs needed)")
    return 0


def _cmd_config(args: list[str]) -> int:
    """Unified configuration."""
    import simba.config_cli

    return simba.config_cli.main(args)


def _cmd_markers(args: list[str]) -> int:
    """Discover, audit, and update SIMBA markers."""
    import simba.markers_cli

    return simba.markers_cli.main(args)


def _cmd_preflight(args: list[str]) -> int:
    """`simba preflight <task>` — surface the right approach BEFORE acting (spec 28).

    Returns the intent-relevant doctrine (project-scoped recall), the applicable
    TOOL_RULEs + redirect rules, and a short brief; sets the per-turn preflight
    flag (so the PreToolUse gate sees it) and emits the 🦁☑ ledger. Mostly
    intent-keyed recall + a rules/redirects lookup.
    """
    import simba.db
    import simba.doctrine.preflight as preflight
    import simba.guardian.preflight_flag as preflight_flag
    import simba.hooks._memory_client
    import simba.redirect.store as redirect_store

    session_id = _parse_opt_value(args, "--session") or ""
    project_path = _parse_opt_value(args, "--project-path") or str(pathlib.Path.cwd())

    # Task is everything that isn't a --flag or its value.
    skip_next = False
    task_parts: list[str] = []
    for tok in args:
        if skip_next:
            skip_next = False
            continue
        if tok.startswith("--"):
            skip_next = True
            continue
        task_parts.append(tok)
    task = " ".join(task_parts).strip()

    if not task:
        print("Usage: simba preflight [--session SID] <task>", file=sys.stderr)
        print("Error: a task description is required", file=sys.stderr)
        return 1

    cwd = pathlib.Path.cwd()

    # (a) intent-relevant doctrine (project-scoped recall — spec 26).
    doctrine_mems = simba.hooks._memory_client.recall_memories(
        task, project_path=project_path
    )
    doctrine_lines = [
        (m.get("content") or "").strip()
        for m in doctrine_mems
        if (m.get("content") or "").strip()
    ]

    # (b) applicable TOOL_RULEs + redirect rules for the project.
    project_id = simba.db.resolve_project_id(cwd)
    tool_rule_mems = simba.hooks._memory_client.recall_memories(
        task,
        project_path=project_id,
        filters={"types": ["TOOL_RULE"]},
    )
    tool_rules = preflight.tool_rule_lines(tool_rule_mems)
    try:
        redirects = preflight.redirect_lines(
            redirect_store.load_rules(cwd, project_path=project_id)
        )
    except Exception:
        redirects = []

    # (c) the brief + the 🦁☑ ledger.
    brief = preflight.build_brief(
        task=task,
        doctrine_lines=doctrine_lines,
        tool_rules=tool_rules,
        redirects=redirects,
    )
    print(brief)

    # Side effect: set the per-turn flag so the PreToolUse mandate gate clears.
    preflight_flag.set_preflight(session_id, task=task)
    return 0


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

    if args and args[0] == "bench":
        return _eval_bench(args[1:])
    if args and args[0] == "halumem":
        return _eval_halumem(args[1:])
    if args and args[0] == "triage":
        return _eval_triage(args[1:])
    if args and args[0] == "ambiguity":
        return _eval_ambiguity(args[1:])
    if args and args[0] == "leaderboard":
        return _eval_leaderboard(args[1:])
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


def _eval_triage(args: list[str]) -> int:
    """Run the built-in retrieval-triage fixture."""
    import json as _json

    import simba.eval.recall_triage as recall_triage_eval

    path = _parse_opt_value(args, "--path")
    as_json = "--json" in args
    allowed = {"--path", "--json"}
    i = 0
    while i < len(args):
        if args[i] == "--path" and i + 1 < len(args):
            i += 2
            continue
        if args[i] in allowed:
            i += 1
            continue
        print(f"eval triage: unknown option {args[i]!r}", file=sys.stderr)
        print("Usage: simba eval triage [--path CASES.jsonl] [--json]")
        return 1

    cases = recall_triage_eval.load_cases(pathlib.Path(path) if path else None)
    result = recall_triage_eval.evaluate(cases)
    if as_json:
        print(_json.dumps(result, indent=2))
    else:
        print(
            "recall-triage eval: "
            f"n={result['n']} accuracy={result['accuracy']:.3f} "
            f"false_negatives={result['false_negatives']} "
            f"false_positives={result['false_positives']} "
            f"gate={result['gate']}"
        )
        for row in result["cases"]:
            if not row["ok"]:
                print(
                    "  miss: "
                    f"expected={row['expected']} actual={row['actual']} "
                    f"reason={row['reason']} prompt={row['prompt']!r}"
                )
    return 0 if result["gate"] == "pass" else 1


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

    print(f"generating queries for up to {n} of {len(mems)} memories…", file=sys.stderr)
    dataset = build.build_from_memories(
        mems, client=client, name=f"real-corpus-{n}", max_cases=n
    )
    pathlib.Path(out).write_text(_json.dumps(dataset.to_dict(), indent=2))
    print(f"wrote {out}: {len(dataset.corpus)} memories, {len(dataset.cases)} cases")
    return 0


def _eval_ambiguity(args: list[str]) -> int:
    """Run executable ambiguity smoke cases.

    Usage:
      simba eval ambiguity [--path PATH] [--backend python|souffle|clingo] [--json]
      simba eval ambiguity --generate python|souffle|clingo [--artifact-dir DIR]
      simba eval ambiguity --fail18 [--repair] [--path PATH] [--corpus PATH]
                           [--backend python|souffle|clingo]
    """
    import json as _json

    import simba.eval.ambiguity as ambiguity
    import simba.eval.ambiguity_backends as ambiguity_backends
    import simba.eval.ambiguity_codegen as ambiguity_codegen
    import simba.eval.ambiguity_fail18 as ambiguity_fail18

    path = ""
    backend = "python"
    as_json = False
    fail18 = False
    repair = False
    corpus = ""
    generate_language = ""
    artifact_dir = ""
    i = 0
    while i < len(args):
        if args[i] == "--path" and i + 1 < len(args):
            path = args[i + 1]
            i += 2
        elif args[i] == "--backend" and i + 1 < len(args):
            backend = args[i + 1]
            i += 2
        elif args[i] == "--fail18":
            fail18 = True
            i += 1
        elif args[i] == "--repair":
            repair = True
            i += 1
        elif args[i] == "--corpus" and i + 1 < len(args):
            corpus = args[i + 1]
            i += 2
        elif args[i] == "--json":
            as_json = True
            i += 1
        elif args[i] == "--generate" and i + 1 < len(args):
            generate_language = args[i + 1]
            i += 2
        elif args[i] == "--artifact-dir" and i + 1 < len(args):
            artifact_dir = args[i + 1]
            i += 2
        else:
            print(f"eval ambiguity: unknown option {args[i]!r}", file=sys.stderr)
            print(
                "Usage: simba eval ambiguity [--fail18] [--path PATH] "
                "[--corpus PATH] [--backend python|souffle|clingo] "
                "[--repair] [--generate LANGUAGE] [--artifact-dir DIR] [--json]",
                file=sys.stderr,
            )
            return 1

    try:
        if fail18 and generate_language:
            print(
                "eval ambiguity: --generate is for ambiguity JSON cases, not fail18",
                file=sys.stderr,
            )
            return 1
        if fail18:
            summary = ambiguity_fail18.summarize(
                pathlib.Path(path) if path else ambiguity_fail18.DEFAULT_MANIFEST,
                backend=backend,
                repair=repair,
                corpus_path=(
                    pathlib.Path(corpus) if corpus else ambiguity_fail18.DEFAULT_CORPUS
                ),
            )
            if as_json:
                print(_json.dumps(summary.to_dict(), indent=2))
            else:
                label = "repaired " if repair else ""
                print(
                    f"clingo_fail18 {label}ambiguity range coverage "
                    f"({summary.backend}): "
                    f"{summary.contains_gold}/{summary.gold_known} known gold "
                    f"inside range; misses={summary.misses_gold}; total={summary.total}"
                )
                for item in summary.results:
                    mark = (
                        "?"
                        if item.contains_gold is None
                        else "ok"
                        if item.contains_gold
                        else "MISS"
                    )
                    print(
                        f"  {mark:<4} {item.question_id} "
                        f"gold={item.gold_numeric} range={item.answer_space} "
                        f"type={item.answer_type} mode={item.failure_mode}"
                    )
            return 0

        case_path = (
            pathlib.Path(path)
            if path
            else pathlib.Path(__file__).parent / "eval" / "datasets" / "ambiguity.json"
        )
        if generate_language:
            generated = []
            ok = True
            for case in ambiguity.load_cases(case_path):
                program, run = ambiguity_codegen.generate_and_run(
                    case, language=generate_language
                )
                if artifact_dir:
                    ambiguity_codegen.save_program(program, artifact_dir)
                generated.append(
                    {
                        "case_id": case.id,
                        "language": program.language,
                        "answer_space": run.answer_space,
                        "ok": run.ok,
                        "stderr": run.stderr,
                    }
                )
                ok = ok and run.ok
            if as_json:
                print(_json.dumps(generated, indent=2))
            else:
                for item in generated:
                    mark = "ok" if item["ok"] else "FAIL"
                    print(
                        f"{item['case_id']}: generated {item['language']} "
                        f"{mark} {item['answer_space']}"
                    )
            return 0 if ok else 1
        reports = ambiguity.evaluate_all(
            ambiguity.load_cases(case_path), backend=backend
        )
        if as_json:
            print(
                _json.dumps(
                    [
                        {
                            "case_id": report.case_id,
                            "question": report.question,
                            "answer_space": report.answer_space,
                            "interpretations": [
                                {
                                    "interpretation_id": item.interpretation_id,
                                    "answer": item.answer,
                                    "reliability": item.reliability,
                                    "evidence_ids": item.evidence_ids,
                                }
                                for item in report.interpretations
                            ],
                        }
                        for report in reports
                    ],
                    indent=2,
                )
            )
        else:
            for report in reports:
                print(f"{report.case_id}: {report.answer_space}")
        return 0
    except ambiguity_backends.BackendUnavailableError as exc:
        print(f"eval ambiguity: backend unavailable: {exc}", file=sys.stderr)
        return 2
    except ambiguity_codegen.CodegenError as exc:
        print(f"eval ambiguity: codegen failed: {exc}", file=sys.stderr)
        return 2


def _eval_bench(args: list[str]) -> int:
    """simba eval bench locomo|longmemeval [--qa] [--n N|--per N|all]
    [--k K] [--split dev|test] [--path PATH] [--json]
    [--baseline] [--cache PATH] [--abstention] [--full]
    [--compare-readback] [--driver-report PATH] [--driver-loop PATH]
    [--persona-start N]
    """
    import dataclasses
    import json as _json
    import time

    import simba.config
    import simba.eval.bench_config  # registers the "bench" section
    import simba.eval.bench_results as bench_results
    import simba.eval.benchmarks.hotpotqa as hotpotqa
    import simba.eval.benchmarks.locomo as locomo
    import simba.eval.benchmarks.longmemeval as lme
    import simba.eval.benchmarks.run as bench_run
    import simba.eval.benchmarks.subtlememory as subtlememory
    import simba.eval.run as run
    import simba.memory.embedding_cache as ec

    usage = (
        "Usage: simba eval bench locomo|longmemeval|hotpotqa|subtlememory [--qa] "
        "[--n N | --per N | all] [--k K] [--split dev|test] "
        "[--path PATH] [--json] [--baseline] [--cache PATH] "
        "[--abstention] [--full] [--persona-limit L] [--compare-readback] "
        "[--driver-report PATH] [--driver-loop PATH] [--persona-start N]"
    )

    if not args or args[0].startswith("--"):
        print(usage, file=sys.stderr)
        return 1

    dataset_name = args[0]
    if dataset_name not in ("locomo", "longmemeval", "hotpotqa", "subtlememory"):
        print(
            f"eval bench: unknown dataset {dataset_name!r}; "
            "choose locomo, longmemeval, hotpotqa, or subtlememory",
            file=sys.stderr,
        )
        return 1

    run_qa_flag = False
    n_mode = "n"
    n_val = 0
    k = 0
    split_arg = ""
    path_arg = ""
    as_json = False
    want_baseline = False
    abstention_flag = False
    full_flag = False
    compare_readback = False
    cache_arg = ""
    driver_report_path = ""
    driver_loop_path = ""
    persona_limit = -1  # -1 = use config default (subtlememory only)
    persona_start = 0

    i = 1
    while i < len(args):
        if args[i] == "--qa":
            run_qa_flag = True
            i += 1
        elif args[i] == "--baseline":
            want_baseline = True
            i += 1
        elif args[i] == "--abstention":
            abstention_flag = True
            i += 1
        elif args[i] == "--full":
            full_flag = True
            i += 1
        elif args[i] == "--compare-readback":
            compare_readback = True
            i += 1
        elif args[i] == "--cache" and i + 1 < len(args):
            cache_arg = args[i + 1]
            i += 2
        elif args[i] == "--driver-report" and i + 1 < len(args):
            driver_report_path = args[i + 1]
            i += 2
        elif args[i] == "--driver-loop" and i + 1 < len(args):
            driver_loop_path = args[i + 1]
            i += 2
        elif args[i] == "--n" and i + 1 < len(args):
            n_mode, n_val = "n", int(args[i + 1])
            i += 2
        elif args[i] == "--per" and i + 1 < len(args):
            n_mode, n_val = "per", int(args[i + 1])
            i += 2
        elif args[i] == "all":
            n_mode = "all"
            i += 1
        elif args[i] == "--k" and i + 1 < len(args):
            k = int(args[i + 1])
            i += 2
        elif args[i] == "--split" and i + 1 < len(args):
            split_arg = args[i + 1]
            i += 2
        elif args[i] == "--path" and i + 1 < len(args):
            path_arg = args[i + 1]
            i += 2
        elif args[i] == "--persona-limit" and i + 1 < len(args):
            persona_limit = int(args[i + 1])
            i += 2
        elif args[i] == "--persona-start" and i + 1 < len(args):
            persona_start = int(args[i + 1])
            i += 2
        elif args[i] == "--json":
            as_json = True
            i += 1
        else:
            print(f"eval bench: unknown option {args[i]!r}", file=sys.stderr)
            return 1

    if compare_readback and dataset_name != "subtlememory":
        print(
            "eval bench: --compare-readback is only supported for subtlememory",
            file=sys.stderr,
        )
        return 1
    if driver_report_path and dataset_name != "subtlememory":
        print(
            "eval bench: --driver-report is only supported for subtlememory",
            file=sys.stderr,
        )
        return 1
    if driver_loop_path and dataset_name != "subtlememory":
        print(
            "eval bench: --driver-loop is only supported for subtlememory",
            file=sys.stderr,
        )
        return 1
    if driver_loop_path and driver_report_path:
        print(
            "eval bench: choose either --driver-loop or --driver-report, not both",
            file=sys.stderr,
        )
        return 1
    if driver_loop_path and run_qa_flag:
        print("eval bench: --driver-loop does not run --qa", file=sys.stderr)
        return 1

    bcfg = simba.config.load("bench")
    mcfg = simba.config.load("memory")
    mcfg = dataclasses.replace(mcfg, **bcfg.eval_memory_config_overrides())

    if dataset_name == "locomo":
        dataset_path = path_arg or bcfg.locomo_path
    elif dataset_name == "hotpotqa":
        dataset_path = path_arg or bcfg.hotpotqa_path
    elif dataset_name == "subtlememory":
        dataset_path = path_arg or bcfg.subtlememory_path
    else:
        dataset_path = path_arg or bcfg.longmemeval_path
    if not dataset_path:
        print(
            f"eval bench: no path for {dataset_name} "
            f"(set bench.{dataset_name}_path or pass --path)",
            file=sys.stderr,
        )
        return 1

    def _resolve_bench_path(s: str) -> pathlib.Path:
        p = pathlib.Path(s)
        return p if p.is_absolute() else pathlib.Path.cwd() / p

    try:
        embed_cache = ec.EmbeddingCache(_resolve_bench_path(bcfg.embedding_cache_path))
        embed_doc, embed_query = run.sync_embedders(mcfg, cache=embed_cache)
    except Exception as exc:  # model download/load failure
        print(f"eval bench: could not load the embedding model: {exc}", file=sys.stderr)
        return 1

    if dataset_name == "locomo":
        datasets = locomo.load_locomo(dataset_path)
    elif dataset_name == "hotpotqa":
        datasets = hotpotqa.load_hotpotqa(dataset_path)
    elif dataset_name == "subtlememory":
        plimit = (
            persona_limit if persona_limit >= 0 else bcfg.subtlememory_persona_limit
        )
        datasets = subtlememory.load_subtlememory(
            dataset_path, persona_limit=plimit, persona_start=persona_start
        )
    else:
        datasets = lme.load_longmemeval(
            dataset_path, include_abstention=abstention_flag
        )
    if n_mode == "n" and n_val > 0:
        datasets = datasets[:n_val]

    import simba.llm.client as llm_client

    # One client, threaded into retrieval so the reranker (memory.llm_rerank_*)
    # and LLM-HyDE (memory.hyde_mode="llm") levers can actually fire under the
    # bench; unused when those are off, so the baseline is unchanged.
    bench_llm = llm_client.get_client()

    driver_report = None
    driver_report_out = None
    driver_loop_report = None
    driver_loop_out = None
    readback_report = None

    def _run_subtle_driver_case(
        name: str,
        cfg,
        overrides: dict[str, object],
    ) -> dict[str, object]:
        recall, ranked = subtlememory.run_recall_with_ranked(
            datasets,
            embed_doc=embed_doc,
            embed_query=embed_query,
            cfg=cfg,
            llm_client=bench_llm,
        )
        ledger = subtlememory.build_failure_ledger(datasets, ranked)
        readback = subtlememory.compare_readback_ceiling(recall, datasets)
        return {
            "name": name,
            "config_overrides": overrides,
            "recall": recall,
            "metric_snapshot": subtlememory.recall_metric_snapshot(recall),
            "readback": readback,
            "driver": ledger,
            "driver_summary": ledger["summary"],
        }

    if driver_loop_path:
        baseline_overrides = {"session_expansion_enabled": False}
        baseline_cfg = dataclasses.replace(mcfg, **baseline_overrides)
        baseline_run = _run_subtle_driver_case(
            "baseline", baseline_cfg, baseline_overrides
        )
        variants: list[tuple[str, dict[str, object]]] = [
            (
                "session_top2_w2",
                {
                    "session_expansion_enabled": True,
                    "session_expansion_top_sessions": 2,
                    "session_expansion_weight": 2.0,
                },
            ),
            (
                "session_top1_w2",
                {
                    "session_expansion_enabled": True,
                    "session_expansion_top_sessions": 1,
                    "session_expansion_weight": 2.0,
                },
            ),
            (
                "session_top3_w2",
                {
                    "session_expansion_enabled": True,
                    "session_expansion_top_sessions": 3,
                    "session_expansion_weight": 2.0,
                },
            ),
            (
                "session_top2_w1",
                {
                    "session_expansion_enabled": True,
                    "session_expansion_top_sessions": 2,
                    "session_expansion_weight": 1.0,
                },
            ),
        ]
        variant_runs = [
            _run_subtle_driver_case(
                name, dataclasses.replace(mcfg, **overrides), overrides
            )
            for name, overrides in variants
        ]
        winner = max(
            variant_runs,
            key=lambda item: subtlememory.driver_objective(item["recall"]),
        )
        baseline_obj = subtlememory.driver_objective(baseline_run["recall"])
        winner_obj = subtlememory.driver_objective(winner["recall"])
        winner_positive = winner_obj > baseline_obj
        winner_delta = subtlememory.metric_snapshot_delta(
            winner["metric_snapshot"], baseline_run["metric_snapshot"]
        )
        promotion_gate = subtlememory.driver_promotion_gate(
            winner_positive=winner_positive,
            winner_delta=winner_delta,
        )
        driver_loop_report = {
            "mode": "subtlememory_driver_loop",
            "dataset_path": dataset_path,
            "persona_limit": (
                persona_limit if persona_limit >= 0 else bcfg.subtlememory_persona_limit
            ),
            "persona_start": persona_start,
            "objective": [
                "contradictory.recall@10",
                "overall.recall@10",
                "contradictory.mrr",
                "overall.mrr",
            ],
            "summary": {
                "baseline_recommendation": baseline_run["driver_summary"][
                    "recommendation"
                ],
                "winner": winner["name"],
                "winner_positive": winner_positive,
                "winner_config_overrides": winner["config_overrides"],
                "winner_delta": winner_delta,
                "promotion_gate": promotion_gate,
                "promotion_gate_passed": promotion_gate["passed"],
                "next_action": (
                    "run held-out personas and cross-bench gates"
                    if promotion_gate["passed"]
                    else "try the baseline driver recommendation's next lever"
                ),
            },
            "baseline": baseline_run,
            "variants": variant_runs,
        }
        driver_loop_out = str(
            subtlememory.write_failure_ledger(driver_loop_report, driver_loop_path)
        )
        recall_report = baseline_run["recall"]
        readback_report = baseline_run["readback"]
    elif driver_report_path:
        recall_report, ranked_by_case = subtlememory.run_recall_with_ranked(
            datasets,
            embed_doc=embed_doc,
            embed_query=embed_query,
            cfg=mcfg,
            llm_client=bench_llm,
        )
        driver_report = subtlememory.build_failure_ledger(datasets, ranked_by_case)
        driver_report_out = str(
            subtlememory.write_failure_ledger(driver_report, driver_report_path)
        )
    else:
        recall_report = bench_run.run_recall(
            datasets,
            embed_doc=embed_doc,
            embed_query=embed_query,
            cfg=mcfg,
            llm_client=bench_llm,
        )
        readback_report = None
    if compare_readback and readback_report is None:
        readback_report = subtlememory.compare_readback_ceiling(recall_report, datasets)

    qa_report = None
    if run_qa_flag:
        import simba.eval.benchmarks.judge as judge
        import simba.eval.benchmarks.judge_cache as jc
        import simba.llm.judge_config as jcfg

        if n_mode == "per":
            qa_datasets = judge.sample_cases(datasets, per_category=n_val)
        elif n_mode == "n" and n_val > 0:
            qa_datasets = judge.sample_cases(datasets, n=n_val)
        else:
            qa_datasets = datasets
        k_val = k or bcfg.default_k
        # Separate answerer (bench_llm) from judge so the model never grades its
        # own answer (B1: get_judge_client defaults to a different local model).
        judge_client = jcfg.get_judge_client()
        cache_path = cache_arg or bcfg.judge_cache_path
        # eval_cfg carries eval.ircot_enabled -> run_qa routes multi-hop cases
        # through the IRCoT answer-time loop when enabled.
        eval_cfg = simba.config.load("eval")
        qa_report = judge.run_qa(
            qa_datasets,
            embed_doc=embed_doc,
            embed_query=embed_query,
            cfg=mcfg,
            llm=bench_llm,
            judge=judge_client,
            k=k_val,
            include_abstention=abstention_flag,
            cache=jc.JudgeCache(_resolve_bench_path(cache_path)),
            eval_cfg=eval_cfg,
            judge_model=judge_client._cfg.model,
            judge_style=bcfg.judge_style,
            reader_style=bcfg.reader_style,
            preference_synthesis=bcfg.preference_synthesis,
            temporal_codegen=bcfg.temporal_codegen,
        )

    git_sha = bench_results.current_git_sha()
    config_snap = bench_results.config_snapshot(
        mcfg,
        bcfg,
        llm_cfg=bench_llm._cfg,
        judge_cfg=judge_client._cfg if run_qa_flag else None,
    )
    excluded_count = int((qa_report or {}).get("n_skipped", 0))
    abstained_count = int(((qa_report or {}).get("abstention") or {}).get("n", 0))
    record = {
        "timestamp": time.time(),
        "git_sha": git_sha,
        "dataset": dataset_name,
        "split": split_arg or None,
        "config": config_snap,
        "provenance": bench_results.build_provenance(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            split=split_arg or None,
            config=config_snap,
            git_sha=git_sha,
            answerer_cfg=bench_llm._cfg,
            judge_cfg=judge_client._cfg if run_qa_flag else None,
            excluded_count=excluded_count,
            abstained_count=abstained_count,
        ),
        "recall": recall_report,
        "readback": readback_report,
        "driver": (
            {"path": driver_report_out, "summary": driver_report["summary"]}
            if driver_report is not None
            else None
        ),
        "driver_loop": (
            {"path": driver_loop_out, "summary": driver_loop_report["summary"]}
            if driver_loop_report is not None
            else None
        ),
        "qa": qa_report,
    }
    bench_results.append_result(_resolve_bench_path(bcfg.results_path), record)

    if want_baseline:
        import simba.eval.benchmarks.baseline_store as baseline_store

        # `_s` distinguishes the full longmemeval_s haystack from the oracle set
        # so their baselines don't mix in one jsonl.
        suffix = "_s" if (dataset_name == "longmemeval" and full_flag) else ""
        meta = {
            "split": split_arg or None,
            "n_mode": n_mode,
            "n_val": n_val,
            "k": k or bcfg.default_k,
            "abstention": abstention_flag,
            "full": full_flag,
        }
        baseline_store.append_baseline(
            f"{dataset_name}{suffix}_recall", recall_report, metadata=meta
        )
        if qa_report is not None:
            baseline_store.append_baseline(
                f"{dataset_name}{suffix}_qa", qa_report, metadata=meta
            )

    if as_json:
        print(
            _json.dumps(
                {
                    "recall": recall_report,
                    "readback": readback_report,
                    "driver": (
                        {"path": driver_report_out, "summary": driver_report["summary"]}
                        if driver_report is not None
                        else None
                    ),
                    "driver_loop": (
                        {
                            "path": driver_loop_out,
                            "summary": driver_loop_report["summary"],
                        }
                        if driver_loop_report is not None
                        else None
                    ),
                    "qa": qa_report,
                },
                indent=2,
            )
        )
    else:
        o = recall_report["overall"]
        print(
            f"\n{dataset_name} recall ({recall_report['n_conversations']} "
            f"conversations, {recall_report['n_cases']} questions)"
        )
        print(
            f"  OVERALL  recall@1={o['recall@1']:.3f} recall@3={o['recall@3']:.3f} "
            f"recall@5={o['recall@5']:.3f} recall@10={o['recall@10']:.3f} "
            f"mrr={o['mrr']:.3f}"
        )
        for cat, m in recall_report["by_category"].items():
            print(
                f"  {cat:<18} n={m['n']:<4} r@1={m['recall@1']:.3f} "
                f"r@5={m['recall@5']:.3f} r@10={m['recall@10']:.3f} "
                f"mrr={m['mrr']:.3f}"
            )
        if "latency" in recall_report:
            lat = recall_report["latency"]
            print(f"  p50={lat['p50_ms']:.0f}ms p95={lat['p95_ms']:.0f}ms")
        if readback_report is not None:
            ceiling = readback_report["ceiling"]
            ro = ceiling["overall"]
            delta = readback_report["delta_vs_recall"]["overall"]
            diag = ceiling["diagnostics"]
            print("\nsubtlememory readback ceiling")
            print(
                f"  CEILING recall@5={ro['recall@5']:.3f} "
                f"recall@10={ro['recall@10']:.3f} mrr={ro['mrr']:.3f}"
            )
            print(
                f"  DELTA   recall@5={delta.get('recall@5', 0.0):+.3f} "
                f"recall@10={delta.get('recall@10', 0.0):+.3f} "
                f"mrr={delta.get('mrr', 0.0):+.3f}"
            )
            print(
                f"  gold ids avg={diag['avg_gold_ids']:.1f} "
                f"max={diag['max_gold_ids']} "
                f"gold>10={diag['gold_gt_k'].get('10', 0)}/{ceiling['n_cases']}"
            )
        if driver_report is not None:
            summary = driver_report["summary"]
            print(f"\nsubtlememory driver report: {driver_report_out}")
            print(
                f"  recommendation={summary['recommendation']} "
                f"k={summary['analysis_k']} cases={summary['n_cases']}"
            )
            for label, count in summary["gap_counts"].items():
                print(f"  {label:<24} {count}")
        if driver_loop_report is not None:
            summary = driver_loop_report["summary"]
            delta = summary["winner_delta"]
            print(f"\nsubtlememory driver loop: {driver_loop_out}")
            print(
                f"  baseline_recommendation={summary['baseline_recommendation']} "
                f"winner={summary['winner']} positive={summary['winner_positive']} "
                f"gate={summary['promotion_gate_passed']}"
            )
            print(
                f"  OVERALL  r@10={delta['overall']['recall@10']:+.3f} "
                f"mrr={delta['overall']['mrr']:+.3f}"
            )
            print(
                f"  CONTRA   r@10={delta['contradictory']['recall@10']:+.3f} "
                f"mrr={delta['contradictory']['mrr']:+.3f}"
            )
            print(f"  next={summary['next_action']}")
        if qa_report is not None:
            print(
                f"\n{dataset_name} QA accuracy (graded={qa_report['n_graded']}, "
                f"skipped={qa_report['n_skipped']})"
            )
            print(f"  OVERALL  accuracy={qa_report['overall']['accuracy']:.3f}")
            if "abstention" in qa_report:
                ab = qa_report["abstention"]
                print(f"  ABSTENTION n={ab['n']:<4} accuracy={ab['accuracy']:.3f}")
            if "latency" in qa_report:
                lat = qa_report["latency"]
                print(f"  p50={lat['p50_ms']:.0f}ms p95={lat['p95_ms']:.0f}ms")
    return 0


def _eval_halumem(args: list[str]) -> int:
    """simba eval halumem [--user-num N] [--k K] [--path PATH] [--json]

    Memory-hallucination eval (docs/plans/10). Unlike recall@k, the metrics
    (accuracy / hallucination_rate / boundary-abstention) reward NOT surfacing
    wrong/stale memories — so Phase-6 dormancy / Phase-7 contradiction-resolution
    can be ablated by toggling their config and re-running.
    """
    import dataclasses
    import json as _json
    import time

    import simba.config
    import simba.eval.bench_config  # registers the "bench" section
    import simba.eval.bench_results as bench_results
    import simba.eval.benchmarks.halumem as halumem
    import simba.eval.benchmarks.halumem_qa as halumem_qa
    import simba.eval.run as run
    import simba.llm.client as llm_client
    import simba.llm.judge_config as jcfg
    import simba.memory.embedding_cache as ec

    user_num = 0
    k = 0
    path_arg = ""
    as_json = False
    i = 0
    while i < len(args):
        if args[i] == "--user-num" and i + 1 < len(args):
            user_num, i = int(args[i + 1]), i + 2
        elif args[i] == "--k" and i + 1 < len(args):
            k, i = int(args[i + 1]), i + 2
        elif args[i] == "--path" and i + 1 < len(args):
            path_arg, i = args[i + 1], i + 2
        elif args[i] == "--json":
            as_json, i = True, i + 1
        else:
            print(f"eval halumem: unknown option {args[i]!r}", file=sys.stderr)
            return 1

    bcfg = simba.config.load("bench")
    mcfg = simba.config.load("memory")
    mcfg = dataclasses.replace(mcfg, **bcfg.eval_memory_config_overrides())

    def _resolve(s: str) -> pathlib.Path:
        p = pathlib.Path(s)
        return p if p.is_absolute() else pathlib.Path.cwd() / p

    dataset_path = path_arg or bcfg.halumem_path
    if not dataset_path or not _resolve(dataset_path).exists():
        print(
            f"eval halumem: no dataset at {dataset_path!r} "
            "(fetch via scripts/fetch_benchmarks.sh or set bench.halumem_path)",
            file=sys.stderr,
        )
        return 1

    try:
        embed_cache = ec.EmbeddingCache(_resolve(bcfg.embedding_cache_path))
        embed_doc, embed_query = run.sync_embedders(mcfg, cache=embed_cache)
    except Exception as exc:
        print(f"eval halumem: could not load the embedder: {exc}", file=sys.stderr)
        return 1

    users = halumem.load_halumem(
        _resolve(dataset_path), user_limit=user_num or bcfg.halumem_user_limit
    )
    answerer = llm_client.get_client()
    judge_client = jcfg.get_judge_client()
    # If answerer/judge use an auto-spawn local server (mlx-server / llama-server),
    # ensure it's up (loads the model once, so the run doesn't reload per call).
    # No-op for openai-http / CLI providers. Fail-open.
    import simba.llm.local_server as local_server

    local_server.ensure_for_config(answerer._cfg)
    local_server.ensure_for_config(judge_client._cfg)
    report = halumem_qa.run_halumem_qa(
        users,
        embed_doc=embed_doc,
        embed_query=embed_query,
        cfg=mcfg,
        llm=answerer,
        judge=judge_client,
        k=k or bcfg.default_k,
    )

    git_sha = bench_results.current_git_sha()
    config_snap = bench_results.config_snapshot(
        mcfg, bcfg, llm_cfg=answerer._cfg, judge_cfg=judge_client._cfg
    )
    record = {
        "timestamp": time.time(),
        "git_sha": git_sha,
        "dataset": "halumem",
        "split": None,
        "config": config_snap,
        "provenance": bench_results.build_provenance(
            dataset_name="halumem",
            dataset_path=dataset_path,
            split=None,
            config=config_snap,
            git_sha=git_sha,
            answerer_cfg=answerer._cfg,
            judge_cfg=judge_client._cfg,
            excluded_count=int(report.get("n_skipped", 0)),
        ),
        "halumem": report,
    }
    bench_results.append_result(_resolve(bcfg.results_path), record)

    if as_json:
        print(_json.dumps(report, indent=2))
    else:
        o = report["overall"]
        b = report["boundary"]
        print(
            f"\nhalumem QA ({len(users)} users, graded={report['n_graded']}, "
            f"skipped={report['n_skipped']})"
        )
        print(
            f"  OVERALL  accuracy={o['accuracy']:.3f} "
            f"hallucination={o['hallucination_rate']:.3f} "
            f"omission={o['omission_rate']:.3f}"
        )
        print(
            f"  BOUNDARY n={b['n']:<4} abstention-accuracy={b['accuracy']:.3f} "
            f"hallucination={b['hallucination_rate']:.3f}"
        )
        for qt, m in report["by_type"].items():
            print(
                f"  {qt:<22} n={m['n']:<4} acc={m['accuracy']:.3f} "
                f"halluc={m['hallucination_rate']:.3f}"
            )
    return 0


def _eval_leaderboard(args: list[str]) -> int:
    """simba eval leaderboard [--json] [--no-write]"""
    import simba.config
    import simba.db
    import simba.eval.bench_config  # registers the "bench" section
    import simba.eval.bench_results as bench_results
    import simba.eval.leaderboard as lb

    no_write = "--no-write" in args

    bcfg = simba.config.load("bench")
    root = simba.db.find_repo_root(pathlib.Path.cwd()) or pathlib.Path.cwd()

    def _resolve_from_root(s: str) -> pathlib.Path:
        p = pathlib.Path(s)
        return p if p.is_absolute() else root / p

    results_path = _resolve_from_root(bcfg.results_path)
    output_path = _resolve_from_root(bcfg.leaderboard_path)

    if not results_path.exists():
        print(
            "leaderboard: no results found (run simba eval bench first)",
            file=sys.stderr,
        )
        return 1

    records = bench_results.load_results(results_path)
    groups = bench_results.latest_two_by_group(records)

    if no_write:
        print(lb.render_stdout(groups))
        return 0

    lb.write_leaderboard(results_path, output_path)
    print(lb.render_stdout(groups))
    print(f"\nWrote {output_path}")
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
    if not args or args[0] not in ("digest", "complete", "run-llm"):
        print(
            "Usage: simba rlm digest <transcript_id|--latest>\n"
            "       simba rlm complete <transcript_id> [--stored N]",
            file=sys.stderr,
        )
        return 1

    if args[0] == "run-llm":
        # Internal worker spawned by the llm-cli engine: completes the prompt
        # file with `llm -m <model>`, parses a JSON array of memories, stores
        # each, and (for a digest) marks the rlm_jobs row done.
        rest = args[1:]
        prompt_file = _parse_opt_value(rest, "--prompt-file")
        if not prompt_file:
            print(
                "Usage: simba rlm run-llm --prompt-file PATH [--cwd P] "
                "[--session-source ID] [--mark-rlm-complete]",
                file=sys.stderr,
            )
            return 1
        cwd = _parse_opt_value(rest, "--cwd") or str(pathlib.Path.cwd())
        session_source = _parse_opt_value(rest, "--session-source") or ""
        mark = "--mark-rlm-complete" in rest
        import simba.rlm.engine

        n = simba.rlm.engine.run_completion_from_file(
            prompt_file, cwd=cwd, session_source=session_source, mark_rlm=mark
        )
        print(f"llm-digest: stored {n}")
        return 0

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


def _cmd_transcript(args: list[str]) -> int:
    """Project-scoped transcript resolution for learning extraction.

    `simba transcript pending [--project P] [--json]` — newest pending-extraction
    transcript for the current project (not the global latest.json, which cross-wires
    sessions). `simba transcript mark-extracted <session_id>` — flip its status.
    """
    import simba.transcripts as _tr

    sub = args[0] if args else ""
    if sub == "pending":
        project = str(pathlib.Path.cwd())
        if "--project" in args:
            i = args.index("--project")
            if i + 1 < len(args):
                project = args[i + 1]
        meta = _tr.find_pending(project)
        if not meta:
            print("{}" if "--json" in args else f"No pending transcript for {project}")
            return 1
        if "--json" in args:
            clean = {k: v for k, v in meta.items() if not k.startswith("_")}
            print(json.dumps(clean))
        else:
            print(f"transcript_path: {meta.get('transcript_path', '')}")
            print(f"session_id: {meta.get('session_id', '')}")
            print(f"project_path: {meta.get('project_path', '')}")
            print(f"metadata_path: {meta.get('_metadata_path', '')}")
        return 0
    if sub == "mark-extracted":
        if len(args) < 2:
            print("usage: simba transcript mark-extracted <session_id>")
            return 1
        meta_path = _tr.default_transcripts_dir() / args[1] / "metadata.json"
        if _tr.mark_extracted(meta_path):
            print(f"marked extracted: {args[1]}")
            return 0
        print(f"could not mark extracted: {meta_path}")
        return 1
    if sub == "distill":
        return _cmd_transcript_distill(args[1:])
    print(
        "usage: simba transcript {pending [--project P] [--json] "
        "| mark-extracted <session_id> "
        "| distill <jsonl> [--session-id X] [--out DIR] [--project-path P]}"
    )
    return 1


def _cmd_transcript_distill(args: list[str]) -> int:
    """`simba transcript distill <jsonl> [--session-id X] [--out DIR]
    [--project-path P]` -- bounded single-pass distillation (see
    ``transcripts/distill.py``) plus persisting any failure->fix arcs it
    found into the project's ``failure_arc`` sidecar table.
    """
    if not args or args[0].startswith("--"):
        print(
            "usage: simba transcript distill <jsonl> [--session-id X] "
            "[--out DIR] [--project-path P]",
            file=sys.stderr,
        )
        return 1

    source = pathlib.Path(args[0])
    session_id = _parse_opt_value(args, "--session-id") or source.stem
    project_path = _parse_opt_value(args, "--project-path") or str(pathlib.Path.cwd())
    out_arg = _parse_opt_value(args, "--out")

    import simba.config
    import simba.hooks.config  # registers "hooks"
    import simba.transcripts as _tr
    import simba.transcripts.arcs as _arcs
    import simba.transcripts.distill as _distill

    out_dir = (
        pathlib.Path(out_arg) if out_arg else _tr.default_transcripts_dir() / session_id
    )

    cfg = simba.config.load("hooks")
    result = _distill.distill_transcript(
        source,
        out_dir=out_dir,
        session_id=session_id,
        project_path=project_path,
        max_output_mb=cfg.distill_max_output_mb,
    )

    for arc in result.arcs:
        _arcs.upsert_arc(
            session_source=session_id,
            harness=result.stats.harness,
            tool=arc.tool,
            signature=arc.signature,
            error_head=arc.error_head,
            failed_args_head=arc.failed_args_head,
            fix_args_head=arc.fix_args_head,
            resolved=arc.resolved,
            repeat_count=arc.repeat_count,
            project_path=project_path,
            cwd=pathlib.Path(project_path),
        )

    if result.skipped:
        print(f"distill: skipped (marker matches) -- {result.md_path}")
    else:
        print(
            f"distill: wrote {result.md_path} "
            f"({result.stats.output_bytes} bytes, "
            f"{len(result.arcs)} arcs, "
            f"{result.stats.elapsed_seconds:.2f}s)"
        )
    return 0


def _free_arg_text(
    args: list[str],
    *,
    value_options: set[str],
    flag_options: set[str] | None = None,
) -> str:
    flag_options = flag_options or set()
    skip_next = False
    parts: list[str] = []
    for tok in args:
        if skip_next:
            skip_next = False
            continue
        if tok in value_options:
            skip_next = True
            continue
        if tok in flag_options:
            continue
        if tok.startswith("--"):
            continue
        parts.append(tok)
    return " ".join(parts).strip()


def _sessions_usage() -> str:
    return (
        "usage: simba sessions index (--latest | --path PATH) "
        "[--project-path P] [--session-id SID] [--source SRC] [--json]\n"
        "       simba sessions search <query> "
        "[--limit N] [--project-path P] [--json]"
    )


def _cmd_sessions(args: list[str]) -> int:
    """Index and search raw transcript messages."""
    import simba.sessions.messages as session_messages

    sub = args[0] if args else ""
    if sub == "index":
        latest = "--latest" in args
        path_arg = _parse_opt_value(args, "--path")
        if latest == bool(path_arg):
            print(_sessions_usage(), file=sys.stderr)
            return 1

        project_path = _parse_opt_value(args, "--project-path") or ""
        session_id = _parse_opt_value(args, "--session-id") or ""
        source = _parse_opt_value(args, "--source") or ""
        if latest:
            meta = _latest_transcript_metadata()
            if not meta:
                print("No latest transcript metadata found.", file=sys.stderr)
                return 1
            path_arg = str(meta.get("transcript_path") or "")
            project_path = project_path or str(meta.get("project_path") or "")
            session_id = session_id or str(meta.get("session_id") or "")
            source = source or str(meta.get("source") or "")

        if not path_arg:
            print(_sessions_usage(), file=sys.stderr)
            return 1
        transcript_path = pathlib.Path(path_arg).expanduser()
        if not transcript_path.is_file():
            print(f"Transcript not found: {transcript_path}", file=sys.stderr)
            return 1
        project_path = project_path or str(pathlib.Path.cwd())

        result = session_messages.index_transcript(
            transcript_path,
            project_path=project_path,
            session_id=session_id,
            source=source,
            cwd=pathlib.Path.cwd(),
        )
        payload = {
            "session_id": result.session_id,
            "project_path": result.project_path,
            "transcript_path": result.transcript_path,
            "source": result.source,
            "message_count": result.message_count,
        }
        if "--json" in args:
            print(json.dumps(payload))
        else:
            print(
                "indexed "
                f"{result.message_count} messages: "
                f"session={result.session_id} transcript={result.transcript_path}"
            )
        return 0

    if sub == "search":
        query = _free_arg_text(
            args[1:],
            value_options={"--limit", "--project-path"},
            flag_options={"--json"},
        )
        if not query:
            print(_sessions_usage(), file=sys.stderr)
            return 1
        limit_raw = _parse_opt_value(args, "--limit")
        try:
            limit = (
                int(limit_raw)
                if limit_raw is not None
                else session_messages.default_search_limit(cwd=pathlib.Path.cwd())
            )
        except ValueError:
            print(f"Invalid --limit value: {limit_raw}", file=sys.stderr)
            return 1
        project_path = _parse_opt_value(args, "--project-path") or ""

        rows = session_messages.search(
            query,
            project_path=project_path,
            limit=limit,
            cwd=pathlib.Path.cwd(),
        )
        if "--json" in args:
            print(json.dumps(rows))
            return 0
        if not rows:
            print("No indexed session messages matched.")
            return 0
        for row in rows:
            text = re.sub(r"\s+", " ", str(row.get("text", ""))).strip()
            if len(text) > 220:
                text = text[:217].rstrip() + "..."
            span = row.get("message_span") or [
                row.get("message_index", 0),
                row.get("message_index", 0),
            ]
            print(
                f"{row.get('session_id', '')}:{span[0]}-{span[1]} "
                f"[{row.get('role', '')}] {text}"
            )
            if row.get("file_refs"):
                print(f"  files: {', '.join(row['file_refs'])}")
            print(f"  transcript: {row.get('transcript_path', '')}")
        return 0

    print(_sessions_usage(), file=sys.stderr)
    return 1


def _values_for(args: list[str], key: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == key and i + 1 < len(args):
            out.append(args[i + 1])
            i += 2
            continue
        i += 1
    return out


def _split_values(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def _git_branch(cwd: pathlib.Path) -> str:
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return ""
    return proc.stdout.strip()


def _task_usage() -> str:
    return (
        "usage: simba task snapshot save --task TEXT [--summary TEXT] "
        "[--next-step TEXT] [--file PATH] [--blocker TEXT]\n"
        "       simba task snapshot show [--project-path P] [--session SID] [--json]\n"
        "       simba task snapshot clear [--reason TEXT] "
        "[--project-path P] [--session SID]"
    )


def _task_project(args: list[str]) -> str:
    return _parse_opt_value(args, "--project-path") or str(pathlib.Path.cwd().resolve())


def _cmd_task(args: list[str]) -> int:
    """Manage compact active task snapshots."""
    if len(args) < 2 or args[0] != "snapshot":
        print(_task_usage(), file=sys.stderr)
        return 1

    import simba.db
    import simba.task_snapshot as snapshots

    sub = args[1]
    rest = args[2:]
    project_path = _task_project(rest)
    session_id = _parse_opt_value(rest, "--session") or ""
    cwd = pathlib.Path.cwd()

    if sub == "save":
        task = _parse_opt_value(rest, "--task") or _free_arg_text(
            rest,
            value_options={
                "--task",
                "--summary",
                "--next-step",
                "--branch",
                "--worktree",
                "--project-path",
                "--session",
                "--file",
                "--files",
                "--blocker",
                "--blockers",
            },
        )
        if not task:
            print(_task_usage(), file=sys.stderr)
            return 1
        files = _split_values(_values_for(rest, "--file"))
        files += _split_values(_values_for(rest, "--files"))
        blockers = _split_values(_values_for(rest, "--blocker"))
        blockers += _split_values(_values_for(rest, "--blockers"))
        with simba.db.connect(cwd):
            row = snapshots.save(
                project_path=project_path,
                session_id=session_id,
                task=task,
                summary=_parse_opt_value(rest, "--summary") or "",
                branch=_parse_opt_value(rest, "--branch") or _git_branch(cwd),
                worktree=_parse_opt_value(rest, "--worktree") or str(cwd.resolve()),
                files=files,
                blockers=blockers,
                next_step=_parse_opt_value(rest, "--next-step") or "",
                now=time.time(),
            )
        if "--json" in rest:
            print(json.dumps(snapshots.to_dict(row)))
        else:
            print(f"saved task snapshot: {row.id}")
        return 0

    if sub == "show":
        with simba.db.connect(cwd):
            row = snapshots.latest(project_path=project_path, session_id=session_id)
            payload = snapshots.to_dict(row) if row is not None else {}
            rendered = snapshots.render(row) if row is not None else ""
        if "--json" in rest:
            print(json.dumps(payload))
        elif rendered:
            print(rendered)
        else:
            print(f"No active task snapshot for {project_path}")
        return 0 if row is not None else 1

    if sub == "clear":
        reason = _parse_opt_value(rest, "--reason") or _free_arg_text(
            rest,
            value_options={"--reason", "--project-path", "--session"},
            flag_options={"--json"},
        )
        with simba.db.connect(cwd):
            row = snapshots.clear(
                project_path=project_path,
                session_id=session_id,
                reason=reason,
                now=time.time(),
            )
        if "--json" in rest:
            print(json.dumps(snapshots.to_dict(row)))
        else:
            print(f"cleared task snapshot: {row.id}")
        return 0

    print(_task_usage(), file=sys.stderr)
    return 1


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
    elif cmd == "codex-curate":
        sys.exit(_cmd_codex_curate(rest))
    elif cmd == "codex-recall":
        sys.exit(_cmd_codex_recall(rest))
    elif cmd == "codex-finalize":
        sys.exit(_cmd_codex_finalize(rest))
    elif cmd == "codex-automation":
        sys.exit(_cmd_codex_automation(rest))
    elif cmd == "hook":
        sys.exit(_cmd_hook(rest))
    elif cmd == "hook-canonical":
        sys.exit(_cmd_hook_canonical(rest))
    elif cmd == "pi-install":
        sys.exit(_cmd_pi_install(rest))
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
    elif cmd == "preflight":
        sys.exit(_cmd_preflight(rest))
    elif cmd == "rlm":
        sys.exit(_cmd_rlm(rest))
    elif cmd == "eval":
        sys.exit(_cmd_eval(rest))
    elif cmd == "episodes":
        sys.exit(_cmd_episodes(rest))
    elif cmd == "db":
        sys.exit(_cmd_db(rest))
    elif cmd == "sessions":
        sys.exit(_cmd_sessions(rest))
    elif cmd == "task":
        sys.exit(_cmd_task(rest))
    elif cmd == "transcript":
        sys.exit(_cmd_transcript(rest))
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
