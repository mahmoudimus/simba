---
name: simba-onboard
description: Analyze project markdown instruction files and generate consolidated SIMBA core instructions with markers. Use when Codex needs to onboard a repo by reading CLAUDE.md/AGENTS.md and .claude docs, then producing .claude/rules/CORE_INSTRUCTIONS.md (or configured filename), updating core reference blocks, and verifying markers.
---

# Simba Onboard

Set up SIMBA markers for a project by analyzing existing markdown files and generating consolidated core instructions.

Follow these steps in order.

## Step 0: Resolve Core Filename

Run:

```bash
simba config get guardian.core_filename
```

Use the returned filename as `${CORE_FILE}` for all following steps. Default is `CORE_INSTRUCTIONS.md`.

## Step 1: Discover Existing Files

Collect onboarding inputs:

1. Read `CLAUDE.md` if it exists.
2. Read `AGENTS.md` if it exists.
3. Read `.md` files under `.claude/`, excluding `.claude/handoffs/` and `.claude/notes/`.
4. Run `simba markers audit`.
5. Run `simba markers list`.

Report all files found, approximate sizes, and any existing markers or `<!-- CORE -->` tags.

## Step 2: Analyze Content

Extract and categorize instructions from discovered files:

| Category | What to extract | SIMBA section |
| --- | --- | --- |
| Critical Constraints | Non-negotiable rules and security requirements | `constraints` |
| Build & Test | Build/test commands and CI steps | `build_commands` |
| Environment | Paths, ports, machines, deployment targets | `environment` |
| Code Style | Formatting and naming conventions | `code_style` |
| Workflow | Git/PR conventions and process rules | `workflow` |
| Agent Rules | Agent/subagent dispatch and context rules | `agent_rules` |

Rules:

- Create only categories with real content.
- Preserve original wording where possible.
- Deduplicate repeated rules.

Report a table of category, source files, and item counts.

## Step 3: Draft `.claude/rules/${CORE_FILE}`

Prepare a full draft with SIMBA markers:

- Use one marker pair per section: `<!-- BEGIN SIMBA:{name} -->` and `<!-- END SIMBA:{name} -->`.
- Keep `SIMBA:core` tight and critical only.
- Separate sections with `---`.

`SIMBA:core` is re-injected repeatedly by guardian logic, so keep it concise.

## Step 4: Ask for Verification Before Writing

Show the full proposed file first.

For each section include:

1. Marker name.
2. Proposed content.
3. Source file(s).
4. Explicit note for `SIMBA:core` that it is repeatedly re-injected.

Request explicit user approval before writing files.

## Step 5: Apply Changes

After approval:

1. Ensure `.claude/rules/` exists.
2. Write `.claude/rules/${CORE_FILE}`.
3. Update `CLAUDE.md` with a core reference block if missing:

```markdown
<!-- BEGIN SIMBA:core_ref -->
**Read `.claude/rules/${CORE_FILE}` for rules that apply to ALL contexts (main session + subagents).**

When dispatching subagents, inject the contents of that file into the prompt.
<!-- END SIMBA:core_ref -->
```

4. Update `AGENTS.md` similarly if it exists:

```markdown
<!-- BEGIN SIMBA:core_ref -->
**All agents must follow `.claude/rules/${CORE_FILE}` before executing.**

When dispatching write-capable agents, read `.claude/rules/${CORE_FILE}` and inject its contents into the dispatch prompt.
<!-- END SIMBA:core_ref -->
```

## Step 6: Verify

Run:

```bash
simba markers audit
simba markers list
```

Report final marker status and next recommended edits.
