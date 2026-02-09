---
name: simba-onboard
description: Analyze project markdown files and generate consolidated core instructions with SIMBA markers
---

# Simba Onboard

Set up SIMBA markers for this project by analyzing existing markdown files and generating consolidated core instructions.

**Follow these steps in order. Do not skip steps.**

---

## Step 0: Resolve Core Filename

Run this command to get the configured filename for the core instructions file:

```bash
simba config get guardian.core_filename
```

This defaults to `CORE_INSTRUCTIONS.md`. Legacy projects may configure it to `CORE.md` or another name. Use the returned value as `${CORE_FILE}` throughout the remaining steps. The full output path will be `.claude/rules/${CORE_FILE}`.

---

## Step 1: Discover Existing Files

Read all project instruction files to understand the current state:

1. Read `CLAUDE.md` (if it exists)
2. Read `AGENTS.md` (if it exists)
3. Read all `.md` files under `.claude/` (excluding `.claude/handoffs/` and `.claude/notes/`)
4. Run `simba markers audit` to check for any existing SIMBA markers
5. Run `simba markers list` to see all markers across the project

**Report to the user**: List every file you found and its approximate size. Note any existing SIMBA markers or `<!-- CORE -->` tags.

---

## Step 2: Analyze Content

For each file read in Step 1, extract and categorize content into these groups:

### Categories to Extract

| Category | What to look for | SIMBA section name |
|----------|------------------|--------------------|
| **Critical Constraints** | Non-negotiable rules, security requirements, things that must ALWAYS be true | `constraints` |
| **Build & Test** | Build commands, test commands, CI steps, make targets | `build_commands` |
| **Environment** | Machine names, SSH info, paths, ports, deployment targets | `environment` |
| **Code Style** | Formatting rules, naming conventions, patterns to follow | `code_style` |
| **Workflow** | Git workflow, commit conventions, PR process, stuck detection | `workflow` |
| **Agent Rules** | Subagent dispatch rules, agent-specific constraints, core injection requirements | `agent_rules` |

**Important**:
- Not every project will have content for every category. Only create sections that have real content.
- Preserve the original wording where possible. Don't over-summarize — these are instructions Claude needs to follow exactly.
- If content is already well-organized in the source file, keep that structure.
- Deduplicate: if the same rule appears in multiple files, include it once.

**Report to the user**: Show a summary table of what you found:
```
Category           | Source File(s)        | Items Found
-------------------|-----------------------|------------
Critical Constraints | CLAUDE.md, AGENTS.md | 5 rules
Build & Test       | CLAUDE.md             | 3 command groups
...
```

---

## Step 3: Generate .claude/rules/${CORE_FILE}

Create the file `.claude/rules/${CORE_FILE}` with this structure:

```markdown
# Core Instructions

These instructions apply to ALL contexts (main session + subagents).
Managed by SIMBA markers — run `simba markers audit` to check health.

This file lives in `.claude/rules/` so Claude Code auto-loads it every session.
Critical rules are wrapped in `SIMBA:core` markers so the guardian hook
re-injects them after context compaction.

---

<!-- BEGIN SIMBA:core -->
## Critical Constraints

[The most important rules go here — these survive compaction via guardian hook]
<!-- END SIMBA:core -->

---

<!-- BEGIN SIMBA:build_commands -->
## Build & Test Commands

[extracted build/test commands here]
<!-- END SIMBA:build_commands -->

[...additional sections as needed...]
```

### Marker Strategy

**`SIMBA:core`** — Wrap the most critical rules (constraints, must-never-violate rules) in this marker. The guardian hook extracts these and re-injects them on EVERY prompt, even after context compaction. Keep this section tight — every token here costs context on every message.

**Other SIMBA markers** (`SIMBA:build_commands`, `SIMBA:environment`, etc.) — These are discoverable and auditable via `simba markers`, but NOT re-injected by the guardian. They're reference content that Claude reads from the file when needed. Put the bulk of the instructions here.

### Rules for Content

- Each section gets its own SIMBA marker pair
- Use the section names from the Category table above
- Include `---` horizontal rules between sections for readability
- Only include sections that have actual content (skip empty categories)
- Content inside markers should be complete and standalone — a subagent reading only this file should have everything it needs
- Keep `SIMBA:core` lean: only rules that MUST survive compaction (aim for <500 tokens)

---

## Step 4: Present to User for Verification

**Do NOT write the file yet.** First, show the user the complete generated content.

For each section:
1. Show the section name and SIMBA marker
2. Show the content that will be written
3. Show which source file(s) it was extracted from
4. For `SIMBA:core`: explicitly note that this content will be re-injected on every prompt

Ask the user:
- "Does this look accurate? Should I add, remove, or modify anything?"
- "Is the `SIMBA:core` section the right set of critical rules? Everything in there costs context on every message."
- Wait for explicit approval before proceeding

If the user requests changes, apply them and show the updated version.

---

## Step 5: Write Files

Once the user approves:

1. **Create** `.claude/rules/` directory if it doesn't exist

2. **Write** `.claude/rules/${CORE_FILE}` with the approved content

3. **Update CLAUDE.md** — Add a reference block (if not already present):
   ```markdown
   <!-- BEGIN SIMBA:core_ref -->
   **Read `.claude/rules/${CORE_FILE}` for rules that apply to ALL contexts (main session + subagents).**

   When dispatching subagents, inject the contents of that file into the prompt.
   <!-- END SIMBA:core_ref -->
   ```
   Place this near the top of the file, after any title/heading but before other content.

4. **Update AGENTS.md** (if it exists) — Add a similar reference:
   ```markdown
   <!-- BEGIN SIMBA:core_ref -->
   **All agents must follow `.claude/rules/${CORE_FILE}` before executing.**

   When dispatching write-capable agents (especially implementer), read `.claude/rules/${CORE_FILE}` and inject its contents into the dispatch prompt.
   <!-- END SIMBA:core_ref -->
   ```

---

## Step 6: Verify

1. Run `simba markers audit` to confirm all markers are recognized
2. Run `simba markers list` to show the final state
3. Report the results to the user

**Done.** The user can now:
- Edit `.claude/rules/${CORE_FILE}` directly to refine rules
- Run `simba markers audit` anytime to check marker health
- Run `/simba-onboard` again to re-analyze if files change significantly
- Change the filename for future projects: `simba config set guardian.core_filename CORE.md`

### How It Works After Onboarding

```
Normal session:
  Claude Code auto-loads .claude/rules/${CORE_FILE}        (native)
  + Guardian re-injects SIMBA:core blocks on every prompt  (compaction-safe)

After compaction:
  Claude Code may lose .claude/rules/ context
  Guardian still injects SIMBA:core blocks                 (safety net)

Subagents:
  Must explicitly receive ${CORE_FILE} in their prompt
  CLAUDE.md and AGENTS.md reference the file for this purpose
```
