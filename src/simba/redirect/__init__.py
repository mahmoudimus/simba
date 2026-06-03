"""Tool-call redirect: steer bare commands to more efficient tooling.

A PreToolUse check parses a Bash command, matches each invoked program against
project redirect rules (e.g. cargo -> soldr cargo, python -> uv run python), and
either denies with the corrected command (the model retries) or, when policy
allows, rewrites it. Rules come from both a repo `.simba/redirects.toml` and the
CLI-managed store.
"""
