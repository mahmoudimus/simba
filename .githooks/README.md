# Git hooks

Version-controlled hooks that run the **same checks as CI** so local and CI can't
drift. Both hooks delegate to [`../scripts/checks.sh`](../scripts/checks.sh) — the
single source of truth, also called by `.github/workflows/ci.yml`.

| Hook | Runs | Why |
|------|------|-----|
| `pre-commit` | `checks.sh lint` (ruff, ~0.05s) | instant — catches the lint CI gates on |
| `pre-push`   | `checks.sh all` (ruff + pytest, ~70s) | full CI mirror before code leaves the machine |

## Enable (once per clone)

```sh
git config core.hooksPath .githooks
```

Bypass a single run with `git commit --no-verify` / `git push --no-verify`.

To change what CI **and** the hooks check, edit `scripts/checks.sh` only.
