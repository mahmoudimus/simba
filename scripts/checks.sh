#!/bin/sh
# Single source of truth for the project's checks — called by BOTH the CI
# workflow (.github/workflows/ci.yml) and the git hooks (.githooks/*), so the two
# can never drift. Change a check here and CI + local stay in lockstep.
#
#   scripts/checks.sh lint   ruff (fast; the pre-commit guard)
#   scripts/checks.sh test   pytest
#   scripts/checks.sh all    lint then test (the full CI mirror; pre-push)
set -e
cd "$(dirname "$0")/.."

lint() {
    echo "→ ruff check src/ tests/"
    uv run ruff check src/ tests/
}

run_tests() {
    echo "→ pytest -q"
    uv run pytest -q
}

case "${1:-all}" in
    lint) lint ;;
    test) run_tests ;;
    all) lint && run_tests ;;
    *)
        echo "usage: scripts/checks.sh [lint|test|all]" >&2
        exit 2
        ;;
esac
