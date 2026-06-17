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
    # --timeout guards against a hung test: fail any single test after 120s
    # instead of letting it stall the whole run (a transient pytest hang on PR
    # #80 burned ~2h of CI). The thread method also catches hangs inside native
    # code (llama_cpp / lancedb), and prints a traceback of where it stuck. The
    # job-level timeout-minutes in ci.yml is the outer guard for a whole-runner
    # stall the in-process timeout can't catch.
    echo "→ pytest -q (per-test timeout 120s)"
    uv run pytest -q --timeout=120 --timeout-method=thread
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
