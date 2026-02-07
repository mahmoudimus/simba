from __future__ import annotations

from simba.sync.heuristics import extract_facts


class TestWorkingSolution:
    def test_working_solution_use_for(self):
        result = extract_facts("WORKING_SOLUTION", "use pytest for testing", "", "abc")
        assert result == [("pytest", "solves", "testing", "memory:abc")]

    def test_working_solution_fixes(self):
        result = extract_facts("WORKING_SOLUTION", "ruff fixes linting", "", "abc")
        assert result == [("ruff", "solves", "linting", "memory:abc")]

    def test_working_solution_no_match(self):
        result = extract_facts("WORKING_SOLUTION", "nothing special here", "", "abc")
        assert result == []


class TestGotcha:
    def test_gotcha_causes(self):
        result = extract_facts("GOTCHA", "circular imports causes deadlock", "", "g1")
        assert result == [("circular imports", "has_pitfall", "deadlock", "memory:g1")]

    def test_gotcha_avoid(self):
        result = extract_facts("GOTCHA", "avoid global state", "", "g2")
        assert result == [("global state", "has_pitfall", "warning", "memory:g2")]

    def test_gotcha_no_match(self):
        result = extract_facts("GOTCHA", "everything is fine", "", "g3")
        assert result == []


class TestPattern:
    def test_pattern_uses(self):
        result = extract_facts("PATTERN", "project uses pytest", "", "p1")
        assert result == [("project", "follows_pattern", "pytest", "memory:p1")]

    def test_pattern_convention(self):
        result = extract_facts("PATTERN", "naming convention: snake_case", "", "p2")
        assert result == [("naming", "follows_pattern", "snake_case", "memory:p2")]

    def test_pattern_no_match(self):
        result = extract_facts("PATTERN", "just some text", "", "p3")
        assert result == []


class TestDecision:
    def test_decision_chose(self):
        result = extract_facts("DECISION", "chose FastAPI over Flask", "", "d1")
        assert result == [("project", "chose", "FastAPI (Flask)", "memory:d1")]

    def test_decision_using_for(self):
        result = extract_facts("DECISION", "using pytest for testing", "", "d2")
        assert result == [("project", "chose", "pytest (testing)", "memory:d2")]

    def test_decision_no_match(self):
        result = extract_facts("DECISION", "no decision here", "", "d3")
        assert result == []


class TestFailure:
    def test_failure_doesnt_work(self):
        result = extract_facts("FAILURE", "pip doesn't work", "", "f1")
        assert result == [("pip", "fails_for", "unknown reason", "memory:f1")]

    def test_failure_fails_for(self):
        result = extract_facts("FAILURE", "mypy fails for generics", "", "f2")
        assert result == [("mypy", "fails_for", "generics", "memory:f2")]

    def test_failure_no_match(self):
        result = extract_facts("FAILURE", "all good", "", "f3")
        assert result == []


class TestPreference:
    def test_preference_prefer(self):
        result = extract_facts("PREFERENCE", "prefer ruff", "", "pr1")
        assert result == [("user", "prefers", "ruff", "memory:pr1")]

    def test_preference_over(self):
        result = extract_facts("PREFERENCE", "prefer ruff over flake8", "", "pr2")
        assert result == [("user", "prefers", "ruff", "memory:pr2")]

    def test_preference_no_match(self):
        result = extract_facts("PREFERENCE", "no opinion", "", "pr3")
        assert result == []


class TestGeneral:
    def test_unknown_type(self):
        result = extract_facts("UNKNOWN_TYPE", "use pytest for testing", "", "x1")
        assert result == []

    def test_proof_format(self):
        result = extract_facts(
            "WORKING_SOLUTION", "use pytest for testing", "", "my-id-123"
        )
        assert len(result) == 1
        assert result[0][3] == "memory:my-id-123"

    def test_subject_truncated(self):
        long_subject = "a" * 200
        content = f"{long_subject} fixes something"
        result = extract_facts("WORKING_SOLUTION", content, "", "trunc")
        assert len(result) == 1
        assert len(result[0][0]) == 80
