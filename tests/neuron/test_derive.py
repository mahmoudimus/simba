"""Sub-phase DERIVE: Datalog materialization (Task B.3)."""

from __future__ import annotations

import pathlib
from unittest.mock import patch

import simba.db


def test_derive_disabled_returns_empty() -> None:
    from simba.neuron.config import NeuronConfig
    from simba.neuron.derive import run_derive

    result = run_derive("/proj", cfg=NeuronConfig(derive_enabled=False))
    assert result.candidates == []
    assert result.errors == 0


def test_derive_no_souffle_returns_empty() -> None:
    from simba.neuron.config import NeuronConfig
    from simba.neuron.derive import run_derive

    result = run_derive("/proj", cfg=NeuronConfig(souffle_cmd=""))
    assert result.candidates == []


def test_derive_with_fake_souffle(tmp_path, monkeypatch) -> None:
    """Monkeypatch subprocess.run to return a fake Soufflé output."""
    import simba.kg.store

    db_path = tmp_path / ".simba" / "simba.db"
    monkeypatch.setattr(simba.db, "get_db_path", lambda cwd=None: db_path)
    # Insert two edges that should trigger transitivity rule
    simba.kg.store.kg_add("A", "uses", "B", "test", project_path="/proj")
    simba.kg.store.kg_add("B", "uses", "C", "test", project_path="/proj")

    from simba.neuron.config import NeuronConfig
    from simba.neuron.derive import run_derive

    fake_output = "A\ttransitively_uses\tC\t1\t2\n"
    mock_result = type(
        "R", (), {"returncode": 0, "stdout": fake_output, "stderr": ""}
    )()

    with patch("subprocess.run", return_value=mock_result):
        result = run_derive(
            "/proj", cfg=NeuronConfig(souffle_cmd="souffle", derive_enabled=True)
        )

    assert len(result.candidates) == 1
    assert result.candidates[0].subject == "A"
    assert result.candidates[0].predicate == "transitively_uses"
    assert result.candidates[0].object == "C"


def test_derive_subprocess_error_is_fail_open(monkeypatch) -> None:
    from simba.neuron.config import NeuronConfig
    from simba.neuron.derive import run_derive

    monkeypatch.setattr(
        simba.db,
        "get_db_path",
        lambda cwd=None: pathlib.Path("/nonexistent/simba.db"),
    )
    with patch("subprocess.run", side_effect=FileNotFoundError("no souffle")):
        result = run_derive("/proj", cfg=NeuronConfig(souffle_cmd="souffle"))
    assert result.errors >= 1
    assert result.candidates == []
