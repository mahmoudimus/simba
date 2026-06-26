# Candidate-Unit IR Eval

This eval captures the generic `action(subject, object, verb)` prompt that
worked on the nine fail18 rows where the earlier semantic-graph path missed.

The saved fixture is:

```text
src/simba/eval/datasets/fail9_candidate_unit_claude.json
```

It records:

- the generic compiler contract
- `claude -p` as the provider path
- one output per fail18 row
- candidate units marked `included`, `excluded`, or `merged`
- answer-variable and individuation-policy choices
- facts, query, rationale, gold, and match flag

The deterministic scorer does not call Claude:

```bash
uv run python -m simba.eval.candidate_unit_ir --json
uv run pytest tests/eval/test_candidate_unit_ir.py -q
```

The important behavior is not just the `9/9` score. The useful target is the
candidate-unit audit before aggregation:

```text
question + evidence
  -> answer_variable
  -> individuation_policy
  -> candidate_units
  -> recursive action/type/coreference facts
  -> aggregate only included answer units
```

Future deterministic Datalog/Clingo work should treat this fixture as the
behavioral spec: reproduce the candidate-unit decisions first, then the score.
