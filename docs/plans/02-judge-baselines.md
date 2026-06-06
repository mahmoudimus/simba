# Workstream B — Local Judge + Honest Baselines

## Task B1 — `judge` configurable section (separate judge from answerer)

**Goal.** Eliminate the self-grading loop where the same model answers and grades. Introduce a `judge` config section that is structurally identical to `llm` but independently configured. `score_case` and `run_qa` receive a dedicated judge client built from the `judge` section while the answerer uses the `llm` section. Default the judge to a different provider/model than the answerer default (`llm` defaults to `claude-cli/haiku`; `judge` defaults to `llm-cli/deepseek-r1` — a local option — with a documented caveat that it is still not GPT-4-level calibration). Both clients flow through the existing `get_client(cfg)` path unchanged.

**Files.**

- Create: `src/simba/llm/judge_config.py`
- Modify: `src/simba/eval/benchmarks/judge.py`
- Modify: `scripts/run_qa.py`
- Create: `tests/llm/test_judge_config.py`
- Modify: `tests/eval/benchmarks/test_judge.py`

**Config** (`simba config get/set judge.<key>`).

```
judge.provider       : str   = "llm-cli"        # backend for the grader; default differs from llm.provider
judge.model          : str   = "deepseek-r1"    # model name as the judge CLI expects it
judge.model_path     : str   = ""               # local GGUF/HF repo path; falls back to model when empty
judge.thinking       : str   = ""               # reasoning-effort hint (llm-cli -o reasoning_effort)
judge.base_url       : str   = ""               # Anthropic-compatible proxy endpoint
judge.api_key_env    : str   = "ANTHROPIC_API_KEY"
judge.extra_args     : str   = ""               # shell-split extra flags appended to CLI argv
judge.timeout_seconds: float = 90.0             # grading may need more time than answering
judge.max_tokens     : int   = 512              # judge needs only short verdicts
```

All fields mirror `LlmConfig` exactly so `get_client(cfg)` accepts either without modification.

**Signatures.**

```python
# src/simba/llm/judge_config.py

@simba.config.configurable("judge")
@dataclasses.dataclass
class JudgeConfig:
    provider: str = "llm-cli"
    model: str = "deepseek-r1"
    model_path: str = ""
    thinking: str = ""
    base_url: str = ""
    api_key_env: str = "ANTHROPIC_API_KEY"
    extra_args: str = ""
    timeout_seconds: float = 90.0
    max_tokens: int = 512


def load_judge_config(**overrides: typing.Any) -> JudgeConfig:
    """Load judge config from TOML files then apply keyword overrides."""
    ...


def get_judge_client(cfg: typing.Any | None = None) -> simba.llm.client.LlmClient:
    """Return an LlmClient built from the judge section (or loaded default)."""
    ...
```

```python
# src/simba/eval/benchmarks/judge.py  — modified signatures

def score_case(
    case: EvalCase,
    retriever: Retriever,
    id2content: dict[str, str],
    answerer: typing.Any,          # was: llm — renamed for clarity
    *,
    judge: typing.Any | None = None,  # NEW: separate judge client; falls back to answerer when None
    k: int = 10,
    cache: typing.Any = None,
    judge_model: str = "",
) -> bool | None: ...

def run_qa(
    datasets: list[Dataset],
    *,
    embed_doc: EmbedFn,
    embed_query: EmbedFn,
    cfg: typing.Any,
    llm: typing.Any,               # answerer — unchanged kwarg name for back-compat
    judge: typing.Any | None = None,  # NEW: if None, falls back to llm (old behaviour)
    k: int = 10,
    answerable_only: bool = True,
    cache: typing.Any = None,
    judge_model: str = "",
) -> dict[str, typing.Any]: ...
```

**Implementation steps.**

1. Create `/Users/mahmoud/src/ai/simba/src/simba/llm/judge_config.py`. Copy the field structure from `LlmConfig` verbatim, change the section string to `"judge"`, adjust defaults (`provider="llm-cli"`, `model="deepseek-r1"`, `timeout_seconds=90.0`, `max_tokens=512`). Implement `load_judge_config(**overrides)` mirroring `load_config` in `llm/config.py`. Implement `get_judge_client(cfg=None)` which imports `simba.config`, imports `simba.llm.judge_config` (side-effect: registers the section), calls `simba.config.load("judge")` when `cfg is None`, and passes the result to `LlmClient`.

2. Add a module-level docstring caveat: "The judge is still a local model and may differ from GPT-4 calibration. Scores are more trustworthy than answerer==judge but not directly comparable to published frontier-judge numbers."

3. In `/Users/mahmoud/src/ai/simba/src/simba/eval/benchmarks/judge.py`:
   - Rename the `llm` parameter of `score_case` to `answerer` in the function body. Keep the public kwarg name `llm` as an alias with a deprecation comment, or rename cleanly since this is internal tooling (clean rename is simpler — update the one call site in `run_qa`).
   - Add `judge: typing.Any | None = None` parameter. Inside the function body: `_judge = judge if judge is not None else answerer`. Use `answerer` for `complete(build_answer_prompt(...))` and `_judge` for `complete_json(build_judge_prompt(...))`.
   - Update `judge_model` default: when `judge_model == ""` and `judge is not None`, auto-derive `judge_model = getattr(judge._cfg, "model", "")` for cache keying.
   - Add `judge: typing.Any | None = None` to `run_qa`. Pass it through to `score_case`.

4. In `/Users/mahmoud/src/ai/simba/scripts/run_qa.py`:
   - Import `simba.llm.judge_config as jcfg`.
   - After building `llm = llm_client.get_client()`, build `judge_client = jcfg.get_judge_client()`.
   - Print judge identity alongside answerer: `print(f"judge: provider={...} model={...} available={...}")`.
   - Pass `judge=judge_client` to `judge.run_qa(...)`.
   - Pass `judge_model=judge_client._cfg.model` for cache keying.

**Tests** (`tests/llm/test_judge_config.py`).

```python
# tests/llm/test_judge_config.py  — all RED before implementation

def test_judge_section_registered_in_config_registry() -> None:
    import simba.config
    import simba.llm.judge_config  # side-effect: registers section
    assert "judge" in simba.config.list_sections()

def test_judge_config_defaults_differ_from_llm_defaults() -> None:
    import simba.llm.config as lc
    import simba.llm.judge_config as jc
    import simba.config, pathlib, tempfile
    with tempfile.TemporaryDirectory() as td:
        cfg_j = simba.config.load("judge", root=pathlib.Path(td))
        cfg_l = simba.config.load("llm",   root=pathlib.Path(td))
    assert cfg_j.provider != cfg_l.provider or cfg_j.model != cfg_l.model
    assert cfg_j.timeout_seconds == 90.0
    assert cfg_j.max_tokens == 512

def test_get_judge_client_returns_llm_client_instance() -> None:
    import simba.llm.client as lc
    import simba.llm.judge_config as jc
    client = jc.get_judge_client()
    assert isinstance(client, lc.LlmClient)

def test_load_judge_config_override_takes_precedence() -> None:
    import simba.llm.judge_config as jc
    cfg = jc.load_judge_config(model="my-judge-model")
    assert cfg.model == "my-judge-model"

def test_judge_config_toml_round_trip(tmp_path) -> None:
    import simba.config, simba.llm.judge_config
    simba.config.set_value("judge", "model", "phi-3-mini", scope="local",
                           root=tmp_path)
    loaded = simba.config.load("judge", root=tmp_path)
    assert loaded.model == "phi-3-mini"
```

Additions to `tests/eval/benchmarks/test_judge.py`:

```python
def test_score_case_uses_judge_for_grading_not_answerer() -> None:
    """answerer.complete_json should NOT be called when a separate judge is given."""
    case = EvalCase(id="q1", query="When?", relevant_ids=["c1"], answer="7 May")
    answerer = FakeLlm(answer="7 May", verdict={"correct": True})
    judge_llm = FakeLlm(answer="ignored", verdict={"correct": True})
    correct = judge.score_case(
        case, lambda q: ["c1"], {"c1": "x"}, answerer, judge=judge_llm, k=5
    )
    assert correct is True
    # answerer called once (generate answer), judge called once (grade)
    assert len(answerer.prompts) == 1      # build_answer_prompt only
    assert len(judge_llm.prompts) == 1    # build_judge_prompt only

def test_score_case_judge_none_falls_back_to_answerer() -> None:
    """Legacy path: no judge kwarg -> answerer grades its own answer."""
    case = EvalCase(id="q1", query="When?", relevant_ids=["c1"], answer="7 May")
    llm = FakeLlm(answer="7 May", verdict={"correct": False})
    correct = judge.score_case(case, lambda q: ["c1"], {"c1": "x"}, llm, k=5)
    assert correct is False
    assert len(llm.prompts) == 2  # answer + grade

def test_run_qa_passes_judge_to_score_case() -> None:
    """run_qa with judge kwarg should route grading to the judge client."""
    from simba.eval.dataset import Dataset, EvalCase, Memory
    import simba.eval.benchmarks.judge as jmod
    called_with: list[dict] = []
    original_score = jmod.score_case
    def patched_score(case, retriever, id2content, answerer, *, judge=None, **kw):
        called_with.append({"answerer": answerer, "judge": judge})
        return True
    import unittest.mock
    with unittest.mock.patch.object(jmod, "score_case", patched_score):
        ds = Dataset(
            name="t", corpus=[Memory(id="c1", content="x")],
            cases=[EvalCase(id="q1", query="q", relevant_ids=["c1"], answer="a")]
        )
        answerer = FakeLlm(answer="a")
        judge_llm = FakeLlm(answer="x")
        import simba.memory.config as mc
        cfg = mc.MemoryConfig(llm_rerank_enabled=False, scoring_enabled=False,
                               expansion_enabled=False)
        import simba.eval.run as run_mod
        embed = lambda t: [0.0] * 384
        jmod.run_qa([ds], embed_doc=embed, embed_query=embed, cfg=cfg,
                    llm=answerer, judge=judge_llm)
    assert called_with[0]["answerer"] is answerer
    assert called_with[0]["judge"] is judge_llm
```

**Acceptance.**

- `simba config get judge.provider` prints `llm-cli`.
- `simba config set judge.model phi-3 && simba config get judge.model` prints `phi-3`.
- `uv run pytest tests/llm/test_judge_config.py tests/eval/benchmarks/test_judge.py -x` is green.
- Answerer prompt count and judge prompt count in `test_score_case_uses_judge_for_grading_not_answerer` both equal 1.

**Verify.**

```bash
uv run pytest tests/llm/test_judge_config.py tests/eval/benchmarks/test_judge.py -x -q
simba config get judge.provider
simba config get judge.timeout_seconds
```

**Reuse.**

- `src/simba/llm/config.py:LlmConfig` — field list to mirror exactly.
- `src/simba/llm/client.py:get_client` (line 163) — call pattern for `get_judge_client`.
- `src/simba/config.py:configurable` (line 28) — decorator to register section.

---

## Task B2 — Full LoCoMo QA baseline on test split

**Goal.** Record a reproducible QA-accuracy number on the full LoCoMo answerable question set (test split), using the B1 split answerer/judge, with the judge-verdict cache enabled. The result is written to `.simba/eval/baselines/locomo_qa.json` (append-only; each run appends a timestamped entry). The script `scripts/run_qa.py` gains `--split test` and `--out` flags. A `--baseline` flag writes the result to the canonical path.

**Files.**

- Modify: `scripts/run_qa.py`
- Create: `src/simba/eval/benchmarks/baseline_store.py`
- Create: `tests/eval/benchmarks/test_baseline_store.py`

**Config.** No new config section. `judge.*` from B1 controls the grader. Existing `llm.*` controls the answerer. The baseline output path is derived from the project root (`.simba/eval/baselines/`) — not a hidden constant; it is a field on a new `@configurable` section:

```
eval.baseline_dir: str = ".simba/eval/baselines"  # relative to project root; used by baseline_store
```

Add this to a new or existing eval config. Since `src/simba/eval/config.py` already exists, read it first and add `baseline_dir` there.

```
# Additions to src/simba/eval/config.py
eval.baseline_dir: str = ".simba/eval/baselines"
```

**Signatures.**

```python
# src/simba/eval/benchmarks/baseline_store.py

def append_baseline(
    name: str,          # e.g. "locomo_qa", "locomo_recall", "longmemeval_s_qa"
    report: dict,       # the full report dict from run_qa / run_recall
    *,
    root: pathlib.Path | None = None,   # project root; None -> auto-detect
    metadata: dict | None = None,       # e.g. {"answerer": "haiku", "judge": "deepseek-r1", "k": 10}
) -> pathlib.Path:
    """Append a timestamped baseline entry to .simba/eval/baselines/<name>.jsonl.

    Append-only: never overwrites. Returns the path written to.
    """

def load_baselines(
    name: str,
    *,
    root: pathlib.Path | None = None,
) -> list[dict]:
    """Return all baseline entries for <name> in chronological order."""
```

Script additions to `scripts/run_qa.py`:

```python
# New CLI args parsed from sys.argv after existing positional args:
# --split dev|test|all  (default "all" to match current behaviour)
# --out PATH            write JSON report to this path
# --baseline            also append to .simba/eval/baselines/<bench>_qa.jsonl
# --cache PATH          judge-verdict cache db path (default .simba/eval/judge_cache.db)
```

**Implementation steps.**

1. Read `/Users/mahmoud/src/ai/simba/src/simba/eval/config.py`. Add `baseline_dir: str = ".simba/eval/baselines"` to the existing `@configurable` dataclass (or create one if absent). Register with `@simba.config.configurable("eval")`.

2. Create `/Users/mahmoud/src/ai/simba/src/simba/eval/benchmarks/baseline_store.py`. Use `pathlib` exclusively (no `os.path`). In `append_baseline`: resolve `root` via `simba.config._find_root(root)`; resolve `baseline_dir` from `simba.config.load("eval", root=root).baseline_dir`; construct path as `root / baseline_dir / f"{name}.jsonl"`; `mkdir(parents=True, exist_ok=True)`; build entry `{"ts": datetime.utcnow().isoformat(), "report": report, "metadata": metadata or {}}`; open in `"a"` mode and write one JSON line (append-only). In `load_baselines`: read lines, parse each as JSON, return list.

3. Modify `scripts/run_qa.py`. Parse new flags with a minimal loop (no argparse dependency to keep it a dev script — use `sys.argv` scanning with a helper). After existing report print: if `--out` given, write `json.dumps(report, indent=2)` to that path. If `--baseline` given, call `baseline_store.append_baseline(bench + "_qa", report, metadata={"answerer": llm._cfg.model, "judge": judge_client._cfg.model, "k": k, "n_arg": n_arg})` and print the written path. Wire `--cache` to a `JudgeCache` instance passed to `run_qa`.

4. Wire B1's `judge_client` — already done in B1 step 4.

**Tests** (`tests/eval/benchmarks/test_baseline_store.py`).

```python
def test_append_creates_file_on_first_write(tmp_path) -> None:
    from simba.eval.benchmarks.baseline_store import append_baseline, load_baselines
    p = append_baseline("locomo_qa", {"overall": {"accuracy": 0.72}},
                        root=tmp_path, metadata={"k": 10})
    assert p.exists()
    entries = load_baselines("locomo_qa", root=tmp_path)
    assert len(entries) == 1
    assert entries[0]["report"]["overall"]["accuracy"] == 0.72
    assert entries[0]["metadata"]["k"] == 10
    assert "ts" in entries[0]

def test_append_is_append_only(tmp_path) -> None:
    from simba.eval.benchmarks.baseline_store import append_baseline, load_baselines
    append_baseline("locomo_qa", {"overall": {"accuracy": 0.70}}, root=tmp_path)
    append_baseline("locomo_qa", {"overall": {"accuracy": 0.73}}, root=tmp_path)
    entries = load_baselines("locomo_qa", root=tmp_path)
    assert len(entries) == 2
    assert entries[0]["report"]["overall"]["accuracy"] == 0.70
    assert entries[1]["report"]["overall"]["accuracy"] == 0.73

def test_different_names_go_to_different_files(tmp_path) -> None:
    from simba.eval.benchmarks.baseline_store import append_baseline
    p1 = append_baseline("locomo_qa", {}, root=tmp_path)
    p2 = append_baseline("locomo_recall", {}, root=tmp_path)
    assert p1 != p2
    assert p1.name == "locomo_qa.jsonl"
    assert p2.name == "locomo_recall.jsonl"

def test_baseline_dir_config_respected(tmp_path) -> None:
    import simba.config, simba.eval.config
    simba.config.set_value("eval", "baseline_dir", ".custom/baselines",
                           scope="local", root=tmp_path)
    from simba.eval.benchmarks.baseline_store import append_baseline
    p = append_baseline("test", {}, root=tmp_path)
    assert ".custom/baselines" in str(p)
```

**Acceptance.**

- `uv run pytest tests/eval/benchmarks/test_baseline_store.py -x -q` is green.
- Manual: `uv run python scripts/run_qa.py locomo /tmp/locomo10.json 50 10 --baseline` produces a line in `.simba/eval/baselines/locomo_qa.jsonl` with `ts`, `report`, and `metadata` keys.
- Second run appends a second line; file has exactly 2 lines.

**Verify.**

```bash
uv run pytest tests/eval/benchmarks/test_baseline_store.py -x -q
uv run python scripts/run_qa.py locomo /tmp/locomo10.json all 10 --baseline
wc -l .simba/eval/baselines/locomo_qa.jsonl
python3 -c "import json,pathlib; [print(json.loads(l)['metadata']) for l in pathlib.Path('.simba/eval/baselines/locomo_qa.jsonl').read_text().splitlines()]"
```

**Reuse.**

- `src/simba/eval/benchmarks/judge_cache.py:JudgeCache` — pass instance to `run_qa` in script.
- `src/simba/config.py:_find_root` — use for root resolution in `baseline_store`.
- `src/simba/eval/config.py` — add `baseline_dir` field here.

---

## Task B3 — Full `longmemeval_s` baseline (real haystack, abstention scoring)

**Goal.** Run recall@k + QA accuracy + abstention accuracy on `longmemeval_s` (the full haystack with distractor sessions, not the oracle). Abstention questions (IDs ending `_abs`) test refusal: the model must decline to answer. Correct abstention = the model's answer contains a refusal phrase. Score separately from non-abstention QA, since different judge logic applies. Record results in `.simba/eval/baselines/longmemeval_s_recall.jsonl` and `longmemeval_s_qa.jsonl`.

**Files.**

- Modify: `src/simba/eval/benchmarks/longmemeval.py` — add `is_abstention(qid)` helper.
- Modify: `src/simba/eval/benchmarks/judge.py` — add `score_abstention`, update `run_qa` to handle abstention cases.
- Modify: `scripts/run_longmemeval.py` — add `--full` / `--abstention` / `--baseline` flags.
- Modify: `tests/eval/benchmarks/test_longmemeval.py` — new tests for abstention scoring.
- Modify: `tests/eval/benchmarks/test_judge.py` — new tests for `score_abstention` and `run_qa` abstention path.

**Abstention scoring definition.**

A predicted answer is a correct abstention if it matches any of a fixed list of refusal phrases (case-insensitive substring match). The phrase list is a `@configurable` field so it can be extended without code changes.

```
eval.abstention_phrases: str = "don't know,do not know,no information,cannot find,not in my memories,i have no record"
# comma-separated; stored as a single string; split on load
```

Add this field to the existing `eval` config section (same dataclass as `baseline_dir`).

**Signatures.**

```python
# src/simba/eval/benchmarks/longmemeval.py — additions

def is_abstention(question_id: str) -> bool:
    """Return True when the question_id ends with '_abs'."""
    ...
```

```python
# src/simba/eval/benchmarks/judge.py — additions

def build_abstention_judge_prompt(question: str, predicted: str) -> str:
    """Prompt the judge to decide if the predicted answer is a proper refusal."""
    ...

def score_abstention(
    case: EvalCase,
    retriever: Retriever,
    id2content: dict[str, str],
    answerer: typing.Any,
    *,
    judge: typing.Any | None = None,
    k: int = 10,
    abstention_phrases: list[str] | None = None,
) -> bool | None:
    """Retrieve, generate, then check for refusal.

    Strategy: heuristic-first (phrase match against abstention_phrases),
    then judge-LLM confirmation only when the heuristic is ambiguous.
    Returns True (correctly abstained), False (wrongly answered), or None (unscored).
    """
    ...

def aggregate_with_abstention(
    rows: list[tuple[str, bool]],
    abstention_rows: list[tuple[str, bool]],
) -> dict[str, typing.Any]:
    """Extend aggregate() output with abstention_accuracy block."""
    ...

# run_qa updated signature:
def run_qa(
    datasets: list[Dataset],
    *,
    embed_doc: EmbedFn,
    embed_query: EmbedFn,
    cfg: typing.Any,
    llm: typing.Any,
    judge: typing.Any | None = None,
    k: int = 10,
    answerable_only: bool = True,
    include_abstention: bool = False,      # NEW
    abstention_phrases: list[str] | None = None,  # NEW; None -> load from eval config
    cache: typing.Any = None,
    judge_model: str = "",
) -> dict[str, typing.Any]: ...
```

**Implementation steps.**

1. In `longmemeval.py`, add `is_abstention(question_id: str) -> bool` — one-liner: `return str(question_id).endswith("_abs")`.

2. In `judge.py`, add `build_abstention_judge_prompt(question, predicted)`:
   ```
   "You are judging whether a predicted answer correctly declines to answer a question "
   "that cannot be answered from the available memories. A correct refusal says the "
   "information is unavailable. Reply JSON only: {\"abstained\": true} or {\"abstained\": false}.\n\n"
   f"Question: {question}\nPredicted answer: {predicted}\nJSON:"
   ```

3. Add `score_abstention`: call `retriever(case.query)[:k]`, build contexts, call `answerer.complete(build_answer_prompt(...))`. Heuristic: lower-case the predicted answer, check if any phrase from `abstention_phrases` is a substring — if yes, return `True` without calling the judge. If no phrase matches, call `judge.complete_json(build_abstention_judge_prompt(...))` and extract `verdict.get("abstained")`. Return `None` if predicted is empty or judge verdict is unparseable.

4. Add `aggregate_with_abstention`: call existing `aggregate(rows)`, add `"abstention": {"n": len(abstention_rows), "accuracy": _acc([c for _, c in abstention_rows])}`.

5. Modify `run_qa`: resolve `abstention_phrases` when `None` by loading from config (`simba.config.load("eval").abstention_phrases.split(",")` after stripping whitespace). When `include_abstention=True`, iterate dataset cases: if `is_abstention(case.id)`, call `score_abstention` and append to `abstention_rows`; otherwise follow existing path. At the end, call `aggregate_with_abstention(rows, abstention_rows)` and include result. When `include_abstention=False`, call existing `aggregate(rows)` as before and add `abstention_rows=[]` stub with `"abstention": {"n": 0, "accuracy": 0.0}` so the report shape is always consistent.

6. In `src/simba/eval/config.py`, add to the existing `@configurable("eval")` dataclass:
   ```python
   baseline_dir: str = ".simba/eval/baselines"
   abstention_phrases: str = (
       "don't know,do not know,no information,cannot find,"
       "not in my memories,i have no record"
   )
   ```

7. In `scripts/run_longmemeval.py`, parse `--full` (load `longmemeval_s.json` not oracle), `--abstention` (pass `include_abstention=True` to loader and `run_qa`), `--baseline` (call `baseline_store.append_baseline`), `--qa` (also run QA after recall). When `--qa` given, import `judge.run_qa` and run with the `judge_client` from B1.

**Tests.**

New tests in `tests/eval/benchmarks/test_longmemeval.py`:

```python
def test_is_abstention_detects_abs_suffix() -> None:
    from simba.eval.benchmarks.longmemeval import is_abstention
    assert is_abstention("q1_abs") is True
    assert is_abstention("q1") is False
    assert is_abstention("abs_q1") is False   # only suffix counts
```

New tests in `tests/eval/benchmarks/test_judge.py`:

```python
def test_score_abstention_heuristic_match_returns_true_without_judge() -> None:
    case = EvalCase(id="q1_abs", query="When did I buy a boat?",
                    relevant_ids=["c1"], answer="no information available")
    answerer = FakeLlm(answer="I don't know, no information available.")
    judge_llm = FakeLlm(answer="x", verdict={"abstained": False})  # should NOT be called
    result = judge.score_abstention(
        case, lambda q: ["c1"], {"c1": "x"}, answerer,
        judge=judge_llm, k=5,
        abstention_phrases=["don't know", "no information"],
    )
    assert result is True
    assert len(judge_llm.prompts) == 0   # heuristic short-circuited

def test_score_abstention_no_phrase_match_calls_judge() -> None:
    case = EvalCase(id="q1_abs", query="Q?", relevant_ids=["c1"], answer="n/a")
    answerer = FakeLlm(answer="The answer is 42.")
    judge_llm = FakeLlm(answer="x", verdict={"abstained": False})
    result = judge.score_abstention(
        case, lambda q: ["c1"], {"c1": "x"}, answerer,
        judge=judge_llm, k=5, abstention_phrases=["don't know"],
    )
    assert result is False
    assert len(judge_llm.prompts) == 1

def test_score_abstention_returns_none_on_empty_answer() -> None:
    case = EvalCase(id="q1_abs", query="Q?", relevant_ids=["c1"], answer="n/a")
    answerer = FakeLlm(answer="   ")
    result = judge.score_abstention(
        case, lambda q: ["c1"], {"c1": "x"}, answerer, k=5,
        abstention_phrases=["don't know"],
    )
    assert result is None

def test_aggregate_with_abstention_includes_abstention_block() -> None:
    rows = [("single-hop", True), ("multi-hop", False)]
    abs_rows = [("temporal", True), ("temporal", False)]
    rep = judge.aggregate_with_abstention(rows, abs_rows)
    assert rep["n_graded"] == 2
    assert rep["abstention"]["n"] == 2
    assert rep["abstention"]["accuracy"] == pytest.approx(0.5)

def test_run_qa_abstention_cases_scored_separately() -> None:
    """run_qa with include_abstention=True scores _abs cases via score_abstention."""
    import simba.eval.benchmarks.judge as jmod
    import unittest.mock
    normal_case = EvalCase(id="q1", query="q", relevant_ids=["c1"], answer="a")
    abs_case    = EvalCase(id="q2_abs", query="q_abs", relevant_ids=["c1"], answer="n/a")
    from simba.eval.dataset import Dataset, Memory
    ds = Dataset(name="t", corpus=[Memory(id="c1", content="x")],
                 cases=[normal_case, abs_case])
    answerer = FakeLlm(answer="a")
    import simba.memory.config as mc
    cfg = mc.MemoryConfig(llm_rerank_enabled=False, scoring_enabled=False,
                           expansion_enabled=False)
    embed = lambda t: [0.0] * 384
    with unittest.mock.patch.object(jmod, "score_case", return_value=True) as m_sc, \
         unittest.mock.patch.object(jmod, "score_abstention", return_value=True) as m_sa:
        jmod.run_qa([ds], embed_doc=embed, embed_query=embed, cfg=cfg,
                    llm=answerer, include_abstention=True,
                    abstention_phrases=["don't know"])
    assert m_sc.call_count == 1   # only normal case
    assert m_sa.call_count == 1   # only abs case
```

**Acceptance.**

- `uv run pytest tests/eval/benchmarks/test_judge.py tests/eval/benchmarks/test_longmemeval.py -x -q` is green.
- `simba config get eval.abstention_phrases` prints the default phrase list.
- `simba config set eval.abstention_phrases "i don't know,no data"` persists and is loaded by `score_abstention` when `abstention_phrases=None`.
- Report dict from `run_qa(..., include_abstention=True)` always has `report["abstention"]["n"]` and `report["abstention"]["accuracy"]` keys.

**Verify.**

```bash
uv run pytest tests/eval/benchmarks/test_judge.py tests/eval/benchmarks/test_longmemeval.py -x -q
simba config get eval.abstention_phrases
uv run python scripts/run_longmemeval.py /tmp/longmemeval_s.json --abstention --baseline
```

**Reuse.**

- `src/simba/eval/benchmarks/judge.py:aggregate` (line 83) — call from `aggregate_with_abstention`.
- `src/simba/eval/benchmarks/judge.py:build_answer_prompt` (line 28) — used inside `score_abstention`.
- `src/simba/eval/benchmarks/longmemeval.py:load_longmemeval` (line 84) — `include_abstention` param already exists.

---

## Task B4 — Per-query latency (p50/p95 in CaseResult and run_recall/run_qa reports)

**Goal.** Time each retriever call individually. Store `latency_ms: float` in `CaseResult`. Aggregate `p50_ms` and `p95_ms` in `EvalReport.aggregate` and in `run_recall` / `run_qa` report dicts. No new dependencies — use `time.perf_counter`. No config section needed (latency is always measured; the overhead is negligible). `to_dict()` on `CaseResult` and `EvalReport` must include latency so JSON output is backwards-compatible (new keys added, no keys removed).

**Files.**

- Modify: `src/simba/eval/runner.py` — `CaseResult`, `run_eval`.
- Modify: `src/simba/eval/benchmarks/run.py` — `run_recall` aggregation.
- Modify: `src/simba/eval/benchmarks/judge.py` — `run_qa` aggregation.
- Modify: `tests/eval/test_runner.py` — new latency assertions.
- Modify: `tests/eval/benchmarks/test_judge.py` — latency in `run_qa` report.

**Signatures.**

```python
# src/simba/eval/runner.py

@dataclasses.dataclass
class CaseResult:
    case_id: str
    query: str
    ranked: list[str]
    metrics: dict[str, float]
    latency_ms: float = 0.0      # NEW: wall-clock ms for the retriever call

    def to_dict(self) -> dict[str, typing.Any]:
        # adds "latency_ms" to existing keys — no removals
        ...

@dataclasses.dataclass
class EvalReport:
    dataset_name: str
    n_cases: int
    ks: tuple[int, ...]
    aggregate: dict[str, float]   # gains p50_ms, p95_ms keys
    per_case: list[CaseResult]
    ...

def run_eval(
    dataset: Dataset,
    retriever: Retriever,
    ks: tuple[int, ...] = _DEFAULT_KS,
    *,
    keep_top: int = 20,
    split: str | None = None,
    test_ratio: float = 0.5,
) -> EvalReport: ...    # signature unchanged; latency computed internally
```

```python
# src/simba/eval/benchmarks/run.py

def run_recall(
    datasets: list[Dataset],
    *,
    embed_doc: EmbedFn,
    embed_query: EmbedFn,
    cfg: typing.Any,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    llm_client: typing.Any = None,
) -> dict[str, typing.Any]:
    # report gains top-level "latency": {"p50_ms": ..., "p95_ms": ..., "n": ...}
    ...
```

```python
# src/simba/eval/benchmarks/judge.py

def run_qa(...) -> dict[str, typing.Any]:
    # report gains top-level "latency": {"p50_ms": ..., "p95_ms": ..., "n": ...}
    # latency measured per score_case call (retriever only, not LLM calls)
    ...
```

**Percentile helper** (private, in `runner.py`):

```python
def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile (0–100) of values. Returns 0.0 when empty."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (p / 100) * (len(sorted_v) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_v) - 1)
    return sorted_v[lo] + (sorted_v[hi] - sorted_v[lo]) * (idx - lo)
```

**Implementation steps.**

1. In `/Users/mahmoud/src/ai/simba/src/simba/eval/runner.py`:
   - Add `import time` at the top.
   - Add `latency_ms: float = 0.0` field to `CaseResult` (after `metrics`; default keeps old construction sites working).
   - In `to_dict()`, add `"latency_ms": self.latency_ms`.
   - Add private `_percentile(values, p)` function.
   - In `run_eval`, wrap the retriever call: `t0 = time.perf_counter(); ranked = list(retriever(case.query)); lat = (time.perf_counter() - t0) * 1000`. Pass `latency_ms=lat` to `CaseResult(...)`.
   - After building `aggregate`, compute `latencies = [c.latency_ms for c in per_case]` and add `aggregate["p50_ms"] = _percentile(latencies, 50)` and `aggregate["p95_ms"] = _percentile(latencies, 95)`.
   - `EvalReport.to_dict()` already calls `self.aggregate` — the new keys are included automatically.

2. In `/Users/mahmoud/src/ai/simba/src/simba/eval/benchmarks/run.py`:
   - After building `overall` list of metrics dicts in `run_recall`, collect latencies from `rep.per_case`: `all_latencies.extend(c.latency_ms for c in rep.per_case)` (accumulate across all datasets).
   - Add to the return dict: `"latency": {"p50_ms": _percentile(all_latencies, 50), "p95_ms": _percentile(all_latencies, 95), "n": len(all_latencies)}`.
   - Import `_percentile` from `simba.eval.runner` or duplicate the two-line impl locally (duplication is fine for a private helper; avoid a circular import — check: `runner` does not import `run`, so importing `runner._percentile` is safe). Use `from simba.eval.runner import _percentile` with a type-checking guard to avoid exposing private names — or just re-implement the four-liner locally as `_pct`.

3. In `/Users/mahmoud/src/ai/simba/src/simba/eval/benchmarks/judge.py`:
   - Add `import time`.
   - In `run_qa`, wrap the `score_case` call: `t0 = time.perf_counter(); correct = score_case(...); lat = (time.perf_counter() - t0) * 1000`. Accumulate into `latencies: list[float]`.
   - Note: this times the full retriever+answer+grade cycle. Document in a comment that it is end-to-end latency per question (not retriever-only). If retriever-only is needed later, `score_case` can be split.
   - Add `"latency": {"p50_ms": _pct(latencies, 50), "p95_ms": _pct(latencies, 95), "n": len(latencies)}` to the returned report.

4. Update `scripts/run_qa.py` and `scripts/run_longmemeval.py` print sections to include latency: `print(f"  p50={report['latency']['p50_ms']:.0f}ms p95={report['latency']['p95_ms']:.0f}ms")`.

**Tests** (additions to `tests/eval/test_runner.py`):

```python
def test_case_result_has_latency_ms() -> None:
    import time
    calls: list[float] = []
    def slow_retriever(q: str) -> list[str]:
        time.sleep(0.01)   # 10ms
        return ["m1"]
    rep = runner.run_eval(_DATASET, slow_retriever, ks=(1,))
    for case in rep.per_case:
        assert case.latency_ms >= 5.0   # at least 5ms (generous lower bound)
    assert case.latency_ms < 5000.0     # sanity upper bound

def test_aggregate_has_p50_p95() -> None:
    rep = runner.run_eval(_DATASET, _retriever, ks=(1,))
    assert "p50_ms" in rep.aggregate
    assert "p95_ms" in rep.aggregate
    assert rep.aggregate["p50_ms"] >= 0.0
    assert rep.aggregate["p95_ms"] >= rep.aggregate["p50_ms"]

def test_to_dict_includes_latency_ms() -> None:
    rep = runner.run_eval(_DATASET, _retriever, ks=(1,))
    d = rep.per_case[0].to_dict()
    assert "latency_ms" in d

def test_percentile_correctness() -> None:
    from simba.eval.runner import _percentile
    assert _percentile([], 50) == 0.0
    assert _percentile([10.0], 50) == pytest.approx(10.0)
    assert _percentile([10.0, 20.0, 30.0], 50) == pytest.approx(20.0)
    assert _percentile([10.0, 20.0, 30.0], 100) == pytest.approx(30.0)
    assert _percentile([10.0, 20.0, 30.0], 0) == pytest.approx(10.0)
```

Additions to `tests/eval/benchmarks/test_judge.py`:

```python
def test_run_qa_report_has_latency_block() -> None:
    """run_qa report dict must include latency.p50_ms and latency.p95_ms."""
    from simba.eval.dataset import Dataset, EvalCase, Memory
    import simba.eval.benchmarks.judge as jmod
    ds = Dataset(
        name="t", corpus=[Memory(id="c1", content="x")],
        cases=[EvalCase(id="q1", query="q", relevant_ids=["c1"], answer="a")]
    )
    import simba.memory.config as mc
    cfg = mc.MemoryConfig(llm_rerank_enabled=False, scoring_enabled=False,
                           expansion_enabled=False)
    embed = lambda t: [0.0] * 384
    llm = FakeLlm(answer="a", verdict={"correct": True})
    report = jmod.run_qa([ds], embed_doc=embed, embed_query=embed,
                          cfg=cfg, llm=llm)
    assert "latency" in report
    assert "p50_ms" in report["latency"]
    assert "p95_ms" in report["latency"]
    assert report["latency"]["n"] == 1
```

**Acceptance.**

- `uv run pytest tests/eval/test_runner.py tests/eval/benchmarks/test_judge.py -x -q` is green.
- `CaseResult.to_dict()` output has `"latency_ms"` key.
- `EvalReport.aggregate` has `"p50_ms"` and `"p95_ms"` keys for any non-empty dataset.
- `run_recall` report dict has `"latency"` top-level key with `p50_ms`, `p95_ms`, `n`.
- `run_qa` report dict has `"latency"` top-level key.

**Verify.**

```bash
uv run pytest tests/eval/test_runner.py tests/eval/benchmarks/test_judge.py -x -q
uv run python -c "
import simba.eval.runner as r, simba.eval.dataset as ds
import time
dset = ds.Dataset('t', [ds.Memory(id='m1',content='x')],
                  [ds.EvalCase(id='c1',query='q',relevant_ids=['m1'])])
rep = r.run_eval(dset, lambda q: ['m1'], ks=(1,))
print(rep.aggregate)
print(rep.per_case[0].to_dict())
"
```

**Reuse.**

- `src/simba/eval/runner.py:run_eval` (line 72) — wraps retriever call; add timing here.
- `src/simba/eval/benchmarks/run.py:run_recall` (line 19) — collects `per_case` from `EvalReport`; read `latency_ms` from there.

---

## Full test suite run (after all four tasks)

```bash
uv run pytest tests/llm/test_judge_config.py \
              tests/eval/benchmarks/test_baseline_store.py \
              tests/eval/benchmarks/test_judge.py \
              tests/eval/benchmarks/test_longmemeval.py \
              tests/eval/test_runner.py \
              -x -q
```

All tests must be green. Then run the full suite to confirm no regressions:

```bash
uv run pytest -x -q
```

---

## File index (all paths absolute)

**Create:**

- `/Users/mahmoud/src/ai/simba/src/simba/llm/judge_config.py`
- `/Users/mahmoud/src/ai/simba/src/simba/eval/benchmarks/baseline_store.py`
- `/Users/mahmoud/src/ai/simba/tests/llm/test_judge_config.py`
- `/Users/mahmoud/src/ai/simba/tests/eval/benchmarks/test_baseline_store.py`

**Modify:**

- `/Users/mahmoud/src/ai/simba/src/simba/eval/benchmarks/judge.py` — B1 (`score_case`/`run_qa` judge split), B3 (`score_abstention`, `aggregate_with_abstention`, abstention path in `run_qa`), B4 (latency in `run_qa`)
- `/Users/mahmoud/src/ai/simba/src/simba/eval/config.py` — B2 (`baseline_dir`), B3 (`abstention_phrases`)
- `/Users/mahmoud/src/ai/simba/src/simba/eval/runner.py` — B4 (`CaseResult.latency_ms`, `_percentile`, timing in `run_eval`)
- `/Users/mahmoud/src/ai/simba/src/simba/eval/benchmarks/run.py` — B4 (latency in `run_recall` report)
- `/Users/mahmoud/src/ai/simba/src/simba/eval/benchmarks/longmemeval.py` — B3 (`is_abstention`)
- `/Users/mahmoud/src/ai/simba/scripts/run_qa.py` — B1 (judge client), B2 (`--baseline`, `--cache`, `--out`, `--split`), B4 (print latency)
- `/Users/mahmoud/src/ai/simba/scripts/run_longmemeval.py` — B3 (`--full`, `--abstention`, `--baseline`, `--qa`), B4 (print latency)
- `/Users/mahmoud/src/ai/simba/tests/eval/test_runner.py` — B4 (latency tests)
- `/Users/mahmoud/src/ai/simba/tests/eval/benchmarks/test_judge.py` — B1, B3, B4 new test functions
- `/Users/mahmoud/src/ai/simba/tests/eval/benchmarks/test_longmemeval.py` — B3 (`is_abstention` test)
