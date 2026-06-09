"""Re-grade mem0's saved LoCoMo predictions with simba's EXACT judge path, so the
mem0 number is apples-to-apples with simba's own 0.426 (same build_judge_prompt,
same _extract_json robustness, same deepseek-v4-pro judge). The subagent's grade.py
mis-graded correct answers (brittle JSON parse) -> the 0.090 was invalid.
"""

from __future__ import annotations

import json

import simba.eval.benchmarks.judge as J
import simba.llm.client as c
import simba.llm.config as lc

PRED = "/Users/mahmoud/src/ai/memory/mem0/.locomo_v4_run/predictions.json"
preds = json.load(open(PRED))
judge = c.LlmClient(lc.LlmConfig(provider="llm-cli", model="deepseek-v4-pro", max_tokens=256))

rows: list[tuple[str, bool]] = []
empty = 0
unjudged = 0
for i, p in enumerate(preds):
    pred = (p.get("predicted") or "").strip()
    intent = p.get("intent") or "?"
    if not pred:
        empty += 1  # simba skips empty answerer output (not graded) — match that
        continue
    verdict = judge.complete_json(J.build_judge_prompt(p["question"], p["gold"], pred))
    if not isinstance(verdict, dict) or "correct" not in verdict:
        unjudged += 1
        continue
    rows.append((intent, bool(verdict["correct"])))
    if (i + 1) % 20 == 0:
        print(f"  graded {len(rows)} / {i + 1}…", flush=True)

rep = J.aggregate(rows)
rep["n_empty_skipped"] = empty
rep["n_unjudged"] = unjudged
print("MEM0_REGRADE " + json.dumps(rep, indent=2))
print("DONE")
