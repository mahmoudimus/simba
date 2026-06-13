# How a local-first memory system measured its way to parity on LongMemEval

*simba is a pure-Python, local-first long-term memory layer for coding agents. This is
the honest story of taking it from 0.561 to 0.823 on LongMemEval-S over one weekend —
matching the strongest comparable open system on the same grading axis — and, just as
importantly, what we measured and threw away along the way.*

> **The one-line claim, stated carefully:** on LongMemEval-S, graded by the *official*
> per-type judge, simba answers **0.823** (386/469) with a deepseek-v4-flash answerer.
> The strongest comparable system we could re-run on the identical judge, hebb-mind,
> scores **0.793** with a stronger (v4-pro) answerer. Paired McNemar puts the +3pp gap
> at **p = 0.18 — not statistically significant.** So: **at least at parity with the
> best comparable open system, point-estimate ahead, not a proven win.** Every point of
> the climb has a named, measured cause, and the negative results are published below.

## Why the headline numbers you've seen are mostly mirages

We surveyed 44 LongMemEval-referencing repos. Every QA number above ~90% turned out to
be cloud-only, self-disclaimed, hardcoded, an oracle-variant (evidence handed to the
reader), or a competitor's cited figure. **A benchmark number means nothing without its
(variant + judge).** The one directly comparable external point — hebb-mind, run on the
same DeepSeek-V4 judge family — sat at 0.79. That became the bar.

When we first measured simba on the same data we got **0.561**, and the gap looked like a
22-point chasm. It wasn't. It was a stack of distinct, individually-cheap problems — and
about half of it wasn't even a *modeling* gap. Here are the tactics that took it apart.

## Tactic 1 — Make failure diagnosable before optimizing anything

The first thing we built wasn't a feature; it was a *debugger for retrieval*. Two metrics:

- `pool_complete@N` — is the full evidence set even in the top-N candidates? (the
  first-stage ceiling)
- `complete@k` — did the reranker land it in the usable top-k?

…and a four-bucket classifier that forces every miss into exactly one of
`candidate_generation | reranking | reasoning | success`. The rule we held to all
weekend: **don't optimize "memory quality" — optimize the failure bucket.**

This immediately killed a class of wasted effort. Counting questions, for instance,
turned out to be *breadth-bound* (`pool_complete@20 = 0.40 → @80 = 1.00`): the evidence
was retrieved, just ranked below the context window. No reranker change could ever have
fixed it; widening the candidate pool did.

## Tactic 2 — The benchmark must measure what ships

We found, repeatedly, that our harness was measuring something subtly different from the
product — always in a direction that *understated* us:

- The reader had no **"Current Date"** anchor, so every "how long ago…" question was
  unanswerable. Adding it (the official LongMemEval reader convention; the host agent
  always knows the date in production) was **+0.111 overall, and temporal-reasoning
  doubled 0.417 → 0.833.**
- Our **judge** was a homemade binary "same meaning?" prompt. Every published system uses
  the official *per-type* templates (a rubric judge for preference questions, off-by-one
  tolerance for temporal, etc.). Re-grading on the official judge was **+3.6pp** on
  identical outputs — and it exposed that our own judge had been *deflating* us.

This tactic also caught a regression *we* shipped. v0.7.0 turned on answer-time conflict
surfacing by default — a genuine win on contradiction benchmarks. But the moment the
bench ran the shipping config, knowledge-update questions cratered to **0.25**: a "what
is my *current* address?" query retrieves the old and new value, the detector flags them
as a conflict, and the directive tells the model not to pick a side — exactly wrong when
recency resolves it. Caught within 24 hours *because* the benchmark measured what ships.

## Tactic 3 — LLMs judge, Python computes

The single most-confirmed principle of the whole effort. The LLM is excellent at
*language understanding* and unreliable at *computation*; so we push every computation
out of the model and into deterministic code, and keep the LLM for judgment.

- **Counting**: the LLM enumerates instances, Python calls `len()`. (0.30 → 0.75 on the
  isolating probe.)
- **Temporal arithmetic**: the LLM writes a small Python program with `datetime`, we
  execute it in a sandbox. (+0.14 on temporal questions; 0% execution failures.)
- **Freshness**: when two dated values conflict, `max(date)` in code — not the model's
  judgment — picks the current one.

The boundary is sharp and we mapped it by failing at it: **three separate attempts to
make a *cheap* conflict detector with NLI cross-encoders all failed**, because NLI's
"contradiction" means *same-scene incompatibility* while memory conflict means
*cross-time incompatibility of durable state* — a different relation entirely. Detection
is judgment; it stays with the LLM.

## Tactic 4 — Probe before building, and publish the kills

Every lever got a paired A/B on identical contexts before it earned a line of production
code. The kill list is long, and it *is* the credibility of the 0.823:

| Killed lever | Why |
|---|---|
| Cross-encoder rerank in eval | dropped LME-S recall@5 0.776 → 0.620 |
| Rich-ingest provenance | neutral |
| NLI conflict detection (×3 model families) | wrong contradiction semantics |
| ARM3 date-disjoint conflict carve-out | failed its SubtleMemory gate (0.722 < 0.9) — date-disjointness can't tell an update from a genuine conflict |
| Wide-scope embedding-nominated retrieval | net-negative vs exhaustive |
| Answer-cache worker | recurrence measured too low to justify |
| Tool-calling + self-verify (rows route, all backends) | net +2 — capped by extraction recall, not routing |

That last row is worth dwelling on: we built four increasingly sophisticated answer-time
computation layers (LLM codegen → query-plan + fixed evaluator → a clingo possible-worlds
engine → an LLM tool-calling loop with a proof-anchored "check your work" pass). **All
four landed in the same narrow +2-to-+4 band**, because they all rearranged the same
*incomplete extracted rows*. The evaluator was never the bottleneck. (The verify pass, on
audit: precision 0.50, recall 0.27, zero auto-corrections — the model re-affirms more
than it re-derives, even when handed a deterministic proof to check against.)

## Tactic 5 — One calibrated axis, and run the significance test on yourself

We validated the judge (deepseek-v4-pro ≈ GPT-4o, κ = 0.90) before trusting any number,
re-ran every comparison on that one axis, and — the step most "we beat X" posts skip —
**ran the paired significance test against ourselves.** It said the +3pp lead is real on
the point estimate but within noise at n=468 (McNemar p=0.18). So we don't claim a win;
we claim parity. The reason that distinction matters is the entire point of the post: a
number you'd defend under someone else re-running it is worth more than a number that
sounds good.

## The ladder

| step | change | LME-S |
|---|---|---|
| baseline | shipped config, homemade judge | 0.561 |
| +protocol | official judge + Current-Date + conflict-off + reader rules | 0.749 |
| +breadth | intent-gated context k=80 for multi-session | 0.770 |
| +codegen | temporal questions answered via executed Python | 0.781 |
| +preference reader | intent-gated synthesis prompt for preference questions | **0.823** |

About 12 of the original 22 points were *protocol and a self-inflicted regression* — not
modeling. The rest was breadth, computation-as-code, and reader discipline. None of it
was a novel architecture.

## The convergence (and where we may be slightly ahead)

The striking thing, surfaced when we read the 2026 literature *after* deriving our levers
from failure analysis: the field has independently converged on the same design. WorldDB,
Chronos, TReMu, Zep, ReSSERAct — raw substrate + a dated-event layer + typed
supersession-vs-contradiction edges + deterministic temporal resolution. Chronos's
ablation even reports its event-calendar carries 58.9% of its gain, matching our timeline
result. We got there by debugging buckets; they got there by ablating readers. Same
destination is good evidence the destination is real.

One place the published work hasn't gone, and where simba's remaining experiments sit: an
**ambiguity-preserving runtime.** The text-to-SQL ambiguity literature (AMBROSIA, AmbiQT,
AmbiSQL) *resolves* ambiguity — enumerate interpretations or ask the user — before
emitting rigid code. None compile to a runtime that *keeps* the ambiguity. Our open thread
— the LLM emits a query plan with the vague terms left as named parameters, and a
possible-worlds engine (we prototyped it on clingo: brave/cautious entailment = the
Imieliński-Lipski certain/possible answers) reports the answer *as a function of
interpretation* — appears to sit in a genuinely open gap.

## What we'd believe, and what we wouldn't

- **Believe:** simba matches the strongest comparable open system on the canonical axis,
  from a weaker answerer, with every lever measured and every dead end documented. The
  fix that mattered most to users — the knowledge-update conflict regression — is live in
  v0.7.1.
- **Wouldn't (yet):** that it's a *significant* win (it isn't, at this sample size); that
  the answer-time machinery is the frontier (it's capped by extraction recall); or that
  any of this transfers beyond English single-user LongMemEval without re-measuring.

The next lever with measured headroom isn't more answer-time cleverness — it's moving
extraction to **write time**, per-session, where recall is recoverable. That's the
campaign that could turn the point-estimate lead into a real one. We'll measure it the
same way, and we'll tell you if it doesn't work.

---

*Methodology, per-question verdicts, and the full kill list are in the repo. Numbers
quoted are LongMemEval-S, official per-type judge (deepseek-v4-pro), answerer
deepseek-v4-flash unless noted; ±1pp run-to-run variance; abstention slice excluded on
both sides.*
