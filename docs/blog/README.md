# simba blog — drafts

The evolution of simba, a local-first long-term memory layer for coding agents, told as
a three-part series. All posts are **drafts**. Numbers throughout are LongMemEval-S,
official per-type judge (deepseek-v4-pro), answerer deepseek-v4-flash unless noted.

## The series

1. **[Origins: a local-first evidence layer, and why store-raw](part1-origins.md)** —
   how simba started, the hooks architecture and local hybrid recall, the founding
   store-raw bet, and the first measured lesson ("multi-hop is reasoning, not
   retrieval") that set the evidence-layer-vs-answer-layer thesis.
2. **[The neuro-symbolic bet, and what the data did to it](part2-neurosymbolic-bet.md)**
   — the hypothesis that fact schemas and first-order/symbolic logic would beat
   store-raw; the fact-index experiment and why it failed (the query-supplied
   equivalence relation); where symbolic genuinely wins (LLM structures, code computes);
   the four-variant answer-time-compute cap; and the open ambiguity-preserving-runtime
   frontier.
3. **[Evaluation as a repeatable benchmark: how we earned the numbers](part3-evaluation-as-benchmark.md)**
   — from ad-hoc spikes to a rigorous program; the diagnostic split that made retrieval
   debuggable; axis calibration; the "measure what ships" fidelity fixes that *were* the
   gains (0.561 → 0.823); and the paired significance test we ran on ourselves
   (parity with the strongest comparable system, not a proven win).

## Also

- **[How we measured our way to parity](how-we-measured-our-way-to-parity.md)** — a
  condensed, standalone TL;DR of the whole arc (overlaps Part 3; keep whichever fits the
  venue).

## The honest through-line

simba is an *evidence* layer; the host agent is the *reasoning* layer. Store the raw
turn, retrieve the complete evidence set, let the agent reason — and push deterministic
*computation* (counting, dates, freshness) out of the model while keeping *judgment*
(relevance, contradiction) in it. The series is the story of betting against that thesis
with symbolic machinery, measuring honestly, and watching the thesis hold — culminating
in a benchmark number we'd defend under someone else re-running it.
