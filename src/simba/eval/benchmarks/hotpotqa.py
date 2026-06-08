"""HotpotQA loader: distractor-setting JSON -> simba eval Datasets.

One Dataset per question — each question ships its own 10-paragraph haystack
(2 gold supporting paragraphs + 8 distractors). Corpus units are the context
paragraphs keyed by title; gold evidence is the set of *supporting-fact*
paragraph titles (the bridge). Scored with ``bridge_recall@k`` (all hops in the
top-k), so it measures genuine multi-hop retrieval where LoCoMo couldn't — fully
local, no LLM judge. ``intent`` carries the HotpotQA ``type`` (bridge|comparison).
"""

from __future__ import annotations

import json
import pathlib
import typing

from simba.eval.dataset import Dataset, EvalCase, Memory


def _paragraph(title: str, sentences: list[str]) -> Memory:
    body = " ".join(s.strip() for s in sentences if s and s.strip())
    content = f"{title}. {body}" if body else title
    return Memory(id=title, content=content, type="PATTERN")


def load_hotpotqa_data(raw: list[dict[str, typing.Any]]) -> list[Dataset]:
    """Convert parsed HotpotQA items into one Dataset per question."""
    datasets: list[Dataset] = []
    for item in raw:
        qid = str(item.get("_id", f"hq-{len(datasets)}"))

        corpus: list[Memory] = []
        seen: set[str] = set()
        for entry in item.get("context", []):
            if not entry:
                continue
            title = str(entry[0])
            raw_sents = entry[1] if len(entry) > 1 else []
            sentences = raw_sents if isinstance(raw_sents, list) else []
            if title in seen:  # HotpotQA titles are unique per question; be safe
                continue
            seen.add(title)
            corpus.append(_paragraph(title, [str(s) for s in sentences]))

        # Gold = distinct supporting-fact paragraph titles (the bridge), kept only
        # when present in this question's corpus (drop unresolvable references).
        gold = []
        gold_seen: set[str] = set()
        for sf in item.get("supporting_facts", []):
            if not sf:
                continue
            title = str(sf[0])
            if title in seen and title not in gold_seen:
                gold_seen.add(title)
                gold.append(title)

        if not corpus or not gold:
            continue

        case = EvalCase(
            id=qid,
            query=str(item.get("question", "")),
            relevant_ids=gold,
            intent=str(item.get("type", "")),
            answer=str(item.get("answer") or ""),
        )
        datasets.append(Dataset(name=qid, corpus=corpus, cases=[case]))
    return datasets


def load_hotpotqa_pooled(
    raw: list[dict[str, typing.Any]], *, max_questions: int | None = None
) -> list[Dataset]:
    """Pool many questions' paragraphs into ONE shared corpus (the recall regime).

    Per-question HotpotQA distractor haystacks are only ~10 paragraphs, so recall
    saturates (top-10 = everything). Pooling deduplicates paragraphs by Wikipedia
    title across ``max_questions`` questions into a single large corpus, so each
    question must recall its 2 gold paragraphs from thousands of competitors — a
    genuine recall problem, the setting where an entity-bridge ADD lever can
    surface *missed* multi-hop gold (cf. HotpotQA fullwiki). Returns a single
    Dataset with one case per question over the shared corpus.
    """
    corpus: list[Memory] = []
    seen_titles: set[str] = set()
    cases: list[EvalCase] = []
    for item in raw:
        if max_questions is not None and len(cases) >= max_questions:
            break
        per_q = load_hotpotqa_data([item])
        if not per_q:
            continue
        d = per_q[0]
        for mem in d.corpus:
            if mem.id not in seen_titles:
                seen_titles.add(mem.id)
                corpus.append(mem)
        cases.append(d.cases[0])
    if not corpus or not cases:
        return []
    return [Dataset(name="hotpotqa-pooled", corpus=corpus, cases=cases)]


def load_hotpotqa(path: str | pathlib.Path) -> list[Dataset]:
    """Load + parse a HotpotQA distractor JSON into per-question Datasets."""
    raw = json.loads(pathlib.Path(path).read_text())
    return load_hotpotqa_data(raw)
