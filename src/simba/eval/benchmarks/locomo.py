"""LoCoMo loader: snap-research/locomo `locomo10.json` -> simba eval Datasets.

One Dataset per conversation (each conversation is a self-contained haystack
shared by its ~199 questions). Corpus units are dialogue turns keyed by their
``dia_id`` (e.g. "D1:3"); gold evidence is the list of supporting ``dia_id``s,
so the existing recall harness scores recall@k of evidence directly.
"""

from __future__ import annotations

import json
import pathlib
import re
import typing

from simba.eval.dataset import Dataset, EvalCase, Memory

_SESSION_KEY = re.compile(r"^session_\d+$")

# Integer category codes -> names (from snap-research/locomo task_eval).
_CATEGORY = {
    1: "multi-hop",
    2: "single-hop",
    3: "open-domain",
    4: "single-hop-factual",
    5: "adversarial",
}


def _conversation_turns(conversation: dict[str, typing.Any]) -> list[Memory]:
    turns: list[Memory] = []
    for key, value in conversation.items():
        if not _SESSION_KEY.match(key) or not isinstance(value, list):
            continue
        # LoCoMo turns use relative time ("yesterday"); the absolute session date
        # lives in a sibling "<key>_date_time" field. Prefix it so the gold
        # (absolute) answers are resolvable — both at recall and QA-judge time.
        date = str(conversation.get(f"{key}_date_time", "")).strip()
        for turn in value:
            dia_id = turn.get("dia_id")
            text = (turn.get("text") or "").strip()
            if not dia_id or not text:
                continue
            speaker = turn.get("speaker", "")
            body = f"{speaker}: {text}" if speaker else text
            content = f"[{date}] {body}" if date else body
            turns.append(Memory(id=str(dia_id), content=content, type="PATTERN"))
    return turns


def load_locomo_data(
    raw: list[dict[str, typing.Any]], *, include_adversarial: bool = True
) -> list[Dataset]:
    """Convert parsed LoCoMo samples into one Dataset per conversation."""
    datasets: list[Dataset] = []
    for sample in raw:
        sample_id = str(sample.get("sample_id", f"conv-{len(datasets)}"))
        corpus = _conversation_turns(sample.get("conversation", {}))
        corpus_ids = {m.id for m in corpus}

        cases: list[EvalCase] = []
        for i, qa in enumerate(sample.get("qa", [])):
            category = qa.get("category", 0)
            if category == 5 and not include_adversarial:
                continue
            evidence = [str(e) for e in qa.get("evidence", [])]
            gold = [e for e in evidence if e in corpus_ids]
            if not gold:  # drop questions whose evidence we can't resolve
                continue
            cases.append(
                EvalCase(
                    id=f"{sample_id}_q{i}",
                    query=str(qa.get("question", "")),
                    relevant_ids=gold,
                    intent=_CATEGORY.get(category, str(category)),
                    answer=str(qa.get("answer") or ""),
                )
            )
        if corpus and cases:
            datasets.append(Dataset(name=sample_id, corpus=corpus, cases=cases))
    return datasets


def load_locomo(
    path: str | pathlib.Path, *, include_adversarial: bool = True
) -> list[Dataset]:
    """Load + parse locomo10.json into per-conversation Datasets."""
    raw = json.loads(pathlib.Path(path).read_text())
    return load_locomo_data(raw, include_adversarial=include_adversarial)
