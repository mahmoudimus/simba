"""Entropy-gated exact-term matching for recall.

simba's keyword arm is a TRIGRAM FTS — great for fuzzy/substring matching on prose,
but it *collides* high-information tokens: ``50815`` becomes ``508``/``081``/``815``,
which overlap ``50806``/``50858``/``50835``, so an exact error code can't discriminate
itself (measured: a bare "INTERR 50815" ranked the exact-code memory #14). The fix is
to route only the *high-information* query tokens to a WHOLE-WORD exact match, leaving
prose to trigrams.

"Information" is **df-surprisal** — ``-log2(df/N)``, the self-information of observing a
term in the corpus. A rare token (error code, symbol, SHA, hapax) scores high; a common
word scores low. Crucially this is NOT character-entropy, which wrongly flags common
all-distinct words like "debug". An UNSEEN token (``df=0``) scores maximal — exactly the
novel-error-code case, so a brand-new code routes to exact before it's ever stored.

Pure + deterministic. ``high_entropy_terms`` selects the tokens; ``exact_boost`` stably
re-orders a candidate pool so memories containing one (whole-word) rank first. Off by
default at the call site; this module never raises.
"""
from __future__ import annotations

import math
import re
import typing

try:  # maintained general-English frequency table (pure-Python + bundled data)
    import wordfreq as _wordfreq
except ImportError:  # fail-open: fall back to the embedded COMMON_WORDS stoplist
    _wordfreq = None

# A token with Zipf frequency >= this is treated as COMMON English (low information).
# Zipf 3.0 ≈ once per 1000 words; cleanly above the discriminators (codes/symbols are
# zipf 0) and below the pollution (how 6.2, control 5.4, error 4.6, block 4.9).
DEFAULT_ZIPF_COMMON = 3.0

# Identifier-aware tokenizer: keeps codes (50815), dotted names (verify.cpp), and
# leading-% / underscored symbols (%var_1DC) as single tokens. Case PRESERVED so the
# lexical-novelty test can see internal/all caps (CamelCase, INTERR).
_TOKEN = re.compile(r"%?[A-Za-z0-9][A-Za-z0-9_.]*")

# General-English common-word prior (frequency list, NOT a spelling dictionary — a
# spelling dict would mark rare real words valid and wrongly exclude proper-noun
# identifiers like Tigress/Hodur). A common word is LOW information regardless of how
# rare it happens to be in a terse technical corpus. Lowercase; compared lowercased.
_COMMON_WORDS_RAW = """
the be to of and a in that have i it for not on with he as you do at this but his by
from they we say her she or an will my one all would there their what so up out if
about who get which go me when make can like time no just him know take people into
year your good some could them see other than then now look only come its over think
also back after use two how our work first well way even new want because any these
give day most us is are was were been being am has had did does doing then once here
why where while whom whose each few more much many such own same other another every
how what when where why who which whom both either neither
debug error errors internal problem problems issue issues fix fixes fixed bug bugs
control consistency consistent value values valid invalid check checks checking state
states block blocks code codes case cases test tests testing run running runs call
calls called set sets reset wrong correct fail fails failed failure success result
results return returns add adds added remove removes removed change changes changed
pass passes passed handle handles handled cause caused causes trigger triggers
function functions method methods rule rules apply applies applied root use uses used
need needs needed try tries tried find finds found verify verifies expected actual
type types size sizes operand operands instruction instructions edge edges node nodes
"""
# .split() of a named variable (not a string literal), so no SIM905; the multiline
# form keeps the wordlist readable + easy to extend with a vetted public stoplist.
COMMON_WORDS = frozenset(_COMMON_WORDS_RAW.split())


def tokenize(text: str, *, lower: bool = True) -> list[str]:
    toks = _TOKEN.findall(text or "")
    return [t.lower() for t in toks] if lower else toks


def _has_identifier_shape(token: str) -> bool:
    """True for codes / symbols / paths / CamelCase / ALL-CAPS markers: any digit, any
    of ``._%-/:`` , an internal uppercase (CamelCase), or an all-caps token (len>=2)."""
    if any(c.isdigit() for c in token) or any(c in "._%-/:" for c in token):
        return True
    if token[1:].lower() != token[1:]:  # an uppercase past position 0 -> CamelCase
        return True
    return token.isupper() and len(token) >= 2


def _is_common_english(token: str, zipf_common: float) -> bool:
    """True if ``token`` is common English. Uses ``wordfreq`` (maintained frequency
    table) when available — a real Zipf prior over ~hundreds of thousands of words —
    else falls back to the embedded :data:`COMMON_WORDS` stoplist."""
    low = token.lower()
    if _wordfreq is not None:
        return _wordfreq.zipf_frequency(low, "en") >= zipf_common
    return low in COMMON_WORDS


def is_lexically_novel(token: str,
                       *, zipf_common: float = DEFAULT_ZIPF_COMMON) -> bool:
    """Lexical (general-English) novelty: identifier-shaped, OR not a common English
    word. Kills common-English pollution (how/internal); keeps codes/symbols/CamelCase
    and rare proper nouns (Tigress/Hodur)."""
    return _has_identifier_shape(token) or not _is_common_english(token, zipf_common)


def surprisal(df: int, n_docs: int) -> float:
    """Self-information ``-log2(df/N)`` in bits, add-0.5 smoothed so ``df=0`` is finite
    yet maximal (the novel-token case) and ``df≈N`` tends to ~0."""
    n = max(1, n_docs)
    p = (df + 0.5) / (n + 1)
    return -math.log2(p)


def high_entropy_terms(
    query: str,
    df_map: dict[str, int] | None = None,
    n_docs: int = 0,
    *,
    min_bits: float = 0.0,
    zipf_common: float = DEFAULT_ZIPF_COMMON,
) -> list[str]:
    """Query tokens worth an exact (non-trigram) match — the CONJUNCTION of two gates:

    1. lexical novelty (general-English): identifier-shaped, or a word not in the common
       stoplist — kills common-English pollution (how/internal/error), and
    2. corpus rarity (df-surprisal ≥ ``min_bits``) — kills corpus-common markers that
       are shape-novel but everywhere (e.g. INTERR at df=191).

    The corpus-rarity gate is OPTIONAL: pass ``df_map`` to enable it (absent token =>
    df 0 => maximal surprisal, the novel-code case). With ``df_map=None`` only the
    lexical gate runs — the daemon path uses this and lets the rarity-WEIGHTED
    ``exact_boost`` discriminate corpus-common markers instead of an at-recall df
    source. Returns deduped lowercase terms in first-seen order."""
    out: list[str] = []
    seen: set[str] = set()
    for tok in tokenize(query, lower=False):  # original case for the shape test
        low = tok.lower()
        if low in seen:
            continue
        if not is_lexically_novel(tok, zipf_common=zipf_common):
            continue
        if df_map is not None and surprisal(df_map.get(low, 0), n_docs) < min_bits:
            continue
        seen.add(low)
        out.append(low)
    return out


_ZIPF_CEIL = 8.0  # ~max English Zipf; rarity weight = ceil - zipf (rarer term => more)


def _contains_whole(text: str, term: str) -> bool:
    """Whole-token containment: ``50815`` matches "INTERR 50815" but not "508150"."""
    pat = rf"(?<![A-Za-z0-9_.%]){re.escape(term)}(?![A-Za-z0-9_.%])"
    return re.search(pat, text or "", re.IGNORECASE) is not None


def contains_whole(text: str, term: str) -> bool:
    """Public whole-token containment (the usage-signal citation check shares
    exact_boost's matching semantics — spec 33)."""
    return _contains_whole(text, term)


def _rarity(term: str) -> float:
    """Rarity weight: rarer (lower Zipf) terms weigh more. wordfreq when available,
    else a flat 1.0 (so scoring degrades to a plain match count)."""
    if _wordfreq is None:
        return 1.0
    return max(0.0, _ZIPF_CEIL - _wordfreq.zipf_frequency(term.lower(), "en"))


def exact_boost(
    candidates: list[dict[str, typing.Any]],
    terms: list[str],
    *,
    fields: tuple[str, ...] = ("content", "context"),
) -> list[dict[str, typing.Any]]:
    """Re-order ``candidates`` by how many high-entropy ``terms`` they contain (whole-
    word, case-insensitive, across ``fields``), each weighted by the term's RARITY — so
    a memory matching more, rarer terms ranks first. This self-discriminates a precise
    code (matches its code + marker) above a corpus-common marker alone, with no
    at-recall df source. Stable: equal scores (incl. zero) keep their original order, so
    non-matching candidates are undisturbed. No terms ⇒ identity."""
    if not terms:
        return candidates
    weights = {t: _rarity(t) for t in terms}

    def score(m: dict[str, typing.Any]) -> float:
        blob = " ".join(str(m.get(f) or "") for f in fields)
        return sum(w for t, w in weights.items() if _contains_whole(blob, t))

    return sorted(candidates, key=lambda m: -score(m))


def build_df_map(texts: typing.Iterable[str]) -> tuple[dict[str, int], int]:
    """Document frequency per token over a corpus; returns ``(df_map, n_docs)``.
    A token counts once per document. Used to feed :func:`high_entropy_terms`."""
    df: dict[str, int] = {}
    n = 0
    for text in texts:
        n += 1
        for tok in set(tokenize(text)):
            df[tok] = df.get(tok, 0) + 1
    return df, n
