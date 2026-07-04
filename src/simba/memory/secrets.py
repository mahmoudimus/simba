"""Secret veto for memory ingestion (spec 33 v2, hippo-memory borrow).

hippo-memory shipped the same curated-markdown→daemon import bridge simba has
and then hotfixed it: credential-bearing memory files were ingested and
auto-shared. This module is the veto — a compact, precision-first pattern set
run over a file's FULL text before anything reaches ``/store``.

Precision over recall by design: ordinary prose ABOUT credentials ("rotate
the API key quarterly") must not match; the false-positive cost is a silently
missing memory, which is worse to debug than a visible veto line. Pure
stdlib, no I/O, never raises.
"""

from __future__ import annotations

import re

# Ordered: the first match names the veto kind (most specific first).
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private-key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY"),
    ),
    ("aws-access-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "github-token",
        re.compile(r"\bghp_[A-Za-z0-9]{36}\b|\bgithub_pat_[A-Za-z0-9_]{22,}\b"),
    ),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    # JWT: three base64url segments, header starting {"alg" → eyJ.
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]+")),
    ("api-secret-key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    # Assignment shapes: password/api_key/secret/token = "8+ non-space chars".
    # The quoted-or-bare value floor kills prose and short examples.
    (
        "credential-assignment",
        re.compile(
            r"(?i)\b(?:password|passwd|api[_-]?key|secret|token)\s*[:=]\s*"
            r"['\"][^'\"\s]{8,}['\"]"
        ),
    ),
)


def detect_secret(text: str) -> str | None:
    """Return the veto kind for the first credential pattern in ``text``.

    ``None`` = clean. Callers veto the whole file on any hit — a memory file
    is small enough that partial salvage isn't worth the leak risk.
    """
    if not text:
        return None
    for kind, pattern in _PATTERNS:
        if pattern.search(text):
            return kind
    return None
