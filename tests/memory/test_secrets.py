"""Secret veto for memory ingestion (spec 33 v2, hippo-memory borrow).

hippo-memory shipped the same curated-markdown import bridge and had to
hotfix credential-bearing files being ingested and auto-shared. Detection is
regex-precision-first: credentials match, ordinary prose about credentials
does not.
"""

from __future__ import annotations

import pytest

from simba.memory.secrets import detect_secret


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("-----BEGIN RSA PRIVATE KEY-----\nMIIE...", "private-key"),
        ("-----BEGIN OPENSSH PRIVATE KEY-----", "private-key"),
        ("aws_access_key_id = AKIAIOSFODNN7EXAMPLE", "aws-access-key"),
        ("token: ghp_" + "a1B2" * 9, "github-token"),
        ("github_pat_11ABCDEFG0_abcdefghijklmnopqrstuv", "github-token"),
        ("xoxb-123456789012-abcdefghijklm", "slack-token"),
        ("OPENAI_API_KEY=sk-abcdefghijklmnopqrstuv123456", "api-secret-key"),
        ("maps key AIzaSyA-1234567890abcdefghijklmnopqrstu", "google-api-key"),
        ('password: "correct-horse-battery"', "credential-assignment"),
        ("api_key = 'zk9f2m4x8q1w7e5r'", "credential-assignment"),
        (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIn0.sig",
            "jwt",
        ),
    ],
)
def test_detects_credentials(text: str, kind: str) -> None:
    assert detect_secret(text) == kind


@pytest.mark.parametrize(
    "text",
    [
        "",
        "rotate the API key quarterly per the security policy",
        "the password field validates length before hashing",
        "use `simba config set memory.port 8741` to change it",
        "git push branch:refs/heads/branch avoids the push.default trap",
        "password: short",  # under the 8-char credential floor
        "the token bucket rate limiter refills at 10/s",
    ],
)
def test_ignores_ordinary_prose(text: str) -> None:
    assert detect_secret(text) is None
