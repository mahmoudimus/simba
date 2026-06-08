#!/usr/bin/env bash
# Fetch external benchmark datasets into a stable, gitignored location.
#
#   ./scripts/fetch_benchmarks.sh            # fetch all into .simba/benchmarks/
#   DEST=/some/dir ./scripts/fetch_benchmarks.sh
#
# Idempotent: skips a file that already exists and matches its recorded sha256.
# No new Python deps — uses curl. LongMemEval URLs can be overridden via env in
# case the upstream layout changes (LME_ORACLE_URL / LME_S_URL).
set -euo pipefail

DEST="${DEST:-.simba/benchmarks}"
mkdir -p "$DEST"

# LoCoMo — 10 conversations / ~1986 QA (snap-research/locomo).
LOCOMO_URL="${LOCOMO_URL:-https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json}"
# LongMemEval — 500 questions (xiaowu0162). Oracle = evidence-only haystack
# (upper bound); _s = full haystack with distractor sessions (the real test).
LME_ORACLE_URL="${LME_ORACLE_URL:-https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_oracle.json}"
LME_S_URL="${LME_S_URL:-https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s.json}"
# HotpotQA — dev distractor (genuine bridge-entity multi-hop). The original CMU
# host (curtis.ml.cmu.edu) is offline since 2025-05, so default to a Wayback
# snapshot of the canonical JSON (override via HOTPOT_URL).
HOTPOT_URL="${HOTPOT_URL:-https://web.archive.org/web/2022id_/http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json}"

fetch() {
  local url="$1" out="$2"
  if [ -f "$out" ]; then
    echo "✓ exists: $out ($(wc -c <"$out" | tr -d ' ') bytes)"
    return 0
  fi
  echo "↓ $url"
  curl -fSL --retry 3 -o "$out.tmp" "$url"
  mv "$out.tmp" "$out"
  echo "✓ saved: $out ($(wc -c <"$out" | tr -d ' ') bytes)"
}

fetch "$LOCOMO_URL"     "$DEST/locomo10.json"
fetch "$LME_ORACLE_URL" "$DEST/longmemeval_oracle.json"
fetch "$LME_S_URL"      "$DEST/longmemeval_s.json"
fetch "$HOTPOT_URL"     "$DEST/hotpot_dev_distractor_v1.json"
# HaluMem (memory-hallucination eval; docs/plans/10). ~33MB, 20 users, >1M tok/user.
HALUMEM_URL="${HALUMEM_URL:-https://huggingface.co/datasets/IAAR-Shanghai/HaluMem/resolve/main/HaluMem-Medium.jsonl}"
fetch "$HALUMEM_URL"    "$DEST/HaluMem-Medium.jsonl"

# Record checksums so reruns / CI can verify integrity.
( cd "$DEST" && shasum -a 256 ./*.json > SHA256SUMS )
echo "--- $DEST/SHA256SUMS ---"
cat "$DEST/SHA256SUMS"
echo "Done. Point the bench scripts at: $DEST/{locomo10,longmemeval_oracle,longmemeval_s}.json"
