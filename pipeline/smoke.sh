#!/usr/bin/env bash
# Run every step of the pipeline on a deliberately small slice.
#
# The point is to watch the whole chain produce output before committing to full scale.
# It uses the smallest real sources and two stand-ins, both of which label what they
# produce:
#
#   * step 2 links by exact article-title match instead of ReFinED -- real linking, no
#     model, much worse;
#   * steps 6 and 7 use a scripted stand-in instead of a served model, so the questions
#     are placeholders. Anything they produce is marked STUB.
#
# What it proves is that the chain runs and every step writes something. It says nothing
# about question quality -- for that, serve the models in config/models.toml and drop
# --stub.
#
# Usage: bash pipeline/smoke.sh [work-dir]
set -euo pipefail

WORK="${1:-${OSINT_DATA:-./data}}"
export OSINT_DATA="$WORK"
LIMIT="${SMOKE_LIMIT:-2000}"

step() { printf '\n=== %s ===\n' "$1"; }

step "1/9 sources"
# cablegate is the private leg and there is no smaller one -- 1.7 GB, downloaded once.
# sanctions and parliament are the small public sources.
uv run python pipeline/01_sources.py cablegate ucdp

step "1b/9 the public entity index (needs the Wikipedia SQL dumps, 2.8 GB)"
if [ -f "$WORK/docs/wikipedia_index.jsonl" ]; then
  echo "already built, skipping"
else
  uv run python pipeline/01_sources.py wikipedia_index
fi

step "2/9 link"
# Both sides: an entity has to be named privately and publicly to bridge.
# Private prose by title match (no model); the public side is tabular, so its names are
# reconciled against the live Wikidata endpoint instead.
# Public side first: its names are reconciled against the live Wikidata endpoint.
uv run python pipeline/02_link.py ucdp --limit "$LIMIT" --stride 401
# Then the private prose, looking only for the entities the public side named. Matching
# all 7.5M article titles against ALL-CAPS cable text finds a title for almost any phrase.
uv run python pipeline/02_link.py cablegate --dictionary --restrict-to ucdp --limit "$LIMIT" --stride 53

step "3/9 graph"
uv run python pipeline/03_graph.py

step "4/9 public evidence for the bridge entities"
uv run python pipeline/04_public.py --limit 25

step "5/9 pair"
uv run python pipeline/05_pair.py

step "6/9 generate (STUB: placeholder questions)"
uv run python pipeline/06_generate.py --stub --limit 25

step "7/9 necessity (STUB: placeholder flags)"
uv run python pipeline/07_necessity.py --stub

step "8/9 review page"
uv run python pipeline/08_review.py

step "9/9 release"
uv run python pipeline/09_release.py --version smoke --no-fingerprints

printf '\nSmoke run complete. Questions are placeholders: steps 6 and 7 ran with a stand-in.\n'
