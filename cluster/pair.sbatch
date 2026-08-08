#!/usr/bin/env bash
#SBATCH --job-name=osint-pair
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log
#
# Build the bridge map and the pairs for one private corpus against one public one.
#
# No GPU and no model. What it answers is how many bridges a pairing has and what kind of
# entities they are -- the question that decides whether generating from it is worth GPU
# time at all. Dodis against the sanctions list produced zero, because a 1947 diplomat is
# not on a 2014 sanctions list, and that cost a two-model job to discover.
#
# Link files can be supplied rather than rebuilt. Linking the parliamentary record costs an
# hour and 90 GB of mapping, and it produces the same file every time.
#
# Usage:
#   sbatch --partition=<p> --qos=<q> \
#     --export=ALL,OSINT_PRIVATE=dodis,OSINT_PUBLIC=parliament,\
#OSINT_LINKS=$HOME/osint-parliament-13502 cluster/pair.sbatch
set -euo pipefail

PRIVATE="${OSINT_PRIVATE:-dodis}"
PUBLIC="${OSINT_PUBLIC:-parliament}"
LINKS="${OSINT_LINKS:-}"
export WORK="/scratch/$USER/osint-pair-$SLURM_JOB_ID"
export OUT="$HOME/osint-pair-$SLURM_JOB_ID"
export CACHE="/scratch/$USER/osint-cache"

export UV_INSTALL_DIR="$WORK/bin" UV_CACHE_DIR="$WORK/uv-cache"
export UV_PYTHON_INSTALL_DIR="$WORK/uv-python" OSINT_DATA="$WORK/data"
export OSINT_RAW="$CACHE/raw" HF_HOME="$CACHE/hf"
export PATH="$UV_INSTALL_DIR:$PATH" PYTHONUNBUFFERED=1
export NLTK_DISABLE_IMPORT_SECURITY=1
export OSINT_DODIS_OCR="$HOME/dodis_ocr" OSINT_DODIS_NT="$HOME/dodis-lod.nt"

echo "node    $(hostname)"
echo "pairing $PRIVATE (confidential) against $PUBLIC (public)"
echo "links   ${LINKS:-none supplied, everything relinked}"
mkdir -p "$UV_INSTALL_DIR" "$OUT" "$OSINT_RAW" "$OSINT_DATA/links"
trap 'rm -rf "$WORK"; echo "cleaned up $WORK (results in $OUT)"' EXIT

echo "=== installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1

echo "=== cloning ==="
git clone -q https://github.com/Jibril-Frej/osint-benchmark.git "$WORK/repo"
cd "$WORK/repo"
uv sync -q
uv run pytest -q

echo "=== 1/4 sources ==="
uv run python pipeline/01_sources.py "$PRIVATE" "$PUBLIC" wikipedia_index

echo "=== 2/4 link ==="
for source in "$PRIVATE" "$PUBLIC"; do
  if [ -n "$LINKS" ] && [ -f "$LINKS/$source.jsonl" ]; then
    echo "reusing $LINKS/$source.jsonl"
    cp "$LINKS/$source.jsonl" "$OSINT_DATA/links/$source.jsonl"
    cp "$LINKS/$source.jsonl.provenance.json" "$OSINT_DATA/links/" 2>/dev/null || true
  else
    uv run python pipeline/02_link.py "$source" --index wikipedia_index
  fi
done
wc -l "$OSINT_DATA"/links/*.jsonl

echo "=== 3/4 graph ==="
uv run python pipeline/03_graph.py

echo "=== 4/4 public evidence and pairs ==="
uv run python pipeline/04_public.py --index wikipedia_index
uv run python pipeline/05_pair.py

for name in graph/bridges.jsonl pairs/pairs.jsonl facts/wikidata.jsonl; do
  cp -f "$OSINT_DATA/$name" "$OUT/$(basename "$name")" 2>/dev/null || true
done

echo "=== what the bridges are ==="
uv run python - <<'PY' || true
import collections, json, os, pathlib

def read(name):
    path = pathlib.Path(os.environ["OSINT_DATA"]) / name
    return [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []

bridges = read("graph/bridges.jsonl")
labels = {r["qid"]: r.get("label", r["qid"]) for r in read("facts/wikidata.jsonl")}
print(f"{len(bridges)} bridges")
by_docs = sorted(bridges, key=lambda b: -(b["private_count"] + b["public_count"]))
for b in by_docs[:20]:
    print(f"  {labels.get(b['qid'], b['qid'])[:44]:44s} "
          f"private {b['private_count']:4d}  public {b['public_count']:4d}")
pairs = read("pairs/pairs.jsonl")
same = sum(1 for p in pairs if p["same_period"])
print(f"\n{len(pairs)} pairs, {same} contemporaneous")
if pairs:
    days = sorted(abs(p["days_apart"]) for p in pairs if p["days_apart"] is not None)
    if days:
        print(f"days apart: min {days[0]}, median {days[len(days)//2]}, max {days[-1]}")
PY
ls -la "$OUT"
echo done
