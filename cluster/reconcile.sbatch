#!/usr/bin/env bash
#SBATCH --job-name=osint-reconcile
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log
#
# Measure what matching aliases buys on the public side of the join.
#
# No GPU. The public leg is reconciled by name against a SPARQL endpoint, and it is the
# ceiling on the whole benchmark: 284 distinct entities resolved from 8,604 sanctions
# targets is the most bridges that can ever exist, however good the private linker is and
# however many private corpora are added.
#
# It runs the same reconciliation twice, once with aliases and once without, and prints
# both. A number is only worth having next to the number it is an improvement on.
#
# The entity index is the expensive part and is cached on the node from earlier jobs.
#
# Usage:
#   sbatch --partition=<p> --qos=<q> cluster/reconcile.sbatch
set -euo pipefail

export WORK="/scratch/$USER/osint-reconcile-$SLURM_JOB_ID"
export CACHE="/scratch/$USER/osint-cache"
export UV_INSTALL_DIR="$WORK/bin" UV_CACHE_DIR="$WORK/uv-cache"
export UV_PYTHON_INSTALL_DIR="$WORK/uv-python" OSINT_DATA="$WORK/data"
export OSINT_RAW="$CACHE/raw" HF_HOME="$CACHE/hf"
export PATH="$UV_INSTALL_DIR:$PATH" PYTHONUNBUFFERED=1
export NLTK_DISABLE_IMPORT_SECURITY=1

echo "node    $(hostname)"
echo "cache   $CACHE ($(du -sh "$CACHE" 2>/dev/null | cut -f1) already here)"
mkdir -p "$UV_INSTALL_DIR" "$OSINT_RAW"
trap 'rm -rf "$WORK"; echo "cleaned up $WORK"' EXIT

echo "=== installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1

echo "=== cloning ==="
git clone -q https://github.com/Jibril-Frej/osint-benchmark.git "$WORK/repo"
cd "$WORK/repo"
uv sync -q
uv run pytest -q

echo "=== 1/2 sources ==="
uv run python pipeline/01_sources.py sanctions wikipedia_index

# Two parameter sets differing in one line, so the comparison is of the setting and not of
# anything else that might have drifted between two hand-written configs.
cp -r config "$WORK/config-aliases"
cp -r config "$WORK/config-titles-only"
sed -i 's/^aliases = true$/aliases = false/' "$WORK/config-titles-only/link.toml"
grep -H "^aliases" "$WORK"/config-*/link.toml

for variant in titles-only aliases; do
  echo "=== 2/2 reconcile: $variant ==="
  OSINT_CONFIG="$WORK/config-$variant" \
    uv run python pipeline/02_link.py sanctions --index wikipedia_index
  cp -f "$OSINT_DATA/links/sanctions.jsonl" "$WORK/sanctions-$variant.jsonl"
done

echo "=== what aliases bought ==="
uv run python - <<'PY'
import json, os
from pathlib import Path

def entities(path):
    rows = [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]
    return rows, {e["qid"] for r in rows for e in r["entities"]}

work = os.environ["WORK"]
rows_a, titles = entities(f"{work}/sanctions-titles-only.jsonl")
rows_b, both = entities(f"{work}/sanctions-aliases.jsonl")
linked_a = sum(1 for r in rows_a if r["entities"])
linked_b = sum(1 for r in rows_b if r["entities"])
print(f"records:            {len(rows_a)}")
print(f"linked, titles:     {linked_a}")
print(f"linked, +aliases:   {linked_b}")
print(f"entities, titles:   {len(titles)}")
print(f"entities, +aliases: {len(both)}")
print(f"gained:             {len(both - titles)}")
# Not a subset by construction: an alias match can make a name ambiguous that a title
# match resolved cleanly, and the caller drops anything ambiguous.
print(f"lost:               {len(titles - both)}")
PY
echo done
