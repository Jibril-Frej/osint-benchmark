#!/usr/bin/env bash
#SBATCH --job-name=osint-parliament
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.log
#
# Link the Swiss parliamentary record with WikiNeural and mGENRE.
#
# The only source needing a linker that is not ReFinED: the record is German and French.
# It is also the only public source contemporaneous with both private corpora -- from 1978
# against Cablegate's 2003-2010 and Dodis's 1941-2001 -- which makes it the only route to
# a question about timing, and the only public counterpart Dodis has at all. Dodis paired
# against the sanctions list produced zero bridges: a 1947 diplomat is not on a
# contemporary sanctions list.
#
# 200 GB of memory, and that is the reason: the (language, title) -> QID mapping is 3.7 GB
# on disk and roughly 90 GB once loaded. gpu-node2 has 470 GB, gpu-node1 has 1.5 TB.
#
# The linker self-tests the mapping before touching the corpus. A mapping whose key shape
# is not what to_qid tries links nothing and reports no error, and finding that out after
# an hour is the expensive way.
#
# Usage:
#   sbatch --partition=<p> --qos=<q> --gpus=<type> cluster/parliament.sbatch
#
#   OSINT_LINK_LIMIT   link only the first N items, for a first look
set -euo pipefail

LIMIT="${OSINT_LINK_LIMIT:-}"
export WORK="/scratch/$USER/osint-parliament-$SLURM_JOB_ID"
export OUT="$HOME/osint-parliament-$SLURM_JOB_ID"
export CACHE="/scratch/$USER/osint-cache"

export UV_INSTALL_DIR="$WORK/bin" UV_CACHE_DIR="$WORK/uv-cache"
export UV_PYTHON_INSTALL_DIR="$WORK/uv-python" OSINT_DATA="$WORK/data"
export OSINT_RAW="$CACHE/raw" HF_HOME="$CACHE/hf"
export TORCH_HOME="$WORK/torch" NLTK_DATA="$WORK/nltk"
export PATH="$UV_INSTALL_DIR:$PATH" PYTHONUNBUFFERED=1
export NLTK_DISABLE_IMPORT_SECURITY=1 TRANSFORMERS_VERBOSITY=error
export OSINT_MGENRE_MAPPING="$HOME/lang_title2wikidataID.pkl"

echo "node    $(hostname)"
echo "memory  $(free -g | awk '/^Mem:/{print $2}') GB"
echo "mapping $(ls -lh "$OSINT_MGENRE_MAPPING" 2>/dev/null | awk '{print $5}' || echo ABSENT)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
mkdir -p "$UV_INSTALL_DIR" "$NLTK_DATA" "$OUT" "$OSINT_RAW"

keep() {
  cp -f "$OSINT_DATA/links/parliament.jsonl" "$OUT/" 2>/dev/null || true
  cp -f "$OSINT_DATA/links/parliament.jsonl.provenance.json" "$OUT/" 2>/dev/null || true
}
trap 'keep; rm -rf "$WORK"; echo "cleaned up $WORK (results in $OUT)"' EXIT

echo "=== installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1

echo "=== cloning ==="
git clone -q https://github.com/Jibril-Frej/osint-benchmark.git "$WORK/repo"
cd "$WORK/repo"
uv sync -q --extra link
uv run pytest -q

echo "=== 1/2 sources ==="
# The parliamentary record fetches itself from a paged API; the entity index is cached.
uv run python pipeline/01_sources.py parliament wikipedia_index

echo "=== 2/2 link ==="
uv run python pipeline/02_link.py parliament --index wikipedia_index --device cuda \
  ${LIMIT:+--limit "$LIMIT"}
keep

echo "=== what was linked ==="
uv run python - <<'PY' || true
import collections, json, os, pathlib
path = pathlib.Path(os.environ["OSINT_DATA"]) / "links" / "parliament.jsonl"
rows = [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []
linked = [r for r in rows if r["entities"]]
qids = collections.Counter(e["qid"] for r in rows for e in r["entities"])
forms = collections.Counter(e["surface_form"] for r in rows for e in r["entities"])
print(f"{len(rows)} items, {len(linked)} linked ({100 * len(linked) / max(len(rows), 1):.1f}%)")
print(f"{sum(qids.values())} mentions over {len(qids)} distinct entities")
print("most-named:", forms.most_common(15))
PY
echo done
