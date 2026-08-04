#!/usr/bin/env bash
#SBATCH --job-name=osint-link
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.log
#
# Link the private corpus with ReFinED, and measure what that buys over the no-model
# fallback.
#
# This is the step the benchmark's ceiling is set by: an entity the linker does not find is
# an entity no question can bridge on. The dictionary matcher found 3 cables in 2,000,
# which is not a benchmark. So the run links the same slice twice -- once by title match,
# once with ReFinED -- and prints both, because "ReFinED found more" is only worth
# believing next to the number it is more than.
#
# Everything lives in the node's own /scratch: the repo, the corpora, the ReFinED weights
# and the entity index. Nothing lands on the shared filesystem except the counts and a
# sample of the links, copied to $HOME at the end.
#
# The GPU is requested at submit time, not here: sites differ on whether a bare count is
# accepted or a type is required (this one rejects --gpus=1 and wants --gpus=a100).
#
# Usage:
#   sbatch --partition=<p> --qos=<q> --gpus=<type> cluster/link.sbatch
#
#   OSINT_LINK_STRIDE   take every Nth cable (default 12, ~21k of 251k)
#   OSINT_LINK_INDEX    entity index to scope to (default wikipedia_index, the full 7.5M)
set -euo pipefail

STRIDE="${OSINT_LINK_STRIDE:-12}"
INDEX="${OSINT_LINK_INDEX:-wikipedia_index}"
export WORK="/scratch/$USER/osint-link-$SLURM_JOB_ID"
export OUT="$HOME/osint-link-$SLURM_JOB_ID"

export UV_INSTALL_DIR="$WORK/bin" UV_CACHE_DIR="$WORK/uv-cache"
export UV_PYTHON_INSTALL_DIR="$WORK/uv-python" OSINT_DATA="$WORK/data"
export PATH="$UV_INSTALL_DIR:$PATH" PYTHONUNBUFFERED=1
# ReFinED downloads ~2 GB of weights and the entity description store on first use, and
# truecase carries an NLTK tokenizer. Both go to the node, not to $HOME.
export HF_HOME="$WORK/hf" TORCH_HOME="$WORK/torch" NLTK_DATA="$WORK/nltk"
export TRANSFORMERS_VERBOSITY=error
# NLTK refuses to import anything resolving inside the working directory or below it, and
# uv puts the virtualenv at .venv inside the project -- so from the repo root NLTK blocks
# its own dependencies and takes ReFinED down with it. Set here as well as in the library,
# because the checks below import truecase and ReFinED directly. See allow_nltk_imports().
export NLTK_DISABLE_IMPORT_SECURITY=1

echo "node    $(hostname)"
echo "stride  $STRIDE"
echo "index   $INDEX"
echo "scratch $(df -h /scratch | tail -1 | awk '{print $4}') free"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
mkdir -p "$UV_INSTALL_DIR" "$NLTK_DATA" "$OUT"

cleanup() {
  rm -rf "$WORK"
  echo "cleaned up $WORK (results kept in $OUT)"
}
trap cleanup EXIT

echo "=== installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
uv --version

echo "=== cloning ==="
git clone -q https://github.com/Jibril-Frej/osint-benchmark.git "$WORK/repo"
cd "$WORK/repo"

echo "=== installing, with the linker ==="
# --extra link pulls torch, transformers 4.46 and ReFinED itself. This is the install that
# historically cost the most time; if it fails, it fails here and not after an hour of
# downloading corpora.
uv sync -q --extra link
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
uv run python -c "from refined.inference.processor import Refined; print('refined imports')"
# truecase tokenises with NLTK, which fetches its tokenizer on first use.
uv run python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"
uv run python -c "import truecase; print(truecase.get_true_case('THE SECRETARY MET RICE'))"
uv run pytest -q

echo "=== does the linker actually build? ==="
# Before any corpus is downloaded. Importing ReFinED is not the same as loading a model:
# job 13338 imported it fine, spent 24 minutes fetching corpora, and then failed building
# the tokenizer. This check needs no data and settles it in three minutes.
uv run python pipeline/02_link.py --check --device cuda

echo "=== 1/5 sources ==="
uv run python pipeline/01_sources.py cablegate sanctions "$INDEX"

echo "=== 2/5 link: the public side ==="
uv run python pipeline/02_link.py sanctions --index "$INDEX"

echo "=== 2/5 link: the private side, by title match (the baseline to beat) ==="
uv run python pipeline/02_link.py cablegate --index "$INDEX" \
  --dictionary --restrict-to sanctions --stride "$STRIDE"
cp "$OSINT_DATA/links/cablegate.jsonl" "$WORK/cablegate.dictionary.jsonl"

echo "=== 2/5 link: the private side, with ReFinED ==="
# No --restrict-to: the point of a model linker is that it does not need to be told what
# to look for. --device cuda, or this silently runs on the CPUs and takes a day.
uv run python pipeline/02_link.py cablegate --index "$INDEX" --device cuda --stride "$STRIDE"

echo "=== what the two linkers found on the same cables ==="
uv run python - <<'PY'
import json, os, collections
from pathlib import Path

def read(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]

for label, path in (
    ("title match", os.environ["WORK"] + "/cablegate.dictionary.jsonl"),
    ("ReFinED", os.environ["OSINT_DATA"] + "/links/cablegate.jsonl"),
):
    rows = read(path)
    linked = [r for r in rows if r["entities"]]
    qids = collections.Counter(e["qid"] for r in rows for e in r["entities"])
    forms = collections.Counter(e["surface_form"] for r in rows for e in r["entities"])
    print(f"\n--- {label} ---")
    print(f"{len(rows)} cables, {len(linked)} linked ({100 * len(linked) / max(len(rows), 1):.1f}%)")
    print(f"{sum(qids.values())} mentions over {len(qids)} distinct entities")
    print("most-named surface forms:", forms.most_common(15))
PY

echo "=== 3/5 graph ==="
uv run python pipeline/03_graph.py

echo "=== 4/5 public evidence ==="
uv run python pipeline/04_public.py --index "$INDEX" --limit 200

echo "=== 5/5 pair ==="
uv run python pipeline/05_pair.py

echo "=== keeping the results ==="
cp -f "$OSINT_DATA/links/cablegate.jsonl" "$OUT/cablegate-refined.jsonl" 2>/dev/null || true
cp -f "$OSINT_DATA/links/sanctions.jsonl" "$OUT/sanctions-links.jsonl" 2>/dev/null || true
cp -f "$OSINT_DATA/graph/bridges.jsonl" "$OUT/bridges.jsonl" 2>/dev/null || true
cp -f "$OSINT_DATA/pairs/pairs.jsonl" "$OUT/pairs.jsonl" 2>/dev/null || true
cp -f "$WORK/cablegate.dictionary.jsonl" "$OUT/" 2>/dev/null || true
ls -la "$OUT"
echo done
