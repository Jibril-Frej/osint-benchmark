#!/usr/bin/env bash
#SBATCH --job-name=osint-review
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.log
#
# Render one review page from every question a set of runs produced.
#
# No GPU and no model: the questions and their necessity verdicts already exist, and this
# only lays them beside the documents they rest on. It needs step 1 because the page shows
# the evidence, and the evidence lives in the corpora -- which is also why this runs on the
# cluster rather than the workstation. The Dodis OCR is here and not there, and a page
# rendered without it says "(not available)" under a third of the questions while still
# looking complete.
#
# Usage:
#   sbatch --partition=<p> --qos=<q> \
#     --export=ALL,OSINT_ITEMS="$HOME/a/measured.jsonl $HOME/b/measured.jsonl" \
#     cluster/review.sbatch
#
#   OSINT_ITEMS      space-separated measured.jsonl files to merge, in order
#   OSINT_MODELS     path to a FILE holding what produced them, shown at the top of the
#                    page. A path and not the text: sbatch --export splits its value on
#                    commas, so a sentence passed directly arrives truncated at the first
#                    one -- which is how the page came to say only "phrased by Qwen/QwQ-32B".
#   OSINT_FACTS      a directory holding articles.jsonl. The association and resolution
#                    types cite an entity's Wikipedia lead as their public evidence, and
#                    without it 296 of 515 questions showed "(not available)" on the public
#                    side while the page still looked complete.
#   OSINT_CALIBRATE  also render the calibration page over this many questions. It asks a
#                    different and prior question from the review page -- not "is this
#                    question good" but "can the pipeline's own verdicts be believed" --
#                    and neither the judge nor the ablations has ever been checked against
#                    a person.
set -euo pipefail

ITEMS="${OSINT_ITEMS:?set OSINT_ITEMS to one or more measured.jsonl paths}"
export WORK="/scratch/$USER/osint-review-$SLURM_JOB_ID"
export OUT="$HOME/osint-review-$SLURM_JOB_ID"
export CACHE="/scratch/$USER/osint-cache"

export UV_INSTALL_DIR="$WORK/bin" UV_CACHE_DIR="$WORK/uv-cache"
export UV_PYTHON_INSTALL_DIR="$WORK/uv-python" OSINT_DATA="$WORK/data"
export OSINT_RAW="$CACHE/raw" HF_HOME="$CACHE/hf"
export PATH="$UV_INSTALL_DIR:$PATH" PYTHONUNBUFFERED=1
export NLTK_DISABLE_IMPORT_SECURITY=1
export OSINT_DODIS_OCR="$HOME/dodis_ocr" OSINT_DODIS_NT="$HOME/dodis-lod.nt"

echo "node   $(hostname)"
echo "items  $ITEMS"
mkdir -p "$UV_INSTALL_DIR" "$OUT" "$OSINT_RAW" "$OSINT_DATA/items"
trap 'cp -f "$OSINT_DATA"/review.html "$OSINT_DATA"/calibration.html "$OUT/" 2>/dev/null || true; rm -rf "$WORK"; echo "results in $OUT"' EXIT

echo "=== installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1

echo "=== cloning ==="
git clone -q https://github.com/Jibril-Frej/osint-benchmark.git "$WORK/repo"
cd "$WORK/repo"
uv sync -q
uv run pytest -q

# One file, deduplicated on item_id: a question rebuilt by a later run should appear once,
# and the later run's verdicts are the ones to keep.
cat $ITEMS > "$WORK/all.jsonl"
uv run python - <<'PY'
import json, os, pathlib
seen, kept = {}, 0
for line in pathlib.Path(os.environ["WORK"], "all.jsonl").open(encoding="utf-8"):
    row = json.loads(line)
    seen[row["item_id"]] = row
    kept += 1
out = pathlib.Path(os.environ["OSINT_DATA"], "items", "measured.jsonl")
with out.open("w", encoding="utf-8") as fh:
    for row in seen.values():
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
# The page shows which models produced the questions, and reads that from the sidecar. A
# merge of several runs has no single one, so it is written here from what the runs declared
# -- without it the page silently omits the block and looks complete.
note_file = os.environ.get("OSINT_MODELS", "")
note = pathlib.Path(note_file).read_text(encoding="utf-8").strip() if note_file else ""
if note:
    pathlib.Path(str(out) + ".provenance.json").write_text(
        json.dumps(
            {
                "output": str(out),
                "kind": "derived",
                "source": os.environ.get("OSINT_ITEMS", ""),
                "source_fields": ["item_id", "question", "answer", "necessity"],
                "kept": {"item_id": "item_id"},
                "dropped": {},
                "rows_in": kept,
                "rows_out": len(seen),
                "note": note,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
print(f"{kept} questions in, {len(seen)} distinct -> {out}")
PY

if [ -n "${OSINT_FACTS:-}" ]; then
  mkdir -p "$OSINT_DATA/facts"
  cp "$OSINT_FACTS/articles.jsonl" "$OSINT_DATA/facts/" 2>/dev/null \
    && echo "article leads: $(wc -l < "$OSINT_DATA/facts/articles.jsonl")" \
    || echo "WARNING: no articles.jsonl in $OSINT_FACTS; enwiki evidence will be missing" >&2
fi

echo "=== 1/2 sources (the page shows the documents) ==="
uv run python pipeline/01_sources.py cablegate dodis parliament ucdp

echo "=== 2/2 review page ==="
uv run python pipeline/08_review.py
cp -f "$OSINT_DATA/review.html" "$OUT/" 2>/dev/null || true
if [ -n "${OSINT_CALIBRATE:-}" ]; then
  echo "=== calibration page ==="
  uv run python pipeline/08_review.py --calibrate "$OSINT_CALIBRATE"
  cp -f "$OSINT_DATA/calibration.html" "$OUT/" 2>/dev/null || true
fi

uv run python - <<'PY'
import collections, json, os, pathlib
rows = [
    json.loads(x)
    for x in pathlib.Path(os.environ["OSINT_DATA"], "items", "measured.jsonl").open(
        encoding="utf-8"
    )
]
need = collections.Counter()
total = collections.Counter()
for row in rows:
    necessity = row.get("necessity") or {}
    total[row["question_type"]] += 1
    if not any(necessity.get(k) for k in ("closed_book", "public_only", "private_only")):
        need[row["question_type"]] += 1
print(f"{'type':14s} {'needs both':>12s}")
for kind in sorted(total):
    print(f"{kind:14s} {need[kind]:5d}/{total[kind]:<5d} {need[kind] / total[kind]:.0%}")
print(f"{'TOTAL':14s} {sum(need.values()):5d}/{len(rows):<5d} {sum(need.values()) / len(rows):.0%}")
sources = collections.Counter(
    e["source"] for row in rows for e in row["evidence"] if e["side"] == "private"
)
print("private corpora:", dict(sources))
PY
ls -la "$OUT"
echo done
