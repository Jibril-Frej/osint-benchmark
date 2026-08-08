#!/usr/bin/env bash
#SBATCH --job-name=osint-typed
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.log
#
# Build the questions whose answers are computed rather than written: association and
# resolution, over the two confidential corpora.
#
# Four stages, and each one's output is copied to $HOME so the next run can skip it. They
# cost very different things, which is why they can be skipped separately:
#
#   link    GPU, ~1.5 h at stride 3 (ReFinED over ~84k cables). Dodis needs no linker.
#   facts   CPU and network, ~1.5 h. Every linked entity's Wikidata record and article
#           lead, plus a second hop for the organisations the association type answers
#           with. Nothing on the GPU: reuse this if you can.
#   phrase  GPU, minutes per hundred questions.
#   measure GPU, the necessity ablations and the review page.
#
# The facts stage holds two A100s idle for an hour and a half if it runs in the same job as
# the others. That is the price of one job rather than three, and it is why OSINT_FACTS
# exists: run this once, then point every later run at what it produced.
#
# Usage:
#   sbatch --partition=<p> --qos=<q> --gpus=a100:2 cluster/typed.sbatch
#
#   OSINT_LINK_STRIDE   take every Nth cable (default 3, ~84k of 251k)
#   OSINT_LINKS         a directory holding cablegate.jsonl and dodis.jsonl from an earlier
#                       run; skips the linking stage entirely
#   OSINT_FACTS         a directory holding wikidata.jsonl and articles.jsonl. On a model
#                       run it skips the fetching stage; with OSINT_STOP_AFTER=facts it is
#                       instead the partial slice to carry on from, so a fetch that ran out
#                       of time can be finished rather than repeated. Only valid with the
#                       link files it was fetched for -- a wider link set needs a wider
#                       slice, and a narrower one carries entities no document names.
#   OSINT_PER_TYPE      candidates of each type to phrase (default 150)
#   OSINT_TYPES         which types to build (default: all five)
#   OSINT_GEN_MODEL     the phraser (default Qwen/QwQ-32B)
#   OSINT_WORKERS       requests in flight at once (default 8). The server batches
#                       continuously; one at a time leaves both GPUs mostly idle.
#   OSINT_STUB          set to 1 to run the model stages against the scripted stand-in and
#                       skip vLLM entirely. Submit this WITHOUT --gpus: what it answers is
#                       how many candidates the corpora yield and where the rest were lost,
#                       which is what decides the stride for a real run, and no part of that
#                       needs a model. The questions it writes are placeholders.
#   OSINT_STOP_AFTER    stop once a stage is done: "link" or "facts". This is how the run
#                       avoids holding two A100s through three hours of network: submit
#                       with --gpus for the linking, then with no GPU at all for the
#                       fetching, then with --gpus again for the model stages.
set -euo pipefail

STRIDE="${OSINT_LINK_STRIDE:-3}"
PER_TYPE="${OSINT_PER_TYPE:-150}"
TYPES="${OSINT_TYPES:-association,resolution,chronology,posture,event}"
MODEL="${OSINT_GEN_MODEL:-Qwen/QwQ-32B}"
WORKERS="${OSINT_WORKERS:-8}"
STUB="${OSINT_STUB:-}"
STOP_AFTER="${OSINT_STOP_AFTER:-}"
LINKS="${OSINT_LINKS:-}"
FACTS="${OSINT_FACTS:-}"
PORT="${OSINT_SERVE_PORT:-8000}"
export WORK="/scratch/$USER/osint-typed-$SLURM_JOB_ID"
export OUT="$HOME/osint-typed-$SLURM_JOB_ID"
export CACHE="/scratch/$USER/osint-cache"

export UV_INSTALL_DIR="$WORK/bin" UV_CACHE_DIR="$WORK/uv-cache"
export UV_PYTHON_INSTALL_DIR="$WORK/uv-python" OSINT_DATA="$WORK/data"
export OSINT_RAW="$CACHE/raw" HF_HOME="$CACHE/hf"
export PATH="$UV_INSTALL_DIR:$PATH" PYTHONUNBUFFERED=1
export TORCH_HOME="$WORK/torch" NLTK_DATA="$WORK/nltk"
export NLTK_DISABLE_IMPORT_SECURITY=1 TRANSFORMERS_VERBOSITY=error
# The Dodis text is the OCR of the scans, not the editorial summary: the summary is about
# 266 characters, which is a caption rather than a document to build a question from.
export OSINT_DODIS_OCR="$HOME/dodis_ocr" OSINT_DODIS_NT="$HOME/dodis-lod.nt"
if [ -r "$HOME/.config/hf-token" ]; then
  HF_TOKEN="$(tr -d '[:space:]' < "$HOME/.config/hf-token")"
  export HF_TOKEN
  echo "hugging face token: loaded from ~/.config/hf-token"
fi
# Both tensor-parallel workers are on this node, so their collectives go over loopback, and
# these A100s are PCIe cards with no NVLink -- peer-to-peer over PCIe hangs here.
export GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo VLLM_HOST_IP=127.0.0.1
export NCCL_P2P_DISABLE=1

echo "node    $(hostname)"
echo "model   $MODEL"
echo "types   $TYPES, $PER_TYPE candidates each, $WORKERS requests in flight"
echo "links   ${LINKS:-none supplied, cablegate will be relinked at stride $STRIDE}"
echo "facts   ${FACTS:-none supplied, the whole slice will be fetched}"
echo "stop     ${STOP_AFTER:-nothing, the whole chain runs}"
echo "models   ${STUB:+STUB -- no model is served, the questions are placeholders}${STUB:-$MODEL served with vLLM}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
mkdir -p "$UV_INSTALL_DIR" "$NLTK_DATA" "$OUT" "$OSINT_RAW" "$HF_HOME" \
         "$OSINT_DATA/links" "$OSINT_DATA/facts"

SERVER_PID=""
keep() {
  # After every stage, so a job killed at the time limit still leaves what it finished.
  for path in links/cablegate.jsonl links/dodis.jsonl facts/wikidata.jsonl \
              facts/articles.jsonl items/accepted.jsonl items/rejected.jsonl \
              items/measured.jsonl review.html; do
    cp -f "$OSINT_DATA/$path" "$OUT/$(basename "$path")" 2>/dev/null || true
  done
}
cleanup() {
  keep
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  cp -f "$WORK/vllm.log" "$OUT/vllm.log" 2>/dev/null || true
  cp -f "$WORK/model-calls.jsonl" "$OUT/model-calls.jsonl" 2>/dev/null || true
  rm -rf "$WORK"
  echo "cleaned up $WORK (results in $OUT)"
}
trap cleanup EXIT

echo "=== installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1

echo "=== cloning ==="
git clone -q https://github.com/Jibril-Frej/osint-benchmark.git "$WORK/repo"
cd "$WORK/repo"
uv sync -q --extra link
uv run pytest -q

export OSINT_CONFIG="$WORK/repo/config/qwq"
export OSINT_SERVED_MODEL="$MODEL"
export OSINT_TRANSCRIPT="$WORK/model-calls.jsonl"

echo "=== 1/6 sources ==="
# parliament as well as the two confidential corpora: the chronology and posture types are
# built on the join between a cable and a parliamentary business item, so both sides have to
# be parsed, not merely linked.
uv run python pipeline/01_sources.py cablegate dodis parliament ucdp wikipedia_index

echo "=== 2/6 link ==="
if [ -n "$LINKS" ]; then
  # Whatever link files the directory holds, rather than a fixed list: a run that adds a
  # corpus should not need this script edited, and a missing one is visible in the count
  # printed below.
  cp "$LINKS"/*.jsonl "$OSINT_DATA/links/" 2>/dev/null || true
  cp "$LINKS"/*.provenance.json "$OSINT_DATA/links/" 2>/dev/null || true
  # The slice files live in the same directory on a reused run; they are not link files.
  rm -f "$OSINT_DATA/links/wikidata.jsonl" "$OSINT_DATA/links/articles.jsonl"
  echo "reused the link files in $LINKS"
  # The slice is only valid for the link set it was fetched against, and --resume decides
  # what to fetch by comparing the two. A mismatch is silent otherwise.
  wc -l "$OSINT_DATA"/links/*.jsonl
else
  uv run python pipeline/02_link.py --check --device cuda
  uv run python pipeline/02_link.py cablegate --index wikipedia_index --device cuda \
    --stride "$STRIDE"
  # Catalogued by archivists: the entities come with the documents, so no model runs. The
  # names still have to be reconciled against Wikidata, which is network-bound, not GPU.
  uv run python pipeline/02_link.py dodis --index wikipedia_index
  # Tabular and country-keyed, so this is a lookup rather than a linker. It bridges nothing,
  # and the event type does not need it to: it matches on an anchor and a date window.
  uv run python pipeline/02_link.py ucdp --index wikipedia_index
fi
wc -l "$OSINT_DATA"/links/*.jsonl
keep
if [ "$STOP_AFTER" = link ]; then echo "stopping after the link stage; results in $OUT"; exit 0; fi

echo "=== 3/6 facts: every linked entity, and what they point at ==="
if [ -n "$FACTS" ]; then
  # Each file separately and none of them required: a fetch that ran out of time during the
  # article stage left a wikidata.jsonl and no articles.jsonl, and that is exactly the case
  # this branch exists to carry on from. Insisting on both would fail the job that needs it.
  for name in wikidata.jsonl articles.jsonl; do
    cp "$FACTS/$name" "$OSINT_DATA/facts/$name" 2>/dev/null \
      && echo "reusing $FACTS/$name ($(wc -l < "$FACTS/$name") records)" \
      || echo "no $name in $FACTS"
  done
  cp "$FACTS"/*.provenance.json "$OSINT_DATA/facts/" 2>/dev/null || true
fi
if [ -z "$FACTS" ] || [ "$STOP_AFTER" = facts ]; then
  # --scope linked, not bridges: an association is two people in ONE document and neither
  # need appear in a public corpus, so scoping to bridges would starve the type entirely.
  # --neighbours adds the second hop, which is where the answers live.
  # Four in flight. Measured against the live APIs: 1.45x on the slice and 1.6x on the
  # article leads, both limited by the far end rather than the round trip.
  # --resume so a job that runs out of time in the middle of three hours of network keeps
  # what it fetched: rerun it with OSINT_FACTS pointing at its own output.
  uv run python pipeline/04_public.py --index wikipedia_index \
    --scope linked --min-confidence 0.9 --neighbours --workers 4 --resume
fi
wc -l "$OSINT_DATA"/facts/*.jsonl
keep
if [ "$STOP_AFTER" = facts ]; then echo "stopping after the facts stage; results in $OUT"; exit 0; fi

if [ -n "$STUB" ]; then
  echo "=== 4/6 phrase with the scripted stand-in (no model, no GPU) ==="
  uv run python pipeline/06_generate.py --types "$TYPES" --per-type "$PER_TYPE" --stub
  keep
  echo "=== 5/6 necessity with the stand-in ==="
  uv run python pipeline/07_necessity.py --stub
  keep
  echo "=== 6/6 review page ==="
  uv run python pipeline/08_review.py
  keep
  ls -la "$OUT"
  echo "done (STUB: the questions are placeholders; the counts above are real)"
  exit 0
fi

echo "=== installing vLLM (node-local) ==="
uv venv "$WORK/vllm-venv" >/dev/null
VLLM_PY="$WORK/vllm-venv/bin/python"
uv pip install --python "$VLLM_PY" -q vllm ninja
export PATH="$WORK/vllm-venv/bin:$PATH"
"$VLLM_PY" -c "import vllm; print('vllm', vllm.__version__)"

echo "=== fetching the weights ==="
"$VLLM_PY" - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
print("weights at", snapshot_download(sys.argv[1], max_workers=8), flush=True)
PY

echo "=== starting vLLM: $MODEL on 2 GPUs ==="
"$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 2 --max-model-len 24576 --gpu-memory-utilization 0.92 \
  --disable-custom-all-reduce \
  > "$WORK/vllm.log" 2>&1 &
SERVER_PID=$!

echo "=== waiting for it to answer ==="
deadline=$(( SECONDS + 1800 ))
until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM exited before becoming ready:" >&2; tail -60 "$WORK/vllm.log" >&2; exit 1
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "vLLM not ready in 1800s of loading:" >&2; tail -60 "$WORK/vllm.log" >&2; exit 1
  fi
  sleep 20
done
echo "ready after ${SECONDS}s"
export OSINT_MODEL_ENDPOINT="http://127.0.0.1:$PORT"

echo "=== 4/6 phrase, $PER_TYPE of each type ==="
uv run python pipeline/06_generate.py --types "$TYPES" --per-type "$PER_TYPE" \
  --workers "$WORKERS"
keep

echo "=== control: can the solver answer when the evidence contains the answer? ==="
# Before any necessity number is worth reading. A solver that fails this makes every
# question look perfectly necessary.
uv run python pipeline/07_necessity.py --control || echo "CONTROL FAILED -- necessity below is not trustworthy"

echo "=== 5/6 necessity ==="
uv run python pipeline/07_necessity.py --workers "$WORKERS"
keep

echo "=== 6/6 review page ==="
uv run python pipeline/08_review.py
keep

echo "=== what came out ==="
uv run python - <<'PY' || true
import collections, json, os, pathlib

def read(name):
    path = pathlib.Path(os.environ["OSINT_DATA"]) / name
    return [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []

for name in ("items/accepted.jsonl", "items/rejected.jsonl"):
    rows = read(name)
    kinds = collections.Counter(r.get("question_type") for r in rows)
    print(f"{name}: {len(rows)} {dict(kinds)}")
gates = collections.Counter(
    gate for row in read("items/rejected.jsonl")
    for gate, ok in (row.get("gates") or {}).items() if not ok
)
print("gate failures:", dict(gates))
measured = read("items/measured.jsonl")
need = collections.Counter(
    tuple((row.get("necessity") or {}).get(k) for k in ("closed_book", "public_only", "private_only"))
    for row in measured
)
print(f"{len(measured)} measured, (closed, public, private) -> count: {dict(need)}")
PY
ls -la "$OUT"
echo done
