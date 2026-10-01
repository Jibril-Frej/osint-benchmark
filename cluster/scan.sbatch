#!/usr/bin/env bash
#SBATCH --job-name=osint-scan
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log
#
# Build the page-scan questions (office_holder, addressee) and measure their necessity with
# the image in front of a vision model.
#
# Two models are served at once, because step 7 with --vision asks both in the same run:
# QwQ-32B writes the questions and answers every text condition, Qwen2.5-VL-7B answers the
# private-only condition from the scan. QwQ gets every GPU but the last, tensor-parallel
# across them; the vision model gets the last. So the GPU count decides the layout:
#
#   --gpus=h200:2   QwQ on one H200, the vision model on the other
#   --gpus=a100:3   QwQ across two A100s (it does not fit on one), the vision model on a third
#
# Two A100s are not enough: QwQ alone fills both.
#
# Needs no link files and no facts slice. The candidates come from the Dodis records, the
# PDFs and live Wikidata lookups, which is why this is its own job rather than a type in
# typed.sbatch.
#
# Usage:
#   sbatch --partition=<p> --qos=<q> --gpus=h200:2 \
#     --export=ALL,OSINT_DODIS_SCANS=<dir of dodis-<id>.pdf> cluster/scan.sbatch
#
#   OSINT_DODIS_SCANS   the Dodis PDFs (default ~/dodis_scans)
#   OSINT_PER_TYPE      candidates of each type to phrase (default 100)
#   OSINT_WORKERS       requests in flight at once (default 8)
set -euo pipefail

PER_TYPE="${OSINT_PER_TYPE:-100}"
WORKERS="${OSINT_WORKERS:-8}"
PORT="${OSINT_SERVE_PORT:-8000}"
VISION_PORT=$(( PORT + 1 ))
export OSINT_DODIS_SCANS="${OSINT_DODIS_SCANS:-$HOME/dodis_scans}"
export WORK="/scratch/$USER/osint-scan-$SLURM_JOB_ID"
export OUT="$HOME/osint-scan-$SLURM_JOB_ID"
export CACHE="/scratch/$USER/osint-cache"

export UV_INSTALL_DIR="$WORK/bin" UV_CACHE_DIR="$WORK/uv-cache"
export UV_PYTHON_INSTALL_DIR="$WORK/uv-python" OSINT_DATA="$WORK/data"
export OSINT_RAW="$CACHE/raw" HF_HOME="$CACHE/hf"
export PATH="$UV_INSTALL_DIR:$PATH" PYTHONUNBUFFERED=1
export OSINT_DODIS_OCR="$HOME/dodis_ocr" OSINT_DODIS_NT="$HOME/dodis-lod.nt"
if [ -r "$HOME/.config/hf-token" ]; then
  HF_TOKEN="$(tr -d '[:space:]' < "$HOME/.config/hf-token")"
  export HF_TOKEN
  echo "hugging face token: loaded from ~/.config/hf-token"
fi
export GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo VLLM_HOST_IP=127.0.0.1
# PCIe A100s have no NVLink, and peer-to-peer over PCIe hangs. Harmless on the H200s.
export NCCL_P2P_DISABLE=1

# The GPUs Slurm gave this job, as the indices CUDA_VISIBLE_DEVICES uses.
IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:?no GPUs allocated: submit with --gpus}"
if [ "${#GPUS[@]}" -lt 2 ]; then
  echo "need at least two GPUs, one for each model; got ${#GPUS[@]}" >&2; exit 1
fi
VISION_GPU="${GPUS[-1]}"
TEXT_GPUS=$(IFS=,; echo "${GPUS[*]:0:${#GPUS[@]}-1}")
TEXT_TP=$(( ${#GPUS[@]} - 1 ))

echo "node    $(hostname)"
echo "scans   $OSINT_DODIS_SCANS ($(find "$OSINT_DODIS_SCANS" -name 'dodis-*.pdf' | wc -l) PDFs)"
echo "types   office_holder,addressee, $PER_TYPE candidates each, $WORKERS requests in flight"
echo "layout  QwQ on GPU $TEXT_GPUS (tp=$TEXT_TP), vision model on GPU $VISION_GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
mkdir -p "$UV_INSTALL_DIR" "$OUT" "$OSINT_RAW" "$HF_HOME"

PIDS=()
keep() {
  # After every stage, so a job killed at the time limit still leaves what it finished.
  for path in items/accepted.jsonl items/rejected.jsonl items/measured.jsonl \
              facts/records.jsonl review.html; do
    cp -f "$OSINT_DATA/$path" "$OUT/$(basename "$path")" 2>/dev/null || true
  done
  # The scans the questions are asked over, which accepted.jsonl names relative to items/.
  if [ -d "$OSINT_DATA/items/images" ]; then
    mkdir -p "$OUT/images" && cp -f "$OSINT_DATA"/items/images/* "$OUT/images/" 2>/dev/null || true
  fi
}
cleanup() {
  keep
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${PIDS[@]}"; do
    for _ in $(seq 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  done
  cp -f "$WORK"/vllm-*.log "$OUT/" 2>/dev/null || true
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
uv sync -q
uv run pytest -q

export OSINT_CONFIG="$WORK/repo/config/scan"
TEXT_MODEL=$(uv run python -c "from osint_benchmark.models import settings; print(settings.load('solver').model)")
VISION_MODEL=$(uv run python -c "from osint_benchmark.models import settings; print(settings.load('vision_solver').model)")
export OSINT_TRANSCRIPT="$WORK/model-calls.jsonl"
echo "models  $TEXT_MODEL (text), $VISION_MODEL (vision)"

echo "=== 1/4 sources ==="
uv run python pipeline/01_sources.py dodis

echo "=== installing vLLM (node-local) ==="
uv venv "$WORK/vllm-venv" >/dev/null
VLLM_PY="$WORK/vllm-venv/bin/python"
uv pip install --python "$VLLM_PY" -q vllm ninja
export PATH="$WORK/vllm-venv/bin:$PATH"
"$VLLM_PY" -c "import vllm; print('vllm', vllm.__version__)"

echo "=== fetching the weights ==="
"$VLLM_PY" - "$TEXT_MODEL" "$VISION_MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
for model in sys.argv[1:]:
    print("weights at", snapshot_download(model, max_workers=8), flush=True)
PY

echo "=== starting vLLM: $TEXT_MODEL and $VISION_MODEL ==="
CUDA_VISIBLE_DEVICES="$TEXT_GPUS" "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model "$TEXT_MODEL" --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size "$TEXT_TP" --max-model-len 24576 --gpu-memory-utilization 0.92 \
  --disable-custom-all-reduce \
  > "$WORK/vllm-text.log" 2>&1 &
PIDS+=($!)
# A scan page is a few thousand image tokens at Qwen2.5-VL's default resolution.
CUDA_VISIBLE_DEVICES="$VISION_GPU" "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model "$VISION_MODEL" --host 127.0.0.1 --port "$VISION_PORT" \
  --max-model-len 32768 --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{"image": 1}' \
  > "$WORK/vllm-vision.log" 2>&1 &
PIDS+=($!)

echo "=== waiting for both to answer ==="
deadline=$(( SECONDS + 1800 ))
for port in "$PORT" "$VISION_PORT"; do
  until curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; do
    for pid in "${PIDS[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "a vLLM server exited before becoming ready:" >&2
        tail -40 "$WORK"/vllm-*.log >&2; exit 1
      fi
    done
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "vLLM not ready in 1800s of loading:" >&2; tail -40 "$WORK"/vllm-*.log >&2; exit 1
    fi
    sleep 20
  done
done
echo "ready after ${SECONDS}s"
export OSINT_MODEL_ENDPOINT="http://127.0.0.1:$PORT"
export OSINT_VISION_ENDPOINT="http://127.0.0.1:$VISION_PORT"

echo "=== 2/4 phrase, $PER_TYPE of each type ==="
uv run python pipeline/06_generate.py --types office_holder,addressee --per-type "$PER_TYPE" \
  --workers "$WORKERS"
keep

echo "=== control: can the solver answer when the evidence contains the answer? ==="
uv run python pipeline/07_necessity.py --control || echo "CONTROL FAILED -- necessity below is not trustworthy"

echo "=== 3/4 necessity, private-only on the image ==="
uv run python pipeline/07_necessity.py --vision --workers "$WORKERS"
keep

echo "=== 4/4 review page ==="
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
    print(f"{name}: {len(rows)} {dict(collections.Counter(r.get('question_type') for r in rows))}")
measured = read("items/measured.jsonl")
keys = ("closed_book", "public_only", "private_only", "ocr_only")
need = collections.Counter(
    tuple((row.get("necessity") or {}).get(k) for k in keys) for row in measured
)
print(f"{len(measured)} measured, {keys} -> count: {dict(need)}")
PY
ls -la "$OUT"
echo done
