#!/usr/bin/env bash
#SBATCH --job-name=osint-measure
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log
#
# Measure necessity over questions that already exist, and render the review page.
#
# Steps 7 and 8 only. Writing the questions is the expensive part -- four hours of linking
# and drafting on job 13397 -- and it had already succeeded when the measurement failed on
# a single bad request. Repeating it to recover two stages worth half an hour of model time
# is waste, so this reads an accepted.jsonl and starts there.
#
# It needs step 1, because the ablations re-read the source documents. Which corpora those
# are is read off the items themselves: their evidence ids are namespaced, so an item
# declares what it needs and nothing has to be guessed from whichever link files happen to
# be present.
#
# Usage:
#   sbatch --partition=<p> --qos=<q> --gpus=a100:2 \
#     --export=ALL,OSINT_ITEMS=$HOME/osint-generate-13397/accepted.jsonl cluster/measure.sbatch
set -euo pipefail

ITEMS="${OSINT_ITEMS:?set OSINT_ITEMS to an accepted.jsonl}"
MODEL="${OSINT_GEN_MODEL:-Qwen/QwQ-32B}"
PORT="${OSINT_SERVE_PORT:-8000}"
export WORK="/scratch/$USER/osint-measure-$SLURM_JOB_ID"
export OUT="$HOME/osint-measure-$SLURM_JOB_ID"
export CACHE="/scratch/$USER/osint-cache"

export UV_INSTALL_DIR="$WORK/bin" UV_CACHE_DIR="$WORK/uv-cache"
export UV_PYTHON_INSTALL_DIR="$WORK/uv-python" OSINT_DATA="$WORK/data"
export OSINT_RAW="$CACHE/raw" HF_HOME="$CACHE/hf"
export PATH="$UV_INSTALL_DIR:$PATH" PYTHONUNBUFFERED=1
export TORCH_HOME="$WORK/torch" NLTK_DATA="$WORK/nltk"
export NLTK_DISABLE_IMPORT_SECURITY=1 TRANSFORMERS_VERBOSITY=error
# A gated model needs a token, and it must not be in this repository -- it is public.
# HF_TOKEN rather than a login file: HF_HOME points at node-local scratch, so anything
# `huggingface-cli login` wrote into $HOME would never be found. Never echoed.
if [ -r "$HOME/.config/hf-token" ]; then
  HF_TOKEN="$(tr -d '[:space:]' < "$HOME/.config/hf-token")"
  export HF_TOKEN
  echo "hugging face token: loaded from ~/.config/hf-token"
else
  echo "hugging face token: none (ungated models only)"
fi
export GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo VLLM_HOST_IP=127.0.0.1
export NCCL_P2P_DISABLE=1

echo "node    $(hostname)"
echo "items   $ITEMS ($(wc -l < "$ITEMS") questions)"
echo "cache   $CACHE ($(du -sh "$CACHE" 2>/dev/null | cut -f1) already here)"
mkdir -p "$UV_INSTALL_DIR" "$NLTK_DATA" "$OUT" "$OSINT_RAW" "$OSINT_DATA/items"

SERVER_PID=""
keep() {
  for path in items/measured.jsonl review.html; do
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
uv sync -q
uv run pytest -q

export OSINT_CONFIG="$WORK/repo/config/qwq"
export OSINT_TRANSCRIPT="$WORK/model-calls.jsonl"

# The questions being measured, put where step 7 looks for them.
cp "$ITEMS" "$OSINT_DATA/items/accepted.jsonl"

echo "=== 1/3 sources (the ablations re-read the documents) ==="
uv run python pipeline/01_sources.py cablegate sanctions

echo "=== installing vLLM (node-local) ==="
uv venv "$WORK/vllm-venv" >/dev/null
VLLM_PY="$WORK/vllm-venv/bin/python"
uv pip install --python "$VLLM_PY" -q vllm ninja
export PATH="$WORK/vllm-venv/bin:$PATH"
"$VLLM_PY" -c "import vllm; print('vllm', vllm.__version__)"

echo "=== fetching the weights ==="
"$VLLM_PY" - <<PY
from huggingface_hub import snapshot_download
print("weights at", snapshot_download("$MODEL", max_workers=8), flush=True)
PY

echo "=== starting vLLM: $MODEL on 2 GPUs ==="
"$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 2 --max-model-len 24576 --gpu-memory-utilization 0.92 \
  --disable-custom-all-reduce \
  > "$WORK/vllm.log" 2>&1 &
SERVER_PID=$!

echo "=== waiting for it to answer ==="
deadline=$(( SECONDS + 1200 ))
until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM exited before becoming ready:" >&2; tail -60 "$WORK/vllm.log" >&2; exit 1
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "vLLM not ready in 1200s of loading:" >&2; tail -60 "$WORK/vllm.log" >&2; exit 1
  fi
  sleep 20
done
echo "ready after ${SECONDS}s"
export OSINT_MODEL_ENDPOINT="http://127.0.0.1:$PORT"

echo "=== control ==="
uv run python pipeline/07_necessity.py --control || echo "CONTROL FAILED -- numbers below are not trustworthy"

echo "=== 2/3 necessity ==="
uv run python pipeline/07_necessity.py
keep

echo "=== 3/3 review page ==="
uv run python pipeline/08_review.py
keep

ls -la "$OUT"
echo done
