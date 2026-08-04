#!/usr/bin/env bash
#SBATCH --job-name=osint-smoke-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log
#
# The whole pipeline on a small slice, with steps 6 and 7 run against a real model.
#
# cluster/smoke.sbatch proves the chain runs. This proves the *model* half of it: that the
# client reaches vLLM, that the prompts come back as parseable JSON, that the verdict
# extraction finds its word, that repeat-and-agree behaves. It does not say the questions
# are any good -- a small model writes poor ones, which is expected and is why the
# datasheet records the model that produced a run.
#
# Everything lives in the node's own /scratch, including the vLLM install, so nothing
# lands on the shared filesystem and there is nothing to clean up afterwards. The model
# weights are read from the existing Hugging Face cache, which is on the shared
# filesystem and therefore visible from any node.
#
# Usage:
#   sbatch --partition=<p> --qos=<q> cluster/smoke_gpu.sbatch
#
#   OSINT_SMOKE_MODEL   what to serve. Default matches config/smoke/models.toml.
set -euo pipefail

MODEL="${OSINT_SMOKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT="${OSINT_SERVE_PORT:-8000}"
WORK="/scratch/$USER/osint-smoke-gpu-$SLURM_JOB_ID"

export UV_INSTALL_DIR="$WORK/bin" UV_CACHE_DIR="$WORK/uv-cache"
export UV_PYTHON_INSTALL_DIR="$WORK/uv-python" OSINT_DATA="$WORK/data"
export PATH="$UV_INSTALL_DIR:$PATH" PYTHONUNBUFFERED=1
# Read the weights already on disk; never try to fetch them.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1

echo "node    $(hostname)"
echo "model   $MODEL"
echo "scratch $(df -h /scratch | tail -1 | awk '{print $4}') free"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
mkdir -p "$UV_INSTALL_DIR"

echo "=== installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
uv --version

echo "=== cloning ==="
git clone -q https://github.com/Jibril-Frej/osint-benchmark.git "$WORK/repo"
cd "$WORK/repo"
uv sync -q
uv run pytest -q

echo "=== installing vLLM (node-local, nothing on the shared filesystem) ==="
uv venv "$WORK/vllm-venv" >/dev/null
VLLM_PY="$WORK/vllm-venv/bin/python"
uv pip install --python "$VLLM_PY" -q vllm
"$VLLM_PY" -c "import vllm; print('vllm', vllm.__version__)"

SERVER_PID=""
cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "stopping vLLM ($SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -rf "$WORK"
  echo "cleaned up $WORK"
}
trap cleanup EXIT

echo "=== starting vLLM ==="
"$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --max-model-len 8192 > "$WORK/vllm.log" 2>&1 &
SERVER_PID=$!

# Readiness is "it answers", not "the process exists": vLLM loads weights and captures
# CUDA graphs long after the process appears, and a step that starts too early fails on
# its first question rather than where the problem is.
echo "=== waiting for it to answer ==="
deadline=$(( SECONDS + 1200 ))
until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM exited before becoming ready:" >&2; tail -40 "$WORK/vllm.log" >&2; exit 1
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "vLLM not ready in time:" >&2; tail -40 "$WORK/vllm.log" >&2; exit 1
  fi
  sleep 5
done
echo "ready after ${SECONDS}s"

export OSINT_MODEL_ENDPOINT="http://127.0.0.1:$PORT"
export OSINT_CONFIG="$WORK/repo/config/smoke"

echo "=== control: can the solver answer when the evidence plainly contains the answer? ==="
# Run this before trusting any necessity number. A solver that cannot answer THIS is
# broken, and a broken solver makes every question look perfectly necessary.
uv run python pipeline/07_necessity.py --control || echo "CONTROL FAILED -- necessity numbers below are not trustworthy"

echo "=== the pipeline, steps 6 and 7 against the real model ==="
SMOKE_STUB=0 bash pipeline/smoke.sh

echo "=== what came out ==="
head -3 "$OSINT_DATA/items/accepted.jsonl" 2>/dev/null || echo "(no accepted items)"
cp -f "$WORK/repo/releases/smoke/datasheet.json" "$HOME/osint-smoke-gpu-datasheet.json" 2>/dev/null || true
echo done
