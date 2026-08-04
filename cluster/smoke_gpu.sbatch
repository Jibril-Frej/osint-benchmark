#!/usr/bin/env bash
#SBATCH --job-name=osint-smoke-gpu
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
# The GPU is requested at submit time, not here: sites differ on whether a bare count is
# accepted or a type is required (this one rejects --gpus=1 and wants --gpus=a100).
#
# Usage:
#   sbatch --partition=<p> --qos=<q> --gpus=<type> cluster/smoke_gpu.sbatch
#
#   OSINT_SMOKE_MODEL   what to serve. Default matches config/smoke/models.toml.
set -euo pipefail

MODEL="${OSINT_SMOKE_MODEL:-swiss-ai/Apertus-8B-Instruct-2509}"
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
# ninja is a real dependency and not a declared one: vLLM's inductor backend shells out
# to it during torch.compile, and the wheel does not ship the binary. Without it the
# engine dies with FileNotFoundError: 'ninja' after the weights have already loaded.
uv pip install --python "$VLLM_PY" -q vllm ninja
"$VLLM_PY" -c "import vllm; print('vllm', vllm.__version__)"
# The venv's bin must be on PATH, not just importable: vLLM invokes ninja as a
# subprocess, so installing the package is not enough if nothing can find the binary.
# Calling "$VLLM_PY" directly rather than activating the venv is what hid this.
export PATH="$WORK/vllm-venv/bin:$PATH"
command -v ninja || { echo "ninja still not on PATH" >&2; exit 1; }

SERVER_PID=""
cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "stopping vLLM ($SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  # Save the server log BEFORE deleting the working directory. The first version of this
  # wiped it, so a startup failure left nothing to diagnose but the tail of a traceback
  # whose own message said "see root cause above".
  cp -f "$WORK/vllm.log" "$HOME/osint-vllm-${SLURM_JOB_ID}.log" 2>/dev/null || true
  rm -rf "$WORK"
  echo "cleaned up $WORK (server log kept at ~/osint-vllm-${SLURM_JOB_ID}.log)"
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
    echo "vLLM exited before becoming ready. First error and context:" >&2
    grep -nE "Error|error:|Traceback|Cannot|unsupported|not supported|requires" "$WORK/vllm.log" | head -20 >&2
    echo "--- last 60 lines ---" >&2; tail -60 "$WORK/vllm.log" >&2
    exit 1
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "vLLM not ready in ${READY_TIMEOUT}s:" >&2; tail -60 "$WORK/vllm.log" >&2; exit 1
  fi
  sleep 5
done
echo "ready after ${SECONDS}s"

export OSINT_MODEL_ENDPOINT="http://127.0.0.1:$PORT"
export OSINT_CONFIG="$WORK/repo/config/smoke"
# Record every prompt and reply. A run can produce nothing and give no reason; without
# this, working out why costs another job.
export OSINT_TRANSCRIPT="$WORK/model-calls.jsonl"

echo "=== control: can the solver answer when the evidence plainly contains the answer? ==="
# Run this before trusting any necessity number. A solver that cannot answer THIS is
# broken, and a broken solver makes every question look perfectly necessary.
uv run python pipeline/07_necessity.py --control || echo "CONTROL FAILED -- necessity numbers below are not trustworthy"

echo "=== the pipeline, steps 6 and 7 against the real model ==="
SMOKE_STUB=0 bash pipeline/smoke.sh

echo "=== what came out ==="
echo "accepted: $(wc -l < "$OSINT_DATA/items/accepted.jsonl" 2>/dev/null || echo 0)"
echo "rejected: $(wc -l < "$OSINT_DATA/items/rejected.jsonl" 2>/dev/null || echo 0)"
head -2 "$OSINT_DATA/items/accepted.jsonl" 2>/dev/null || true

echo "=== what the model actually said ==="
uv run python - <<'PY' || true
import json, os, collections
path = os.environ.get("OSINT_TRANSCRIPT")
rows = [json.loads(x) for x in open(path, encoding="utf-8")] if path and os.path.exists(path) else []
print(f"{len(rows)} calls:", dict(collections.Counter(r["role"] for r in rows)))
print("empty replies:", sum(1 for r in rows if r["reply_chars"] == 0))
for role in ("solver", "phraser", "judge"):
    sample = next((r for r in rows if r["role"] == role), None)
    if sample:
        print(f"--- {role} reply ({sample['reply_chars']} chars) ---")
        print(sample["reply"][:600])
PY
cp -f "$OSINT_TRANSCRIPT" "$HOME/osint-model-calls-${SLURM_JOB_ID}.jsonl" 2>/dev/null || true
cp -f "$WORK/repo/releases/smoke/datasheet.json" "$HOME/osint-smoke-gpu-datasheet.json" 2>/dev/null || true
echo done
