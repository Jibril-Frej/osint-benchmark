#!/usr/bin/env bash
#SBATCH --job-name=osint-generate
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.log
#
# Build the corpora, link them, and write questions with a real model.
#
# What this is for is reading the questions. Every number it prints about their quality
# comes from a judge that is the same model as the phraser (see config/qwq/models.toml),
# which is not evidence -- the evidence is the review page, and a person reading it.
#
# Results are copied out after each stage rather than at the end, so a job that runs out of
# time still leaves everything it had finished.
#
# The raw downloads are kept in $HOME/osint-raw and reused: they are ~4.5 GB, they do not
# change, and re-fetching them cost an hour of GPU time on the last run. Everything else --
# the repo, the parsed corpora, vLLM, the model weights -- lives on the node and is
# discarded.
#
# GPUs are requested at submit time. This needs two: QwQ-32B is ~66 GB in bf16.
#
# Usage:
#   sbatch --partition=<p> --qos=<q> --gpus=a100:2 cluster/generate.sbatch
#
#   OSINT_GEN_LIMIT     pairs to draft from (default 80)
#   OSINT_LINK_STRIDE   take every Nth cable (default 12, ~21k of 251k)
set -euo pipefail

LIMIT="${OSINT_GEN_LIMIT:-80}"
STRIDE="${OSINT_LINK_STRIDE:-12}"
MODEL="${OSINT_GEN_MODEL:-Qwen/QwQ-32B}"
PORT="${OSINT_SERVE_PORT:-8000}"
export WORK="/scratch/$USER/osint-generate-$SLURM_JOB_ID"
export OUT="$HOME/osint-generate-$SLURM_JOB_ID"

export UV_INSTALL_DIR="$WORK/bin" UV_CACHE_DIR="$WORK/uv-cache"
export UV_PYTHON_INSTALL_DIR="$WORK/uv-python" OSINT_DATA="$WORK/data"
export PATH="$UV_INSTALL_DIR:$PATH" PYTHONUNBUFFERED=1
# Kept between jobs. Everything else is not.
export OSINT_RAW="$HOME/osint-raw"
# The weights are node-local: 66 GB does not belong on a filesystem that is 100% full.
export HF_HOME="$WORK/hf" TORCH_HOME="$WORK/torch" NLTK_DATA="$WORK/nltk"
export TRANSFORMERS_VERBOSITY=error
export NLTK_DISABLE_IMPORT_SECURITY=1

echo "node    $(hostname)"
echo "model   $MODEL"
echo "limit   $LIMIT pairs"
echo "scratch $(df -h /scratch | tail -1 | awk '{print $4}') free"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
mkdir -p "$UV_INSTALL_DIR" "$NLTK_DATA" "$OUT" "$OSINT_RAW"

SERVER_PID=""
keep() {
  # Copy whatever exists. Called after every stage, so a job killed at the time limit
  # still leaves its questions behind.
  for path in items/accepted.jsonl items/rejected.jsonl items/measured.jsonl \
              pairs/pairs.jsonl graph/bridges.jsonl review.html; do
    cp -f "$OSINT_DATA/$path" "$OUT/$(basename "$path")" 2>/dev/null || true
  done
}
cleanup() {
  keep
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  cp -f "$WORK/vllm.log" "$OUT/vllm.log" 2>/dev/null || true
  cp -f "$WORK/model-calls.jsonl" "$OUT/model-calls.jsonl" 2>/dev/null || true
  rm -rf "$WORK"
  echo "cleaned up $WORK (results in $OUT)"
}
trap cleanup EXIT

echo "=== installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
uv --version

echo "=== cloning ==="
git clone -q https://github.com/Jibril-Frej/osint-benchmark.git "$WORK/repo"
cd "$WORK/repo"
uv sync -q --extra link
uv run pytest -q

export OSINT_CONFIG="$WORK/repo/config/qwq"
export OSINT_TRANSCRIPT="$WORK/model-calls.jsonl"

echo "=== does the linker build? ==="
uv run python pipeline/02_link.py --check --device cuda

echo "=== 1/8 sources (raw reused from $OSINT_RAW) ==="
uv run python pipeline/01_sources.py cablegate sanctions wikipedia_index

echo "=== 2/8 link ==="
uv run python pipeline/02_link.py sanctions --index wikipedia_index
uv run python pipeline/02_link.py cablegate --index wikipedia_index --device cuda --stride "$STRIDE"

echo "=== 3/8 graph ==="
uv run python pipeline/03_graph.py

echo "=== 4/8 public evidence ==="
uv run python pipeline/04_public.py --index wikipedia_index

echo "=== 5/8 pair ==="
uv run python pipeline/05_pair.py
keep

echo "=== installing vLLM (node-local) ==="
uv venv "$WORK/vllm-venv" >/dev/null
VLLM_PY="$WORK/vllm-venv/bin/python"
# ninja is a real dependency and not a declared one: vLLM's inductor backend shells out to
# it during torch.compile and the wheel does not ship the binary. The venv's bin must also
# be on PATH -- installing the package is not enough if nothing can find the executable.
uv pip install --python "$VLLM_PY" -q vllm ninja
export PATH="$WORK/vllm-venv/bin:$PATH"
command -v ninja >/dev/null || { echo "ninja not on PATH" >&2; exit 1; }
"$VLLM_PY" -c "import vllm; print('vllm', vllm.__version__)"

echo "=== starting vLLM: $MODEL on 2 GPUs (~66 GB of weights to fetch first) ==="
"$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 2 --max-model-len 8192 --gpu-memory-utilization 0.92 \
  > "$WORK/vllm.log" 2>&1 &
SERVER_PID=$!

# Readiness is "it answers", not "the process exists". The weights download, then load,
# then CUDA graphs capture, long after the process appears.
echo "=== waiting for it to answer ==="
deadline=$(( SECONDS + 5400 ))
until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM exited before becoming ready:" >&2
    grep -nE "Error|error:|Traceback|Cannot|unsupported|out of memory" "$WORK/vllm.log" | head -20 >&2
    echo "--- last 60 lines ---" >&2; tail -60 "$WORK/vllm.log" >&2
    exit 1
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "vLLM not ready in 5400s:" >&2; tail -60 "$WORK/vllm.log" >&2; exit 1
  fi
  sleep 20
done
echo "ready after ${SECONDS}s"
export OSINT_MODEL_ENDPOINT="http://127.0.0.1:$PORT"

echo "=== control: can the solver answer when the evidence contains the answer? ==="
# Before any necessity number is worth reading. A solver that fails this makes every
# question look perfectly necessary.
uv run python pipeline/07_necessity.py --control || echo "CONTROL FAILED -- necessity below is not trustworthy"

echo "=== 6/8 generate, $LIMIT pairs ==="
uv run python pipeline/06_generate.py --limit "$LIMIT"
keep

echo "=== 7/8 necessity ==="
uv run python pipeline/07_necessity.py
keep

echo "=== 8/8 review page ==="
uv run python pipeline/08_review.py
keep

echo "=== what came out ==="
echo "accepted: $(wc -l < "$OSINT_DATA/items/accepted.jsonl" 2>/dev/null || echo 0)"
echo "rejected: $(wc -l < "$OSINT_DATA/items/rejected.jsonl" 2>/dev/null || echo 0)"
uv run python - <<'PY' || true
import collections, json, os, pathlib

def read(path):
    path = pathlib.Path(path)
    return [json.loads(x) for x in path.open(encoding="utf-8")] if path.exists() else []

data = os.environ["OSINT_DATA"]
rejected = read(f"{data}/items/rejected.jsonl")
gates = collections.Counter(
    name for row in rejected for name, ok in (row.get("gates") or {}).items() if not ok
)
print("gate failures:", dict(gates))
calls = read(os.environ.get("OSINT_TRANSCRIPT", ""))
print(f"{len(calls)} model calls:", dict(collections.Counter(c["role"] for c in calls)))
print("empty replies:", sum(1 for c in calls if c["reply_chars"] == 0))
PY
ls -la "$OUT"
echo done
