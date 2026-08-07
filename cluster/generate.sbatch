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
# Two things are kept between jobs because they never change and re-fetching them costs
# real GPU time: the raw corpus downloads (~4.5 GB, an hour) and the model weights (~66 GB,
# twenty minutes). Both live in a stable directory on the node's own /scratch, NOT in
# $HOME.
#
# $HOME is Kerberos-authenticated NFS and a batch job's credential expires underneath a
# long write: job 13343 spent 75 minutes downloading the cables to $HOME and died on
# `OSError: [Errno 127] Key has expired` with the file at exactly its pinned size. Short
# writes at the end of a job do work, which is why the results are still copied there.
#
# The cache survives as long as jobs land on the same node; on a different one they
# download again. That is the price of not touching NFS during a run.
#
# GPUs are requested at submit time. This needs two: QwQ-32B is ~66 GB in bf16. If tensor
# parallelism proves unreliable here, the fallback is one GPU with --quantization fp8,
# which halves the weights to ~33 GB and removes the cross-worker collectives entirely at
# some cost in fidelity.
#
# --disable-custom-all-reduce is set alongside the loopback interface variables below.
# Either alone may be enough; both are cheap, and the failure they prevent costs 40
# minutes to observe.
#
# Usage:
#   sbatch --partition=<p> --qos=<q> --gpus=a100:2 cluster/generate.sbatch
#
#   OSINT_GEN_LIMIT     pairs to draft from (default 80)
#   OSINT_LINK_STRIDE   take every Nth cable (default 12, ~21k of 251k)
#   OSINT_PAIRS         reuse a pairs.jsonl from an earlier run and skip linking, for an
#                       experiment that changes only a prompt. bridges.jsonl must sit
#                       beside it.
#   OSINT_JUDGE_MODEL   a second model to judge with, served after the phraser is
#                       torn down. Set it and the run uses config/split; leave it
#                       unset and one model does both roles, which is a wiring
#                       check rather than evidence.
set -euo pipefail

LIMIT="${OSINT_GEN_LIMIT:-80}"
STRIDE="${OSINT_LINK_STRIDE:-12}"
MODEL="${OSINT_GEN_MODEL:-Qwen/QwQ-32B}"
# Set to serve a second model for judging, after the phraser is torn down.
JUDGE_MODEL="${OSINT_JUDGE_MODEL:-}"
PORT="${OSINT_SERVE_PORT:-8000}"
export WORK="/scratch/$USER/osint-generate-$SLURM_JOB_ID"
export OUT="$HOME/osint-generate-$SLURM_JOB_ID"

export UV_INSTALL_DIR="$WORK/bin" UV_CACHE_DIR="$WORK/uv-cache"
export UV_PYTHON_INSTALL_DIR="$WORK/uv-python" OSINT_DATA="$WORK/data"
export PATH="$UV_INSTALL_DIR:$PATH" PYTHONUNBUFFERED=1
# Node-local and kept between jobs. Not deleted by the cleanup trap.
export CACHE="/scratch/$USER/osint-cache"
export OSINT_RAW="$CACHE/raw" HF_HOME="$CACHE/hf"
export TORCH_HOME="$WORK/torch" NLTK_DATA="$WORK/nltk"
export TRANSFORMERS_VERBOSITY=error
# Gated repos -- Gemma, Llama, Mistral -- need a Hugging Face token, and the token must not
# be in this repository: it is public. Read it from a file outside git, if there is one.
#
# HF_TOKEN rather than a login file, because these jobs point HF_HOME at node-local
# scratch and the library looks for `token` under HF_HOME -- so whatever
# `huggingface-cli login` wrote into $HOME would never be found. The variable is read
# regardless of HF_HOME.
#
# Never echoed. A Slurm log is world-readable on a shared filesystem.
if [ -r "$HOME/.config/hf-token" ]; then
  HF_TOKEN="$(tr -d '[:space:]' < "$HOME/.config/hf-token")"
  export HF_TOKEN
  echo "hugging face token: loaded from ~/.config/hf-token"
else
  echo "hugging face token: none (ungated models only)"
fi
export NLTK_DISABLE_IMPORT_SECURITY=1
# Both tensor-parallel workers are on this one node, so their collectives go over
# loopback. Left to itself gloo enumerates the node's interfaces and can pick net1, which
# has no IPv4 address: job 13344 sat in a barrier for its full 30-minute timeout and died
# with "Application timeout caused pair closure", 40 minutes after the weights had loaded.
export GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo VLLM_HOST_IP=127.0.0.1
# These A100s are PCIe cards with no NVLink between them. Peer-to-peer over PCIe hangs on
# hosts where the switch has ACS enabled, and job 13348 did exactly that: it got past the
# gloo barrier, initialised NCCL on both ranks, and then printed nothing for ninety
# minutes without fetching a single byte of weights. Shared memory instead -- there is
# 252 GB of it on this node and the model is 66 GB.
export NCCL_P2P_DISABLE=1

echo "node    $(hostname)"
echo "model   $MODEL"
echo "limit   $LIMIT pairs"
echo "scratch $(df -h /scratch | tail -1 | awk '{print $4}') free"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
mkdir -p "$UV_INSTALL_DIR" "$NLTK_DATA" "$OUT" "$OSINT_RAW" "$HF_HOME"
echo "cache   $CACHE ($(du -sh "$CACHE" 2>/dev/null | cut -f1) already here)"

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
    # Then insist. A server wedged inside an NCCL collective ignores SIGTERM, and a bare
    # `wait` on it never returns: job 13348 held two A100s for a further 35 minutes after
    # it had already given up and printed why.
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
uv --version

echo "=== cloning ==="
git clone -q https://github.com/Jibril-Frej/osint-benchmark.git "$WORK/repo"
cd "$WORK/repo"
uv sync -q --extra link
uv run pytest -q

# config/split names two models; config/qwq puts one in all three roles.
[ -n "$JUDGE_MODEL" ] && PROFILE=split || PROFILE=qwq
export OSINT_CONFIG="$WORK/repo/config/$PROFILE"
echo "profile $PROFILE${JUDGE_MODEL:+ (judge: $JUDGE_MODEL)}"
export OSINT_TRANSCRIPT="$WORK/model-calls.jsonl"

echo "=== does the linker build? ==="
uv run python pipeline/02_link.py --check --device cuda

if [ -n "${OSINT_PAIRS:-}" ]; then
  # Reuse a pair set from an earlier run. Linking 83k cables is 1.5 GPU-hours and produces
  # the same pairs every time, so an experiment that changes only a prompt should not pay
  # for it. What still has to be rebuilt is the corpora (the drafts quote them) and the
  # public facts (the bridge label goes into the prompt, and only bridges.jsonl is saved).
  echo "=== reusing pairs from $OSINT_PAIRS ==="
  mkdir -p "$OSINT_DATA/pairs" "$OSINT_DATA/graph"
  cp "$OSINT_PAIRS" "$OSINT_DATA/pairs/pairs.jsonl"
  cp "$(dirname "$OSINT_PAIRS")/bridges.jsonl" "$OSINT_DATA/graph/bridges.jsonl"
  wc -l < "$OSINT_DATA/pairs/pairs.jsonl" | xargs echo "pairs:"

  echo "=== 1/8 sources (raw cached in $OSINT_RAW) ==="
  uv run python pipeline/01_sources.py cablegate sanctions wikipedia_index

  echo "=== 4/8 public evidence (labels for the bridge entities) ==="
  uv run python pipeline/04_public.py --index wikipedia_index
else
  echo "=== 1/8 sources (raw cached in $OSINT_RAW) ==="
  uv run python pipeline/01_sources.py cablegate sanctions wikipedia_index

  echo "=== 2/8 link ==="
  # The public side first: the title matcher below is scoped to the entities it names.
  uv run python pipeline/02_link.py sanctions --index wikipedia_index
  # Both linkers, merged. They miss different entities -- measured on job 13340, the matcher
  # found 22 bridge entities ReFinED did not and ReFinED found 15 the matcher did not -- so
  # the union is 62 bridges against 40, for no extra model and no extra GPU time.
  uv run python pipeline/02_link.py cablegate --index wikipedia_index --device cuda \
    --stride "$STRIDE" --and-dictionary --restrict-to sanctions

  echo "=== 3/8 graph ==="
  uv run python pipeline/03_graph.py

  echo "=== 4/8 public evidence ==="
  uv run python pipeline/04_public.py --index wikipedia_index

  echo "=== 5/8 pair ==="
  uv run python pipeline/05_pair.py
fi
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

echo "=== fetching the weights before starting anything ==="
# Separately, so the two things that take a long time can be told apart. Job 13348 sat for
# ninety minutes inside a readiness loop that covers both, and only an inspection of the
# cache directory showed it had never downloaded a byte -- it was hung in NCCL. Fetching
# first makes the download visible and resumable, and lets the readiness deadline below be
# short enough that a hang costs minutes.
for model in "$MODEL" ${JUDGE_MODEL:+"$JUDGE_MODEL"}; do
  "$VLLM_PY" - "$model" <<'PY'
import sys
from huggingface_hub import snapshot_download
print("weights at", snapshot_download(sys.argv[1], max_workers=8), flush=True)
PY
done
du -sh "$HF_HOME/hub" 2>/dev/null || true

# Serving is a function because the two-pass path does it twice: the phraser writes the
# questions, is torn down, and the judge takes its place. 62 GB of QwQ plus 54 GB of Gemma
# does not fit in the 80 GB of two A100s, and a judge from the phraser's own family is
# scoring its own output -- so they take turns.
#
# --max-model-len 24576, not the 8192 of job 13349: a reasoning model needs room to reason
# before it answers, and 8192 minus the evidence left QwQ truncated mid-thought on 79 of 80
# questions.
serve() {
  local model="$1"
  echo "=== starting vLLM: $model on 2 GPUs ==="
  "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
    --model "$model" --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size 2 --max-model-len 24576 --gpu-memory-utilization 0.92 \
    --disable-custom-all-reduce \
    >> "$WORK/vllm.log" 2>&1 &
  SERVER_PID=$!

  # Readiness is "it answers", not "the process exists": the weights load and CUDA graphs
  # capture long after the process appears.
  echo "=== waiting for it to answer ==="
  local deadline=$(( SECONDS + 1800 ))
  until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "vLLM exited before becoming ready:" >&2
      grep -nE "Error|error:|Traceback|Cannot|unsupported|out of memory" "$WORK/vllm.log" | tail -20 >&2
      echo "--- last 60 lines ---" >&2; tail -60 "$WORK/vllm.log" >&2
      exit 1
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "vLLM not ready in 1800s of loading:" >&2; tail -60 "$WORK/vllm.log" >&2; exit 1
    fi
    sleep 20
  done
  echo "ready after ${SECONDS}s"
  export OSINT_MODEL_ENDPOINT="http://127.0.0.1:$PORT"
}

unserve() {
  # The GPUs have to be genuinely free before the next model loads, and a server wedged in
  # a collective ignores SIGTERM.
  [ -n "$SERVER_PID" ] || return 0
  kill "$SERVER_PID" 2>/dev/null || true
  for _ in $(seq 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
  sleep 10
  nvidia-smi --query-gpu=memory.used --format=csv,noheader || true
}

serve "$MODEL"

if [ -n "$JUDGE_MODEL" ]; then
  echo "=== 6/8 generate: drafting $LIMIT pairs with $MODEL ==="
  uv run python pipeline/06_generate.py --limit "$LIMIT" --draft
  cp -f "$OSINT_DATA/items/drafted.jsonl" "$OUT/drafted.jsonl" 2>/dev/null || true
  unserve
  serve "$JUDGE_MODEL"
  echo "=== 6/8 generate: judging with $JUDGE_MODEL ==="
  uv run python pipeline/06_generate.py --verify
  keep
else
  echo "=== 6/8 generate, $LIMIT pairs ==="
  uv run python pipeline/06_generate.py --limit "$LIMIT"
  keep
fi

echo "=== control: can the solver answer when the evidence contains the answer? ==="
# Before any necessity number is worth reading. A solver that fails this makes every
# question look perfectly necessary.
uv run python pipeline/07_necessity.py --control || echo "CONTROL FAILED -- necessity below is not trustworthy"

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
