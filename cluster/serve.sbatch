#!/usr/bin/env bash
#SBATCH --job-name=osint-serve
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.log
#
# Serve a model with vLLM, run a pipeline step against it, then shut the server down.
#
# Steps 6 and 7 need a served model. This starts one, waits until it actually answers,
# points the pipeline at it, and cleans up afterwards -- so a run cannot end with a model
# still holding a GPU.
#
# Usage:
#   sbatch cluster/serve.sbatch python pipeline/06_generate.py --limit 20
#   sbatch cluster/serve.sbatch python pipeline/07_necessity.py --control
#
# Configure with environment variables rather than by editing this file:
#   OSINT_SERVE_MODEL   what to serve, e.g. Qwen/QwQ-32B. Must match the `model` in
#                       config/models.toml for the role being run.
#   OSINT_SERVE_ARGS    extra vLLM arguments, e.g. "--tensor-parallel-size 2 --quantization awq"
#   OSINT_SERVE_PORT    default 8000
#
# The environment running vLLM needs `ninja` on PATH: the inductor backend shells out to
# it during torch.compile and the wheel does not ship the binary.
#
# Slurm directives above are deliberately generic. Partition, QoS and the GPU request
# differ per site -- some reject a bare count and require a type -- so pass them at submit
# time:
#   sbatch --partition=<p> --qos=<q> --gpus=<type> cluster/serve.sbatch ...
set -euo pipefail

MODEL="${OSINT_SERVE_MODEL:?set OSINT_SERVE_MODEL to the model to serve}"
PORT="${OSINT_SERVE_PORT:-8000}"
EXTRA="${OSINT_SERVE_ARGS:-}"
READY_TIMEOUT="${OSINT_SERVE_READY_TIMEOUT:-1800}"

if [ "$#" -eq 0 ]; then
  echo "usage: sbatch cluster/serve.sbatch <command...>" >&2
  exit 2
fi

echo "node   $(hostname)"
echo "model  $MODEL"
echo "port   $PORT"

# The server is a child of this job, so it dies with it. The trap covers the ordinary
# path, where the step finishes and the GPU should be released at once rather than held
# until the walltime expires.
SERVER_PID=""
cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "stopping vLLM ($SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "=== starting vLLM ==="
# shellcheck disable=SC2086
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --host 127.0.0.1 --port "$PORT" $EXTRA > "vllm-$PORT.log" 2>&1 &
SERVER_PID=$!

echo "=== waiting for it to answer ==="
# Readiness is "it responds", not "the process exists": vLLM loads weights and captures
# CUDA graphs long after the process appears, and a step that starts too early fails on
# its first question rather than at the point the problem occurred.
deadline=$(( SECONDS + READY_TIMEOUT ))
until curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM exited before becoming ready; last lines:" >&2
    tail -30 "vllm-$PORT.log" >&2
    exit 1
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "vLLM not ready after ${READY_TIMEOUT}s; last lines:" >&2
    tail -30 "vllm-$PORT.log" >&2
    exit 1
  fi
  sleep 10
done
echo "ready after ${SECONDS}s"

export OSINT_MODEL_ENDPOINT="http://127.0.0.1:$PORT"
export PYTHONUNBUFFERED=1

echo "=== running: $* ==="
"$@"
