#!/usr/bin/env bash
#SBATCH --job-name=osint-smoke
#SBATCH --partition=nodes
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log
#
# Build the benchmark from a clean clone, end to end, on one compute node.
#
# Everything happens in the node's own /scratch: the clone, the toolchain, the corpora.
# That is deliberate -- a login node is shared, and parsing seven million entities on one
# is antisocial and gets the job killed anyway. /scratch is node-local, so a job that
# spans nodes cannot see its own intermediate files; this runs the whole chain in one job
# for that reason.
#
# CPU only. No GPU is requested and none is needed: steps 6 and 7 run with a stand-in.
#
# Submit with:  sbatch cluster/smoke.sbatch
set -euo pipefail

WORK="/scratch/$USER/osint-smoke-$SLURM_JOB_ID"
export UV_INSTALL_DIR="$WORK/bin"
export UV_CACHE_DIR="$WORK/uv-cache"
export UV_PYTHON_INSTALL_DIR="$WORK/uv-python"
export OSINT_DATA="$WORK/data"
export PATH="$UV_INSTALL_DIR:$PATH"

echo "node    $(hostname)"
echo "work    $WORK"
echo "scratch $(df -h /scratch | tail -1 | awk '{print $4}') free"
mkdir -p "$UV_INSTALL_DIR"

echo "=== installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
uv --version

echo "=== cloning ==="
git clone -q https://github.com/Jibril-Frej/osint-benchmark.git "$WORK/repo"
cd "$WORK/repo"
uv sync -q
uv run python -V
uv run pytest -q

echo "=== running the pipeline ==="
bash pipeline/smoke.sh

echo "=== what was built ==="
find "$OSINT_DATA" -name '*.jsonl' -printf '%10s  %p\n' | sort -k2
echo "=== sidecars missing (a missing one means that step died) ==="
for f in "$OSINT_DATA"/docs/*.jsonl; do
  [ -f "$f.provenance.json" ] || echo "MISSING: $f"
done
echo "done"

# The node's scratch is not shared, so nothing here is visible afterwards. Copy the
# summary somewhere that is, then remove the working data.
cp -f "$OSINT_DATA/../repo/releases/smoke/datasheet.json" "$HOME/osint-smoke-datasheet.json" 2>/dev/null || true
rm -rf "$WORK"
echo "cleaned up $WORK"
