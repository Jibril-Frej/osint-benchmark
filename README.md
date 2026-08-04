# OSINT Benchmark

A question-answering benchmark in which every question needs two documents to answer: one
confidential, one public. Neither side suffices alone, and that property is verified
rather than assumed.

The corpora are **not redistributed here**. This repo ships the scripts that fetch them
from their original homes and the checksums that prove you rebuilt the same thing. See
[docs/architecture.md](docs/architecture.md) for the full pipeline.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Free disk space: TBD

## Installation

```bash
git clone https://github.com/Jibril-Frej/osint-benchmark.git
cd osint-benchmark
uv sync
uv run pytest -q
```

## Try it end to end

```bash
bash pipeline/smoke.sh
```

Runs every step on a small slice and writes something at each, in about five minutes. It
uses two stand-ins, both labelled in what they produce: step 2 links by article-title match
instead of ReFinED, and steps 6–7 use a scripted stand-in instead of a served model, so the
questions are placeholders marked `STUB`. It proves the chain runs; it says nothing about
question quality.

`SMOKE_STUB=0` runs steps 6 and 7 against a real model instead.

## The pipeline

Run in order. Each step reads the previous step's output from `data/`.

| step | what it does | needs |
| --- | --- | --- |
| `01_sources.py` | fetch, parse and verify the corpora | network |
| `02_link.py` | annotate documents with Wikidata entities | a GPU and `uv sync --extra link`, or `--dictionary` |
| `03_graph.py` | invert links into entity↔document bridges | — |
| `04_public.py` | fetch public evidence for the bridge entities | network |
| `05_pair.py` | pair one confidential document with one public record | — |
| `06_generate.py` | write each question and its answer | a model served with vLLM |
| `07_necessity.py` | measure whether both sides are needed | a model served with vLLM |
| `08_review.py` | render the questions for the human pass | — |
| `09_release.py` | freeze a version | — |

```bash
uv run python pipeline/01_sources.py            # all sources
uv run python pipeline/01_sources.py cablegate  # just one
```

### Step 2

ReFinED needs a GPU and its own extra. Without `--device cuda` it runs on the CPUs and
takes a day.

```bash
uv sync --extra link
uv run python pipeline/02_link.py cablegate --device cuda
uv run python pipeline/02_link.py cablegate --dictionary --restrict-to sanctions  # no model
```

Settings are in [`config/link.toml`](config/link.toml). On Slurm, one job links the slice
both ways and prints the two side by side:

```bash
sbatch --partition=<p> --qos=<q> --gpus=<type> cluster/link.sbatch
```

### Steps 6 and 7

They need a model served with vLLM, named in [`config/models.toml`](config/models.toml).
Point the pipeline at it and check the solver works before trusting any number:

```bash
export OSINT_MODEL_ENDPOINT=http://127.0.0.1:8000
uv run python pipeline/07_necessity.py --control   # a solver that fails this makes
uv run python pipeline/06_generate.py --limit 20   # every question look necessary
```

`--stub` runs both against a scripted stand-in instead, for exercising the chain with no
GPU. Set `OSINT_TRANSCRIPT=calls.jsonl` to record every prompt and reply — off by default,
because prompts carry corpus text and replies carry gold answers.

On Slurm, one job does the lot: clone, install, serve, run, tear down.

```bash
sbatch --partition=<p> --qos=<q> cluster/smoke.sbatch                 # CPU, stand-in models
sbatch --partition=<p> --qos=<q> --gpus=<type> cluster/smoke_gpu.sbatch   # a real model
```

## Sources

| source | leg | from |
| --- | --- | --- |
| `cablegate` | confidential | [Internet Archive](https://archive.org/details/wikileaks-cables-csv) |
| `gdelt` | public | data.gdeltproject.org |
| `parliament` | public | Curia Vista OData |
| `sanctions` | public | sesam.search.admin.ch |
| `ucdp` | public | ucdp.uu.se |
| `wikipedia_index` | public | dumps.wikimedia.org |
| `wikipedia_index_simple` | public | dumps.wikimedia.org — 44 MB against 2.85 GB, for exercising the chain |

Every URL, size and SHA-256 is in [`pins/sources.toml`](pins/sources.toml). Re-pinning a
dump to a newer date is an edit to that file, not a code change.

`01_sources.py` exits non-zero if a download does not match its checksum, or if a rebuilt
corpus does not match the fingerprint in [`pins/corpora.toml`](pins/corpora.toml) — one
line per source, a hash over every record plus a count. That check is the point of shipping
scripts instead of data: it is how you confirm your corpora are the ones the benchmark's
answers were written against.

## Prompts and parameters

Every prompt sent to a model is a file in [`prompts/`](prompts/). Every parameter is in
`config/`:

| file | governs |
| --- | --- |
| [`config/models.toml`](config/models.toml) | temperature, token ceiling, sample count, which model in which role |
| [`config/link.toml`](config/link.toml) | the linker: checkpoint, entity set, confidence floor, batch size |

Read `prompts/` and `config/models.toml` together; neither is correct alone. No prompt text
appears in any `.py` file, and a test fails if it reappears there.

`OSINT_CONFIG` points at a whole alternative set — [`config/smoke/`](config/smoke/) is the
one the smoke run uses.

## Where the data goes

```
data/raw/     downloads, never modified
data/docs/    parsed corpora, one JSONL per source, each with a .provenance.json
data/links/   entities per document        data/graph/  entity↔document bridges
data/facts/   public evidence per entity   data/pairs/  paired documents
data/items/   questions                    releases/    frozen versions
```

All gitignored. Four environment variables move the roots: `OSINT_DATA` (everything
derived), `OSINT_RAW`, `OSINT_DOCS`, `OSINT_PINS`.

## Repository layout

| directory | holds |
| --- | --- |
| `pipeline/` | the nine numbered entry points, run in order |
| `osint_benchmark/` | the library each of them calls |
| `config/` | every tunable parameter |
| `prompts/` | every prompt sent to a model |
| `pins/` | source checksums and corpus fingerprints |
| `cluster/` | Slurm jobs |
| `tests/` | `uv run pytest -q` |
| `docs/` | why the pipeline is shaped this way |
