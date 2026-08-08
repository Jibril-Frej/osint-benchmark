# OSINT Benchmark

A question-answering benchmark in which every question needs two documents to answer: one
confidential, one public. Neither side suffices alone, and that property is verified
rather than assumed.

The corpora are **not redistributed here**. This repo ships the scripts that fetch them
from their original homes and the checksums that prove you rebuilt the same thing — with
one exception, `dodis`, noted under Sources. See
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
| `06_generate.py --types` | build the questions whose answers are computed | a model served with vLLM |
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
uv run python pipeline/02_link.py --check --device cuda   # run this first
uv run python pipeline/02_link.py cablegate --device cuda
uv run python pipeline/02_link.py cablegate --dictionary --restrict-to sanctions  # no model
```

`--check` builds the linker and links one sentence. It needs no corpora, so a broken
install costs three minutes rather than the half hour of downloads that precedes the real
run. Importing ReFinED is not the same as loading a model, and only the second one fails.

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

### Question types

`06_generate.py` writes one type by default and two more on request.

| type | the answer | how it is arrived at |
| --- | --- | --- |
| `bridge` | written by the model from two documents | judged against both documents, unanimous verdicts only |
| `association` | the organisation two people in one document both belong to | computed from the Wikidata slice |
| `resolution` | the full name of someone a document calls by family name alone | the linker's resolution, checked against the passage |

The two typed builders need facts for every linked entity and for the entities those point
at, not just for the bridges:

```bash
uv run python pipeline/04_public.py --scope linked --min-confidence 0.9 --neighbours
uv run python pipeline/06_generate.py --types association,resolution --per-type 150 --workers 8
```

`--workers` puts that many requests in flight; steps 6 and 7 both take it. vLLM batches
continuously, so one request at a time leaves the GPUs mostly idle.

They take no judge and no `--draft`/`--verify` pass: the answer is computed, so there is
nothing for a model to have got wrong. The phraser is never shown it.

On Slurm, one job does the lot: clone, install, serve, run, tear down.

```bash
sbatch --partition=<p> --qos=<q> cluster/smoke.sbatch                    # CPU, stand-in models
sbatch --partition=<p> --qos=<q> --gpus=<type> cluster/smoke_gpu.sbatch  # a real model
sbatch --partition=<p> --qos=<q> --gpus=a100:2 cluster/generate.sbatch   # corpora to questions
sbatch --partition=<p> --qos=<q> --gpus=a100:2 cluster/typed.sbatch      # the typed questions
sbatch --partition=<p> --qos=<q> --gpus=<type> cluster/parliament.sbatch # link German and French
```

`typed.sbatch` copies each stage's output to `$HOME`. Pass `OSINT_LINKS` and `OSINT_FACTS`
to reuse them: fetching a record and an article for every linked entity takes an hour and a
half of network with the GPUs doing nothing, and it produces the same files every time.

`generate.sbatch` takes `OSINT_JUDGE_MODEL` to judge with a second model, served after the
phraser is torn down — a judge from the phraser's own family is scoring its own output, and
two 30B models do not fit on two A100s together. `OSINT_PAIRS` reuses a saved pair set and
skips linking, which turns a prompt experiment from five hours into two.

`parliament.sbatch` asks for 200 GB of memory: mGENRE's (language, title) → QID mapping is
3.7 GB on disk and roughly 90 GB loaded.

`generate.sbatch` uses [`config/qwq/`](config/qwq/), which serves one model in all three
roles. That makes the judge the same model as the phraser — read its verdicts as a wiring
check, not as evidence.

Give a reasoning model room. `max_tokens` too low truncates the reply inside its `<think>`
block, before any answer: the run reports questions that were never written and gives no
reason. Step 6 prints where every lost pair went.

## Sources

| source | leg | linked by | from |
| --- | --- | --- | --- |
| `cablegate` | confidential | ReFinED | [Internet Archive](https://archive.org/details/wikileaks-cables-csv) |
| `dodis` | confidential | the archive's own curated links | **not fetchable — see below** |
| `parliament` | public | WikiNeural + mGENRE (German, French) | Curia Vista OData |
| `sanctions` | public | name reconciliation | sesam.search.admin.ch |
| `gdelt` | public | country codes | data.gdeltproject.org |
| `ucdp` | public | country codes | ucdp.uu.se |
| `wikipedia_index` | — | the entity universe | dumps.wikimedia.org |
| `wikipedia_index_simple` | — | the entity universe | dumps.wikimedia.org — 44 MB against 2.85 GB, for exercising the chain |

**`dodis` cannot be rebuilt from a clone.** Dodis offers no bulk download of its scans and
defends against crawlers, so the 4,065 PDFs were gathered by a crawler that is not in this
repository, and OCR'd separately. Point `OSINT_DODIS_OCR` at the text and `OSINT_DODIS_NT`
at the open-data N-Triples export; without them the source says what is missing.

**`gdelt` and `ucdp` produce no bridges.** They name countries, and a country co-occurs
with everything, so step 5 excludes it as a bridge anchor. That is a fact about the bridge
type, not about the sources: the previous project built questions on both by matching a
cable to public events sharing an anchor *and* falling within ±21 days of its date, which
is discriminative where the entity alone is not. That type is not ported yet.

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
