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
```

Check it worked:

```bash
uv run pytest -q
```

## Building the corpora
To fetch and process all source datasets, run:  

```bash
uv run python pipeline/01_sources.py
```


Output for Cablegate:

```
data/raw/cablegate/cables.csv              the untouched download
data/docs/cablegate.jsonl                  251,287 documents, one JSON object per line
data/docs/cablegate.jsonl.provenance.json  what the parser kept from the source, and dropped
```

Everything under `data/` is gitignored — it is large, and the source licences differ per
corpus.

## Sources

| source | leg | from | status |
| --- | --- | --- | --- |
| Cablegate | confidential | [Internet Archive](https://archive.org/details/wikileaks-cables-csv) | built |

