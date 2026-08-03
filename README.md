# OSINT Benchmark

A question-answering benchmark in which every question needs two documents to answer: one
confidential, one public. Neither side suffices alone, and that property is verified
rather than assumed.

The corpora are **not redistributed here**. This repo ships the scripts that fetch them
from their original homes and the checksums that prove you rebuilt the same thing. See
[docs/architecture.md](docs/architecture.md) for the full pipeline.

## Requirements

- Linux or macOS, `git`, and an internet connection
- [uv](https://docs.astral.sh/uv/) — install with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- Python 3.13. You do not need to install it: `uv sync` downloads it if your system
  Python is older.
- Free disk space for the corpora you build. Cablegate needs about **4 GB** (1.7 GB
  downloaded, 1.8 GB parsed).

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

```bash
uv run python pipeline/01_sources.py
```

That fetches every source, parses it into normalised documents, and checks the result
against the published hashes. To build one source instead of all of them, name it:

```bash
uv run python pipeline/01_sources.py cablegate
```

Cablegate parses in about 35 seconds; the download is what takes the time, and how long
depends entirely on the Internet Archive — measured between 1 and 10 MB/s in one sitting,
so budget anywhere from 3 minutes to half an hour for the 1.7 GB.

The step is re-runnable, which is what makes that variance tolerable: a file already
downloaded is not fetched again, and an interrupted download resumes from where it
stopped rather than starting over.

Output:

```
data/raw/cablegate/cables.csv              the untouched download
data/docs/cablegate.jsonl                  251,287 documents, one JSON object per line
data/docs/cablegate.jsonl.provenance.json  what the parser kept from the source, and dropped
```

Everything under `data/` is gitignored — it is large, and the source licences differ per
corpus.

### Building somewhere else

Four environment variables move the roots. The useful one is `OSINT_RAW`, which points at
corpora you already have so nothing is downloaded twice:

```bash
OSINT_RAW=/mnt/big/raw uv run python pipeline/01_sources.py cablegate
```

| variable | default | holds |
| --- | --- | --- |
| `OSINT_DATA` | `./data` | everything derived |
| `OSINT_RAW` | `$OSINT_DATA/raw` | downloads, never modified |
| `OSINT_DOCS` | `$OSINT_DATA/docs` | parsed documents |
| `OSINT_PINS` | `./pins` | source checksums and document hashes |

A source's raw file belongs at `<OSINT_RAW>/<source>/<filename>`, e.g.
`data/raw/cablegate/cables.csv`.

## Sources

| source | leg | from | status |
| --- | --- | --- | --- |
| Cablegate | confidential | [Internet Archive](https://archive.org/details/wikileaks-cables-csv) | built |
| Dodis | confidential | opendata.dodis.ch | not yet |
| Wikipedia | public | Wikimedia dumps | not yet |
| Swiss parliament | public | Curia Vista OData | not yet |
| SECO sanctions | public | seco.admin.ch | not yet |
| UCDP events | public | ucdp.uu.se | not yet |

Every source's URL, size and SHA-256 lives in [`pins/sources.toml`](pins/sources.toml).
Re-pinning a dump to a new date is an edit to that file, not a code change.

`01_sources.py` exits non-zero if a downloaded file does not match its pinned checksum, or
if a rebuilt corpus contradicts the published per-document hashes. That check is the point
of shipping scripts instead of data: it is how you confirm your Cablegate is byte-identical
to the one the benchmark's answers were written against. Before the first release there
are no published hashes yet, so the run reports that and passes.
