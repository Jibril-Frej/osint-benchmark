# Architecture

> **Status, 2026-08-04.** All nine steps are implemented and run end to end; see the
> README for how. Two of them — 06 generate and 07 necessity — have only been run with a
> scripted stand-in, never with a served model, so they are the least proven. Entity
> linking has been run with a dictionary matcher, not ReFinED. Dodis, and the Wikidata
> subset that a retrieval surface would need, are not implemented.

## What the repo produces

A question-answering benchmark in which every question needs two documents: one
confidential, one public. Neither side suffices, and that property is measured rather
than assumed.

The benchmark exists to evaluate agents split across a trust boundary — a private
cluster trusted with sensitive documents but short on compute, a public cluster with
compute but no trust. It measures two things: whether such an agent can retrieve across
the boundary, and whether it keeps the two sides separated.

This repo **builds** the benchmark. Running an agent against it is the POC repo's job.

## The pipeline

Nine steps, run in order. The numbering is the documentation: `pipeline/01_sources.py`
is the first thing to run, and the file listing shows the order without anyone reading
prose.

| step | does | runs on | output |
| --- | --- | --- | --- |
| `01_sources` | fetch + parse + verify the bulk corpora | workstation | `docs/*.jsonl` |
| `02_link` | ReFinED / mGENRE → Wikidata QIDs | cluster | `links/*.jsonl` |
| `03_graph` | invert links into entity↔document relations | workstation | `graph/bridge_map.json` |
| `04_public` | the entity-driven fetches: commercial register, Wikidata statements, article text | workstation + live | `docs/shab.jsonl`, `facts/*.jsonl` |
| `05_pair` | one private document + one public record | workstation | `pairs/*.jsonl` |
| `06_generate` | candidates → gate suite → items | cluster, GPU | `items/*.jsonl` |
| `07_necessity` | the three-way ablation | cluster, GPU | items + flags |
| `08_review` | self-contained HTML page, human verdicts merged back | workstation | `review.html` |
| `09_release` | freeze, hash, datasheet | workstation | `releases/v1/` |

Fetching appears twice on purpose. Step 1 is everything that depends on nothing; step 4
is everything that cannot be fetched until the linker has said which entities matter.
Conflating them is what produced the coverage bug in the previous project: the Wikidata
slice was fetched against a corpus subset before the linker was rerun at full scale, and
covered properties for 44,477 of 126,903 entities — 35% — with no downstream symptom
beyond filters silently matching nothing.

Steps 02, 06 and 07 need the cluster. The rest run on the workstation.

## Layout

```
osint_benchmark/          the logic — imported, tested, never run directly
    config.py             the one module that knows where data lives
    artifacts.py          jsonl read/write + the provenance sidecar
    schema.py             Document, Record, Item, Gold, Necessity
    models/               completion backends behind one protocol; prompts as files
    sources/              01
    link/                 02
    graph/                03
    pair/                 05
    generate/             06
    necessity/            07
    review/               08
    release/              09

pipeline/                 the entry points — run in order, thin
    01_sources.py … 09_release.py

tests/                    mirrors osint_benchmark/, not pipeline/
pins/                     committed: source URLs, checksums, document hashes
data/                     gitignored
docs/
```

There is no `src/`. It exists to stop tests importing the source tree instead of the
installed package; with an editable install that problem does not arise, and a directory
holding exactly one thing is indirection.

There is no `scripts/` either. `python pipeline/01_sources.py` does not put the package
on `sys.path`, which is why the previous project's scripts open with
`sys.path.insert(0, ...)`. Setting `package = true` in `pyproject.toml` so `uv sync`
installs the package editable makes the problem not exist. (Module names cannot begin
with a digit, so numbered files are run, never imported — which is all they are for.)

The package name is kept as an umbrella because without it the top-level modules would
be `sources`, `link`, `graph`, `models`, `config` — all plausible installed-package
names, and one candidate, `io`, would shadow the standard library.

## Data

### Wikidata plays two roles

**At build time it is queried live.** Every access is anchored: either a QID is already
known and its statements are read, or a name or country code is resolved to a QID. Both
are indexed lookups. No build stage scans the graph, and nothing is downloaded for it.

**In the benchmark it is a retrieval surface** — an agent calls `search_entity` /
`get_properties` at answer time, as `src/eval/osint_wikidata.py` does today. That makes
it public evidence, and the fair-corpus rule below applies to it exactly as it does to
Wikipedia: the searchable space must be defined by a property of Wikidata, not by which
entities became questions. The universe is **the entities holding an English Wikipedia
sitelink** — the same universe as the Wikipedia corpus, which is easier to defend than
two differently-scoped public sources, and which the QID↔title index already enumerates.

That subset is the one thing a dump is needed to build: resolving ~7M entities by key
over the API is not practical, so it is filtered out of a truthy RDF dump once, at build
time. Users receive the built subset or the script, never the dump. Live querying still
serves everything else — reconciliation, gold computation, and tiers 0 and 1.

Measured against the public QLever endpoint (`https://qlever.dev/api/wikidata`, which
`qlever.cs.uni-freiburg.de` redirects to) on 2026-08-03:

| query | for | result |
| --- | --- | --- |
| `?s wdt:P901 "SZ"` | code-keyed reconciliation | Q39, 5 ms |
| `VALUES ?l {…5 names} ?s rdfs:label ?l . ?s wdt:P31 ?type` | name reconciliation | 106 ms |
| `MAX(schema:dateModified)` | index freshness | 2026-08-03T11:14:01Z over 127,682,900 entities |

The batched label query is the one the previous project's local index could not plan —
its comment records a sort over the whole 172M-triple alias relation. The public instance
answers it in 106 ms. The existing code already speaks QLever SPARQL through an
environment variable, so this is a change of endpoint, not of code.

`schema:dateModified` is present per entity, so "has this entity changed since it was
pinned?" is a cheap query.

Costs, accepted knowingly: the endpoint is live, so it is not a frozen snapshot; it is a
third-party research service with no SLA, so the endpoint stays configurable and hosting
a dump remains the documented fallback.

### There is no frozen Wikidata to query instead

Checked. The Query Service and QLever both serve live data. The
[History Query Service](https://www.wikidata.org/wiki/Wikidata:History_Query_Service)
would have answered as-of-date queries but was loaded only to 2019-07-01 and is a
stalled experiment. Dated dumps age off `dumps.wikimedia.org` within months and the
Internet Archive mirror is patchy. What *is* permanent is per-entity revision
addressing: `Special:EntityData/Q42.json?revision=<revid>` returns those exact bytes
indefinitely.

So reproducibility is bought where it matters rather than wholesale: every gold answer
records the revision it was read from. Across the previous project's 404 questions only
134 distinct QIDs appear on the gold path, against a candidate pool of 126,903 entities —
three orders of magnitude apart. Pinning the gold path costs kilobytes; pinning the pool
would cost a dump.

### Wikipedia is the corpus and is not shrunk

The public corpus stays the full English Wikipedia at a stated dump date. The articles of
bridge entities are what questions are *written* from, but a retrieval corpus built from
the answer key is not a retrieval corpus: every distractor would be a near-miss bridge
candidate and the gold article would be guaranteed present.

The same rule applies on the private side, which is already satisfied — the corpus is all
251k cables and all of Dodis, not only the documents that produced questions.

Wikidata is a retrieval surface too, so the same rule binds it — see above for the
universe it is scoped to.

### What is shipped and what is specified

| | size | who can obtain it |
| --- | --- | --- |
| private text (cables + Dodis regesten + OCR) | 1.8 GB | only from this project |
| enwiki raw dump | 25 GB | anyone, permanently |
| derived Wikipedia layers (extraction, chunks, shards, BM25) | 92 GB | rebuildable |

Nothing is redistributed. The release ships build scripts, and `pins/` carries the source
URLs with sizes and SHA-256s so a rebuilt corpus is verifiable — a user can confirm their
cable is byte-identical to the one the gold was written against without receiving a
single cable. Dodis is open data and needs none of this.

Everything the project authored — questions, gold answers, verdicts, necessity flags,
pairings, evidence offsets — is unencumbered and ships freely. Evidence is carried as
`(doc_id, offsets)` rather than inlined text, so no release file contains corpus text.

Because users build the corpus themselves, a silent parse regression becomes their wrong
numbers. `verify` is therefore a first-class step, not a convenience: it checks built
documents against the shipped hashes and fails loudly. The failure it exists for is real —
a missing `escapechar` in the Cablegate CSV parse cost 68% of the corpus text and was
invisible.

### Distribution tiers

The measurement the project is about — whether an agent keeps the two sides separated —
depends on the query that crosses the boundary, not on how large the public corpus is.
That makes the security axis cheap and lets the benchmark be usable at three weights:

| tier | download | measures | does not measure |
| --- | --- | --- | --- |
| 0 | ~50 MB + a build step | reasoning, the necessity property | retrieval, separation |
| 1 | ~2 GB | private retrieval, query formulation, **separation** | public retrieval quality |
| 2 | +25 GB dump, the Wikidata subset, and an index build | end-to-end, fully reproducible | — |

Tier 1 is the headline: it measures the thing the project is about, and live Wikidata
fits it with no download. Tier 2 is the condition the paper's numbers are reported on.
Every reported number names its tier; a number quoted without one is the failure mode.

## Stage 1 in detail

Six bulk sources, one module each, behind one contract.

| module | source | output |
| --- | --- | --- |
| `cablegate` | `cables.csv` | documents |
| `dodis` | `dodis-lod.nt` | documents (regesten) |
| `wikipedia` | enwiki dump | documents **and** the QID↔title index |
| `parliament` | Curia Vista OData, paged | documents |
| `sanctions` | SECO `sesam.xml` | records |
| `ucdp` | GED CSV | records |
| `wikidata` | truthy RDF dump, filtered to the enwiki-sitelinked QIDs | the public fact surface |

`wikidata` belongs here — not in step 4 — because its scope is a property of Wikidata,
not of the questions: it is built from the QID↔title index, which is itself a step-1
output, so it runs after `wikipedia` and before anything entity-driven. It is the only
source in the pipeline that needs a dump, and only because it is a *corpus*; the
build-time lookups it does not serve go to the live endpoint.

Not here: the commercial register and article text (entity-driven, step 4), Dodis OCR (no committed implementation in the previous project — only the
bake-off harness survives, PaddleOCR won on a 39-document gold set — so it needs writing
and gets its own step).

```
osint_benchmark/sources/
    __init__.py      registry: name -> Source
    base.py          the Source dataclass, generic fetch(), generic verify()
    cablegate.py     SOURCE + parse()
    dodis.py
    parliament.py    + a pagination fetch override
    sanctions.py
    ucdp.py
    wikidata.py
    wikipedia/       a folder, because it has two outputs
        __init__.py  SOURCE
        text.py      article extraction
        index.py     QID <-> title from page / page_props
```

A module by default; a folder only when a source has more than one output.

The split is not fetch/parse/verify per source, because those are not equally
source-specific. **verify** is identical everywhere — hash the built documents, compare
against `pins/hashes/` — and six copies would be six chances for one to be weakened
differently. **fetch** varies only in its URL list and, for Curia Vista, its pagination.
**parse** is where everything source-specific lives: the Cablegate record segmentation
and `escapechar` recovery, the N-Triples handling that keeps a multi-line summary from
splitting a record, the enwiki extraction.

So: one shared fetch engine, one shared verify, a per-source parser. URLs and checksums
stay out of the Python, in `pins/sources.toml`, so re-pinning a dump date is a data edit.

Fetch and parse are separate functions because raw bytes are downloaded once and parsed
many times, and because it puts every test on the parse side with no network.

Record shapes: a shared `Document {doc_id, source, date, lang, title, text, meta}` for
anything with prose, so the linker does not care where a document came from. The
genuinely tabular sources — sanctions, the register, event tables — keep their own
shapes; forcing those into a Document is where the "the public side is a row, not a
document" awkwardness came from.

### Order of work

Cablegate first, alone, end to end. It has a committed test, carries the worst parse
hazard, and is the one source whose text cannot be redistributed — so it exercises fetch,
parse, provenance, pins and verify at once. Dodis second: different format, different
hazard, which proves the contract generalises rather than being one script wearing an
interface. Then Wikipedia, whose fetch is 25 GB and must be resumable and must no-op
against the copy already on the workstation. Parliament, sanctions and UCDP are mechanical
after that.

Stage 1 deliberately does not chunk or index — that is the consumer's job, and it is why
92 of those 118 GB exist — and does no entity resolution of any kind.

## Conventions

**Every derived file carries a provenance sidecar** (`<output>.provenance.json`) naming
its source, the fields kept, and the fields dropped *with a reason*. This is enforced by
the artifact writer, not by per-script discipline. It exists because two corpora were
quietly reduced and nothing downstream could tell: the event store lost casualties and
party names, the sanctions parse lost the programme and the justification. Both
projections were right for the job they were written for, wrong for the job they were
later used for, and invisible either way. A drift check then fails on any field that
disappears without a reason.

**Candidate builders cannot write output.** They return candidates; a single `emit()`
runs the gate suite and writes. This is the structural fix for the previous project's
worst failure — each question stream had its own prompt and its own gate, so the
dependency check that verifies the private side is load-bearing ran on one stream only,
and 82% of the resulting set had a decorative private hop. A new type cannot bypass a
gate because it has no path to the file.

**Model-written gold is verified, not trusted.** Both the question and its answer are
written by a model reading the two documents, so nothing external checks the answer — the
previous project's generator gave a Federal Councillor who left office in 1893 an exit
year of 1943. Three things stand in for the check that computed gold got for free: a
verification pass that re-reads the source documents and rejects an answer not supported
by them; repeat-and-agree on every model judgement, asking *n*=3 times and keeping only a
unanimous verdict (with sampling enabled — unanimity is meaningless at temperature 0);
and the necessity ablation, which now carries more weight than it did when five of seven
types read their gold out of the public record. The generator and the judge stay in
different model families to avoid self-enhancement bias.

**Models are injected, prompts are files.** One completion protocol, a llama-server
backend for the cluster, stubs in tests. Every deterministic stage then tests with no
GPU, which is already true of most of the ported tests and is worth preserving
deliberately.

**No module-level data paths.** The previous project has `DATA = Path("data/osint")` in a
dozen files, which is why nothing runs from another directory and why the cluster path
needs edits. One `config.py`, one environment variable, defaulting to the existing
location — never a copy.

**Every step is re-runnable and skips completed work**, so a failure at step 6 does not
cost steps 1–5.

**Port with tests.** If a file has a test in the previous project, the test comes with it.

## Decisions taken

**Wikidata is a retrieval surface**, not only a build-time reference. It is therefore
scoped like a corpus — the enwiki-sitelinked universe — and is the one source built from
a dump. Build-time lookups still go to the live endpoint.

**Questions and their answers are both model-written** from the two documents, rather
than gold being read out of the public record before a model is involved. This buys the
property the computed path could not give: an answer that comes from combining the two
sides, instead of a public fact the private document merely helps locate. The previous
project's measurements show what that fixes — the two types whose gold was a register
field were also the two most answerable from the public side alone, at 10 of 34 for
trajectory and 6 of 10 for officer. What it costs is the free correctness check, which
the verification pass and repeat-and-agree above are there to replace.

## Open decisions

1. **Does the property backfill leave the workstation?** Name reconciliation sends
   public-list names (sanctions, commercial register), which is uncontroversial. The
   backfill sends 126,903 QIDs derived from linking the cables. The previous project
   already does this against the public API, so there is precedent, but it should be a
   recorded decision rather than an oversight.

2. **Cablegate redistribution** is avoided by shipping build scripts, which settles the
   licensing question but not availability: the pinned source must be reachable years
   from now. Archiving the pinned inputs — the 25 GB enwiki dump, the truthy Wikidata
   dump and `cables.csv`, not the 118 GB of derived layers — is what makes tier 2
   rebuildable later.
