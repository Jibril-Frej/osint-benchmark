# OSINT Benchmark

A question-answering benchmark in which **every question requires two documents to
answer: one confidential, one public.** Neither side is sufficient on its own.

The questions are built from real archives. The confidential side is diplomatic
reporting that was not written for publication; the public side is an open record —
a parliamentary debate, a company register, a conflict-event database, an
encyclopedia — that happens to concern the same people, organisations and events.

## Why this benchmark exists

Retrieval systems are increasingly split across a boundary: a sensitive corpus that
must stay on-premises, and a public corpus reached over the network. Evaluating such
a system requires questions that genuinely exercise the split. A question answerable
from public sources alone measures nothing about the private side; a question
answerable from the private corpus alone never needs to cross the boundary at all.

Existing multi-hop benchmarks are built entirely from public text, so both hops sit
on the same side of any realistic trust boundary. This benchmark is built so that the
two hops fall on *opposite* sides, and so that this property is verified rather than
assumed.

## What makes a question valid

Every candidate question is re-solved three times, each time with part of the
evidence withheld:

| condition | evidence given | required outcome |
| --- | --- | --- |
| closed-book | none | model must fail |
| public only | the public document | model must fail |
| private only | the confidential document | model must fail |

If the model succeeds in any of the three, the question does not test the split:
either it was already known, or one side alone is enough. Only questions that fail
all three are kept.

Surviving questions then go through **human validation** — a person reads each
question beside both documents and confirms that the answer is right and that
answering really does need both.

## Sources

**Confidential side**

| corpus | content | language |
| --- | --- | --- |
| Cablegate | US diplomatic cables, 2003–2010 | English |
| Dodis | Swiss diplomatic documents, read by OCR from the scans | German, French |

**Public side**

| corpus | content |
| --- | --- |
| Swiss parliament | motions, interpellations and Federal Council replies |
| Swiss commercial register | company entries, officers, publications |
| UCDP | georeferenced armed-conflict events |
| Wikidata | entity relations |
| Wikipedia | article text |

Neither the source corpora nor the generated questions are redistributed in this
repository; it contains the code that builds the benchmark and the instructions for
obtaining each source. Licences differ per corpus and are documented per source.

## Question types

| type | asks | confidential side | public side |
| --- | --- | --- | --- |
| **posture** | does the public position match what was reported privately? | cable | parliamentary record |
| **event** | does the public record bear out the incident described privately? | cable | conflict-event database |
| **chronology** | how many days separate two events, one reported on each side? | cable | parliamentary record |
| **resolution** | which of several public figures does the private document mean? | cable, Dodis | Wikidata |
| **trajectory** | what became of a company named in the reporting? | cable | commercial register |
| **officer** | who signs for a company named in the reporting? | cable | commercial register |
| **association** | what publicly connects two people the reporting puts together? | cable, Dodis | Wikidata |

## How a question is built

1. **Find the entities.** Every document on both sides is annotated with the
   Wikidata entities it mentions, using a neural entity linker.
2. **Link entities and documents.** The annotations form a graph connecting each
   entity to every document that mentions it, on either side of the boundary.
3. **Pair one confidential and one public document** that share entities, are on the
   same subject, and are close in time.
4. **Write the question with an LLM.** The model is shown both documents and asked
   for a question that cannot be answered without combining them — and, for the
   question types whose answer is a judgement rather than a lookup, for the answer
   too.
5. **Prove both sides are needed** — the three-way check above.
6. **Validate by hand** — the human pass.

## Installation

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:Jibril-Frej/osint-benchmark.git
cd osint-benchmark
uv sync
```

Building the benchmark from scratch additionally requires GPU access for the
entity linking, OCR and question-writing stages. Each stage is a separate step that
can be run independently, so a partial rebuild does not require repeating the
expensive ones.

## Status

Under construction. This repository is being assembled from an earlier prototype;
the pipeline described above is the target design. See the issue tracker for what is
in place and what is not.

## Citation

A paper describing this benchmark is in preparation. Citation details will be added
here on publication.
