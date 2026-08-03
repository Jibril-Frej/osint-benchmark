"""Step 2: annotate documents with the Wikidata entities they mention. Needs the cluster.

Prose is linked by a model (ReFinED for English); tabular sources are reconciled by name
or external code against the live Wikidata endpoint. Everything is filtered to the public
entity set built in step 1, since an entity with no English Wikipedia article cannot
bridge to a public corpus scoped to entities that have one.

Usage::

    uv run python pipeline/02_link.py cablegate           # needs ReFinED: uv sync --extra link
    uv run python pipeline/02_link.py cablegate --limit 500   # a subset, for a smoke run

ReFinED is an optional dependency because it pulls torch. Without it this step reports how
to install it rather than failing obscurely.
"""

from __future__ import annotations

import argparse
import sys

from osint_benchmark import paths
from osint_benchmark.artifacts import Provenance, read_jsonl, write_records
from osint_benchmark.link import refined
from osint_benchmark.sources import base, get_source

PROSE = {"cablegate": "private", "dodis": "private", "parliament": "public"}


def entity_set() -> frozenset[str]:
    """Return the QIDs holding an English Wikipedia article."""
    index = paths.docs_dir() / "wikipedia_index.jsonl"
    if not index.exists():
        raise SystemExit(f"{index} is missing: run pipeline/01_sources.py wikipedia_index first")
    return frozenset(row["qid"] for row in read_jsonl(index))


def main(argv: list[str] | None = None) -> int:
    """Link the requested sources; return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sources", nargs="*", help=f"default: all of {', '.join(PROSE)}")
    parser.add_argument("--limit", type=int, help="link only the first N documents")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    try:
        linker = refined.load(device=args.device)
    except ImportError as exc:
        print(exc, file=sys.stderr)
        return 1

    universe = entity_set()
    print(f"public entity set: {len(universe)} entities with an English article")

    for name in args.sources or list(PROSE):
        source = get_source(name)
        documents = read_jsonl(base.output_path(source))
        if args.limit:
            documents = (row for _, row in zip(range(args.limit), documents, strict=False))

        output = paths.data_dir() / "links" / f"{name}.jsonl"
        rows = write_records(
            output,
            refined.link_documents(documents, linker, PROSE[name], universe),
            Provenance(
                source=f"{name} documents linked by ReFinED",
                source_fields=("doc_id", "text"),
                kept={"doc_id": "doc_id", "text": "entities (mentions resolved from it)"},
                kind="index",
                note=(
                    f"Confidence floor {refined.DEFAULT_CONFIDENCE}, filtered to entities "
                    "holding an English Wikipedia article."
                    + (f" Limited to the first {args.limit} documents." if args.limit else "")
                ),
            ),
        )
        linked = sum(1 for row in read_jsonl(output) if row["entities"])
        print(f"{name}: {rows} documents, {linked} with at least one entity -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
