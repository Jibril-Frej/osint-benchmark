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
from osint_benchmark.link import dictionary, refined, tabular
from osint_benchmark.sources import base, get_source

PROSE = {"cablegate": "private", "dodis": "private", "parliament": "public"}
TABULAR = {"sanctions": "public", "ucdp": "public", "gdelt": "public"}
SIDES = {**PROSE, **TABULAR}


def main(argv: list[str] | None = None) -> int:
    """Link the requested sources; return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sources", nargs="*", help=f"default: all of {', '.join(SIDES)}")
    parser.add_argument("--limit", type=int, help="link only N documents")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help=(
            "take every Nth document rather than the first N. The corpora are sorted, so "
            "the first N cables are all from the 1960s and the first N events are all one "
            "country -- a biased slice that bridges nothing"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dictionary",
        action="store_true",
        help="link by exact article-title match instead of ReFinED: no model, much worse",
    )
    parser.add_argument(
        "--restrict-to",
        metavar="SOURCE",
        help=(
            "with --dictionary, look only for entities already linked on the other side. "
            "Matching all 7.5M titles against ALL-CAPS cable text finds a title for almost "
            "any phrase; restricting to entities the public side names is both far cleaner "
            "and closer to what a bridge is"
        ),
    )
    args = parser.parse_args(argv)

    index = list(read_jsonl(paths.docs_dir() / "wikipedia_index.jsonl"))
    universe = frozenset(row["qid"] for row in index)
    print(f"public entity set: {len(universe)} entities with an English article")

    wanted = args.sources or list(SIDES)
    linker, method = None, ""
    if any(name in PROSE for name in wanted):
        if args.dictionary:
            titles = index
            method = "exact article-title match (no model)"
            if args.restrict_to:
                other = paths.data_dir() / "links" / f"{args.restrict_to}.jsonl"
                if not other.exists():
                    raise SystemExit(f"{other} is missing: link {args.restrict_to} first")
                allowed = {e["qid"] for row in read_jsonl(other) for e in row["entities"]}
                titles = [row for row in index if row["qid"] in allowed]
                method += f", restricted to the {len(allowed)} entities {args.restrict_to} names"
            linker = dictionary.linker(dictionary.build_dictionary(titles))
            print(f"linking prose by title match over {len(titles)} candidate entities")
        else:
            try:
                linker = refined.load(device=args.device)
            except ImportError as exc:
                print(exc, file=sys.stderr)
                return 1
            method = "ReFinED"

    for name in wanted:
        source = get_source(name)
        records = read_jsonl(base.output_path(source))
        if args.stride > 1:
            records = (row for i, row in enumerate(records) if i % args.stride == 0)
        if args.limit:
            records = (row for _, row in zip(range(args.limit), records, strict=False))

        if name in PROSE:
            how = method
            rows_iter = refined.link_documents(records, linker, PROSE[name], universe)
        else:
            how = "name reconciliation against the live Wikidata endpoint"
            rows_iter = tabular.link_records(records, name, TABULAR[name], universe)

        output = paths.data_dir() / "links" / f"{name}.jsonl"
        rows = write_records(
            output,
            rows_iter,
            Provenance(
                source=f"{name} linked by {how}",
                source_fields=("doc_id", "text"),
                kept={"doc_id": "doc_id", "text": "entities (mentions resolved from it)"},
                kind="index",
                note=(
                    f"Linked by {how}, filtered to entities holding an English "
                    "Wikipedia article."
                    + (f" Limited to the first {args.limit} documents." if args.limit else "")
                ),
            ),
        )
        linked = sum(1 for row in read_jsonl(output) if row["entities"])
        print(f"{name}: {rows} documents, {linked} with at least one entity -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
