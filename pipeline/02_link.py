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
from functools import partial

from osint_benchmark import paths
from osint_benchmark.artifacts import Provenance, read_jsonl, write_records
from osint_benchmark.link import dictionary, refined, tabular
from osint_benchmark.link import settings as link_settings
from osint_benchmark.sources import base, get_source

PROSE = {"cablegate": "private", "dodis": "private", "parliament": "public"}
TABULAR = {"sanctions": "public", "ucdp": "public", "gdelt": "public"}
SIDES = {**PROSE, **TABULAR}


# A sentence in the shape the linker meets: a cable, upper-cased, opening on its routing
# preamble. If the entities in it come back, the environment is sound.
CHECK_CABLE = (
    "VZCZCXRO1234\nRR RUEHWEB\nSUBJECT: MEETING\n\n"
    "1. (C) THE AMBASSADOR MET SIEMENS AND THE INTERNATIONAL ATOMIC ENERGY AGENCY "
    "IN VIENNA ON 3 MARCH.\nSMITH"
)


def check(device: str) -> int:
    """Build the linker and link one sentence; return a process exit code."""
    try:
        linker = refined.load(device=device)
    except ImportError as exc:
        print(exc, file=sys.stderr)
        return 1
    text = refined.prepare(CHECK_CABLE)
    print(f"prepared: {text!r}")
    for mention in linker([text])[0]:
        print(f"  {mention.qid:<12} {mention.confidence:.3f}  {mention.surface_form}")
    return 0


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
    parser.add_argument(
        "--index",
        default="wikipedia_index",
        help="which entity index to scope to; wikipedia_index_simple is the small one",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "load the linker, link one sentence and exit. Run this first: building the "
            "model is where the install actually fails, and it needs no corpora, so a "
            "broken environment costs three minutes rather than the half hour of "
            "downloads that would otherwise come before it"
        ),
    )
    args = parser.parse_args(argv)

    if args.check:
        return check(args.device)

    index_path = paths.docs_dir() / f"{args.index}.jsonl"
    if not index_path.exists():
        raise SystemExit(f"{index_path} is missing: run pipeline/01_sources.py {args.index} first")
    index = list(read_jsonl(index_path))
    universe = frozenset(row["qid"] for row in index)
    print(f"public entity set: {len(universe)} entities with an English article")

    settings = link_settings.refined()
    min_title_words, min_title_chars = link_settings.dictionary()

    wanted = args.sources or list(SIDES)
    linker, method = None, ""
    prepare = None
    if any(name in PROSE for name in wanted):
        if args.dictionary:
            titles = index
            method = "exact article-title match (no model)"
            # Unrestricted, single-word titles are unusable: Wikipedia has articles called
            # "Told" and "Right". Restricted to a handful of known entities the danger is
            # gone and the requirement only hurts -- "Afghanistan" is one word.
            min_words = min_title_words
            if args.restrict_to:
                other = paths.data_dir() / "links" / f"{args.restrict_to}.jsonl"
                if not other.exists():
                    raise SystemExit(f"{other} is missing: link {args.restrict_to} first")
                allowed = {e["qid"] for row in read_jsonl(other) for e in row["entities"]}
                titles = [row for row in index if row["qid"] in allowed]
                min_words = 1
                method += f", restricted to the {len(allowed)} entities {args.restrict_to} names"
            linker = dictionary.linker(
                dictionary.build_dictionary(titles, min_chars=min_title_chars, min_words=min_words)
            )
            print(f"linking prose by title match over {len(titles)} candidate entities")
        else:
            try:
                linker = refined.load(
                    model_name=settings.model,
                    device=args.device,
                    entity_set=settings.entity_set,
                )
            except ImportError as exc:
                print(exc, file=sys.stderr)
                return 1
            method = f"ReFinED {settings.model} over the {settings.entity_set} entity set"
            # Truecasing and narrative extraction, not decoration: the cables are ALL CAPS
            # and open with routing lines. Linking them raw is what makes a model linker
            # perform no better than the dictionary it was meant to replace.
            prepare = partial(refined.prepare, max_chars=settings.max_chars)
            print(f"linking prose with {method}, batch {settings.batch_size}")

    for name in wanted:
        source = get_source(name)
        records = read_jsonl(base.output_path(source))
        if args.stride > 1:
            records = (row for i, row in enumerate(records) if i % args.stride == 0)
        if args.limit:
            records = (row for _, row in zip(range(args.limit), records, strict=False))

        if name in PROSE:
            how = method
            rows_iter = refined.link_documents(
                records,
                linker,
                PROSE[name],
                universe,
                confidence=settings.confidence,
                batch_size=settings.batch_size,
                prepare_text=prepare,
            )
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
        linked = mentions = 0
        distinct: set[str] = set()
        for row in read_jsonl(output):
            linked += bool(row["entities"])
            mentions += len(row["entities"])
            distinct.update(entity["qid"] for entity in row["entities"])
        print(
            f"{name}: {rows} documents, {linked} with at least one entity, "
            f"{mentions} mentions over {len(distinct)} distinct entities -> {output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
