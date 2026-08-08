"""Step 5: pair one confidential document with one public record. Runs on the workstation.

Every pair shares an entity. Whether the two are close enough in time to be about the same
events is recorded rather than assumed -- Dodis is archival and its pairs are normally not
contemporaneous, so the interval-based question types do not apply to them.

Usage::

    uv run python pipeline/05_pair.py
    uv run python pipeline/05_pair.py --window-days 30
"""

from __future__ import annotations

import argparse

from osint_benchmark import paths
from osint_benchmark.artifacts import Provenance, read_jsonl, write_records
from osint_benchmark.generate.evidence import linked_sources
from osint_benchmark.graph import entity_types
from osint_benchmark.pair import join
from osint_benchmark.sources import base, get_source, refs


def document_dates(sources: list[str]) -> dict[str, str | None]:
    """Return ``source:doc_id -> date`` for the given corpora.

    The date is read under whatever key each source calls it. Reading ``date`` from every
    record left every sanctions listing undated -- it calls the field ``list_date`` -- so
    no pair had an interval and every one was recorded as non-contemporaneous. That was
    reported as a fact about the corpora covering different eras.
    """
    dates: dict[str, str | None] = {}
    for name in sources:
        output = base.output_path(get_source(name))
        if not output.exists():
            continue
        for row in read_jsonl(output):
            dates[refs.ref(name, row["doc_id"])] = refs.record_date(name, row)
    return dates


def main(argv: list[str] | None = None) -> int:
    """Pair private documents with public records over their shared entities."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window-days", type=int, default=join.DEFAULT_WINDOW_DAYS)
    parser.add_argument(
        "--max-pairs-per-entity",
        type=int,
        default=50,
        help="cap the pairs one entity may produce; 42 bridges once produced 30,416 pairs",
    )
    parser.add_argument(
        "--any-entity-type",
        action="store_true",
        help=(
            "pair on every bridge, including countries. The first real run bridged only on "
            "countries and asked what connects Cameroon and Canada; the answer was that "
            "both documents mention international relations"
        ),
    )
    args = parser.parse_args(argv)

    bridges_path = paths.data_dir() / "graph" / "bridges.jsonl"
    if not bridges_path.exists():
        raise SystemExit(f"{bridges_path} is missing: run pipeline/03_graph.py first")

    sources = linked_sources()
    print(f"dating documents from: {', '.join(sources) or 'nothing linked'}")
    dates = document_dates(sources)
    bridges = list(read_jsonl(bridges_path))

    if args.any_entity_type:
        note = "Pairs formed on every bridge, including countries."
    else:
        facts_path = paths.data_dir() / "facts" / "wikidata.jsonl"
        if not facts_path.exists():
            raise SystemExit(f"{facts_path} is missing: run pipeline/04_public.py first")
        facts = list(read_jsonl(facts_path))
        # Resolve organisation-hood through the class hierarchy once, for every class these
        # entities are an instance of. A flat list of class ids cannot do it: Associated
        # Press is a news agency, which is a subclass of organisation.
        organisations, places = entity_types.kinds_present(entity_types.classes_in(facts))
        print(f"entity classes present: {len(organisations)} organisations, {len(places)} places")
        counts = entity_types.summarise(facts, organisations, places)
        allowed = entity_types.bridgeable_qids(facts, organisations=organisations, places=places)
        before = len(bridges)
        bridges = [b for b in bridges if b["qid"] in allowed]
        print(f"entity types: {counts}; {len(bridges)} of {before} bridges may anchor a pair")
        note = (
            f"Only entities whose Wikidata class can anchor a question: {counts}. "
            "Only people and organisations may anchor a question. Places are excluded "
            "however they are classified: they co-occur with everything and so "
            "distinguish nothing."
        )

    pairs = join.pair_documents(
        bridges,
        dates,
        dates,
        window_days=args.window_days,
        max_per_entity=args.max_pairs_per_entity,
    )

    output = paths.data_dir() / "pairs" / "pairs.jsonl"
    written = write_records(
        output,
        (pair.to_json() for pair in pairs),
        Provenance(
            source=f"bridges from {bridges_path}",
            source_fields=("qid", "private", "public"),
            kept={
                "qid": "qid",
                "private": "private_id (one pair per document)",
                "public": "public_id (one pair per record)",
            },
            kind="derived",
            note=(
                f"One row per (private document, public record, shared entity), at most "
                f"{args.max_pairs_per_entity} per entity. same_period is a "
                f"{args.window_days}-day window; days_apart is signed, public minus "
                f"private. {note}"
            ),
        ),
    )
    contemporaneous = sum(1 for row in read_jsonl(output) if row["same_period"])
    print(f"{written} pairs, {contemporaneous} contemporaneous -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
