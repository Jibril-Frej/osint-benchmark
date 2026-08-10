"""Step 4: fetch the public evidence for the bridge entities. Workstation, live sources.

Everything here depends on the graph, which is why it is not step 1: there is nothing to
ask for until the linker has said which entities matter. Fetching before that is what left
the previous project with statements for 35% of its entities and no symptom.

Usage::

    uv run python pipeline/04_public.py
    uv run python pipeline/04_public.py --limit 50      # a subset, for a smoke run

Every record carries the revision it was read from: these sources are live, so a gold
answer taken from one is only correct against a particular version of it.
"""

from __future__ import annotations

import argparse
import sys
from itertools import chain

from osint_benchmark import paths, public
from osint_benchmark.artifacts import Provenance, read_jsonl, write_records
from osint_benchmark.generate import association
from osint_benchmark.public import articles, wikidata


def bridge_entities(limit: int | None = None) -> list[str]:
    """Return the QIDs named on both sides of the boundary."""
    path = paths.data_dir() / "graph" / "bridges.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} is missing: run pipeline/03_graph.py first")
    qids = [row["qid"] for row in read_jsonl(path)]
    return qids[:limit] if limit else qids


def linked_entities(confidence: float = 0.0, limit: int | None = None) -> list[str]:
    """Return every QID any linked document names, ordered and deduplicated.

    Wider than the bridges, and the typed question builders need it to be. An association
    is two people in *one* document who share a public neighbour -- neither of them has to
    appear in a public corpus at all, so scoping the fetch to bridges would starve exactly
    the type that does not use them.

    The previous project's slice was 62,497 entities for the whole of Cablegate and Dodis,
    which is 1,250 batched requests. This is scoped the same way: everything linked, and
    linking already restricts to entities holding an English Wikipedia article.
    """
    links = paths.data_dir() / "links"
    if not links.is_dir():
        raise SystemExit(f"{links} is missing: run pipeline/02_link.py first")
    seen: dict[str, None] = {}
    for path in sorted(links.glob("*.jsonl")):
        for row in read_jsonl(path):
            for entity in row.get("entities", []):
                if entity.get("confidence", 1.0) >= confidence:
                    seen.setdefault(entity["qid"])
    qids = list(seen)
    return qids[:limit] if limit else qids


def neighbours_of(records: list[dict], predicates: frozenset[str]) -> list[str]:
    """Return the entities the fetched ones point at through the given predicates.

    A second hop, and the association type cannot work without it. What that type asks for
    is the organisation two people both belong to — and an organisation nobody wrote a
    document about is never linked, so it is absent from the first hop by construction. Its
    label is the answer and its article is the public evidence, so both have to be fetched.
    """
    seen = {record["qid"] for record in records}
    out: dict[str, None] = {}
    for record in records:
        for predicate, values in (record.get("statements") or {}).items():
            if predicate not in predicates:
                continue
            for value in values:
                if isinstance(value, str) and value.startswith("Q") and value not in seen:
                    out.setdefault(value)
    return list(out)


def already(name: str) -> tuple[list[dict], set[str]]:
    """Return the records a previous run of this step left, and which entities they cover.

    Fetching the slice for a full-scale link set is hours of network. A job that runs out of
    time after two of them should not start again from nothing, and the records are already
    on disk in the shape they will be rewritten in.
    """
    path = paths.data_dir() / "facts" / name
    if not path.exists():
        return [], set()
    rows = list(read_jsonl(path))
    return rows, {row["qid"] for row in rows if row.get("qid")}


def titles_for(qids: list[str], index_name: str = "wikipedia_index") -> list[tuple[str, str]]:
    """Return (QID, article title) for the entities that have an article."""
    index = paths.docs_dir() / f"{index_name}.jsonl"
    if not index.exists():
        raise SystemExit(f"{index} is missing: run pipeline/01_sources.py {index_name} first")
    wanted = set(qids)
    return [(row["qid"], row["title"]) for row in read_jsonl(index) if row["qid"] in wanted]


def main(argv: list[str] | None = None) -> int:
    """Fetch Wikidata statements and article text for the bridge entities."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="fetch only the first N entities")
    parser.add_argument(
        "--scope",
        choices=("bridges", "linked"),
        default="bridges",
        help=(
            "which entities to fetch facts for. 'bridges' is enough for the generic "
            "question type; the typed builders need 'linked', because an association is "
            "two people in one document and neither need appear in a public corpus"
        ),
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="with --scope linked, ignore mentions the linker was less sure of than this",
    )
    parser.add_argument("--index", default="wikipedia_index", help="which entity index to read")
    parser.add_argument(
        "--neighbours",
        action="store_true",
        help=(
            "also fetch the entities the first hop points at through a relational "
            "predicate. The association type answers with an organisation that may itself "
            "appear in no document, so it is reachable only this way"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "keep what an earlier run of this step already fetched and ask only for the "
            "rest. The slice for a full link set is hours of network, and a job that runs "
            "out of time should not start again from nothing"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "how many requests to have in flight. The article leads are latency-bound and "
            "gain almost linearly; the Wikidata slice is bandwidth-bound and gains about "
            "half again"
        ),
    )
    args = parser.parse_args(argv)

    if args.scope == "linked":
        qids = linked_entities(args.min_confidence, args.limit)
    else:
        qids = bridge_entities(args.limit)
    print(f"{len(qids)} {args.scope} entities")
    failures: list[str] = []

    def note(key: str, exc: Exception) -> None:
        failures.append(f"{key}: {exc}")

    kept, have = already("wikidata.jsonl") if args.resume else ([], set())
    extra_asked: list[str] = []
    if kept:
        print(f"resuming: {len(kept)} entities already fetched")
    # Held rather than streamed to disk: the second hop is chosen from what the first one
    # said, so the records have to be read before any of them can be written. The previous
    # project's whole slice was 24 MB.
    wanted = [qid for qid in qids if qid not in have]
    records = kept + list(
        wikidata.fetch_entities(
            wanted,
            on_error=note,
            workers=args.workers,
            progress=public.reporting("wikidata", len(wanted)),
        )
    )
    if args.neighbours:
        extra = [qid for qid in neighbours_of(records, association.RELATIONAL) if qid not in have]
        extra_asked = extra
        print(f"{len(extra)} entities reachable through a relational predicate")
        records += list(
            wikidata.fetch_entities(
                extra,
                on_error=note,
                workers=args.workers,
                progress=public.reporting("neighbours", len(extra)),
            )
        )
        qids += extra

    facts = paths.data_dir() / "facts" / "wikidata.jsonl"
    written = write_records(
        facts,
        records,
        Provenance(
            source="wikidata.org Special:EntityData",
            source_fields=("labels", "descriptions", "claims", "sitelinks", "aliases"),
            kept={
                "labels": "label (English)",
                "descriptions": "description (English)",
                "claims": f"statements ({len(wikidata.KEEP_PROPERTIES)} selected properties)",
            },
            dropped={
                "sitelinks": "the entity set is already scoped by wikipedia_index",
                "aliases": "the linker resolves surface forms; no question asks for them",
            },
            kind="derived",
            note="Every record carries the lastrevid it was read at; Wikidata is live.",
        ),
        # Everything a record could have come from: the entities asked for this run, plus
        # the ones an earlier run already fetched. Counting only the former made rows_out
        # exceed rows_in and failed the provenance check -- correctly, because a resumed run
        # carries records for entities the new link set no longer asks about (the previous
        # run's second hop), and they are inputs too.
        rows_in=len(set(qids) | have),
    )
    print(f"wikidata: {written} written, {len(wanted) + len(extra_asked)} newly fetched -> {facts}")

    have_text, have_articles = already("articles.jsonl") if args.resume else ([], set())
    if have_text:
        print(f"resuming: {len(have_text)} articles already fetched")
    # Titles for everything on disk, not only for what this run asked about: a resumed run's
    # kept records are as entitled to an article as the ones it fetched.
    pairs = [
        pair
        for pair in titles_for(sorted(set(qids) | have), args.index)
        if pair[0] not in have_articles
    ]
    text = paths.data_dir() / "facts" / "articles.jsonl"
    written = write_records(
        text,
        chain(
            have_text,
            articles.fetch_articles(
                pairs,
                on_error=note,
                workers=args.workers,
                progress=public.reporting("articles", len(pairs)),
            ),
        ),
        Provenance(
            source="en.wikipedia.org action API (prop=extracts|revisions|info, exintro)",
            source_fields=("title", "pageid", "extract", "revisions", "length", "missing"),
            kept={
                "title": "title",
                "pageid": "page_id",
                "extract": "text (lead section, plain text)",
                "revisions": "revision and revision_date",
                "length": "article_bytes (the whole article's size, the prominence signal)",
            },
            dropped={"missing": "used as a filter: an absent article yields no record"},
            kind="derived",
            note=(
                "Lead sections only. The full corpus a system searches is a different "
                "artefact and is specified, not built here."
            ),
        ),
        # Both halves of what is written, for the same reason as the slice above: this
        # stream is the kept articles followed by the newly fetched ones. Counting only the
        # second made rows_out exceed rows_in -- the identical mistake, one write later.
        rows_in=len(pairs) + len(have_text),
    )
    print(f"articles: {written} written, {len(pairs)} newly fetched -> {text}")

    for failure in failures[:10]:
        print(f"  failed {failure}", file=sys.stderr)
    if failures:
        print(f"{len(failures)} fetches failed (unknown data, not absent)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
