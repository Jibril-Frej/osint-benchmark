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

from osint_benchmark import paths
from osint_benchmark.artifacts import Provenance, read_jsonl, write_records
from osint_benchmark.public import articles, wikidata


def bridge_entities(limit: int | None = None) -> list[str]:
    """Return the QIDs named on both sides of the boundary."""
    path = paths.data_dir() / "graph" / "bridges.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} is missing: run pipeline/03_graph.py first")
    qids = [row["qid"] for row in read_jsonl(path)]
    return qids[:limit] if limit else qids


def titles_for(qids: list[str]) -> list[tuple[str, str]]:
    """Return (QID, article title) for the entities that have an article."""
    index = paths.docs_dir() / "wikipedia_index.jsonl"
    if not index.exists():
        raise SystemExit(f"{index} is missing: run pipeline/01_sources.py wikipedia_index first")
    wanted = set(qids)
    return [(row["qid"], row["title"]) for row in read_jsonl(index) if row["qid"] in wanted]


def main(argv: list[str] | None = None) -> int:
    """Fetch Wikidata statements and article text for the bridge entities."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="fetch only the first N bridge entities")
    args = parser.parse_args(argv)

    qids = bridge_entities(args.limit)
    print(f"{len(qids)} bridge entities")
    failures: list[str] = []

    def note(key: str, exc: Exception) -> None:
        failures.append(f"{key}: {exc}")

    facts = paths.data_dir() / "facts" / "wikidata.jsonl"
    written = write_records(
        facts,
        wikidata.fetch_entities(qids, on_error=note),
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
        rows_in=len(qids),
    )
    print(f"wikidata: {written} of {len(qids)} entities -> {facts}")

    pairs = titles_for(qids)
    text = paths.data_dir() / "facts" / "articles.jsonl"
    written = write_records(
        text,
        articles.fetch_articles(pairs, on_error=note),
        Provenance(
            source="en.wikipedia.org action API (prop=extracts, exintro)",
            source_fields=("title", "pageid", "extract", "revisions", "missing"),
            kept={
                "title": "title",
                "pageid": "page_id",
                "extract": "text (lead section, plain text)",
                "revisions": "revision and revision_date",
            },
            dropped={"missing": "used as a filter: an absent article yields no record"},
            kind="derived",
            note=(
                "Lead sections only, for bridge entities only. The full corpus a system "
                "searches is a different artefact and is specified, not built here."
            ),
        ),
        rows_in=len(pairs),
    )
    print(f"articles: {written} of {len(pairs)} entities -> {text}")

    for failure in failures[:10]:
        print(f"  failed {failure}", file=sys.stderr)
    if failures:
        print(f"{len(failures)} fetches failed (unknown data, not absent)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
