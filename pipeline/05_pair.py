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
from osint_benchmark.pair import join
from osint_benchmark.sources import base, get_source


def document_dates(sources: list[str]) -> dict[str, str | None]:
    """Return ``doc_id -> date`` for the given corpora."""
    dates: dict[str, str | None] = {}
    for name in sources:
        output = base.output_path(get_source(name))
        if not output.exists():
            continue
        for row in read_jsonl(output):
            dates[row["doc_id"]] = row.get("date")
    return dates


def main(argv: list[str] | None = None) -> int:
    """Pair private documents with public records over their shared entities."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window-days", type=int, default=join.DEFAULT_WINDOW_DAYS)
    args = parser.parse_args(argv)

    bridges_path = paths.data_dir() / "graph" / "bridges.jsonl"
    if not bridges_path.exists():
        raise SystemExit(f"{bridges_path} is missing: run pipeline/03_graph.py first")

    sources = linked_sources()
    print(f"dating documents from: {', '.join(sources) or 'nothing linked'}")
    dates = document_dates(sources)
    pairs = join.pair_documents(
        read_jsonl(bridges_path), dates, dates, window_days=args.window_days
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
                f"One row per (private document, public record, shared entity). "
                f"same_period is a {args.window_days}-day window; days_apart is signed, "
                "public minus private."
            ),
        ),
    )
    contemporaneous = sum(1 for row in read_jsonl(output) if row["same_period"])
    print(f"{written} pairs, {contemporaneous} contemporaneous -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
