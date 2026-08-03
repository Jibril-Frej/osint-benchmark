"""Step 3: invert the links into the entity-document graph. Runs on the workstation.

One row per entity naming documents on both sides of the trust boundary. Those are the
bridges the pairing step draws from.

Usage::

    uv run python pipeline/03_graph.py
    uv run python pipeline/03_graph.py --max-private 50
"""

from __future__ import annotations

import argparse

from osint_benchmark import paths
from osint_benchmark.artifacts import Provenance, read_jsonl, write_records
from osint_benchmark.graph import bridge

DEFAULT_MAX_PRIVATE = 200


def main(argv: list[str] | None = None) -> int:
    """Build the bridge map from every link file present."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--max-private",
        type=int,
        default=DEFAULT_MAX_PRIVATE,
        help="drop entities named in more private documents than this (0 for no limit)",
    )
    args = parser.parse_args(argv)

    links_dir = paths.data_dir() / "links"
    files = sorted(links_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no link files in {links_dir}: run pipeline/02_link.py first")

    rows = [row for path in files for row in read_jsonl(path)]
    bridges = bridge.build(rows)
    cap = args.max_private or None

    output = paths.data_dir() / "graph" / "bridges.jsonl"
    written = write_records(
        output,
        bridge.bridging(bridges, max_private=cap),
        Provenance(
            source=f"links from {', '.join(p.stem for p in files)}",
            source_fields=("doc_id", "side", "entities"),
            kept={
                "doc_id": "private / public document id lists",
                "entities": "qid, and surface_forms with counts",
                "side": "which list a document joins",
            },
            kind="index",
            note=(
                f"Only entities named on both sides are kept, from {len(rows)} linked "
                f"documents and {len(bridges)} distinct entities."
                + (f" Entities in more than {cap} private documents dropped." if cap else "")
            ),
        ),
        rows_in=len(bridges),
    )
    print(f"{len(bridges)} entities, {written} bridging both sides -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
