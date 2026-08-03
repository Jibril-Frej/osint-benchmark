"""Step 8: render the questions into one page for the human pass. Runs on the workstation.

The gates decide whether a question is well-formed; only a person can decide whether it is
well-founded. This lays each question beside the evidence both claims rest on.

Usage::

    uv run python pipeline/08_review.py
    uv run python pipeline/08_review.py --verdicts verdicts.json   # merge a review back

One self-contained HTML file, no server and no external asset, so questions derived from
the confidential corpora never leave the workstation.
"""

from __future__ import annotations

import argparse
import json

from osint_benchmark import paths
from osint_benchmark.generate.evidence import evidence_texts
from osint_benchmark.release.load import load_items
from osint_benchmark.review import page


def main(argv: list[str] | None = None) -> int:
    """Render the review page, or merge a completed review back."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verdicts", help="a verdicts.json exported from the page")
    args = parser.parse_args(argv)

    items_dir = paths.data_dir() / "items"
    source = items_dir / "measured.jsonl"
    if not source.exists():
        source = items_dir / "accepted.jsonl"
    if not source.exists():
        raise SystemExit(f"no items in {items_dir}: run pipeline/06_generate.py first")

    items = load_items(source)

    if args.verdicts:
        decisions = json.loads(paths.ROOT.joinpath(args.verdicts).read_text(encoding="utf-8"))
        kept = [i for i in items if decisions.get(i.item_id) == "keep"]
        output = items_dir / "reviewed.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for item in kept:
                handle.write(json.dumps(item.to_json(), ensure_ascii=False, sort_keys=True) + "\n")
        print(f"{len(kept)} kept of {len(items)} reviewed -> {output}")
        return 0

    html = page.render(items, evidence_texts())
    output = paths.data_dir() / "review.html"
    output.write_text(html, encoding="utf-8")
    print(f"{len(items)} questions -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
