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
from osint_benchmark.generate.evidence import evidence_texts, sources_for
from osint_benchmark.release.load import load_items
from osint_benchmark.review import calibrate, page


def main(argv: list[str] | None = None) -> int:
    """Render the review page, or merge a completed review back."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verdicts", help="a verdicts.json exported from the page")
    parser.add_argument(
        "--calibrate",
        type=int,
        metavar="N",
        help=(
            "render N questions, half from each side of the necessity verdict, asking "
            "whether the pipeline got them right rather than whether they are good. The "
            "judge and the ablations decide every figure this benchmark reports and "
            "neither has been checked against a person"
        ),
    )
    args = parser.parse_args(argv)

    items_dir = paths.data_dir() / "items"
    source = items_dir / "measured.jsonl"
    if not source.exists():
        source = items_dir / "accepted.jsonl"
    if not source.exists():
        raise SystemExit(f"no items in {items_dir}: run pipeline/06_generate.py first")

    items = load_items(source)
    # Which corpora to read is taken from the items, not from whichever link files happen
    # to be on disk. Rendering from an items file alone otherwise silently produced a page
    # whose every document read "(not available)" -- the questions were there, the evidence
    # they rest on was not, and the page still looked complete.
    texts = evidence_texts(sources_for(items))

    if args.verdicts:
        decisions = json.loads(paths.ROOT.joinpath(args.verdicts).read_text(encoding="utf-8"))
        kept = [i for i in items if decisions.get(i.item_id) == "keep"]
        dropped = [i for i in items if decisions.get(i.item_id) == "drop"]
        undecided = len(items) - len(kept) - len(dropped)
        output = items_dir / "reviewed.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for item in kept:
                handle.write(json.dumps(item.to_json(), ensure_ascii=False, sort_keys=True) + "\n")
        print(f"{len(kept)} kept, {len(dropped)} dropped of {len(items)} -> {output}")
        # A partial review is the normal case -- skimming twenty of a hundred is the point.
        # Reporting "kept of all" made 111 questions nobody had looked at indistinguishable
        # from 111 somebody had rejected.
        if undecided:
            print(f"{undecided} were never reviewed, and are not in {output.name}")
        return 0

    if args.calibrate:
        chosen = calibrate.sample(items, args.calibrate)
        if not chosen:
            raise SystemExit(f"no measured items in {source}: run pipeline/07_necessity.py first")
        output = paths.data_dir() / "calibration.html"
        output.write_text(calibrate.render(chosen, texts), encoding="utf-8")
        needs = sum(1 for i in chosen if i.necessity.needs_both)
        print(f"{len(chosen)} questions ({needs} the pipeline calls necessary) -> {output}")
        return 0

    # The models that produced these are a fact about the run, recorded in the artefact's
    # provenance sidecar rather than on each item. A reviewer weighing a necessity verdict
    # needs to know whose verdict it is, so the page carries it rather than leaving it a
    # file away.
    sidecar = source.with_suffix(source.suffix + ".provenance.json")
    note = ""
    if sidecar.exists():
        note = json.loads(sidecar.read_text(encoding="utf-8")).get("note", "")
    html = page.render(items, texts, note)
    output = paths.data_dir() / "review.html"
    output.write_text(html, encoding="utf-8")
    print(f"{len(items)} questions -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
