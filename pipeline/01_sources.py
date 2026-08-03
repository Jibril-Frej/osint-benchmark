"""Step 1: fetch, parse and verify the bulk corpora. Run this first.

The bulk sources are the ones that depend on nothing — no entity list, no linker output —
so they can all be built before anything else exists. The entity-driven fetches (the
commercial register, article text for bridge entities) come later, at step 4.

Usage::

    uv run python pipeline/01_sources.py                      # every bulk source
    uv run python pipeline/01_sources.py cablegate            # one of them
    uv run python pipeline/01_sources.py --step fetch         # only acquire
    uv run python pipeline/01_sources.py --write-pins         # record a hash baseline

Each step is re-runnable: a raw file already on disk is never re-downloaded, so a failure
here costs only the work that had not finished.
"""

from __future__ import annotations

import argparse
import sys

from osint_benchmark import config
from osint_benchmark.sources import ALL, base, get_source

STEPS = ("fetch", "parse", "verify")


def main(argv: list[str] | None = None) -> int:
    """Run the requested steps over the requested sources; return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "sources", nargs="*", default=None, help=f"default: all of {', '.join(ALL)}"
    )
    parser.add_argument("--step", choices=(*STEPS, "all"), default="all")
    parser.add_argument(
        "--write-pins",
        action="store_true",
        help="record the built documents' hashes as the baseline instead of checking them",
    )
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="skip checksumming the raw files (they are large; the size check still runs)",
    )
    args = parser.parse_args(argv)

    names = args.sources or list(ALL)
    steps = STEPS if args.step == "all" else (args.step,)
    print(f"raw={config.raw_dir()}  docs={config.docs_dir()}  pins={config.pins_dir()}")

    failed = False
    for name in names:
        source = get_source(name)

        if "fetch" in steps:
            try:
                for path in base.fetch(source, check_hash=not args.skip_hash):
                    print(f"{name}: raw ok  {path}")
            except base.SourceUnavailable as exc:
                print(f"{name}: {exc}", file=sys.stderr)
                failed = True
                continue

        if "parse" in steps:
            output = base.parse(source)
            print(f"{name}: parsed   {output}")

        if "verify" in steps:
            report = base.verify(source, write_pins=args.write_pins)
            print(report.summary())
            for doc_id in report.changed[:5]:
                print(f"{name}:   changed {doc_id}", file=sys.stderr)
            failed = failed or not (report.ok or args.write_pins)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
