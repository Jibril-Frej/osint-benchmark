"""Step 1: fetch, parse and verify the bulk corpora. Run this first.

The bulk sources are the ones that depend on nothing — no entity list, no linker output —
so they can all be built before anything else exists. The entity-driven fetches (the
commercial register, article text for bridge entities) come later, at step 4.

Usage::

    uv run python pipeline/01_sources.py              # every bulk source
    uv run python pipeline/01_sources.py cablegate    # just one

Re-runnable: a raw file already on disk is not downloaded again, and an interrupted
download resumes, so a failed run costs only the work that had not finished.
"""

from __future__ import annotations

import argparse
import sys

from osint_benchmark import paths
from osint_benchmark.sources import ALL, base, get_source


def main(argv: list[str] | None = None) -> int:
    """Build every requested source; return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sources", nargs="*", help=f"default: all of {', '.join(ALL)}")
    args = parser.parse_args(argv)

    print(f"raw={paths.raw_dir()}  docs={paths.docs_dir()}  pins={paths.pins_dir()}")
    failed = False
    for name in args.sources or ALL:
        source = get_source(name)
        try:
            for path in base.fetch(source):
                print(f"{name}: raw ok  {path}")
        except base.SourceUnavailable as exc:
            print(f"{name}: {exc}", file=sys.stderr)
            failed = True
            continue

        print(f"{name}: parsed   {base.parse(source)}")

        report = base.verify(source)
        print(report.summary())
        failed = failed or not report.ok

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
