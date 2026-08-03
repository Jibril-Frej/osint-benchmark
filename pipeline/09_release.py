"""Step 9: freeze a version of the benchmark. Runs on the workstation.

Ships what this project authored -- questions, answers, gate outcomes, necessity flags,
evidence pointers -- and no corpus text. A user rebuilds the corpora from pins/ instead.

Usage::

    uv run python pipeline/09_release.py --version v1

Also publishes the corpus fingerprints, so a rebuilder can confirm they built the corpora
this release's answers were written against.
"""

from __future__ import annotations

import argparse
import tomllib

from osint_benchmark import paths
from osint_benchmark.release import freeze
from osint_benchmark.release.load import load_items
from osint_benchmark.sources import ALL, base, get_source


def publish_fingerprints() -> dict[str, dict]:
    """Record every built corpus's fingerprint and return them."""
    for name in ALL:
        if base.output_path(get_source(name)).exists():
            base.write_fingerprint(get_source(name))
    path = paths.pins_dir() / "corpora.toml"
    return tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main(argv: list[str] | None = None) -> int:
    """Freeze the reviewed questions into a versioned release."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default="v1")
    args = parser.parse_args(argv)

    items_dir = paths.data_dir() / "items"
    for name in ("reviewed.jsonl", "measured.jsonl", "accepted.jsonl"):
        source = items_dir / name
        if source.exists():
            break
    else:
        raise SystemExit(f"no items in {items_dir}: run pipeline/06_generate.py first")

    items = load_items(source)
    corpora = publish_fingerprints()
    out_dir = paths.ROOT / "releases" / args.version
    sheet = freeze.freeze(items, corpora, out_dir)

    print(f"{sheet['questions']} questions from {source.name} -> {out_dir}")
    print(f"  sha256 {sheet['sha256'][:16]}  types {sheet['types']}")
    if sheet["necessity"]["measured"]:
        need = sheet["necessity"]
        print(f"  necessity: {need['needs_both']} of {need['measured']} need both sides")
    else:
        print("  necessity: not measured -- run pipeline/07_necessity.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
