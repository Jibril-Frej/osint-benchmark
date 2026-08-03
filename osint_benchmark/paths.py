"""Where the data lives.

The previous project put ``DATA = Path("data/osint")`` at module level in a dozen files,
which is why nothing there ran from another directory and why the cluster needed path
edits. Every path in this project is resolved here instead, at call time, so a test or a
cluster job can redirect the whole pipeline by setting one environment variable.

Four roots, each overridable:

* ``OSINT_DATA`` — everything this project derives (gitignored).
* ``OSINT_RAW`` — immutable downloads. Point this at an existing copy of the corpora to
  read them in place rather than fetching 135 GB a second time.
* ``OSINT_DOCS`` — parsed, normalised documents.
* ``OSINT_PINS`` — the committed source checksums and per-document hashes.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    """Return the root for everything this project derives."""
    return Path(os.environ.get("OSINT_DATA", ROOT / "data"))


def raw_dir() -> Path:
    """Return the root for immutable downloads."""
    return Path(os.environ.get("OSINT_RAW", data_dir() / "raw"))


def docs_dir() -> Path:
    """Return the root for parsed, normalised documents."""
    return Path(os.environ.get("OSINT_DOCS", data_dir() / "docs"))


def pins_dir() -> Path:
    """Return the root for the committed pins (source checksums, document hashes)."""
    return Path(os.environ.get("OSINT_PINS", ROOT / "pins"))
