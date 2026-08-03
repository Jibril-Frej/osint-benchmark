"""Check that the project skeleton is wired up.

Placeholder so `uv run pytest` exercises something before any stage is ported; delete
it once real tests exist.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_targets_the_supported_python():
    """The project pins Python 3.13, which the ruff target must agree with."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["project"]["requires-python"] == ">=3.13"
    assert config["tool"]["ruff"]["target-version"] == "py313"


def test_src_is_on_the_import_path():
    """`pythonpath = ["src"]` puts the source tree on sys.path for the suite."""
    import sys

    assert str(ROOT / "src") in sys.path
