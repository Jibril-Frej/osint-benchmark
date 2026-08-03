"""Check that the project skeleton is wired up."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_targets_the_supported_python():
    """The project pins Python 3.13, which the ruff target must agree with."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["project"]["requires-python"] == ">=3.13"
    assert config["tool"]["ruff"]["target-version"] == "py313"


def test_the_package_is_installed_rather_than_path_hacked():
    """`uv sync` installs the package, so the numbered pipeline files can just import it.

    This is what lets `pipeline/01_sources.py` run from any directory without the
    `sys.path.insert` the previous project's scripts needed.
    """
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["uv"]["package"] is True

    import osint_benchmark

    assert osint_benchmark.__file__ is not None
