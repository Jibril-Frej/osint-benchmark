"""Read the linking parameters out of ``config/link.toml``.

Same contract as :mod:`osint_benchmark.models.settings`: nothing here holds a default
value, because a missing setting is a mistake to report rather than one to paper over with
a plausible number. The module-level constants in :mod:`osint_benchmark.link.refined` are
the *code's* defaults for callers that pass nothing — a run reads this file.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from osint_benchmark import paths


@dataclass(frozen=True)
class RefinedSettings:
    """What the ReFinED linker is configured to do.

    Attributes:
        model: The checkpoint name.
        entity_set: ``wikipedia`` or ``wikidata`` — which pool it may resolve into.
        confidence: Floor below which a resolution is discarded.
        batch_size: Documents per forward pass.
        max_chars: Characters of each document the model is shown.
    """

    model: str
    entity_set: str
    confidence: float
    batch_size: int
    max_chars: int


def _table(section: str, config_file: Path | None = None) -> dict:
    """Return one section of the link config.

    Raises:
        KeyError: If the section is absent.
    """
    path = config_file or (paths.config_dir() / "link.toml")
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    if section not in config:
        raise KeyError(f"no [{section}] section in {path}; known: {', '.join(sorted(config))}")
    return config[section]


def refined(config_file: Path | None = None) -> RefinedSettings:
    """Return the ReFinED settings.

    Raises:
        KeyError: If a setting is missing.
    """
    section = _table("refined", config_file)
    missing = {"model", "entity_set", "confidence", "batch_size", "max_chars"} - set(section)
    if missing:
        raise KeyError(f"[refined] in link.toml is missing: {', '.join(sorted(missing))}")
    return RefinedSettings(
        model=section["model"],
        entity_set=section["entity_set"],
        confidence=float(section["confidence"]),
        batch_size=int(section["batch_size"]),
        max_chars=int(section["max_chars"]),
    )


def reconcile(config_file: Path | None = None) -> tuple[bool, str]:
    """Return ``(match aliases, method)`` for name reconciliation.

    Raises:
        KeyError: If a setting is missing.
        ValueError: If the method is not one this project implements.
    """
    section = _table("reconcile", config_file)
    missing = {"aliases", "method"} - set(section)
    if missing:
        raise KeyError(f"[reconcile] in link.toml is missing: {', '.join(sorted(missing))}")
    method = str(section["method"])
    if method not in {"exact", "search"}:
        raise ValueError(f"[reconcile] method must be exact or search, not {method!r}")
    return bool(section["aliases"]), method


def dictionary(config_file: Path | None = None) -> tuple[int, int]:
    """Return ``(min_title_words, min_title_chars)`` for the no-model fallback.

    Raises:
        KeyError: If a setting is missing.
    """
    section = _table("dictionary", config_file)
    missing = {"min_title_words", "min_title_chars"} - set(section)
    if missing:
        raise KeyError(f"[dictionary] in link.toml is missing: {', '.join(sorted(missing))}")
    return int(section["min_title_words"]), int(section["min_title_chars"])
