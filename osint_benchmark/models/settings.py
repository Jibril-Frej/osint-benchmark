"""Read the model parameters out of ``config/models.toml``.

Every knob that changes what a model does lives in that file, so it can be reviewed as a
set alongside the prompts in ``prompts/``. Nothing here holds a default value: a missing
setting is a mistake to report, not one to paper over with a plausible number.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from osint_benchmark import paths


@dataclass(frozen=True)
class Settings:
    """What one model role is configured to do.

    Attributes:
        role: ``phraser``, ``judge`` or ``solver``.
        model: The served model's name, for the record and for the endpoint to route on.
        endpoint: Where it is served. Empty until one is running.
        temperature: Sampling temperature. Repeat-and-agree needs this above 0.
        max_tokens: Reply ceiling. Too low truncates a reasoning model mid-thought.
        samples: How many times to ask before accepting a judgement.
    """

    role: str
    model: str
    endpoint: str
    temperature: float
    max_tokens: int
    samples: int


def load(role: str, config_file: Path | None = None) -> Settings:
    """Return the settings for one model role.

    ``OSINT_MODEL_ENDPOINT`` overrides the configured endpoint, so a cluster job can point
    at whatever it just started serving without editing a committed file.

    Raises:
        KeyError: If the role has no section, or a section is missing a setting.
    """
    path = config_file or (paths.config_dir() / "models.toml")
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    if role not in config:
        raise KeyError(f"no [{role}] section in {path}; known: {', '.join(sorted(config))}")
    section = config[role]
    missing = {"model", "endpoint", "temperature", "max_tokens", "samples"} - set(section)
    if missing:
        raise KeyError(f"[{role}] in {path} is missing: {', '.join(sorted(missing))}")
    return Settings(
        role=role,
        model=section["model"],
        endpoint=os.environ.get("OSINT_MODEL_ENDPOINT") or section["endpoint"],
        temperature=float(section["temperature"]),
        max_tokens=int(section["max_tokens"]),
        samples=int(section["samples"]),
    )
