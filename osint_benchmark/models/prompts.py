"""Load the prompts in ``prompts/``.

**No prompt text belongs in a ``.py`` file.** Every prompt is one file in ``prompts/``, so
the whole set can be read and checked in one place instead of being reconstructed from
f-strings scattered across a dozen modules. This module is the only way a stage reaches
one.

A prompt file is plain text with ``{placeholder}`` slots. Braces that should reach the
model literally are doubled, as in :meth:`str.format`.
"""

from __future__ import annotations

import os
import re
import string
from pathlib import Path

from osint_benchmark import paths

_FIELD = re.compile(r"(?<!\{)\{([a-z_][a-z0-9_]*)\}", re.IGNORECASE)


def prompts_dir() -> Path:
    """Return the directory holding the prompt files."""
    return Path(os.environ.get("OSINT_PROMPTS", paths.ROOT / "prompts"))


def names(directory: Path | None = None) -> list[str]:
    """Return every prompt's name, sorted."""
    return sorted(p.stem for p in (directory or prompts_dir()).glob("*.md"))


def read(name: str, directory: Path | None = None) -> str:
    """Return a prompt's raw text, placeholders unfilled.

    Raises:
        FileNotFoundError: If no prompt by that name exists.
    """
    path = (directory or prompts_dir()) / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"no prompt {name!r} in {path.parent}")
    return path.read_text(encoding="utf-8")


def placeholders(name: str, directory: Path | None = None) -> set[str]:
    """Return the placeholder names a prompt declares."""
    return set(_FIELD.findall(read(name, directory)))


def render(name: str, directory: Path | None = None, **values: object) -> str:
    """Return a prompt with its placeholders filled.

    Every declared placeholder must be supplied and every supplied value must be
    declared. Both directions are checked because both failures are silent otherwise: a
    missing value leaves a literal ``{cable}`` in the text sent to the model, and a
    surplus value means the caller thinks it is providing context that the prompt never
    asks for.

    Raises:
        KeyError: If the placeholders and the supplied values do not correspond.
    """
    declared = placeholders(name, directory)
    supplied = set(values)
    if declared != supplied:
        missing = ", ".join(sorted(declared - supplied)) or "none"
        surplus = ", ".join(sorted(supplied - declared)) or "none"
        raise KeyError(f"prompt {name!r}: missing {missing}; unexpected {surplus}")
    return string.Formatter().vformat(read(name, directory), (), dict(values))
