"""Step 1: the bulk corpora — the sources that depend on nothing and can be fetched first.

Not here, deliberately: the commercial register and the article text for bridge entities,
which cannot be fetched until the linker has said which entities matter (step 4), and
Wikidata's build-time lookups, which are answered live. Conflating the two kinds of fetch
is what produced the previous project's coverage bug — the Wikidata slice was fetched
against a corpus subset before the linker was rerun at full scale and covered 35% of
entities, with no symptom beyond filters silently matching nothing.
"""

from __future__ import annotations

from osint_benchmark.sources import cablegate
from osint_benchmark.sources.base import Source

_SOURCES: dict[str, Source] = {source.name: source for source in (cablegate.SOURCE,)}

ALL = tuple(_SOURCES)


def get_source(name: str) -> Source:
    """Return a source by name.

    Raises:
        KeyError: If no source has that name.
    """
    if name not in _SOURCES:
        raise KeyError(f"unknown source {name!r}; known: {', '.join(ALL)}")
    return _SOURCES[name]
