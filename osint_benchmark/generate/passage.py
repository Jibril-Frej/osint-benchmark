"""Find where a document names someone, and cut out the passage that says so.

The typed question builders need two things the link files do not carry. A curated corpus
records *which* entities a document concerns but not how the document writes them, and every
builder needs the few hundred words around the mention rather than a whole document — the
phraser is asked to describe a situation, and a situation is a paragraph, not a dispatch.

Both come from the previous project, which learned the order the hard way: the full name is
looked for before the family name, because a document that writes the whole name is not being
ambiguous and the resolution type must reject it. Looking for the family name first would
turn every unambiguous mention into a candidate.
"""

from __future__ import annotations

import re

# Characters before and after the mention. Wide enough to hold what the person was doing,
# narrow enough that the phraser writes about the situation rather than the dispatch.
BEFORE, AFTER = 400, 700

# A surface this short is an initialism or OCR noise rather than a name.
MIN_SURFACE = 4

WORD = re.compile(r"[a-z0-9]+")


def normalise(text: str) -> str:
    """Return a string as its lowercase alphanumeric tokens, space-separated."""
    return " ".join(WORD.findall(text.lower()))


def written_as(label: str, text: str) -> str:
    """Return how a document writes an entity's name, or empty if it does not.

    The full label is tried before the family name, and the order is the point: a document
    writing the whole name is not an ambiguous mention, and returning the family name first
    would hide that.
    """
    lowered = text.lower()
    forms = [label]
    tokens = label.split()
    if len(tokens) > 1:
        forms.append(tokens[-1])
    for form in forms:
        if len(form) < MIN_SURFACE:
            continue
        found = lowered.find(form.lower())
        if found >= 0:
            return text[found : found + len(form)]
    return ""


def window(text: str, anchor: str, before: int = BEFORE, after: int = AFTER) -> str:
    """Return the passage around a document's first mention of ``anchor``.

    Falls back to the opening of the document when the anchor cannot be found, so a caller
    always has something to show; callers that require the mention to be present check for
    it with :func:`written_as` instead of inferring it from an empty return.
    """
    found = text.lower().find(anchor.lower()) if anchor else -1
    if found < 0:
        return text[: before + after]
    return text[max(0, found - before) : found + after]


def locate(row: dict, text: str) -> dict:
    """Return a link row whose entities carry the name as the document writes it.

    For the corpora whose entities were catalogued rather than found by a linker. An entity
    whose name appears nowhere in the document text is dropped, not kept with the
    catalogue's spelling: a question about how a document names someone cannot be built on
    a name the document does not use.
    """
    located = []
    for entity in row.get("entities", []):
        surface = written_as(str(entity.get("surface_form", "")), text)
        if surface:
            located.append({**entity, "surface_form": surface})
    return {**row, "entities": located}
