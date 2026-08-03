"""The record shapes every stage passes around.

One :class:`Document` for anything with prose — cables, Dodis regesten, parliamentary
items, Wikipedia articles — so the linker does not care which corpus a document came
from. Genuinely tabular sources (the sanctions list, the commercial register, event
tables) keep their own shapes; forcing those into a Document is where the "the public
side is a row, not a document" awkwardness came from in the previous project.

Whatever a source carries that does not fit the common fields goes in ``meta``, which is
declared field-by-field in the source's :class:`~osint_benchmark.sources.base.Projection`
so it never becomes a bag of undocumented leftovers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    """One piece of prose from one corpus, normalised.

    Attributes:
        doc_id: Identifier, unique within the source (a cable id, a Dodis id).
        source: The source module's name, e.g. ``"cablegate"``.
        text: The prose a retriever indexes. Routing headers and other apparatus stay
            out of it and live in ``meta``.
        date: ISO-8601, or None when the source's date is absent or unparseable. The
            unparsed original stays in ``meta`` so nothing is lost.
        lang: ISO-639-1 code, empty when the source does not say.
        title: A short human-readable label; empty when the source has none.
        meta: Source-specific fields, declared in the source's projection.
    """

    doc_id: str
    source: str
    text: str
    date: str | None = None
    lang: str = ""
    title: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Return the on-disk form, with keys in a fixed order."""
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "date": self.date,
            "lang": self.lang,
            "title": self.title,
            "text": self.text,
            "meta": self.meta,
        }
