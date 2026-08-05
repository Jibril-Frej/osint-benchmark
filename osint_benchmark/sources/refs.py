"""How a document is named once it has left its own corpus.

A ``doc_id`` is only unique *within* a source. Cablegate numbers its cables from the leak's
own sequence; the sanctions export numbers its targets with its own; both are bare
integers in overlapping ranges. The moment two corpora meet — in the bridge map, in a
pair, in the dictionary of evidence texts — a bare ``doc_id`` stops identifying anything.

It went wrong exactly as you would expect and was invisible for four runs. The evidence
lookup was one flat ``doc_id -> text`` dictionary across every source, so the "public
record" handed to the question writer for sanctions target 47703 was *cable* 47703. Every
pair was a cable beside an unrelated cable. The model noticed before we did: it declined
79 of 80 pairs as having no genuine connection, which was the correct answer to the
question it was actually being asked.

So outside a corpus, a document is ``source:doc_id``. The corpora themselves are untouched
— their records keep their own ids, and the fingerprints in ``pins/corpora.toml`` that let
a rebuilder prove they built the same bytes still hold.

Dates live under a different key in every source, which is the same class of mistake one
level down: the pairing step read ``date`` from every record, sanctions calls it
``list_date``, and so every pair was recorded as non-contemporaneous. That was reported as
a fact about the corpora spanning different eras. It was a missing field.
"""

from __future__ import annotations

SEPARATOR = ":"

# Which key carries a record's date, per source, best first. Absent means the source has no
# usable date and its documents are undated rather than wrongly dated.
#
# Sanctions takes two because the obvious one is wrong. `list_date` is the *export's* own
# date attribute: every record in a download carries the same value, so an interval
# computed from it measures when the file was fetched rather than when anything happened.
# `added` is the enactment date of that particular listing, which is the date a question
# about timing would mean. Reading list_date first left every pair non-contemporaneous
# against a constant -- the same symptom as reading no date at all, and I reported it as a
# fact about the corpora twice before looking.
DATE_FIELDS = {
    "cablegate": ("date",),
    "dodis": ("date",),
    "sanctions": ("added", "list_date"),
    "ucdp": ("date_start",),
    "gdelt": ("date",),
    "parliament": ("date",),
}


def ref(source: str, doc_id: str) -> str:
    """Return the cross-corpus name of one document."""
    return f"{source}{SEPARATOR}{doc_id}"


def split(reference: str) -> tuple[str, str]:
    """Return ``(source, doc_id)`` for a reference.

    Splits once, from the left: an id may itself contain the separator (``enwiki:Q42``
    reached the evidence map as an article id long before this existed) and only the
    source is being taken off the front.
    """
    source, _, doc_id = reference.partition(SEPARATOR)
    return source, doc_id


def record_date(source: str, record: dict) -> str | None:
    """Return a record's date, under whatever key this source calls it.

    The first field that has a value wins, so a source can name a precise date and fall
    back to a coarse one rather than to nothing.
    """
    for field in DATE_FIELDS.get(source, ()):
        if value := record.get(field):
            return str(value)
    return None
