"""Combine what two linkers found in the same document.

The two linkers in this project fail differently, and measurably so. Over 20,941 cables
scoped to the 284 entities the sanctions list names, the title matcher found 22 bridge
entities ReFinED did not, and ReFinED found 15 the title matcher did not. Neither is a
subset of the other, and their union is 62 bridges against ReFinED's 40 — more than half
again as many questions, for no model and no GPU time.

Why they miss different things:

* ReFinED reads context and resolves an ambiguous name to one entity. When it picks a
  different entity than the sanctions reconciliation picked for the same name, the bridge
  is silently lost — both sides are confident and they disagree.
* The title matcher cannot disambiguate at all, so it links every spelling that matches
  exactly. Restricted to a small known entity set that stops being a liability and starts
  being a virtue: an exact match against a name the public side actually uses is good
  evidence, and a model second-guessing it is not always right.

Merging keeps both readings. A document that names an entity under either linker names it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator


def merge_entities(*groups: Iterable[dict]) -> list[dict]:
    """Return the union of several entity lists, by QID, sorted.

    Where both linkers resolved the same QID, the higher-confidence mention wins — so the
    surface form recorded is the one whose reading was strongest, not whichever arrived
    last.
    """
    best: dict[str, dict] = {}
    for group in groups:
        for entity in group:
            current = best.get(entity["qid"])
            if current is None or entity.get("confidence", 0) > current.get("confidence", 0):
                best[entity["qid"]] = entity
    return [best[qid] for qid in sorted(best)]


def merge_rows(rows: Iterable[dict], extra: dict[str, list[dict]]) -> Iterator[dict]:
    """Yield link rows with ``extra`` entities folded in by document id.

    ``extra`` is the whole of the other linker's output, held in memory: it is the small
    one by construction, since it only ever looks for entities the public side already
    named. The rows being merged into are streamed, because they are not.
    """
    for row in rows:
        other = extra.get(row["doc_id"])
        if other:
            row = {**row, "entities": merge_entities(row["entities"], other)}
        yield row
