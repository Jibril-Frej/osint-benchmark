"""Link a corpus that already knows its own entities.

Dodis is catalogued by archivists: every document carries the people and places it
concerns, and the places carry Wikidata QIDs outright. There is nothing for a model to
infer, so this corpus skips the linker entirely — no ReFinED, no mGENRE, no GPU, and none
of the mislink rate that makes every other source's bridges worth doubting.

That matters beyond convenience. The linker has been the ceiling on this benchmark
throughout: the title matcher found entities in 0.8% of cables, ReFinED in 99.9% but with
unknown precision, and the sanctions list only resolves 4.7% of its targets. A corpus whose
entity links were made by people who read the documents is a different kind of evidence,
and it is worth knowing whether questions built on it come out better.

Two kinds of curated link, handled differently:

* **Places** arrive as QIDs and are used as they are.
* **People** arrive as names only, so they go through the same name reconciliation the
  tabular sources use — and inherit its conservatism, which is deliberate. A curated name
  that resolves to two Wikidata entities is still ambiguous.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

from osint_benchmark.link.reconcile import by_label
from osint_benchmark.link.tabular import KINDS
from osint_benchmark.sources import refs

# Curated links are archivist judgements rather than model output, so they carry the
# confidence a human assertion deserves. Nothing downstream currently thresholds on this,
# but recording 1.0 for a guess would be a lie waiting to be believed.
CURATED_CONFIDENCE = 1.0


def qids_in(record: dict) -> list[dict]:
    """Return the entities a record names outright, as link entries."""
    found = []
    for place in record.get("meta", {}).get("places", []) or []:
        qid = place.get("qid")
        if qid:
            found.append(
                {
                    "qid": qid,
                    "surface_form": place.get("name") or qid,
                    "confidence": CURATED_CONFIDENCE,
                }
            )
    return found


def names_in(record: dict) -> list[str]:
    """Return the people a record names, for reconciliation."""
    return [name for name in record.get("meta", {}).get("persons", []) or [] if name.strip()]


def link_records(
    records: Iterable[dict],
    source: str,
    side: str,
    entity_set: frozenset[str] | set[str],
    resolve: Callable[..., dict[str, list[str]]] = by_label,
) -> Iterator[dict]:
    """Yield one link row per record, from the catalogue rather than from a model.

    Records are held in memory because the names have to be collected before they can be
    resolved in batches, exactly as for the tabular sources. Dodis is 4,065 documents, so
    that is affordable.
    """
    rows = list(records)
    wanted = {name for row in rows for name in names_in(row)}
    resolved = resolve(sorted(wanted), KINDS.get("dodis", ("Q5",))) if wanted else {}

    for row in rows:
        entities: dict[str, dict] = {}
        for entity in qids_in(row):
            if entity["qid"] in entity_set:
                entities[entity["qid"]] = entity
        for name in names_in(row):
            candidates = [q for q in resolved.get(name, []) if q in entity_set]
            # An ambiguous name is not a resolution, however carefully it was curated.
            if len(candidates) != 1 or candidates[0] in entities:
                continue
            entities[candidates[0]] = {
                "qid": candidates[0],
                "surface_form": name,
                "confidence": CURATED_CONFIDENCE,
            }
        yield {
            "doc_id": refs.ref(source, row["doc_id"]),
            "source": source,
            "side": side,
            "entities": sorted(entities.values(), key=lambda e: e["qid"]),
        }
