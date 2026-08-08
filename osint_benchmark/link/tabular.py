"""Link the sources that carry names rather than prose.

A sanctions listing has a person's name; a UCDP event has a country and an actor. There is
no text to read, so there is no linker to run — the name is looked up.

This is the other half of step 2, and leaving it out is what made the public side of the
graph empty in the first smoke run: the parliamentary record is German, so an
English-title matcher finds nothing in it, and every other public source is tabular.

Conservative on purpose. A candidate must match the name exactly and be in the public
entity set; an ambiguous name resolving to several entities is left unresolved rather than
guessed at. A sanctions list is exactly where a false match is expensive, and most
sanctioned individuals genuinely have no Wikidata entity.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

from osint_benchmark.link import search
from osint_benchmark.link.reconcile import by_label
from osint_benchmark.sources import refs

# Where each tabular source keeps the names worth resolving, and what kind of thing those
# names denote. The type constraint is what makes a name usable: "Afghanistan" alone
# returns thirty entities -- films, ships, historical states -- and picking among them
# would be guessing. Constrained to a country, it returns the country.
NAME_FIELDS = {
    "sanctions": ("names",),
    "ucdp": ("side_a", "side_b", "country"),
    "gdelt": ("actor1_name", "actor2_name"),
}

KINDS = {
    # country, sovereign state, state
    "ucdp": ("Q6256", "Q3624078", "Q7275"),
    "gdelt": ("Q6256", "Q3624078", "Q7275", "Q43229"),
    # human, organisation, business
    "sanctions": ("Q5", "Q43229", "Q4830453"),
}


# A single given name identifies nobody. "Muhammad" and "Khalid" both resolved to exactly
# one Wikidata person and became bridges, which is how a cable mentioning a common name
# ends up paired with an unrelated listing. Names needing this are person-like sources;
# country and organisation names are frequently one word and legitimately so.
MIN_NAME_WORDS = {"sanctions": 2}


def names_in(record: dict, fields: Iterable[str], min_words: int = 1) -> list[str]:
    """Return the names a record carries, from list-valued or scalar fields."""
    found: list[str] = []
    for field in fields:
        value = record.get(field)
        if isinstance(value, list):
            found.extend(str(v).strip() for v in value if str(v).strip())
        elif isinstance(value, str) and value.strip():
            found.append(value.strip())
    return [name for name in found if len(name.split()) >= min_words]


def link_by_search(
    records: Iterable[dict],
    source: str,
    side: str,
    entity_set: frozenset[str] | set[str],
    resolve: Callable[..., str | None] = search.resolve,
) -> Iterator[dict]:
    """Yield one link row per record, resolving each target through Wikidata's search.

    Per record rather than in one batch, because the verification is per target: every
    name variant of *this* listing is tried together, and its birth year and kind decide.
    That is what lifts the sanctions list from 383 resolved targets to a measured 1,232 in
    the previous project, and the public leg is the ceiling on the whole benchmark.
    """
    fields = NAME_FIELDS.get(source, ("name",))
    for row in records:
        names = names_in(row, fields, min_words=1)
        qid = (
            resolve(names, kind=str(row.get("type") or ""), years=row.get("dobs") or ())
            if names
            else None
        )
        entities = []
        if qid and qid in entity_set:
            entities.append({"qid": qid, "surface_form": names[0], "confidence": 1.0})
        yield {
            "doc_id": refs.ref(source, row["doc_id"]),
            "source": source,
            "side": side,
            "entities": entities,
        }


def link_records(
    records: Iterable[dict],
    source: str,
    side: str,
    entity_set: frozenset[str] | set[str],
    resolve: Callable[..., dict[str, list[str]]] = by_label,
) -> Iterator[dict]:
    """Yield one link row per record, resolving its names in a single batched pass.

    Records are read into memory because the names have to be collected before they can be
    resolved in batches — one query per record would be thousands of round trips. These
    sources are the small ones, so that is affordable; the corpora that are not small carry
    prose and go through a linker instead.
    """
    rows = list(records)
    fields = NAME_FIELDS.get(source, ("name",))
    min_words = MIN_NAME_WORDS.get(source, 1)
    wanted = {name for row in rows for name in names_in(row, fields, min_words)}
    resolved = resolve(sorted(wanted), KINDS.get(source, ())) if wanted else {}

    for row in rows:
        entities = []
        seen: set[str] = set()
        for name in names_in(row, fields, min_words):
            candidates = [q for q in resolved.get(name, []) if q in entity_set]
            # An ambiguous name is not a resolution. Guessing is how a sanctions listing
            # ends up attached to the wrong person.
            if len(candidates) != 1 or candidates[0] in seen:
                continue
            seen.add(candidates[0])
            entities.append({"qid": candidates[0], "surface_form": name, "confidence": 1.0})
        yield {
            # Namespaced, because a bare doc_id is unique only inside its own corpus and
            # everything downstream mixes them. See osint_benchmark.sources.refs.
            "doc_id": refs.ref(source, row["doc_id"]),
            "source": source,
            "side": side,
            "entities": sorted(entities, key=lambda e: e["qid"]),
        }
