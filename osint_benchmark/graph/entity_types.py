"""Decide which entities may bridge: people and organisations, and nothing else.

The previous project never had to decide this. Its knowledge base was built *around*
persons and organisations — parliamentarians indexed by name and verified on birth date,
companies with commercial-register entries — so a canton could not become a bridge because
nothing ever proposed one.

This project proposes every shared entity and filters afterwards, which is why the filter
kept losing. It began by blocking countries; then cities; then, when Dodis met the
parliamentary record, the top twenty bridges were Swiss cantons. Each patch was correct
and the next category was already queued behind it.

So the test is now the one the previous project enforced structurally: an entity may
anchor a question if it is a **person** or an **organisation**. Everything else is out,
including places, whether or not anyone remembered to list them.

The first real run bridged only on countries, and every question it produced was of the
form "what connects Cameroon and Canada?" — the answer being that both documents mention
international relations. That is not a bridge, it is a coincidence of two documents each
naming a country.

A country co-occurs with everything, so it distinguishes nothing. What is specific enough
that two documents naming it are probably about the same matter is a person or a named
organisation.

Membership is not a flat list, because Wikidata does not work that way: Associated Press is
an instance of *news agency*, which is a subclass of organisation, and an allowlist of
class ids dropped it. So organisation-hood is resolved through ``P279*`` — the one query
this module makes — while people need no hierarchy at all, since Wikidata types every
human as ``Q5``.

Types come from Wikidata ``P31``/``P279``, not from the linker. The previous project's note
is explicit that ReFinED's own coarse types are unreliable — it labels countries ORG — so
taking types from the graph rather than the model is the only version of this that works.
"""

from __future__ import annotations

from collections.abc import Iterable

HUMAN = "Q5"
ORGANISATION = "Q43229"
# Everything with a location on the earth descends from this, including every
# administrative subdivision. It has to be resolved through the hierarchy exactly as
# organisation-hood is, and it has to win: Wikidata models a canton as a kind of
# organisation, so testing only for organisations admitted all twenty-six of them.
PLACE = "Q2221906"

# A second ancestor, because the first one misses continents. Asia is an instance of
# "continent" and of "geographic region", and neither descends from "geographic location" --
# so it came through as a substantive shared entity and joined a cable about North Korean
# nuclear tests to a parliamentary session on climate protection, on the strength of sharing
# a landmass. "Geographic region" covers everything "geographic location" does and continents
# besides, and still excludes people.
REGION = "Q82794"

# Every ancestor whose descendants are somewhere rather than something.
PLACE_ANCESTORS = (PLACE, REGION)

# Kept as documentation of the kinds this filter is meant to admit. Not a gate: an
# allowlist cannot anticipate the tail of organisation types, and excluding what it failed
# to list dropped Associated Press. What decides is descent from ORGANISATION.
BRIDGEABLE = {
    "Q5": "human",
    "Q43229": "organization",
    "Q4830453": "business",
    "Q783794": "company",
    "Q6881511": "enterprise",
    "Q7278": "political party",
    "Q484652": "international organization",
    "Q1335818": "supranational organisation",
    "Q327333": "government agency",
    "Q2659904": "government organization",
    "Q1156831": "armed organization",
    "Q17127659": "terrorist organisation",
    "Q1656682": "event",
    "Q198": "war",
    "Q180684": "conflict",
    "Q3839081": "disaster",
    "Q2334719": "legal case",
    "Q49848": "document",
}

# Classes that may not, however often they appear. These are the entities that co-occur
# with everything: geography, and the administrative units of geography.
NOT_BRIDGEABLE = {
    "Q6256": "country",
    "Q3624078": "sovereign state",
    "Q7275": "state",
    "Q5107": "continent",
    "Q82794": "geographic region",
    "Q56061": "administrative territorial entity",
    "Q15617994": "designation for an administrative territorial entity",
    "Q1048835": "political territorial entity",
    "Q10864048": "first-level administrative country subdivision",
    "Q515": "city",
    "Q1549591": "big city",
    "Q532": "village",
    "Q5119": "capital city",
    "Q3957": "town",
    "Q34770": "language",
    "Q11563": "number",
    "Q577": "year",
}


def descendants_of(classes: Iterable[str], ancestor: str, query=None) -> set[str]:
    """Return which of these Wikidata classes descend from one, following ``P279*``.

    One query for the whole set. Asked per entity this would be thousands of round trips;
    asked per *class* it is a few dozen, because a bridge map draws on far fewer kinds of
    thing than it has things.
    """
    from osint_benchmark.link import reconcile  # noqa: PLC0415 - avoids a circular import

    wanted = sorted({c for c in classes if c})
    if not wanted:
        return set()
    values = " ".join(f"wd:{c}" for c in wanted)
    rows = (query or reconcile.sparql)(
        f"SELECT DISTINCT ?c WHERE {{ VALUES ?c {{ {values} }} ?c wdt:P279* wd:{ancestor} }}"
    )
    return {row["c"]["value"].rsplit("/", 1)[-1] for row in rows}


def place_classes(classes: Iterable[str], query=None) -> set[str]:
    """Return which of these classes describe somewhere rather than something."""
    wanted = sorted({c for c in classes if c})
    found: set[str] = set()
    for ancestor in PLACE_ANCESTORS:
        found |= descendants_of(wanted, ancestor, query)
    return found


def kinds_present(classes: Iterable[str], query=None) -> tuple[set[str], set[str]]:
    """Return ``(organisation classes, place classes)`` among the ones given."""
    classes = sorted({c for c in classes if c})
    return (descendants_of(classes, ORGANISATION, query), place_classes(classes, query))


def classify(
    statements: dict[str, list[str]],
    organisations: set[str] | None = None,
    places: set[str] | None = None,
) -> str:
    """Return ``bridgeable``, ``blocked`` or ``unknown`` for one entity's statements.

    Three tests, in order, and the first is the one that was missing.

    **A class is not a thing.** "Police", "School", "Company", "Student" and "Bank" all
    resolve to Wikidata entities that are instances of an organisation class, so a check on
    ``instance_of`` alone calls them bridgeable — and they co-occur with everything exactly
    as countries do. What separates them from Associated Press or Vladimir Putin is that
    they are *categories*: they have ``subclass_of`` statements, because other things are
    kinds of them. Named individuals subclass nothing.

    The previous version made this worse by *unioning* the two properties, so "subclass of
    organisation" counted as evidence of being an organisation. That reading turns every
    generic noun in GDELT's actor vocabulary — 896 of them — into a bridge anchor.

    Then geography, which is blocked however it is classified: an entity that is both a
    country and an organisation, as many states are in Wikidata's modelling, is still a
    country for our purposes.
    """
    kinds = set(statements.get("instance_of", ()))
    if statements.get("subclass_of"):
        return "blocked"
    # Geography vetoes, and must be tested before organisation-hood rather than after:
    # Wikidata models a sovereign state and a canton as kinds of organisation, so descent
    # from ORGANISATION alone lets every country and all twenty-six cantons back in
    # through the door just closed on them.
    if kinds & set(NOT_BRIDGEABLE):
        return "blocked"
    if places and kinds & places:
        return "blocked"
    if not kinds:
        return "unknown"
    if HUMAN in kinds:
        return "bridgeable"
    if organisations is None:
        # Without the hierarchy, only the classes named above can be recognised. That is
        # the old allowlist and it drops real organisations, so a caller who cares passes
        # the resolved set.
        return "bridgeable" if kinds & set(BRIDGEABLE) else "blocked"
    return "bridgeable" if kinds & organisations else "blocked"


def bridgeable_qids(
    facts: Iterable[dict],
    keep_unknown: bool = False,
    organisations: set[str] | None = None,
    places: set[str] | None = None,
) -> set[str]:
    """Return the QIDs allowed to anchor a bridge.

    ``keep_unknown`` decides what happens to entities whose type we could not read. The
    default excludes them: an entity nobody can classify is one nobody can vouch for, and
    admitting it is how the country problem returns by another route.
    """
    allowed = set()
    for record in facts:
        verdict = classify(record.get("statements", {}), organisations, places)
        if verdict == "bridgeable" or (keep_unknown and verdict == "unknown"):
            allowed.add(record["qid"])
    return allowed


def summarise(
    facts: Iterable[dict],
    organisations: set[str] | None = None,
    places: set[str] | None = None,
) -> dict[str, int]:
    """Return how many entities fall into each class, for a run to report."""
    counts = {"bridgeable": 0, "blocked": 0, "unknown": 0}
    for record in facts:
        counts[classify(record.get("statements", {}), organisations, places)] += 1
    return counts


def classes_in(facts: Iterable[dict]) -> set[str]:
    """Return every class any of these entities is an instance of."""
    return {qid for record in facts for qid in record.get("statements", {}).get("instance_of", ())}
