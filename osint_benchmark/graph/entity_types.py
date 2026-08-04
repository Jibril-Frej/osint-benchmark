"""Decide which entities may bridge, by what kind of thing they are.

The first real run bridged only on countries, and every question it produced was of the
form "what connects Cameroon and Canada?" — the answer being that both documents mention
international relations. That is not a bridge, it is a coincidence of two documents each
naming a country.

A country co-occurs with everything, so it distinguishes nothing. The entities worth
building a question on are the ones specific enough that two documents naming the same one
are probably about the same thing: a person, a company, an organisation, an event.

Types come from Wikidata ``P31``/``P279``, not from the linker. The previous project's note
is explicit that ReFinED's own coarse types are unreliable — it labels countries ORG — so
taking types from the graph rather than the model is the only version of this that works.
"""

from __future__ import annotations

from collections.abc import Iterable

# Classes that may anchor a bridge. Specific enough that two documents naming the same one
# are plausibly about the same matter.
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


def classify(statements: dict[str, list[str]]) -> str:
    """Return ``bridgeable``, ``blocked`` or ``unknown`` for one entity's statements.

    Blocked wins over bridgeable. An entity that is both a country and an organisation --
    which many states are, in Wikidata's modelling -- is still a country for our purposes,
    and still co-occurs with everything.
    """
    classes = set(statements.get("instance_of", ())) | set(statements.get("subclass_of", ()))
    if classes & set(NOT_BRIDGEABLE):
        return "blocked"
    if classes & set(BRIDGEABLE):
        return "bridgeable"
    return "unknown"


def bridgeable_qids(facts: Iterable[dict], keep_unknown: bool = False) -> set[str]:
    """Return the QIDs allowed to anchor a bridge.

    ``keep_unknown`` decides what happens to entities whose type we could not read. The
    default excludes them: an entity nobody can classify is one nobody can vouch for, and
    admitting it is how the country problem returns by another route.
    """
    allowed = set()
    for record in facts:
        verdict = classify(record.get("statements", {}))
        if verdict == "bridgeable" or (keep_unknown and verdict == "unknown"):
            allowed.add(record["qid"])
    return allowed


def summarise(facts: Iterable[dict]) -> dict[str, int]:
    """Return how many entities fall into each class, for a run to report."""
    counts = {"bridgeable": 0, "blocked": 0, "unknown": 0}
    for record in facts:
        counts[classify(record.get("statements", {}))] += 1
    return counts
