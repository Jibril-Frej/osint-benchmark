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

# Classes that plainly may anchor a bridge. No longer a gate -- an allowlist cannot
# anticipate the tail of organisation types, and excluding what it failed to list dropped
# real named entities. Kept as documentation of what this filter is for, and read by
# nothing that decides.
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
    if kinds & set(NOT_BRIDGEABLE):
        return "blocked"
    if not kinds:
        return "unknown"
    # Anything else that is a specific thing. BRIDGEABLE was an allowlist and is now only
    # a description, because an allowlist cannot anticipate the tail: Associated Press is
    # an instance of news agency, cooperative and nonprofit organisation, none of which
    # were in it, so a real named agency came out unknown and was dropped. That is the
    # same failure the linker's coarse-type allowlist had, and the same fix -- name what
    # may not bridge, and let the rest through.
    return "bridgeable"


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
