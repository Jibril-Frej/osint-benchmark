"""Fetch Wikidata statements for the bridge entities, pinned to the revision read.

Entity-driven, so it belongs to step 4 rather than step 1: there is nothing to ask for
until the graph has said which entities matter. Getting that ordering wrong is what
produced the previous project's coverage bug — its slice was fetched against a corpus
subset before the linker was rerun at full scale and covered properties for 44,477 of
126,903 entities, 35%, with no symptom beyond filters silently matching nothing.

**Every record carries the revision it was read from.** Wikidata is live, so a gold answer
taken from it is only correct against a particular revision — a company's officers change,
an entity is merged, a statement is corrected. ``Special:EntityData`` returns ``lastrevid``
with the entity, so pinning costs nothing at fetch time, and
``Special:EntityData/Q42.json?revision=N`` returns those exact bytes indefinitely. That is
what makes a live source usable for answers that have to stay right.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator

ENTITY_DATA = "https://www.wikidata.org/wiki/Special:EntityData"
USER_AGENT = "osint-benchmark/0.1 (research; https://github.com/Jibril-Frej/osint-benchmark)"

# The properties a question can be built on. Deliberately short: this is the public half of
# an answer, and an entity's whole statement set is mostly identifiers and cross-references.
KEEP_PROPERTIES = {
    "P31": "instance_of",
    "P279": "subclass_of",
    "P569": "birth_date",
    "P570": "death_date",
    "P19": "birth_place",
    "P20": "death_place",
    "P106": "occupation",
    "P39": "position_held",
    "P102": "party",
    "P27": "citizenship",
    "P159": "headquarters",
    "P571": "inception",
    "P576": "dissolved",
    "P112": "founder",
    # Who someone works for. The previous project's association type named ``employer`` among
    # the affiliations it would build a question on, but its slice never fetched P108, so the
    # predicate could not fire and three of the four did the work. Fetched here, so it can.
    "P108": "employer",
    "P169": "chief_executive",
    "P488": "chairperson",
    "P452": "industry",
    "P17": "country",
    "P36": "capital",
    "P463": "member_of",
    "P361": "part_of",
    "P155": "follows",
    "P156": "followed_by",
}

Fetch = Callable[[str], dict]


def fetch_entity(qid: str, revision: int | None = None, timeout: float = 60.0) -> dict:
    """Return one entity's raw JSON, at a specific revision when given."""
    url = f"{ENTITY_DATA}/{qid}.json"
    if revision is not None:
        url += f"?revision={revision}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read())


def snak_value(snak: dict) -> str:
    """Return a statement's value as a string, whatever its datatype.

    Only the shapes a question can use are unpacked. Anything else returns empty rather
    than a stringified dict, so a caller cannot mistake plumbing for an answer.
    """
    value = snak.get("datavalue", {})
    kind, data = value.get("type"), value.get("value")
    if kind == "string":
        return str(data)
    if kind == "wikibase-entityid":
        return str(data.get("id", ""))
    if kind == "time":
        return str(data.get("time", "")).lstrip("+")
    if kind == "quantity":
        return str(data.get("amount", "")).lstrip("+")
    if kind == "monolingualtext":
        return str(data.get("text", ""))
    return ""


def statements(entity: dict) -> dict[str, list[str]]:
    """Return the kept properties of one entity, as ``friendly_name -> values``."""
    claims = entity.get("claims", {})
    out: dict[str, list[str]] = {}
    for prop, name in KEEP_PROPERTIES.items():
        values = [
            value
            for claim in claims.get(prop, [])
            if claim.get("rank") != "deprecated"
            and (value := snak_value(claim.get("mainsnak", {})))
        ]
        if values:
            out[name] = values
    return out


def english(terms: dict) -> str:
    """Return the English term, falling back to the multilingual one.

    Wikidata introduced the ``mul`` language code for labels that are identical across
    languages, and *removes* them from ``en`` when it applies. Douglas Adams has no ``en``
    label at all: reading only ``en`` returns empty for a large share of entities, which
    would look like an entity with no name rather than a lookup in the wrong place.
    """
    for code in ("en", "mul"):
        value = terms.get(code, {}).get("value")
        if value:
            return str(value)
    return ""


def record(qid: str, payload: dict) -> dict:
    """Return the on-disk record for one fetched entity."""
    entity = payload.get("entities", {}).get(qid, {})
    return {
        "doc_id": qid,
        "qid": qid,
        "revision": entity.get("lastrevid"),
        "label": english(entity.get("labels", {})),
        "description": english(entity.get("descriptions", {})),
        "statements": statements(entity),
    }


def fetch_entities(
    qids: Iterable[str],
    fetch: Fetch = fetch_entity,
    pause: float = 0.1,
    on_error: Callable[[str, Exception], None] | None = None,
) -> Iterator[dict]:
    """Yield one record per entity, skipping and reporting the ones that fail.

    A failed fetch is *unknown data*, not absent data. It is reported through ``on_error``
    rather than yielded as an empty record, because a run that silently turns failures into
    empty statement sets is how a third of an entity set goes missing without a symptom.
    """
    for qid in dict.fromkeys(qids):
        try:
            yield record(qid, fetch(qid))
        except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
            if on_error is not None:
                on_error(qid, exc)
            continue
        if pause:
            time.sleep(pause)
