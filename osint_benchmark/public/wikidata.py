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
import urllib.parse
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
BatchFetch = Callable[[list[str]], dict]

API = "https://www.wikidata.org/w/api.php"

# The most ids ``wbgetentities`` accepts in one request.
BATCH = 50


def fetch_entity(qid: str, revision: int | None = None, timeout: float = 60.0) -> dict:
    """Return one entity's raw JSON, at a specific revision when given.

    One entity per request, which is what makes it the way to re-read a pinned revision --
    ``Special:EntityData/Q42.json?revision=N`` returns those exact bytes indefinitely. For
    fetching a whole slice see :func:`fetch_batch`.
    """
    url = f"{ENTITY_DATA}/{qid}.json"
    if revision is not None:
        url += f"?revision={revision}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read())


def fetch_batch(qids: list[str], timeout: float = 60.0) -> dict:
    """Return the raw JSON for up to :data:`BATCH` entities at once.

    ``props=info`` is asked for because that is what carries ``lastrevid``: the pinning the
    whole file exists to preserve survives batching, it is just requested explicitly here
    rather than arriving with the entity.
    """
    params = {
        "action": "wbgetentities",
        "ids": "|".join(qids),
        "props": "info|labels|descriptions|claims",
        "languages": "en|mul",
        "format": "json",
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"
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


def entity_in(payload: dict, qid: str) -> dict | None:
    """Return one entity out of a batch payload, following a redirect if there was one.

    Wikidata merges entities, and asking for a merged id returns the *target* keyed under
    its own id. Looking only under the id asked for would report a merged entity as missing,
    which is the same silent loss the batch fallback below exists to prevent.
    """
    entities = payload.get("entities", {})
    if qid in entities:
        return entities[qid]
    for entity in entities.values():
        if (entity.get("redirects") or {}).get("from") == qid:
            return entity
    return None


def fetch_entities(
    qids: Iterable[str],
    fetch: BatchFetch = fetch_batch,
    pause: float = 0.1,
    on_error: Callable[[str, Exception], None] | None = None,
    one: Fetch = fetch_entity,
) -> Iterator[dict]:
    """Yield one record per entity, skipping and reporting the ones that fail.

    Fetched fifty at a time. ``Special:EntityData`` serves one entity per request, which is
    the right shape for re-reading a pinned revision and the wrong one for a slice: the
    previous project's was 62,497 entities, and one request each would be a working day of
    waiting rather than the twenty minutes 1,250 batched requests take.

    A failed fetch is *unknown data*, not absent data. It is reported through ``on_error``
    rather than yielded as an empty record, because a run that silently turns failures into
    empty statement sets is how a third of an entity set goes missing without a symptom.
    Batching makes that worse in one specific way — a single bad id would cost the other
    forty-nine — so a batch that fails is retried one entity at a time before anything in it
    is given up on.
    """
    order = list(dict.fromkeys(qids))
    for start in range(0, len(order), BATCH):
        chunk = order[start : start + BATCH]
        try:
            payload = fetch(chunk)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            for qid in chunk:
                try:
                    yield record(qid, one(qid))
                except (urllib.error.URLError, OSError, ValueError, KeyError):
                    if on_error is not None:
                        on_error(qid, exc)
                if pause:
                    time.sleep(pause)
            continue
        for qid in chunk:
            entity = entity_in(payload, qid)
            if entity is None:
                if on_error is not None:
                    on_error(qid, KeyError(f"{qid} absent from the batch that asked for it"))
                continue
            yield record(qid, {"entities": {qid: entity}})
        if pause:
            time.sleep(pause)
