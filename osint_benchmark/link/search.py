"""Resolve a name through Wikidata's own search, then verify the candidate.

The exact-match reconciliation in :mod:`osint_benchmark.link.reconcile` asks whether a name
appears verbatim as a label or alias. That resolves 383 of the sanctions list's 8,604
targets. The previous project resolved **1,232** from the same file, and the difference is
entirely in how candidates are found: it asked Wikidata's ``wbsearchentities`` — fuzzy,
ranked, across labels and aliases — and then *verified* what came back.

Which matters, because the public leg is the ceiling on the whole benchmark. Bridges have
scaled roughly with public entities: 284 gave 62, 383 gave 84.

Search alone would be reckless on a sanctions list, where a false match is expensive. So
each candidate must survive three checks, all ported from the previous project:

* **A label must actually match.** The search is fuzzy and will happily return a plausible
  neighbour; the candidate's own label or the matched variant has to agree once normalised.
* **The kind must agree.** A person may not resolve to a company of the same name.
* **Birth years must not disagree.** When both the listing and Wikidata state one and they
  differ, it is a different person — this is the check that makes searching safe rather
  than merely productive.

And every recorded name variant is tried, compared order-independently: a listing writes
"Lukashenka Dzmitry" where Wikidata writes "Dmitry Lukashenka", and only one variant need
match.
"""

from __future__ import annotations

import json
import re
import unicodedata
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable

API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "osint-benchmark/0.1 (research; https://github.com/Jibril-Frej/osint-benchmark)"

# Candidates per name variant. The previous project used five; beyond that the ranking has
# stopped meaning anything and every extra costs a verification lookup.
CANDIDATES = 5
# Variants tried per target. A listing can carry a dozen transliterations and the tail of
# them are near-duplicates.
VARIANTS = 4

PERSON_CLASSES = frozenset({"Q5"})
ORG_CLASSES = frozenset(
    {"Q43229", "Q4830453", "Q783794", "Q6881511", "Q7278", "Q484652", "Q1156831", "Q17127659"}
)

WORD = re.compile(r"[^\w\s]", re.UNICODE)


def normalise(name: str) -> str:
    """Return a name stripped of accents, punctuation and case, for comparison only."""
    folded = unicodedata.normalize("NFKD", name)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return WORD.sub(" ", folded).lower().strip()


def key(name: str) -> frozenset[str]:
    """Return a name as an unordered bag of tokens.

    Order-independent because a sanctions listing writes "Lukashenka Dzmitry" where
    Wikidata writes "Dmitry Lukashenka". Comparing the strings finds nothing.
    """
    return frozenset(normalise(name).split())


def get(url: str, timeout: float = 30.0) -> dict:
    """GET a JSON document, returning an empty dict on failure."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read())
    except Exception:  # noqa: BLE001 - one failed lookup costs one name
        return {}


def search(name: str, limit: int = CANDIDATES) -> list[dict]:
    """Return Wikidata's own ranked candidates for a name."""
    query = urllib.parse.urlencode(
        {
            "action": "wbsearchentities",
            "search": name[:250],
            "language": "en",
            "uselang": "en",
            "format": "json",
            "limit": limit,
            "type": "item",
        }
    )
    return get(f"{API}?{query}").get("search", []) or []


def facts(qids: list[str]) -> dict[str, dict]:
    """Return ``QID -> {label, classes, births}`` for candidates, fifty at a time."""
    found: dict[str, dict] = {}
    for start in range(0, len(qids), 50):
        chunk = qids[start : start + 50]
        query = urllib.parse.urlencode(
            {
                "action": "wbgetentities",
                "ids": "|".join(chunk),
                "props": "claims|labels",
                "languages": "en",
                "format": "json",
            }
        )
        for qid, entity in (get(f"{API}?{query}").get("entities") or {}).items():
            claims = entity.get("claims") or {}
            classes = {
                claim["mainsnak"]["datavalue"]["value"]["id"]
                for claim in claims.get("P31", [])
                if claim.get("mainsnak", {}).get("datavalue")
            }
            births = {
                claim["mainsnak"]["datavalue"]["value"]["time"][1:5]
                for claim in claims.get("P569", [])
                if claim.get("mainsnak", {}).get("datavalue")
            }
            label = ((entity.get("labels") or {}).get("en") or {}).get("value", "")
            found[qid] = {"label": label, "classes": classes, "births": births}
    return found


def resolve(
    names: Iterable[str],
    kind: str = "",
    years: Iterable[str] = (),
    searcher: Callable[[str], list[dict]] = search,
    fetch: Callable[[list[str]], dict[str, dict]] = facts,
) -> str | None:
    """Return the one candidate agreeing on name, kind and birth year, or None.

    Returns None rather than a best guess. Most sanctioned individuals genuinely have no
    Wikidata entity, and that is the correct answer for them.
    """
    variants = [name for name in dict.fromkeys(names) if len(normalise(name).split()) >= 2]
    if not variants:
        return None  # a single token is a collision waiting to happen
    variants = variants[:VARIANTS]

    hits: dict[str, dict] = {}
    for variant in variants:
        for hit in searcher(variant):
            hits.setdefault(hit["id"], hit)
    if not hits:
        return None

    known = fetch(list(hits))
    wanted = {key(variant) for variant in variants}
    years = {str(year) for year in years if year}
    for qid, hit in hits.items():
        info = known.get(qid)
        if not info:
            continue
        # The search is fuzzy and returns plausible neighbours; require a label to agree.
        if key(hit.get("label", "")) not in wanted and key(info["label"]) not in wanted:
            continue
        if kind == "individual" and not (info["classes"] & PERSON_CLASSES):
            continue
        if kind == "entity" and not (info["classes"] & ORG_CLASSES):
            continue
        # Both sides state a birth year and they disagree: a different person.
        if years and info["births"] and not (years & info["births"]):
            continue
        return qid
    return None
