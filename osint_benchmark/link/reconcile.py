"""Resolve names and codes to Wikidata QIDs, for the sources that arrive without them.

The cables and the parliamentary record get their QIDs from a linker reading prose. The
tabular sources do not: a sanctions listing has a name, a UCDP event has a country name and
a Gleditsch-Ward number, a GDELT row has CAMEO and FIPS codes. Those resolve by lookup.

Queries go to a public QLever endpoint over the live graph rather than a local dump.
Measured: an exact code lookup answers in 5 ms and a batched label-plus-type lookup in
about 100 ms, including the alias-relation query the previous project's local index could
not plan. Nothing here is a scan; every query is anchored on a key.

Resolution is deliberately conservative, because a sanctions list is exactly where a false
match is expensive: a candidate must agree on the name *and* on entity kind. Anything else
is left unresolved, which is the correct answer here -- most sanctioned individuals
genuinely have no Wikidata entity.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable

DEFAULT_ENDPOINT = "https://qlever.dev/api/wikidata"
# It is somebody else's research service. Batched queries arrive fast enough to earn a 429
# -- reconciling the sanctions list hit one within five seconds -- so back off when asked
# and pause between batches rather than going as fast as the network allows.
TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})
RETRIES = 5
BACKOFF_SECONDS = 5
BATCH_PAUSE_SECONDS = 0.5

USER_AGENT = "osint-benchmark/0.1 (research; https://github.com/Jibril-Frej/osint-benchmark)"

PREFIXES = (
    "PREFIX wd: <http://www.wikidata.org/entity/> "
    "PREFIX wdt: <http://www.wikidata.org/prop/direct/> "
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
    "PREFIX schema: <http://schema.org/> "
)

# External identifier schemes the tabular sources already carry. Resolving on these is
# exact: no name matching, no ranking, no judgement.
CODE_PROPERTIES = {
    "fips": "P901",  # GDELT ActionGeo country codes
    "iso3": "P298",  # CAMEO actor country codes track ISO-3
    "gwno": "P4133",  # Gleditsch-Ward number, UCDP state actors
}

Query = Callable[[str], list[dict]]


def sparql(query: str, endpoint: str = DEFAULT_ENDPOINT, timeout: float = 60.0) -> list[dict]:
    """Run a SPARQL query and return its result bindings.

    Retries what is worth retrying. A 429 means slow down, not stop, and the alternative is
    a linking run that dies partway through and leaves an entity set nobody can tell is
    incomplete.
    """
    # POST, not GET. A batch of sixty long names in a query string overflows the URI limit
    # and earns a 414 -- which the sanctions list, whose names run to forty characters,
    # does reliably.
    request = urllib.request.Request(
        endpoint,
        data=urllib.parse.urlencode({"query": PREFIXES + query}).encode(),
        headers={
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": USER_AGENT,
        },
    )
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                return json.loads(response.read())["results"]["bindings"]
        except urllib.error.HTTPError as exc:
            if exc.code not in TRANSIENT_STATUS:
                raise
            last = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
        time.sleep(BACKOFF_SECONDS * (attempt + 1))
    raise RuntimeError(f"{endpoint}: giving up after {RETRIES} attempts") from last


QID_ONLY = re.compile(r"^Q\d+$")


def _qid(binding: dict, name: str = "s") -> str:
    """Return the QID from a binding, or empty if it is not an item.

    ``rdfs:label`` matches lexemes and their senses too -- "Afghanistan" returns ten
    ``L…-S1`` ids alongside the entities. A lexeme is a word, not a thing anyone can ask a
    question about.
    """
    identifier = binding[name]["value"].rsplit("/", 1)[-1]
    return identifier if QID_ONLY.match(identifier) else ""


def by_code(scheme: str, codes: Iterable[str], query: Query = sparql) -> dict[str, str]:
    """Return ``code -> QID`` for an external identifier scheme.

    Exact and unambiguous: the code *is* the join key, so there is nothing to rank.

    Raises:
        KeyError: If the scheme has no known Wikidata property.
    """
    if scheme not in CODE_PROPERTIES:
        known = ", ".join(sorted(CODE_PROPERTIES))
        raise KeyError(f"unknown code scheme {scheme!r}; known: {known}")
    prop = CODE_PROPERTIES[scheme]
    wanted = [c for c in dict.fromkeys(codes) if c]
    if not wanted:
        return {}
    values = " ".join(f'"{c}"' for c in wanted)
    rows = query(f"SELECT ?s ?c WHERE {{ VALUES ?c {{ {values} }} ?s wdt:{prop} ?c }}")
    return {row["c"]["value"]: _qid(row) for row in rows}


def by_label(
    names: Iterable[str],
    kinds: Iterable[str] = (),
    query: Query = sparql,
    batch: int = 60,
) -> dict[str, list[str]]:
    """Return ``name -> candidate QIDs`` by exact English label, optionally typed.

    Candidates, not answers: a label match is ambiguous by construction and the caller
    decides. ``kinds`` are ``P31`` values (human, business, sovereign state); when given, a
    candidate must be an instance of one of them, which is what stops a person resolving
    to a company of the same name.

    Batched at 60 names per query -- larger batches drew "remote end closed connection"
    against the previous project's endpoint.
    """
    wanted = [n for n in dict.fromkeys(names) if n and n.strip()]
    found: dict[str, list[str]] = {}
    kind_values = " ".join(f"wd:{k}" for k in kinds)
    for start in range(0, len(wanted), batch):
        chunk = wanted[start : start + batch]
        values = " ".join(json.dumps(n) + "@en" for n in chunk)
        where = f"VALUES ?l {{ {values} }} ?s rdfs:label ?l ."
        if kind_values:
            where += f" VALUES ?k {{ {kind_values} }} ?s wdt:P31 ?k ."
        for row in query(f"SELECT DISTINCT ?s ?l WHERE {{ {where} }}"):
            qid = _qid(row)
            if qid:
                found.setdefault(row["l"]["value"], []).append(qid)
        if BATCH_PAUSE_SECONDS:
            time.sleep(BATCH_PAUSE_SECONDS)
    return found


def modified(qids: Iterable[str], query: Query = sparql) -> dict[str, str]:
    """Return ``QID -> last-modified timestamp``.

    The index carries ``schema:dateModified`` per entity, so "has this changed since it was
    read?" is a cheap query rather than a re-fetch. That is what makes a live endpoint
    usable in a pipeline whose answers have to stay correct.
    """
    wanted = [q for q in dict.fromkeys(qids) if q]
    if not wanted:
        return {}
    values = " ".join(f"<https://www.wikidata.org/wiki/Special:EntityData/{q}>" for q in wanted)
    rows = query(f"SELECT ?s ?d WHERE {{ VALUES ?s {{ {values} }} ?s schema:dateModified ?d }}")
    return {binding["s"]["value"].rsplit("/", 1)[-1]: binding["d"]["value"] for binding in rows}
