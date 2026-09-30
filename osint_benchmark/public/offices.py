"""The public half of the two visual types: who held an office when, and who a name is.

Answered live from the QLever endpoint, like every other build-time Wikidata lookup here.
Three questions are asked of it, each anchored on a key so that none is a scan:

* **which country a place name means** -- the German, French and English names of every
  country that has a dated head of government, and of its capital, so that a letterhead
  reading "Gesandtschaft in Österreich" or a dateline reading "Wien" finds Austria;
* **who headed that country's government, and from when to when** -- the ``P6`` statements
  on the country's item with their start and end qualifiers;
* **which Swiss person a full name is** -- for the addressee, whose full name the archive
  records and the letter abbreviates.

**Why a head of government rather than the head of the Swiss mission.** The natural question
about a letter from a legation is who ran the legation, and Wikidata cannot answer it: the
Swiss ministers of the 1940s and 1950s are recorded, where they are recorded at all, as
holding the undated position "ambassador". Heads of government are the opposite case --
dated to the day, for every country the archive's missions sat in. So the office is the host
country's, and the private document supplies the two things the public record cannot: which
country, and which day.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass

from osint_benchmark.generate.letter import fold
from osint_benchmark.link import reconcile
from osint_benchmark.link.reconcile import Query

QUALIFIERS = (
    "PREFIX p: <http://www.wikidata.org/prop/> "
    "PREFIX ps: <http://www.wikidata.org/prop/statement/> "
    "PREFIX pq: <http://www.wikidata.org/prop/qualifier/> "
)

SWITZERLAND = "Q39"
HUMAN = "Q5"

# Country, historical country, sovereign state.
STATE_CLASSES = ("wd:Q6256", "wd:Q3024240", "wd:Q3624078")

# A place name this short is an abbreviation or a code, and matches inside the letterhead's
# other words -- "Irak" is fine, "UK" is not.
MIN_PLACE_CHARS = 4

# How many ids one VALUES clause carries.
BATCH = 60


@dataclass(frozen=True)
class Term:
    """One stretch of one person heading one country's government.

    Attributes:
        country: The country's QID.
        holder: The office holder's QID.
        label: The holder's English name, which is the answer.
        start: ISO date the term began.
        end: ISO date it ended, empty while it runs.
    """

    country: str
    holder: str
    label: str
    start: str
    end: str


def _id(binding: dict, name: str) -> str:
    """Return the trailing id of a URI binding."""
    return binding[name]["value"].rsplit("/", 1)[-1]


def _day(binding: dict, name: str) -> str:
    """Return a date binding as ``YYYY-MM-DD``, or empty when absent."""
    return binding.get(name, {}).get("value", "")[:10]


def governed(query: Query = reconcile.sparql) -> list[str]:
    """Return the states, present and historical, whose heads of government Wikidata dates.

    Historical states are wanted as much as present ones: a legation in Moscow in 1950 was
    posted to the Soviet Union, and its head of government is not Russia's.
    """
    rows = query(
        QUALIFIERS + f"SELECT DISTINCT ?c WHERE {{ VALUES ?k {{ {' '.join(STATE_CLASSES)} }} "
        "?c wdt:P31 ?k . ?c p:P6 ?st . ?st pq:P580 ?s . }"
    )
    return sorted({_id(row, "c") for row in rows} - {SWITZERLAND})


def gazetteer(countries: Iterable[str], query: Query = reconcile.sparql) -> dict[str, str]:
    """Return ``folded place name -> country QID`` for the given states and their capitals.

    A name that two states share is dropped rather than given to either: "Congo" names two,
    and so does "Berlin" once both German states are in the list, and a letterhead that
    says only that cannot be assigned. Switzerland is never in it, because every letterhead
    in the archive names it -- it is the sender, not the host.

    Asked one batch of states at a time. The same question asked of every state at once
    makes the endpoint find the states first by scanning all labels, and it times out.
    """
    wanted = sorted(set(countries) - {SWITZERLAND})
    owners: dict[str, set[str]] = {}
    for start in range(0, len(wanted), BATCH):
        values = " ".join(f"wd:{qid}" for qid in wanted[start : start + BATCH])
        rows = query(
            f"SELECT ?c ?l WHERE {{ VALUES ?c {{ {values} }} "
            "{ ?c rdfs:label ?l } UNION { ?c wdt:P36 ?cap . ?cap rdfs:label ?l } "
            'FILTER(LANG(?l) IN ("de", "fr", "en")) }'
        )
        for row in rows:
            name = fold(row["l"]["value"]).strip()
            if len(name) >= MIN_PLACE_CHARS:
                owners.setdefault(name, set()).add(_id(row, "c"))
        if reconcile.BATCH_PAUSE_SECONDS:
            time.sleep(reconcile.BATCH_PAUSE_SECONDS)
    return {name: next(iter(qids)) for name, qids in owners.items() if len(qids) == 1}


def terms(countries: Iterable[str], query: Query = reconcile.sparql) -> list[Term]:
    """Return every dated head-of-government term of the given countries.

    A statement without a start date is skipped: it cannot place anyone on a day. A missing
    end date is kept, as a term that runs on -- that is how Wikidata marks an incumbent.
    """
    wanted = sorted(set(countries))
    found: list[Term] = []
    for start in range(0, len(wanted), BATCH):
        values = " ".join(f"wd:{qid}" for qid in wanted[start : start + BATCH])
        rows = query(
            QUALIFIERS + f"SELECT ?c ?h ?hl ?s ?e WHERE {{ VALUES ?c {{ {values} }} "
            "?c p:P6 ?st . ?st ps:P6 ?h ; pq:P580 ?s . OPTIONAL { ?st pq:P582 ?e } "
            'OPTIONAL { ?h rdfs:label ?hl FILTER(LANG(?hl) = "en") } }'
        )
        for row in rows:
            label = row.get("hl", {}).get("value", "")
            if not label:
                continue
            found.append(
                Term(
                    country=_id(row, "c"),
                    holder=_id(row, "h"),
                    label=label,
                    start=_day(row, "s"),
                    end=_day(row, "e"),
                )
            )
        if reconcile.BATCH_PAUSE_SECONDS:
            time.sleep(reconcile.BATCH_PAUSE_SECONDS)
    return found


def swiss_people(names: Iterable[str], query: Query = reconcile.sparql) -> dict[str, list[str]]:
    """Return ``full name -> QIDs`` of the Swiss people Wikidata files under that exact name.

    The archive writes names the way a Swiss catalogue does, and Wikidata's English label for
    a Swiss diplomat is almost always the same string. Citizenship is required so that the
    Alfred Zehnder a Bern letter was addressed to is not matched to a namesake elsewhere;
    anything left ambiguous is the caller's to drop.
    """
    wanted = [name for name in dict.fromkeys(names) if name and name.strip()]
    found: dict[str, list[str]] = {}
    for start in range(0, len(wanted), BATCH):
        chunk = wanted[start : start + BATCH]
        values = " ".join(f'"{name}"@en' for name in (n.replace('"', "") for n in chunk))
        rows = query(
            f"SELECT DISTINCT ?s ?l WHERE {{ VALUES ?l {{ {values} }} ?s rdfs:label ?l . "
            f"?s wdt:P31 wd:{HUMAN} ; wdt:P27 wd:{SWITZERLAND} . }}"
        )
        for row in rows:
            qid = _id(row, "s")
            if qid.startswith("Q"):
                found.setdefault(row["l"]["value"], []).append(qid)
        if reconcile.BATCH_PAUSE_SECONDS:
            time.sleep(reconcile.BATCH_PAUSE_SECONDS)
    return found


def render_terms(country_label: str, held: Iterable[Term]) -> str:
    """Return a country's heads of government as the evidence text a solver is shown."""
    lines = [f"Heads of government of {country_label}:"]
    for term in sorted(held, key=lambda t: t.start):
        lines.append(f"- {term.label}: {term.start} to {term.end or 'present'}")
    return "\n".join(lines)


def english_labels(qids: Iterable[str], query: Query = reconcile.sparql) -> dict[str, str]:
    """Return ``QID -> English label`` for the given items."""
    wanted = sorted(set(qids))
    found: dict[str, str] = {}
    for start in range(0, len(wanted), BATCH):
        values = " ".join(f"wd:{qid}" for qid in wanted[start : start + BATCH])
        rows = query(
            f"SELECT ?s ?l WHERE {{ VALUES ?s {{ {values} }} ?s rdfs:label ?l "
            'FILTER(LANG(?l) = "en") }'
        )
        for row in rows:
            found[_id(row, "s")] = row["l"]["value"]
    return found


def person_records(qids: Iterable[str], query: Query = reconcile.sparql) -> dict[str, dict]:
    """Return ``QID -> {label, description, born, positions}`` for the given people.

    This is the public side of an addressee question: what an analyst holding only a family
    name, a title and a date would look the person up in. Positions are what make the lookup
    possible -- "Chef der Abteilung für Politische Angelegenheiten" on the letter, "head of
    the Division of Political Affairs" here.
    """
    wanted = sorted(set(qids))
    found: dict[str, dict] = {}
    for start in range(0, len(wanted), BATCH):
        values = " ".join(f"wd:{qid}" for qid in wanted[start : start + BATCH])
        rows = query(
            f"SELECT ?s ?l ?d ?b ?pl WHERE {{ VALUES ?s {{ {values} }} "
            '?s rdfs:label ?l FILTER(LANG(?l) = "en") '
            'OPTIONAL { ?s schema:description ?d FILTER(LANG(?d) = "en") } '
            "OPTIONAL { ?s wdt:P569 ?b } "
            'OPTIONAL { ?s wdt:P39 ?p . ?p rdfs:label ?pl FILTER(LANG(?pl) = "en") } }'
        )
        for row in rows:
            record = found.setdefault(
                _id(row, "s"),
                {"label": row["l"]["value"], "description": "", "born": "", "positions": []},
            )
            record["description"] = record["description"] or row.get("d", {}).get("value", "")
            record["born"] = record["born"] or _day(row, "b")[:4]
            position = row.get("pl", {}).get("value", "")
            if position and position not in record["positions"]:
                record["positions"].append(position)
        if reconcile.BATCH_PAUSE_SECONDS:
            time.sleep(reconcile.BATCH_PAUSE_SECONDS)
    return found


def render_person(record: dict) -> str:
    """Return a person's public record as the evidence text a solver is shown."""
    head = record["label"] + (f" (born {record['born']})" if record.get("born") else "")
    lines = [f"{head}: {record.get('description') or 'no description'}."]
    if record.get("positions"):
        lines.append("Positions held: " + "; ".join(sorted(record["positions"])) + ".")
    return "\n".join(lines)
