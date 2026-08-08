"""Pair a confidential document with the parliamentary business item on the same subject.

The join two question types rest on: what a government was told privately, beside what its
parliament was doing about the same thing at the same time. Ported from the previous
project, which built it to size the family before writing a generator for it, and whose
measurements are the reason each condition below exists.

**Lexical overlap is useless here.** The cables are English and the parliamentary record is
German, so two language-independent signals do the work instead:

* **Topic.** Cables carry State Department TAGS (PREL, PARM, ETRD) and 69% of business items
  carry German subject categories (Wirtschaft, Sicherheitspolitik). :data:`CROSSWALK` maps
  between them.
* **Entities.** Both sides are linked to Wikidata, so a shared entity is a shared
  *identifier* rather than a shared spelling. Matching English labels inside German text
  bridged 113 entities; matching QIDs bridges the 2,091 the two corpora actually share.

**A shared entity is a weak signal on its own**, and the previous project measured why:
matching on QIDs doubled the pair count and diluted it, because a cable about WTO
negotiations and a motion on registered partnerships share "European Union" and nothing
else. Two things separate a topical link from a coincidence, and :func:`focused` requires
one of them — the entity appears in the item's own *title*, so the item is about it, or the
pair shares more than one entity and at least one is not a place.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from datetime import date, datetime

# State Department subject TAGS -> Swiss parliament subject categories.
CROSSWALK: dict[str, frozenset[str]] = {
    "PREL": frozenset({"Internationale Politik"}),
    "PGOV": frozenset({"Staatspolitik"}),
    "PARM": frozenset({"Sicherheitspolitik"}),
    "MARR": frozenset({"Sicherheitspolitik"}),
    "MNUC": frozenset({"Sicherheitspolitik"}),
    "KNNP": frozenset({"Sicherheitspolitik"}),
    "PTER": frozenset({"Sicherheitspolitik"}),
    "MOPS": frozenset({"Sicherheitspolitik"}),
    "PHUM": frozenset({"Soziale Fragen", "Recht Allgemein"}),
    "PREF": frozenset({"Migration"}),
    "SMIG": frozenset({"Migration"}),
    "ECON": frozenset({"Wirtschaft"}),
    "ETRD": frozenset({"Wirtschaft"}),
    "EINV": frozenset({"Wirtschaft"}),
    "EIND": frozenset({"Wirtschaft"}),
    "EFIN": frozenset({"Finanzwesen"}),
    "ENRG": frozenset({"Energie"}),
    "SENV": frozenset({"Umwelt"}),
    "TBIO": frozenset({"Gesundheit"}),
    "KFLU": frozenset({"Gesundheit"}),
    "KIPR": frozenset({"Recht Allgemein"}),
    "KCRM": frozenset({"Recht Allgemein"}),
    "KJUS": frozenset({"Recht Allgemein"}),
    "CVIS": frozenset({"Recht Allgemein"}),
    "KPAO": frozenset({"Medien und Kommunikation"}),
    "EAGR": frozenset({"Landwirtschaft"}),
    "EDUC": frozenset({"Bildung"}),
}

# A cable reports on what is in front of it, and a parliamentary item takes months to move.
WINDOW_DAYS = 90

# An entity in more than this share of the confidential corpus carries no topical signal:
# "Switzerland" links a branding postulate to every Swiss cable there is.
MAX_ENTITY_SHARE = 0.15

# Spaces but not newlines. The previous project allowed \s, which runs past the end of the
# line and swallows the start of the next one -- so the last tag came back glued to it,
# failed isalpha, and was silently dropped. Every cable lost its final tag that way.
TAGS = re.compile(r"TAGS:[ \t]*([A-Z0-9_,\t -]{3,200})")
SUBJECT = re.compile(r"SUBJECT:\s*(.{0,120})")
WORD = re.compile(r"[a-z0-9]+")


def normalise(text: str) -> str:
    """Case-fold, strip accents, and reduce to alphanumeric tokens.

    Accents are stripped because the two sides spell the same name differently — Genève and
    Genf and Geneva — and the comparison is on tokens rather than on characters.
    """
    decomposed = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(WORD.findall(stripped.lower()))


def parse_date(value) -> date | None:
    """Return an ISO date from whatever a corpus stored, or None when it cannot be read."""
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def tags_in(text: str) -> set[str]:
    """Return the State Department topic tags a cable declares."""
    found = TAGS.search(text or "")
    if not found:
        return set()
    return {tag.strip() for tag in found.group(1).split(",") if tag.strip().isalpha()}


def subject_of(text: str) -> str:
    """Return a cable's subject line, or empty when it has none."""
    found = SUBJECT.search(text or "")
    return found.group(1).strip() if found else ""


def topics(tags: Iterable[str]) -> set[str]:
    """Return the parliamentary subject categories a cable's tags map onto."""
    mapped: set[str] = set()
    for tag in tags:
        mapped |= CROSSWALK.get(tag, frozenset())
    return mapped


def generic(
    cables: list[dict], labels: dict[str, str], share: float = MAX_ENTITY_SHARE
) -> set[str]:
    """Return the QIDs too common in the private corpus to mean anything."""
    if not cables:
        return set()
    counts: Counter[str] = Counter()
    for cable in cables:
        counts.update(set(cable.get("qids", ())))
    ceiling = share * len(cables)
    return {qid for qid, seen in counts.items() if seen > ceiling and qid in labels}


def focused(pair: dict, places: set[str]) -> bool:
    """Return whether a pair's shared entities amount to a topical link.

    The condition the previous project arrived at after measuring that entity overlap alone
    doubled the pairs and diluted them. Either the item's *title* names a shared entity — so
    the item is about it, not merely adjacent to it — or there is more than one shared entity
    and at least one is not a place, since two documents sharing only countries share only a
    map reference.
    """
    if pair["shared_in_title"]:
        return True
    shared = pair["shared_qids"]
    return len(shared) > 1 and any(qid not in places for qid in shared)


def by_month(items: Iterable[dict]) -> dict[tuple[int, int], list[dict]]:
    """Bucket public items by year and month, so the pairwise scan stays cheap."""
    buckets: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for item in items:
        buckets[(item["date"].year, item["date"].month)].append(item)
    return buckets


def nearby(buckets: dict, when: date, window: int) -> Iterator[dict]:
    """Yield the items within ``window`` days of a date, month bucket by month bucket."""
    months = window // 30 + 2
    for offset in range(-months, months + 1):
        total = when.year * 12 + (when.month - 1) + offset
        year, month = divmod(total, 12)
        for item in buckets.get((year, month + 1), []):
            if abs((item["date"] - when).days) <= window:
                yield item


def join(
    cables: list[dict],
    business: list[dict],
    labels: dict[str, str],
    places: set[str],
    window: int = WINDOW_DAYS,
    max_share: float = MAX_ENTITY_SHARE,
    outcomes: Counter | None = None,
) -> Iterator[dict]:
    """Yield one record per (cable, business item) that shares a topic, a date and an entity.

    Each side is a dict of ``doc_id``, ``date``, ``qids`` and its own text fields. Every pair
    carries ``focused``, which is what the two question types filter on: the unfocused ones
    are kept so the ratio is visible rather than assumed.
    """
    if outcomes is None:
        outcomes = Counter()
    ignore = generic(cables, labels, max_share)
    outcomes["generic_entities_ignored"] = len(ignore)
    buckets = by_month(business)

    for cable in cables:
        wanted = topics(cable.get("tags", ()))
        if not wanted:
            outcomes["no_mapped_topic"] += 1
            continue
        outcomes["cable_with_topic"] += 1
        # Short labels match too much once they are reduced to tokens, and a generic entity
        # links anything to anything.
        usable = [
            qid
            for qid in cable.get("qids", ())
            if qid in labels and len(labels[qid]) > 3 and qid not in ignore
        ]
        seen: set[str] = set()
        for item in nearby(buckets, cable["date"], window):
            if item["doc_id"] in seen or not (wanted & item["cats"]):
                continue
            seen.add(item["doc_id"])
            shared = [qid for qid in usable if qid in item["qids"]]
            if not shared:
                outcomes["no_shared_entity"] += 1
                continue
            title = normalise(item.get("title", ""))
            # Whole tokens, never a substring: "Chad" otherwise matches inside the German
            # "Schlechtwetterentschaedigung" and manufactures a topical link out of nothing.
            in_title = [
                qid
                for qid in shared
                if labels.get(qid) and re.search(rf"\b{re.escape(normalise(labels[qid]))}\b", title)
            ]
            pair = {
                "private_id": cable["doc_id"],
                "private_date": cable["date"].isoformat(),
                "private_origin": cable.get("origin", ""),
                "private_subject": cable.get("subject", "")[:90],
                "private_tags": sorted(cable.get("tags", ()))[:6],
                "public_id": item["doc_id"],
                "public_date": item["date"].isoformat(),
                "public_title": item.get("title", "")[:110],
                "public_type": item.get("type", ""),
                "shared_cats": sorted(wanted & item["cats"]),
                "shared_qids": shared[:4],
                "shared_entities": [labels.get(qid, qid) for qid in shared[:4]],
                "shared_in_title": [labels.get(qid, qid) for qid in in_title[:4]],
                "has_response": item.get("has_response", False),
                "days_apart": abs((item["date"] - cable["date"]).days),
            }
            pair["focused"] = focused(pair, places)
            outcomes["focused" if pair["focused"] else "unfocused"] += 1
            yield pair
