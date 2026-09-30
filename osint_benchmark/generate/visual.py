"""The two question types asked over a page scan rather than over its text.

A Dodis document is a scanned typescript, and the benchmark's other Dodis questions read it
through its OCR. These two are asked over the first page's image instead, beside a public
record, and keep the benchmark's one rule: the answer is computed, the private side says
what to look up and the public side says what it is.

* **office_holder** -- the letterhead says which country the sending Swiss mission sat in
  and the dateline says when; Wikidata's dated heads of government say who led that
  country's government that day. "Who headed the government of the country this letter
  was sent from, on the day it was written?"
* **addressee** -- the address block names a Swiss official by title, initial and family
  name; the archive's own list of the document's people gives the full name, and Wikidata
  confirms there is exactly one such Swiss person and describes them. "What is the full name
  of the official this letter is addressed to?"

What is read from the page is read from the OCR by :mod:`osint_benchmark.generate.letter`,
because a question has to be built before anyone looks at the image; that the image carries
the same thing is measured afterwards, by the necessity step's OCR-only condition.

Every document dropped is counted under the reason it was dropped, as the text builders do.

Two things are recorded that no question asks about, for the system evaluation to score:
``classification`` (the archive's secrecy grading, which a reader sees stamped on the page)
and ``names_on_scan`` (the people the archive lists for the document whose family names are
legible on page one -- what a name-redacting strategy has to hide, and what a transcription
of the image is checked against).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator

from osint_benchmark.generate import letter
from osint_benchmark.generate.typed import Candidate, names
from osint_benchmark.public.offices import Term
from osint_benchmark.sources import refs
from osint_benchmark.visual.scan import ScanUnavailable

# A scan taker: the bare Dodis id in, the scan's on-item form out.
Scanner = Callable[[str], dict]

# The day-precise form of a Dodis date. A year alone ("1947") becomes 1 January on the
# corpus record, which is a day nobody headed anything on in particular.
DAY_PRECISE = re.compile(r"^\s*\d{1,2}\.\d{1,2}\.\d{4}\s*$")

# The public record of an office holder question: one state's heads of government.
OFFICES = "offices"
# The public record of an addressee question: one person's Wikidata description.
PEOPLE = "people"


def display_name(person: str) -> str:
    """Return an archive name in reading order: "Petitpierre, Max" becomes "Max Petitpierre"."""
    family, comma, given = person.partition(",")
    return f"{given.strip()} {family.strip()}" if comma else person.strip()


def names_on_scan(page: str, persons: Iterable[str]) -> list[str]:
    """Return the archive's people whose family names are legible on the page.

    Compared on the folded, closed-up text, so an emphasised "Z e h n d e r" counts.
    """
    folded = letter.fold(letter.unspaced(page))
    found = []
    for person in persons:
        name = display_name(person)
        family = letter.family_of(name)
        if family and re.search(rf"(?<![a-z]){re.escape(family)}(?![a-z])", folded):
            found.append(name)
    return sorted(set(found))


def holding(held: Iterable[Term], day: str) -> list[Term]:
    """Return the terms that include a day, both ends counted."""
    return [term for term in held if term.start <= day and (not term.end or day <= term.end)]


def _common(document: dict) -> tuple[str, str, dict]:
    """Return a document's reference, its page one, and the provenance both types record."""
    doc_id = str(document["doc_id"])
    meta = document.get("meta") or {}
    page = letter.first_page(document.get("text", ""), doc_id)
    return (
        refs.ref("dodis", doc_id),
        page,
        {
            "classification": str(meta.get("classification") or ""),
            "names_on_scan": "; ".join(names_on_scan(page, meta.get("persons") or ())),
        },
    )


def _scan(doc_id: str, scanner: Scanner, outcomes: Counter) -> dict | None:
    """Return the page's scan, or None after counting why there is none."""
    try:
        return scanner(doc_id)
    except (ScanUnavailable, FileNotFoundError):
        outcomes["no_scan"] += 1
        return None


def from_office_holder(
    documents: Iterable[dict],
    places: dict[str, str],
    terms: dict[str, list[Term]],
    country_labels: dict[str, str],
    scanner: Scanner,
    outcomes: Counter | None = None,
) -> Iterator[Candidate]:
    """Yield an office-holder candidate per letter whose host country and day are certain.

    ``places`` maps a folded place name to its country (see
    :func:`osint_benchmark.public.offices.gazetteer`), ``terms`` a country to its heads of
    government, and ``scanner`` a Dodis id to its scan. A letter is dropped when:

    * its first page has no Swiss mission's letterhead;
    * the archive dates it to less than a day;
    * the letterhead names no country, or more than one;
    * no recorded term covers the day, or two do -- a handover day, or overlapping records;
    * the letter names the head of government itself, so the public record adds nothing.
    """
    if outcomes is None:
        outcomes = Counter()
    for document in documents:
        reference, page, provenance = _common(document)
        head = letter.letterhead(page)
        if head is None:
            outcomes["no_letterhead"] += 1
            continue
        raw = str((document.get("meta") or {}).get("date_raw") or "")
        day = document.get("date") or ""
        if not DAY_PRECISE.match(raw) or not day:
            outcomes["date_not_to_the_day"] += 1
            continue
        countries = letter.places_in(head.window, places)
        if len(countries) != 1:
            outcomes["host_country_unclear" if countries else "host_country_not_named"] += 1
            continue
        country = next(iter(countries))
        covering = {term.holder: term for term in holding(terms.get(country, ()), day)}
        if len(covering) != 1:
            outcomes["term_ambiguous" if covering else "no_term_on_date"] += 1
            continue
        term = next(iter(covering.values()))
        if names(document.get("text", ""), term.label):
            outcomes["gold_named_privately"] += 1
            continue
        image = _scan(str(document["doc_id"]), scanner, outcomes)
        if image is None:
            continue
        outcomes["candidate"] += 1
        country_label = country_labels.get(country, "")
        yield Candidate(
            item_id=f"{reference}|office_holder|{term.holder}",
            question_type="office_holder",
            answer=term.label,
            gold_qid=term.holder,
            private_id=reference,
            public_id=refs.ref(OFFICES, country),
            passage="",
            facts={"mission": head.mission},
            provenance={
                **provenance,
                "country_qid": country,
                "country": country_label,
                "date": day,
                "term": f"{term.start} to {term.end or 'present'}",
                # The country and the year are what the scan contributes. A question naming
                # either hands a public-only solver the lookup, so the gate checks for both.
                "withheld": f"{country_label}; {day[:4]}",
                "gold": "computed from the letter's date and Wikidata's dated heads of "
                "government; no model wrote it",
            },
            image=image,
        )


def addressed_to(
    document: dict, page: str | None = None
) -> tuple[letter.Addressee | None, str, str]:
    """Return a letter's addressee, the archive's full name for them, and why not if not.

    The third element is empty when the first two are usable, and otherwise names the
    condition that failed -- the same names :func:`from_addressee` counts under. Split out
    so the build step can collect the names to look up on Wikidata before building.
    """
    if page is None:
        page = letter.first_page(document.get("text", ""), str(document["doc_id"]))
    who = letter.addressee(page)
    if who is None:
        return None, "", "no_address_line"
    if who.given_written:
        return who, "", "given_name_on_letter"
    persons = [display_name(p) for p in (document.get("meta") or {}).get("persons") or ()]
    fitting = sorted(
        {
            name
            for name in persons
            if letter.family_of(name) == who.family and letter.initials_agree(who.initials, name)
        }
    )
    if len(fitting) != 1:
        return who, "", "addressee_ambiguous" if fitting else "addressee_not_catalogued"
    full = fitting[0]
    if names(document.get("text", ""), full):
        return who, full, "full_name_in_document"
    return who, full, ""


def from_addressee(
    documents: Iterable[dict],
    swiss: dict[str, list[str]],
    records: dict[str, dict],
    scanner: Scanner,
    outcomes: Counter | None = None,
) -> Iterator[Candidate]:
    """Yield an addressee candidate per letter whose addressee resolves to one person.

    ``swiss`` maps a full name to the Swiss people Wikidata files under it (see
    :func:`osint_benchmark.public.offices.swiss_people`), ``records`` a QID to that
    person's public record. A letter is dropped when:

    * its first page has no address line naming someone;
    * the address line writes a given name out, so the full name is on the page;
    * none, or more than one, of the archive's people for the document fits the family name
      and initials;
    * the letter spells that full name out anywhere;
    * Wikidata has no Swiss person of that name, or several, or no record to show.
    """
    if outcomes is None:
        outcomes = Counter()
    for document in documents:
        reference, page, provenance = _common(document)
        who, full, reason = addressed_to(document, page)
        if reason or who is None:
            outcomes[reason] += 1
            continue
        qids = swiss.get(full, [])
        if len(qids) != 1:
            outcomes["namesakes_on_wikidata" if qids else "not_on_wikidata"] += 1
            continue
        qid = qids[0]
        record = records.get(qid)
        if not record:
            outcomes["no_public_record"] += 1
            continue
        image = _scan(str(document["doc_id"]), scanner, outcomes)
        if image is None:
            continue
        outcomes["candidate"] += 1
        yield Candidate(
            item_id=f"{reference}|addressee|{qid}",
            question_type="addressee",
            answer=record["label"],
            gold_qid=qid,
            private_id=reference,
            public_id=refs.ref(PEOPLE, qid),
            passage="",
            facts={},
            provenance={
                **provenance,
                "address_line": who.line,
                "archive_name": full,
                # The family name is what the scan contributes; written into the question,
                # it turns an image question into a name lookup.
                "withheld": letter.family_of(full),
                "gold": "the archive's full name for the addressee, confirmed as one Swiss "
                "person on Wikidata; no model wrote it",
            },
            image=image,
        )
