"""The second private leg: Swiss diplomatic documents from Dodis.

Two files make one corpus, and the split is the point.

**The text** is OCR over the scans, not the editorial summary. Dodis publishes a *regest*
of a couple of sentences per document; that is a description of a document rather than a
document, and a question built on one is a question about a catalogue entry. The scans
carry what the diplomat actually wrote.

**The entities** come from the metadata, and they are hand-curated by archivists — places
already carry Wikidata QIDs, people carry names. So this corpus needs no entity linker at
all: no ReFinED, no mGENRE, no GPU. That is worth more than it sounds, because the linker
is the ceiling on every other source.

Where it differs from every other source here
---------------------------------------------

It cannot be fetched. Dodis offers no bulk download of scans and its site defends against
crawlers, so the 4,065 PDFs were gathered by a polite crawler that lives in neither this
repository nor the previous one, and OCR'd in a separate GPU pass. **Both steps are
missing from this project**, which means a stranger cannot rebuild this corpus from a
clone the way they can rebuild Cablegate.

That is recorded rather than hidden. ``OSINT_DODIS_OCR`` and ``OSINT_DODIS_NT`` point at
the two inputs; without them this source says what is missing instead of producing a
smaller corpus that looks complete. Closing the gap means either publishing the scans, if
their licence allows, or bringing the crawler into this repository.

The N-Triples parse is ported from the previous project, including the reason it reads
N-Triples rather than the MySQL dump: the format is line-oriented, so a summary containing
newlines never splits a record — the same hazard that truncated the Cablegate CSV.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

from osint_benchmark.schema import Document
from osint_benchmark.sources.base import Projection, Source

VOCAB = "http://dodis.ch/schema/vocab/"
LINE = re.compile(r"^<([^>]+)>\s+<([^>]+)>\s+(.+)\s\.\s*$")
LITERAL = re.compile(r'^"(.*)"(?:\^\^<[^>]+>|@[\w-]+)?$', re.S)

DEFAULT_LANGS = ("de", "fr")

# The vocab predicates this build consumes. The dump holds ~7M vocab triples and only a
# quarter are these; gating before parsing the object literal roughly halves the pass.
HANDLED = frozenset(
    {
        "document_summary",
        "document_title",
        "document_classification",
        "document_lang_code",
        "document_doc_date",
        "document_has_person_document_id",
        "document_has_person_person_id",
        "document_has_place_document_id",
        "document_has_place_place_id",
        "person_fallback_person_id",
        "person_fallback_first_name",
        "person_fallback_last_name",
        "place_wikidata_id",
        "place_fallback_place_id",
        "place_fallback_name",
    }
)

PROJECTION = Projection(
    source="Dodis open-data N-Triples export, joined to OCR over the document scans",
    source_fields=(
        "document_summary",
        "document_title",
        "document_classification",
        "document_lang_code",
        "document_doc_date",
        "document_has_person",
        "document_has_place",
        "person_fallback",
        "place_wikidata_id",
        "place_fallback",
    ),
    kept={
        "document_doc_date": "date",
        "document_title": "title",
        "document_lang_code": "lang",
        "document_classification": "meta.classification",
        "document_summary": "meta.summary (the regest, kept beside the OCR text)",
        "person_fallback": "meta.persons (curated by archivists, names only)",
        "place_wikidata_id": "meta.places (curated, and already Wikidata QIDs)",
        "<the OCR text>": "text",
    },
    dropped={
        "specimen": "physical description of the artefact, not its content",
        "dossier": "archival grouping, not a property of the document",
        "relationship_type": "internal vocabulary for how records reference each other",
        "person_fallback_first_name": "folded into the person's name",
    },
    note=(
        "The text is OCR over the scans; the summary is Dodis's own regest and is kept in "
        "meta so the two can be compared. Entities are the archive's curated links, not a "
        "linker's output, so this corpus needs no entity linking."
    ),
)


def obj_value(term: str) -> str:
    r"""Return the value of an N-Triples object term.

    URIs come back without their angle brackets; literals are unescaped, since
    N-Triples ``\"``, ``\\``, ``\n`` and ``\uXXXX`` are all JSON-compatible. A malformed
    literal is returned verbatim rather than dropped: one bad triple costs one field.
    """
    term = term.strip()
    if term.startswith("<"):
        return term[1:-1]
    match = LITERAL.match(term)
    if not match:
        return term
    try:
        return json.loads('"' + match.group(1) + '"')
    except json.JSONDecodeError:
        return match.group(1)


def clean(text: str) -> str:
    """Reduce a Dodis title or summary to plain text; the dump carries HTML."""
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def read_metadata(nt_path: Path, langs: tuple[str, ...] = DEFAULT_LANGS) -> dict[str, dict]:
    """Return ``document id -> metadata`` from the N-Triples dump.

    One streaming pass. The dump is a relational export, so the person and place links
    arrive as separate rows keyed by URI and are joined afterwards rather than by seeking
    back through a 546 MB file.
    """
    docs: dict[str, dict] = defaultdict(dict)
    has_person: dict[str, dict] = defaultdict(dict)
    has_place: dict[str, dict] = defaultdict(dict)
    person_parts: dict[str, dict] = defaultdict(dict)
    place_parts: dict[str, dict] = defaultdict(dict)
    place_qid: dict[str, str] = {}

    fields = {
        "document_summary": "summary",
        "document_title": "title",
        "document_classification": "classification",
        "document_lang_code": "lang",
        "document_doc_date": "date",
    }
    with nt_path.open(encoding="utf-8") as handle:
        for line in handle:
            match = LINE.match(line)
            if not match:
                continue
            subject, predicate, raw = match.groups()
            if not predicate.startswith(VOCAB):
                continue
            field = predicate[len(VOCAB) :]
            if field not in HANDLED:
                continue
            value = obj_value(raw)
            if field in fields:
                docs[subject][fields[field]] = value
            elif field == "document_has_person_document_id":
                has_person[subject]["doc"] = value
            elif field == "document_has_person_person_id":
                has_person[subject]["person"] = value
            elif field == "document_has_place_document_id":
                has_place[subject]["doc"] = value
            elif field == "document_has_place_place_id":
                has_place[subject]["place"] = value
            elif field == "person_fallback_person_id":
                person_parts[subject]["person"] = value
            elif field == "person_fallback_first_name":
                person_parts[subject]["first"] = value
            elif field == "person_fallback_last_name":
                person_parts[subject]["last"] = value
            elif field == "place_wikidata_id":
                place_qid[subject] = value
            elif field == "place_fallback_place_id":
                place_parts[subject]["place"] = value
            elif field == "place_fallback_name":
                place_parts[subject]["name"] = value

    # First fallback seen wins: the same person has one row per language and the names
    # agree, so a later row would only overwrite like with like.
    names: dict[str, str] = {}
    for row in person_parts.values():
        uri = row.get("person")
        if uri and uri not in names:
            name = " ".join(part for part in (row.get("first"), row.get("last")) if part).strip()
            if name:
                names[uri] = name
    place_names: dict[str, str] = {}
    for row in place_parts.values():
        uri = row.get("place")
        if uri and uri not in place_names and row.get("name"):
            place_names[uri] = row["name"]

    persons: dict[str, list[str]] = defaultdict(list)
    for row in has_person.values():
        if row.get("doc") and row.get("person") in names:
            persons[row["doc"]].append(names[row["person"]])
    places: dict[str, list[dict]] = defaultdict(list)
    for row in has_place.values():
        place = row.get("place")
        if row.get("doc") and place and (place in place_qid or place in place_names):
            places[row["doc"]].append({"qid": place_qid.get(place), "name": place_names.get(place)})

    wanted = set(langs)
    return {
        uri.rsplit("/", 1)[-1]: {
            "lang": doc.get("lang"),
            "date": doc.get("date"),
            "classification": doc.get("classification"),
            "title": clean(doc.get("title", "")),
            "summary": clean(doc.get("summary", "")),
            "persons": sorted(set(persons.get(uri, []))),
            "places": places.get(uri, []),
        }
        for uri, doc in docs.items()
        if doc.get("lang") in wanted and doc.get("summary")
    }


def iso_date(raw: str | None) -> str | None:
    """Return the ISO-8601 form of a Dodis ``D.M.YYYY`` date, or None.

    Dodis writes dates the Swiss way and not always completely — ``1947`` and ``5.1947``
    both occur. A date that will not parse returns None rather than a guess; the original
    stays in ``meta.date_raw``.
    """
    if not raw:
        return None
    parts = raw.strip().split(".")
    try:
        if len(parts) == 3:
            day, month, year = (int(p) for p in parts)
            return f"{year:04d}-{month:02d}-{day:02d}"
        if len(parts) == 1 and len(parts[0]) == 4:
            return f"{int(parts[0]):04d}-01-01"
    except ValueError:
        return None
    return None


def to_document(doc_id: str, text: str, meta: dict) -> Document:
    """Turn one OCR'd document and its metadata into a record."""
    return Document(
        doc_id=doc_id,
        source="dodis",
        text=text,
        date=iso_date(meta.get("date")),
        lang=meta.get("lang"),
        title=meta.get("title") or "",
        meta={
            "classification": meta.get("classification"),
            "summary": meta.get("summary"),
            "persons": meta.get("persons", []),
            "places": meta.get("places", []),
            "date_raw": meta.get("date"),
        },
    )


def ocr_dir() -> Path:
    """Return where the OCR'd text lives."""
    return Path(os.environ.get("OSINT_DODIS_OCR", Path.home() / "dodis_ocr"))


def nt_path() -> Path:
    """Return where the N-Triples metadata dump lives."""
    return Path(os.environ.get("OSINT_DODIS_NT", Path.home() / "dodis-opendata.nt"))


def parse(raw_dir: Path):
    """Yield one record per OCR'd document that the metadata also describes.

    Raises:
        SystemExit: If either input is missing, naming which one and what it is. Silently
            yielding nothing would look like an empty corpus rather than an absent one.
    """
    ocr, triples = ocr_dir(), nt_path()
    if not ocr.is_dir():
        raise SystemExit(
            f"{ocr} is missing: Dodis needs the OCR'd scans. They are not fetchable from "
            "this repository -- see the module docstring. Set OSINT_DODIS_OCR."
        )
    if not triples.exists():
        raise SystemExit(
            f"{triples} is missing: Dodis needs its open-data N-Triples dump. Set OSINT_DODIS_NT."
        )
    metadata = read_metadata(triples)
    for path in sorted(ocr.glob("dodis-*.txt")):
        doc_id = path.stem.removeprefix("dodis-")
        meta = metadata.get(doc_id)
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        # A scan with no metadata cannot be dated or entity-linked, and one with no text
        # is a failed OCR. Either way there is no question to build from it.
        if meta and text:
            yield to_document(doc_id, text, meta).to_json()


SOURCE = Source(name="dodis", kind="private", parse=parse, projection=PROJECTION)
