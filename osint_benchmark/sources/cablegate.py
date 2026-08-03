r"""The private leg: WikiLeaks Cablegate, all 251k cables.

Two hazards in the dump, both of which cost corpus text in the previous project before
they were found.

**The file cannot be read as one CSV.** ``cables.csv`` is fully double-quoted and its
``body`` field holds multi-line telegram text, which desynchronises a ``csv.reader`` run
over the whole file — it yields ~21M fragments instead of ~250k records. So the file is
segmented into records first, by the quoted record-start pattern
``"<id>","<M/D/YYYY H:MM>","<ref>","<origin>","<class>",``, and each record is parsed in
isolation. One malformed record then cannot corrupt the rest.

**Quotes are escaped with a backslash**, not by doubling (``\"``, ``\'``). Without
``escapechar="\\"`` the reader treats ``\"`` as a field-closing quote: the quote state
inverts, the rest of the body is consumed as later columns, and the cable is silently
truncated at its first quoted phrase. That cost ~68% of the corpus text, and produced a
file that looked fine.

Every cable is kept. The benchmark's private corpus is the whole of Cablegate — a
retrieval corpus narrowed to the documents that produced questions is not a retrieval
corpus — so origin filtering is a downstream concern, not this parser's.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from osint_benchmark.schema import Document
from osint_benchmark.sources.base import Projection, Source

FILENAME = "cables.csv"

_START = re.compile(
    r'^"(\d+)","(\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2})","([^"]*)","([^"]*)","([^"]*)",'
)

# The dump has no header row; these are the eight positional columns, in order.
COLUMNS = (
    "id",
    "date",
    "reference",
    "origin",
    "classification",
    "refs",
    "header",
    "body",
)

PROJECTION = Projection(
    source="WikiLeaks Cablegate cables.csv (~251k cables, 1966-2010)",
    source_fields=COLUMNS,
    kept={
        "id": "doc_id",
        "date": "date (ISO-8601; the original stays in meta.date_raw)",
        "reference": "title, and meta.reference",
        "origin": "meta.origin",
        "classification": "meta.classification",
        "refs": "meta.refs",
        "header": "meta.header",
        "body": "text",
    },
    note=(
        "No cable is dropped and no field is discarded. The routing header (VZCZ.../FM "
        "AMEMBASSY/TO SECSTATE) is kept apart from the body so indexing can ignore it "
        "without having to re-detect it."
    ),
)


def iter_records(csv_path: Path) -> Iterator[tuple[re.Match[str], str]]:
    """Yield ``(start_match, raw_record_text)`` for every cable in the dump.

    Records are delimited by the quoted record-start pattern; everything up to the next
    start line belongs to the current record, multi-line body included.
    """
    match: re.Match[str] | None = None
    buf: list[str] = []
    with open(csv_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            hit = _START.match(line)
            if hit:
                if match is not None:
                    yield match, "".join(buf)
                match, buf = hit, [line]
            elif match is not None:
                buf.append(line)
    if match is not None:
        yield match, "".join(buf)


def parse_cable(match: re.Match[str], text: str) -> dict[str, str]:
    """Turn one raw record into the eight source fields.

    The five metadata fields come from the record-start match; ``refs``, ``header`` and
    ``body`` come from re-parsing the isolated record with ``csv.reader``, which is safe
    because the record is self-contained. A record too malformed even in isolation yields
    empty content rather than raising, so one bad cable costs one cable.
    """
    cid, date, reference, origin, classification = match.groups()
    refs = header = body = ""
    try:
        for row in csv.reader(io.StringIO(text), escapechar="\\"):
            if len(row) >= 8:
                refs, header, body = row[5], row[6], row[7]
                break
    except csv.Error:
        pass
    return {
        "id": cid,
        "date": date,
        "reference": reference,
        "origin": origin,
        "classification": classification,
        "refs": refs,
        "header": header,
        "body": body,
    }


def iso_date(raw: str) -> str | None:
    """Return the ISO-8601 form of a cable's ``M/D/YYYY H:MM`` timestamp, or None.

    Cablegate timestamps are US-ordered (month first). A date that will not parse returns
    None rather than a guess; the original is kept in ``meta.date_raw`` either way.
    """
    try:
        return datetime.strptime(raw, "%m/%d/%Y %H:%M").isoformat()
    except ValueError:
        return None


def to_document(fields: dict[str, str]) -> Document:
    """Turn the eight source fields into a :class:`~osint_benchmark.schema.Document`."""
    return Document(
        doc_id=fields["id"],
        source="cablegate",
        text=fields["body"],
        date=iso_date(fields["date"]),
        lang="en",
        title=fields["reference"],
        meta={
            "reference": fields["reference"],
            "origin": fields["origin"],
            "classification": fields["classification"],
            "refs": fields["refs"],
            "header": fields["header"],
            "date_raw": fields["date"],
        },
    )


def parse(raw_dir: Path) -> Iterator[Document]:
    """Yield every cable in the dump as a document."""
    for match, text in iter_records(raw_dir / "cablegate" / FILENAME):
        yield to_document(parse_cable(match, text))


SOURCE = Source(name="cablegate", kind="private", parse=parse, projection=PROJECTION)
