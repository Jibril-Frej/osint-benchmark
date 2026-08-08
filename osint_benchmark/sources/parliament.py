"""A public leg: the Swiss parliamentary record, from the Curia Vista OData service.

What officials said and did in public, so a cable's account of their private position can
be adjudicated against it. Anonymous, and covers business items from 1978, which overlaps
the cables.

The only source that fetches itself: it is a paged API, not a file, so there is nothing to
download and nothing to checksum. Two things about that paging, both learned the expensive
way in the previous project and both preserved here:

* **Keyset paging, not** ``$skip``. The service answers HTTP 500 once the offset gets deep
  — observed at ``$skip=6000`` on Business — so each page asks for ``ID gt <last seen>``.
* **A short page is not the last page.** The server returns short pages mid-stream.
  Termination is an *empty* page; treating a short one as the end truncates the fetch
  silently.

Six entity sets land in one output, told apart by an ``entity`` field, because they are one
record — the parliamentary register — split across endpoints rather than six corpora.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from osint_benchmark.artifacts import canonical, read_jsonl
from osint_benchmark.sources.base import Projection, Source

ODATA = "https://ws.parlament.ch/odata.svc"
PAGE = 1000
GERMAN = "Language eq 'DE'"
# The cables end in 2010; parliamentary items after that cannot corroborate them.
BUSINESS_END = "2011-01-01"

ENTITIES = (
    ("Person", GERMAN),
    ("MemberCouncil", GERMAN),
    ("PersonInterest", GERMAN),
    ("PersonOccupation", GERMAN),
    ("Party", GERMAN),
    ("Business", f"{GERMAN} and SubmissionDate lt datetime'{BUSINESS_END}'"),
)

PROJECTION = Projection(
    source=f"Curia Vista OData ({ODATA}): {', '.join(name for name, _ in ENTITIES)}",
    source_fields=tuple(name for name, _ in ENTITIES) + ("__metadata", "__deferred"),
    kept={name: f"records with entity={name!r}" for name, _ in ENTITIES},
    dropped={
        "__metadata": "OData envelope: URIs and type names for the row itself",
        "__deferred": "OData navigation links to related entity sets, not data",
    },
    note=(
        "Every field of every entity is kept as returned, minus the OData envelope. "
        "German rows only, and Business is limited to items submitted before "
        f"{BUSINESS_END}, since later items cannot corroborate cables ending in 2010. "
        "doc_id is 'entity:ID', because IDs are only unique within an entity set."
    ),
)


def _get(url: str, tries: int = 4) -> bytes:
    """GET a URL with linear back-off, returning the raw body."""
    last: Exception | None = None
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
                return response.read()
        except Exception as exc:  # noqa: BLE001 - retry on any transport error
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET failed after {tries} tries: {url}") from last


def strip_odata(row: dict) -> dict:
    """Drop the OData envelope and turn ``/Date(ms)/`` stamps into ISO dates."""
    out = {}
    for key, value in row.items():
        if key == "__metadata" or (isinstance(value, dict) and "__deferred" in value):
            continue
        if isinstance(value, str) and value.startswith("/Date(") and value.endswith(")/"):
            millis = value[6:-2].split("+")[0]
            value = time.strftime("%Y-%m-%d", time.gmtime(int(millis) / 1000))
        out[key] = value
    return out


def _page(entity: str, where: str, last_id: int | None, skip: int) -> list[dict]:
    """Fetch one page, by keyset when there is an integer cursor and by offset otherwise."""
    query = where if last_id is None else f"{where} and ID gt {last_id}"
    params = {"$filter": query, "$top": PAGE, "$format": "json", "$orderby": "ID asc"}
    if last_id is None and skip:
        params["$skip"] = skip
    body = json.loads(_get(f"{ODATA}/{entity}?{urllib.parse.urlencode(params)}"))["d"]
    return body["results"] if isinstance(body, dict) else body


def page_entity(entity: str, where: str, out_path: Path) -> int:
    """Page one entity set into JSONL, resuming from what is already there.

    Two of the six entity sets key on a GUID string rather than an integer —
    ``PersonInterest`` and ``PersonOccupation`` — so there is no ordered cursor to compare
    with ``ID gt``. Those page by ``$skip`` instead, which is safe at their size; only
    ``Business``, at 25k rows, goes deep enough to hit the HTTP 500 that made keyset
    paging necessary in the first place.

    Rows are written before the cursor is advanced, so a page can never be fetched and
    then dropped. Requiring an integer cursor to continue is what silently discarded both
    GUID-keyed sets: they returned rows the loop then refused to write.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen, last_id = 0, None
    if out_path.exists():
        rows = list(read_jsonl(out_path))
        ints = [row["ID"] for row in rows if isinstance(row.get("ID"), int)]
        seen, last_id = len(rows), max(ints, default=None)

    with out_path.open("a" if seen else "w", encoding="utf-8") as handle:
        while True:
            rows = _page(entity, where, last_id, seen)
            if not rows:
                break
            for row in rows:
                handle.write(canonical(strip_odata(row)) + "\n")
            seen += len(rows)
            ints = [row["ID"] for row in rows if isinstance(row.get("ID"), int)]
            if ints and last_id is not None and max(ints) <= last_id:
                break  # the server stopped advancing; stop rather than loop forever
            last_id = max(ints) if ints else None
    return seen


def acquire(raw_dir: Path) -> list[Path]:
    """Page every entity set into ``<raw>/parliament/<entity>.jsonl``."""
    paths = []
    for entity, where in ENTITIES:
        out_path = raw_dir / f"{entity}.jsonl"
        count = page_entity(entity, where, out_path)
        print(f"  parliament: {entity} {count} records")
        paths.append(out_path)
    return paths


# The fields carrying prose, per entity set. Everything else is an id, a date, a code or
# an OData envelope, and a linker shown those finds nothing -- which is exactly what
# happened: 2,000 items linked to 0 entities because no record had a `text` field at all.
TEXT_FIELDS = {
    # The prose is here, and it is substantial: the previous project's corpus averages
    # about 5 KB per item across these fields. Title and Description alone are ~100
    # characters, which is what made a truncated run look like a register of names.
    "Business": (
        "Title",
        "Description",
        "InitialSituation",
        "SubmittedText",
        "ReasonText",
        "MotionText",
        "FederalCouncilResponseText",
        "FederalCouncilProposalText",
        "DraftText",
        "DocumentationText",
        "Proceedings",
        "SubmittedBy",
    ),
    "Person": ("FirstName", "LastName", "OfficialName", "PersonNumber"),
    "MemberCouncil": ("FirstName", "LastName", "CantonName", "PartyName", "CouncilName"),
    "PersonInterest": ("Name", "Function", "Description"),
    "PersonOccupation": ("Name", "Description"),
    "Party": ("PartyName", "PartyAbbreviation"),
}

# A field whose value is longer than this is prose even if it is not in the list above.
PROSE_CHARS = 40


def document_text(entity: str, row: dict) -> str:
    """Return the text of one record: the prose fields, in order, one per line.

    Curia Vista is a register rather than a document store, so an item's "text" is its
    title and description -- around a hundred characters. That is thin, and it is what the
    API returns at this endpoint: the full submitted texts live in entity sets this source
    does not fetch. Named entities do appear in titles, which is what makes it linkable at
    all.
    """
    named = TEXT_FIELDS.get(entity, ())
    parts = [str(row[field]).strip() for field in named if isinstance(row.get(field), str)]
    # Anything long that the list did not anticipate: better a field too many than an
    # entity set silently contributing nothing.
    parts += [
        value.strip()
        for key, value in row.items()
        if key not in named
        and not key.startswith("__")
        and isinstance(value, str)
        and len(value) > PROSE_CHARS
    ]
    return "\n".join(part for part in dict.fromkeys(parts) if part)


def parse(raw_dir: Path) -> Iterator[dict]:
    """Yield every fetched row, tagged with the entity set it came from."""
    for entity, _ in ENTITIES:
        for row in read_jsonl(raw_dir / "parliament" / f"{entity}.jsonl"):
            yield {
                "doc_id": f"{entity}:{row.get('ID')}",
                "entity": entity,
                "text": document_text(entity, row),
                **row,
            }


SOURCE = Source(
    name="parliament", kind="public", parse=parse, projection=PROJECTION, acquire=acquire
)
