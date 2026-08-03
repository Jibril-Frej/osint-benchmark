"""A public leg: the SECO sanctions list (SESAM), who Switzerland has listed and why.

Ported from the previous project, including the correction that mattered: the first
version kept only ``{target_id, names, type, dobs}`` and silently discarded the fields
that make a listing answerable — which programme listed the target, the justification, the
addresses, and the dates they were added or removed. Those are exactly what a profile or
trajectory question asks about.

A target names only a sanctions-set id. The programme sits one level up: each
``<sanctions-program>`` *contains* the ``<sanctions-set>`` elements it created, so the set
id is the join from a listed person to the ordinance that listed them.

**This source cannot be pinned.** The export is a live snapshot, not a versioned release —
it was 39,899,536 bytes on 2026-07-27 and 39,920,777 bytes three days later. So
``pins/sources.toml`` carries no checksum for it, and the list's own ``date`` attribute is
recorded on every record instead. Freezing a release means archiving the XML that was
fetched, not pinning a URL.
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from collections.abc import Iterator
from pathlib import Path

from osint_benchmark.sources.base import Projection, Source

FILENAME = "sesam.xml"

PROJECTION = Projection(
    source="SECO SESAM whole-list XML export (sesam.search.admin.ch)",
    source_fields=(
        "swiss-sanctions-list",
        "sanctions-program",
        "program-name",
        "program-key",
        "origin",
        "sanctions-set",
        "sanctions-set-id",
        "target",
        "individual",
        "entity",
        "object",
        "identity",
        "name",
        "name-part",
        "value",
        "spelling-variant",
        "day-month-year",
        "justification",
        "address",
        "address-details",
        "zip-code",
        "c-o",
        "p-o-box",
        "country",
        "modification",
        "added",
        "removed",
        "other-information",
        "relation",
        "nationality",
        "place-of-birth",
        "identification-document",
        "number",
        "issuer",
        "date-of-issue",
        "place-of-issue",
        "expiry-date",
        "foreign-identifier",
        "remark",
        "place",
        "location",
        "location-variant",
        "area",
        "area-variant",
    ),  # fmt: skip
    kept={
        "target": "doc_id (top-level targets only)",
        "individual": "type",
        "entity": "type",
        "object": "type",
        "name": "names",
        "name-part": "names",
        "value": "names",
        "spelling-variant": "names",
        "day-month-year": "dobs",
        "sanctions-set-id": "sanctions_set_id",
        "program-name": "programme",
        "program-key": "programme_key",
        "origin": "origin",
        "sanctions-set": "measures",
        "justification": "justification",
        "address": "addresses",
        "address-details": "addresses",
        "zip-code": "addresses",
        "c-o": "addresses",
        "p-o-box": "addresses",
        "country": "addresses",
        "modification": "modifications (all attributes) + the derived added / removed dates",
        "added": "added (its embedded target snapshot is not emitted)",
        "removed": "removed (its embedded target snapshot is not emitted)",
        "other-information": "other_information",
        "swiss-sanctions-list": "list_date (the export's own date attribute)",
    },
    dropped={
        "sanctions-program": "container element; its fields are resolved onto each target",
        "identity": "container element only",
        "relation": "links identities of the same target, already merged into names",
        "nationality": "not used by any question type",
        "place-of-birth": "not used by any question type",
        "identification-document": "passport and id numbers, deliberately not retained",
        "number": "part of identification-document",
        "issuer": "part of identification-document",
        "date-of-issue": "part of identification-document",
        "place-of-issue": "part of identification-document",
        "expiry-date": "part of identification-document",
        "foreign-identifier": "external list cross-reference, unused",
        "remark": "free-text editorial note, unused",
        "place": "container for address parts, folded into addresses",
        "location": "geographic scope of a programme, not of a target",
        "location-variant": "alternate spelling of a programme location",
        "area": "geographic scope of a programme, not of a target",
        "area-variant": "alternate spelling of a programme area",
    },
    note=(
        "Fields are parsed out of nested XML, so the kept names are ours rather than "
        "source tag names. The export is a live snapshot with no fixed version, so every "
        "record carries the list_date it came from."
    ),
)


def _tag(node) -> str:
    """Return an element's local name, without its namespace."""
    return node.tag.rsplit("}", 1)[-1]


def walk(node) -> Iterator:
    """Yield a node and its descendants, stopping at any nested ``<target>``.

    A listing's ``<modification>`` history embeds a full copy of the target as it stood at
    each change, inside ``<added>``/``<removed>``. Those copies are not separate listings
    and their content is not the current one: ``element.iter()`` walks into them, which
    collects every historical name onto the live record and counts each snapshot as its
    own target. In this export that is 8,470 of 17,074 target elements — the previous
    project's 17,074-row sanctions corpus is that number, and it is about half history.
    """
    yield node
    for child in node:
        if _tag(child) != "target":
            yield from walk(child)


def _joined_values(node) -> str:
    """Return a name's parts joined, since a name is split across <value> elements."""
    return " ".join(
        (part.text or "").strip() for part in walk(node) if _tag(part) == "value" and part.text
    ).strip()


def programmes_by_set(root) -> dict[str, dict]:
    """Map each sanctions-set id to the programme that created it."""
    found: dict[str, dict] = {}
    for programme in root.iter():
        if _tag(programme) != "sanctions-program":
            continue
        info = {"programme": "", "programme_key": "", "origin": "", "measures": ""}
        for child in programme.iter():
            tag, lang = _tag(child), child.get("lang")
            if tag == "program-name" and lang == "eng" and child.text:
                info["programme"] = child.text.strip()
            elif tag == "program-key" and lang == "eng" and child.text:
                info["programme_key"] = child.text.strip()
            elif tag == "origin" and child.text:
                info["origin"] = child.text.strip()
        for child in programme.iter():
            if _tag(child) != "sanctions-set" or not child.get("ssid"):
                continue
            entry = dict(info)
            if child.get("lang") == "eng" and child.text:
                entry["measures"] = child.text.strip()
            previous = found.get(child.get("ssid"))
            if not previous or (entry["measures"] and not previous.get("measures")):
                found[child.get("ssid")] = entry
    return found


def target_record(target, programmes: dict[str, dict], list_date: str) -> dict:
    """Turn one <target> element into a record."""
    record: dict = {
        "doc_id": target.get("ssid"),
        "list_date": list_date,
        "type": None,
        "names": [],
        "dobs": [],
        "sanctions_set_id": None,
        "programme": "",
        "programme_key": "",
        "origin": "",
        "measures": "",
        "justification": "",
        "addresses": [],
        "modifications": [],
        "added": None,
        "removed": None,
        "other_information": [],
    }
    for node in walk(target):
        tag = _tag(node)
        if tag in {"individual", "entity", "object"}:
            record["type"] = tag
        elif tag == "name":
            if full := _joined_values(node):
                record["names"].append(full)
        elif tag == "day-month-year" and node.get("year"):
            record["dobs"].append(node.get("year"))
        elif tag == "sanctions-set-id" and node.text:
            record["sanctions_set_id"] = node.text.strip()
        elif tag == "justification" and node.text:
            record["justification"] = node.text.strip()
        elif tag == "address":
            parts = [(p.text or "").strip() for p in walk(node) if p.text and p.text.strip()]
            if address := " ".join(parts).strip():
                record["addresses"].append(address)
        elif tag == "modification":
            record["modifications"].append(
                {
                    "type": node.get("modification-type"),
                    "enactment_date": node.get("enactment-date"),
                    "publication_date": node.get("publication-date"),
                    "effective_date": node.get("effective-date"),
                }
            )
        elif tag == "other-information" and node.text:
            record["other_information"].append(node.text.strip())
    # The convenience dates every consumer would otherwise re-derive. Publication is the
    # date the listing took public effect, which is the one a question can be asked about.
    for modification in record["modifications"]:
        when = modification["publication_date"] or modification["enactment_date"]
        if modification["type"] == "listed":
            record["added"] = record["added"] or when
        elif modification["type"] == "de-listed":
            record["removed"] = record["removed"] or when
    record.update(programmes.get(record["sanctions_set_id"] or "", {}))
    return record


def parse(raw_dir: Path) -> Iterator[dict]:
    """Yield one record per listed target, skipping targets with no name."""
    root = ElementTree.parse(raw_dir / "sanctions" / FILENAME).getroot()
    programmes = programmes_by_set(root)
    list_date = root.get("date", "")
    # Direct children only: a nested target is a historical snapshot, not a listing.
    for target in root:
        if _tag(target) != "target":
            continue
        record = target_record(target, programmes, list_date)
        if record["names"]:
            yield record


SOURCE = Source(name="sanctions", kind="public", parse=parse, projection=PROJECTION)
