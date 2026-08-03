"""A public leg: GDELT 1.0 events over the Cablegate window.

Machine-coded world events from news wire text — the other public record a cable's account
of an incident can be checked against, alongside UCDP. 63 files: yearly for 2003-2005 and
monthly through 2010-12, which is where the cables stop.

Two things to know before building a question on it.

**Actors are coded by role, not identity.** The commonest actor names are UNITED STATES,
POLICE, GOVERNMENT, PRESIDENT, SCHOOL. GDELT 1.0 says *a president* did something, not
*which* president, so it makes poor bridges to diplomatic reporting; the previous
project's attempt to reconcile actor labels to Wikidata QIDs was rejected outright.

**The event code is the valuable column.** ``event_code`` is CAMEO's taxonomy of what
happened. The previous project's compact event store dropped it and recorded the gap:
"this index says where an event happened and who was involved, but not WHAT happened."
Every column is kept here, so that cannot recur.

The output is gzipped: 91.6M events is 78 GB of JSONL plain and 6.2 GB compressed, and
compression is what makes keeping all 57 columns affordable rather than forcing a
projection nobody would be able to justify field by field.
"""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Iterator
from pathlib import Path

from osint_benchmark.sources.base import Projection, Source

# GDELT 1.0 event files carry no header row, so the schema is positional. These are the
# 57 columns of the documented format, in order.
COLUMNS = (
    "event_id", "date", "month_year", "year", "fraction_date",
    "actor1_code", "actor1_name", "actor1_country", "actor1_known_group",
    "actor1_ethnic", "actor1_religion1", "actor1_religion2",
    "actor1_type1", "actor1_type2", "actor1_type3",
    "actor2_code", "actor2_name", "actor2_country", "actor2_known_group",
    "actor2_ethnic", "actor2_religion1", "actor2_religion2",
    "actor2_type1", "actor2_type2", "actor2_type3",
    "is_root_event", "event_code", "event_base_code", "event_root_code",
    "quad_class", "goldstein", "num_mentions", "num_sources", "num_articles", "avg_tone",
    "actor1_geo_type", "actor1_geo_name", "actor1_geo_country", "actor1_geo_adm1",
    "actor1_geo_lat", "actor1_geo_long", "actor1_geo_feature_id",
    "actor2_geo_type", "actor2_geo_name", "actor2_geo_country", "actor2_geo_adm1",
    "actor2_geo_lat", "actor2_geo_long", "actor2_geo_feature_id",
    "action_geo_type", "action_geo_name", "action_geo_country", "action_geo_adm1",
    "action_geo_lat", "action_geo_long", "action_geo_feature_id",
    "date_added",
)  # fmt: skip

PROJECTION = Projection(
    source="GDELT 1.0 event files, 2003-2010 (data.gdeltproject.org/events)",
    source_fields=COLUMNS,
    kept=dict.fromkeys(COLUMNS, "kept verbatim"),
    kind="corpus",
    note=(
        "Every column is kept; doc_id is added, copied from event_id. Column names are "
        "ours: the files carry no header, so the schema is positional. Rows with fewer "
        "than 57 fields are skipped as malformed and counted in rows_in vs rows_out."
    ),
)


def parse(raw_dir: Path) -> Iterator[dict]:
    """Yield every event across all 63 archives, oldest file first."""
    for archive_path in sorted((raw_dir / "gdelt").glob("*.zip")):
        with zipfile.ZipFile(archive_path) as archive:
            member = archive.namelist()[0]
            with archive.open(member) as handle:
                text = io.TextIOWrapper(handle, encoding="utf-8", errors="replace")
                for row in csv.reader(text, delimiter="\t"):
                    # A short row is a truncated line in the source export, not a schema
                    # change: the column count is fixed and unversioned in GDELT 1.0.
                    if len(row) < len(COLUMNS):
                        continue
                    record = dict(zip(COLUMNS, row, strict=False))
                    yield {"doc_id": record["event_id"], **record}


SOURCE = Source(name="gdelt", kind="public", parse=parse, projection=PROJECTION, compress=True)
