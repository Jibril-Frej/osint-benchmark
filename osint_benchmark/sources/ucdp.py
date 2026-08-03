"""A public leg: UCDP Georeferenced Event Dataset, expert-coded conflict events.

The public record a cable's account of an incident can be checked against — who fought
whom, where, when, and how many died. Distributed as one zipped CSV per release, and a
release is immutable, so unlike the sanctions list it pins exactly.

Every column is kept. The previous project's event store kept seven join keys and lost
casualties and party names, which was the right projection for the matcher it was written
for and the wrong one for the question type that came later. Deciding what a question
needs is not this step's job.
"""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Iterator
from pathlib import Path

from osint_benchmark.sources.base import Projection, Source

FILENAME = "ged251-csv.zip"
MEMBER = "GEDEvent_v25_1.csv"

# The release's 49 columns, listed rather than read from the header on purpose: a
# projection that discovers its own source fields can never fail the check that is the
# point of recording them. A new column in a later release should stop the build.
COLUMNS = (
    "id", "relid", "year", "active_year", "code_status", "type_of_violence",
    "conflict_dset_id", "conflict_new_id", "conflict_name", "dyad_dset_id", "dyad_new_id",
    "dyad_name", "side_a_dset_id", "side_a_new_id", "side_a", "side_b_dset_id",
    "side_b_new_id", "side_b", "number_of_sources", "source_article", "source_office",
    "source_date", "source_headline", "source_original", "where_prec", "where_coordinates",
    "where_description", "adm_1", "adm_2", "latitude", "longitude", "geom_wkt",
    "priogrid_gid", "country", "country_id", "region", "event_clarity", "date_prec",
    "date_start", "date_end", "deaths_a", "deaths_b", "deaths_civilians", "deaths_unknown",
    "best", "high", "low", "gwnoa", "gwnob",
)  # fmt: skip

PROJECTION = Projection(
    source="UCDP GED v25.1 (ged251-csv.zip, GEDEvent_v25_1.csv)",
    source_fields=COLUMNS,
    kept=dict.fromkeys(COLUMNS, "kept verbatim"),
    kind="corpus",
    note="Every column is kept; doc_id is added, copied from id.",
)


def parse(raw_dir: Path) -> Iterator[dict]:
    """Yield every event in the release, one record per row.

    The CSV is read straight out of the zip — 250 MB unzipped against 29 MB zipped, and
    nothing else needs the expanded copy.
    """
    with zipfile.ZipFile(raw_dir / "ucdp" / FILENAME) as archive, archive.open(MEMBER) as member:
        reader = csv.DictReader(io.TextIOWrapper(member, encoding="utf-8"))
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(
                f"{MEMBER} columns differ from the recorded release: "
                f"{sorted(set(reader.fieldnames or ()) ^ set(COLUMNS))}"
            )
        for row in reader:
            yield {"doc_id": row["id"], **row}


SOURCE = Source(name="ucdp", kind="public", parse=parse, projection=PROJECTION)
