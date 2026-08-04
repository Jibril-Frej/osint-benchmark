"""Unit tests for the UCDP event parser."""

from __future__ import annotations

import csv
import io
import zipfile

import pytest

from osint_benchmark.sources.ucdp import COLUMNS, FILENAME, MEMBER, parse


def _write_zip(tmp_path, header, rows):
    """Write a synthetic GED zip where the parser expects it, and return the raw root."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    raw = tmp_path / "raw"
    (raw / "ucdp").mkdir(parents=True)
    with zipfile.ZipFile(raw / "ucdp" / FILENAME, "w") as archive:
        archive.writestr(MEMBER, buffer.getvalue())
    return raw


class TestParse:
    """Events are read from inside the zip, whole."""

    def test_every_column_survives_and_doc_id_comes_from_id(self, tmp_path):
        """No projection: all 49 columns are kept, plus a doc_id copied from id."""
        row = [f"v{i}" for i in range(len(COLUMNS))]
        row[COLUMNS.index("id")] = "12345"
        raw = _write_zip(tmp_path, list(COLUMNS), [row])

        records = list(parse(raw))

        assert len(records) == 1
        assert records[0]["doc_id"] == "12345"
        # date is added alongside, copied from date_start, so the pairing step can compute
        # intervals -- without it every pair came out undated.
        assert set(records[0]) == set(COLUMNS) | {"doc_id", "date"}

    def test_a_changed_column_set_stops_the_build(self, tmp_path):
        """A later release adding or renaming a column must not pass silently.

        The provenance record names all 49 columns; if the release no longer matches it,
        the sidecar would claim an accounting that is no longer true.
        """
        header = [*COLUMNS, "new_column_in_v26"]
        raw = _write_zip(tmp_path, header, [[f"v{i}" for i in range(len(header))]])

        with pytest.raises(ValueError, match="new_column_in_v26"):
            list(parse(raw))
