"""Unit tests for the GDELT event parser."""

from __future__ import annotations

import io
import zipfile

from osint_benchmark.sources.gdelt import COLUMNS, parse


def _row(event_id: str, **overrides) -> list[str]:
    """Build one 57-field GDELT row."""
    row = [f"{name}-v" for name in COLUMNS]
    row[COLUMNS.index("event_id")] = event_id
    for name, value in overrides.items():
        row[COLUMNS.index(name)] = value
    return row


def _write_zips(tmp_path, archives: dict[str, list[list[str]]]):
    """Write synthetic GDELT archives and return the raw root."""
    raw = tmp_path / "raw"
    (raw / "gdelt").mkdir(parents=True)
    for name, rows in archives.items():
        buffer = io.StringIO()
        buffer.writelines("\t".join(row) + "\n" for row in rows)
        with zipfile.ZipFile(raw / "gdelt" / f"{name}.zip", "w") as archive:
            archive.writestr(f"{name}.csv", buffer.getvalue())
    return raw


class TestParse:
    """Events are read positionally, since the files have no header row."""

    def test_all_57_columns_are_named_and_kept(self, tmp_path):
        """No projection, and the event code — the column the old store lost — survives."""
        raw = _write_zips(tmp_path, {"200601": [_row("1", event_code="190", date="20060115")]})

        records = list(parse(raw))

        assert len(records) == 1
        assert set(records[0]) == set(COLUMNS) | {"doc_id"}
        assert records[0]["doc_id"] == "1"
        assert records[0]["event_code"] == "190"
        assert records[0]["date"] == "20060115"

    def test_archives_are_read_oldest_first(self, tmp_path):
        """Sorted by filename, which for yyyy and yyyymm names is chronological order."""
        raw = _write_zips(
            tmp_path,
            {"2003": [_row("a")], "200601": [_row("b")], "201012": [_row("c")]},
        )

        assert [r["doc_id"] for r in parse(raw)] == ["a", "b", "c"]

    def test_a_truncated_row_is_skipped_not_misaligned(self, tmp_path):
        """A short line must not shift every later field by one column."""
        raw = _write_zips(tmp_path, {"2003": [_row("1"), ["only", "three", "fields"], _row("2")]})

        records = list(parse(raw))

        assert [r["doc_id"] for r in records] == ["1", "2"]
        assert records[1]["date_added"] == "date_added-v"
