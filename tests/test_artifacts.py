"""Unit tests for the artifact writer and its provenance sidecar.

The sidecar exists to make a silent projection impossible: in the previous project the
event store lost casualties and party names, and the sanctions parse lost the programme
and the justification, with no way downstream to notice. These tests pin the checks that
make that loss loud.
"""

from __future__ import annotations

import json

import pytest

from osint_benchmark.artifacts import (
    Provenance,
    check_provenance,
    record_hash,
    sidecar_path,
    write_provenance,
    write_records,
)
from osint_benchmark.schema import Document

SOUND = Provenance(
    source="a test source",
    source_fields=("a", "b"),
    kept={"a": "text"},
    dropped={"b": "not used by any question type"},
)


def _written(path):
    """Return the provenance record written beside a file."""
    return json.loads(sidecar_path(path).read_text(encoding="utf-8"))


class TestCheckProvenance:
    """check_provenance catches the ways a projection record can lie."""

    def test_a_sound_record_has_no_problems(self):
        """Every source field is either kept or explained."""
        path = "/tmp/out.jsonl"
        data = {
            "output": path,
            "kind": "corpus",
            "source": "s",
            "source_fields": ["a", "b"],
            "kept": {"a": "text"},
            "dropped": {"b": "not needed"},
        }
        assert check_provenance(data) == []

    def test_a_field_neither_kept_nor_dropped_is_flagged(self):
        """The silent-loss case: a source field nobody accounted for."""
        data = {
            "output": "o",
            "kind": "corpus",
            "source": "s",
            "source_fields": ["a", "b"],
            "kept": {"a": "text"},
            "dropped": {},
        }
        problems = check_provenance(data)
        assert any("neither kept nor explained" in p and "b" in p for p in problems)

    def test_dropping_without_a_reason_is_flagged(self):
        """A blank reason is not an explanation."""
        data = {
            "output": "o",
            "kind": "corpus",
            "source": "s",
            "source_fields": ["a"],
            "kept": {},
            "dropped": {"a": "   "},
        }
        assert any("no reason given" in p for p in check_provenance(data))

    def test_kept_keys_must_be_source_names(self):
        """Mapping ours->theirs instead of theirs->ours is caught, not silently accepted."""
        data = {
            "output": "o",
            "kind": "corpus",
            "source": "s",
            "source_fields": ["a"],
            "kept": {"text": "a"},
            "dropped": {"a": "x"},
        }
        assert any("keys must be source names" in p for p in check_provenance(data))

    def test_more_rows_out_than_in_is_flagged(self):
        """A projection cannot produce more rows than it read."""
        data = {
            "output": "o",
            "kind": "corpus",
            "source": "s",
            "source_fields": ["a"],
            "kept": {"a": "text"},
            "rows_in": 2,
            "rows_out": 3,
        }
        assert any("exceeds rows_in" in p for p in check_provenance(data))


class TestWriteProvenance:
    """The writer refuses to record a guarantee it cannot make."""

    def test_unsound_provenance_raises_rather_than_writing(self, tmp_path):
        """A bad sidecar is worse than none, because it reads as a guarantee."""
        bad = Provenance(source="s", source_fields=("a", "b"), kept={"a": "text"})
        with pytest.raises(ValueError, match="neither kept nor explained"):
            write_provenance(tmp_path / "out.jsonl", bad)
        assert not sidecar_path(tmp_path / "out.jsonl").exists()


class TestWriteRecords:
    """Documents cannot be written without their provenance."""

    def test_writes_jsonl_and_sidecar_together(self, tmp_path):
        """One JSON object per line, and a sidecar recording the row count."""
        output = tmp_path / "docs" / "test.jsonl"
        records = [
            Document(doc_id="1", source="test", text="one").to_json(),
            Document(doc_id="2", source="test", text="two").to_json(),
        ]

        count = write_records(output, records, SOUND, rows_in=2)

        assert count == 2
        assert len(output.read_text(encoding="utf-8").splitlines()) == 2
        assert _written(output)["rows_out"] == 2
        assert _written(output)["dropped"]["b"]

    def test_an_unsound_projection_stops_the_write(self, tmp_path):
        """The output may exist, but no sidecar means the run is not to be trusted."""
        output = tmp_path / "test.jsonl"
        bad = Provenance(source="s", source_fields=("a", "b"), kept={"a": "text"})

        with pytest.raises(ValueError):
            write_records(output, [Document(doc_id="1", source="t", text="x").to_json()], bad)
        assert not sidecar_path(output).exists()


class TestRecordHash:
    """Hashing covers the whole record, not just its text."""

    def test_key_order_does_not_change_the_hash(self):
        """The canonical form sorts keys, so field order is not a difference."""
        assert record_hash({"a": 1, "b": 2}) == record_hash({"b": 2, "a": 1})

    def test_a_metadata_change_changes_the_hash(self):
        """A changed classification is a changed document, not an invisible edit."""
        base = {"doc_id": "1", "text": "same", "meta": {"classification": "CONFIDENTIAL"}}
        other = {"doc_id": "1", "text": "same", "meta": {"classification": "SECRET"}}
        assert record_hash(base) != record_hash(other)
