"""Unit tests for the Dodis source.

Two files make one corpus: OCR over the scans supplies the text, the open-data dump
supplies dates and the archive's own curated entity links. Neither is useful alone, and
the join is where a document can go missing without anyone noticing.
"""

from __future__ import annotations

import pytest

from osint_benchmark.sources import dodis

VOCAB = "http://dodis.ch/schema/vocab/"
DOC = "http://dodis.ch/document/42"


def triple(subject: str, field: str, obj: str) -> str:
    """Return one N-Triples line."""
    return f"<{subject}> <{VOCAB}{field}> {obj} .\n"


@pytest.fixture
def dump(tmp_path):
    """Write a dump holding one document with a person and a place."""
    lines = [
        triple(DOC, "document_summary", '"Ein kurzer Regest."@de'),
        triple(DOC, "document_title", '"Ein <b>Titel</b>"@de'),
        triple(DOC, "document_lang_code", '"de"'),
        triple(DOC, "document_doc_date", '"9.5.1947"'),
        triple(DOC, "document_classification", '"geheim"'),
        triple("http://dodis.ch/dhp/1", "document_has_person_document_id", f"<{DOC}>"),
        triple("http://dodis.ch/dhp/1", "document_has_person_person_id", "<http://dodis.ch/p/7>"),
        triple("http://dodis.ch/pf/1", "person_fallback_person_id", "<http://dodis.ch/p/7>"),
        triple("http://dodis.ch/pf/1", "person_fallback_first_name", '"Walter"'),
        triple("http://dodis.ch/pf/1", "person_fallback_last_name", '"Stampfli"'),
        triple("http://dodis.ch/dhpl/1", "document_has_place_document_id", f"<{DOC}>"),
        triple("http://dodis.ch/dhpl/1", "document_has_place_place_id", "<http://dodis.ch/pl/3>"),
        triple("http://dodis.ch/pl/3", "place_wikidata_id", '"Q39"'),
        triple("http://dodis.ch/pl/3", "place_fallback_place_id", "<http://dodis.ch/pl/3>"),
        triple("http://dodis.ch/pl/3", "place_fallback_name", '"Schweiz"'),
    ]
    path = tmp_path / "dodis.nt"
    path.write_text("".join(lines), encoding="utf-8")
    return path


class TestReadMetadata:
    """One streaming pass over a relational export."""

    def test_a_document_carries_its_curated_entities(self, dump):
        """The reason this corpus needs no entity linker."""
        meta = dodis.read_metadata(dump)["42"]

        assert meta["persons"] == ["Walter Stampfli"]
        assert meta["places"] == [{"qid": "Q39", "name": "Schweiz"}]

    def test_html_is_stripped_from_the_title(self, dump):
        """The dump carries markup; a question should not quote a tag."""
        assert dodis.read_metadata(dump)["42"]["title"] == "Ein Titel"

    def test_a_language_outside_the_request_is_dropped(self, dump):
        """German and French are ~98% of Dodis; the rest are not worth a second parser."""
        assert dodis.read_metadata(dump, langs=("it",)) == {}

    def test_a_document_without_a_summary_is_not_a_document(self, tmp_path):
        """A row with no regest is a stub record in the export."""
        path = tmp_path / "d.nt"
        path.write_text(triple(DOC, "document_lang_code", '"de"'), encoding="utf-8")

        assert dodis.read_metadata(path) == {}

    def test_a_summary_containing_a_newline_stays_one_record(self, tmp_path):
        r"""Why N-Triples rather than the MySQL dump.

        The format escapes newlines inside a literal, so one document is one physical
        line. Reading a format that does not is what truncated the Cablegate CSV.
        """
        path = tmp_path / "d.nt"
        path.write_text(
            triple(DOC, "document_summary", '"erste Zeile\\nzweite Zeile"@de')
            + triple(DOC, "document_lang_code", '"de"'),
            encoding="utf-8",
        )

        assert dodis.read_metadata(path)["42"]["summary"] == "erste Zeile\nzweite Zeile"


class TestDates:
    """Dodis writes dates the Swiss way, and not always completely."""

    def test_a_full_date_becomes_iso(self):
        """Dodis writes day first."""
        assert dodis.iso_date("9.5.1947") == "1947-05-09"

    def test_a_year_alone_is_kept_as_the_first_of_january(self):
        """Better a coarse date than none: the pairing step compares intervals."""
        assert dodis.iso_date("1947") == "1947-01-01"

    def test_an_unparseable_date_is_absent_rather_than_guessed(self):
        """A guessed date would silently decide whether a pair is contemporaneous."""
        assert dodis.iso_date("um 1947") is None
        assert dodis.iso_date(None) is None


class TestParse:
    """The join between OCR text and metadata."""

    def test_a_document_needs_both_halves(self, dump, tmp_path, monkeypatch):
        """A scan with no metadata cannot be dated; a record with no text is a failed OCR."""
        ocr = tmp_path / "ocr"
        ocr.mkdir()
        (ocr / "dodis-42.txt").write_text("Der Bundesrat hat beschlossen.", encoding="utf-8")
        (ocr / "dodis-999.txt").write_text("orphan scan, no metadata", encoding="utf-8")
        (ocr / "dodis-43.txt").write_text("   ", encoding="utf-8")
        monkeypatch.setenv("OSINT_DODIS_OCR", str(ocr))
        monkeypatch.setenv("OSINT_DODIS_NT", str(dump))

        records = list(dodis.parse(tmp_path))

        assert [r["doc_id"] for r in records] == ["42"]
        assert records[0]["text"] == "Der Bundesrat hat beschlossen."
        assert records[0]["date"] == "1947-05-09"

    def test_a_missing_input_says_which_one(self, tmp_path, monkeypatch):
        """Yielding nothing would look like an empty corpus rather than an absent one."""
        monkeypatch.setenv("OSINT_DODIS_OCR", str(tmp_path / "nowhere"))
        monkeypatch.setenv("OSINT_DODIS_NT", str(tmp_path / "nothing.nt"))

        with pytest.raises(Exception, match="OCR"):
            list(dodis.parse(tmp_path))

    def test_the_regest_is_kept_beside_the_text_not_instead_of_it(
        self, dump, tmp_path, monkeypatch
    ):
        """The summary is a description of a document; the OCR is the document."""
        ocr = tmp_path / "ocr"
        ocr.mkdir()
        (ocr / "dodis-42.txt").write_text("the full scanned text", encoding="utf-8")
        monkeypatch.setenv("OSINT_DODIS_OCR", str(ocr))
        monkeypatch.setenv("OSINT_DODIS_NT", str(dump))

        record = next(iter(dodis.parse(tmp_path)))

        assert record["text"] == "the full scanned text"
        assert record["meta"]["summary"] == "Ein kurzer Regest."


class TestCuratedLinking:
    """A corpus catalogued by archivists needs no linker."""

    @staticmethod
    def _record():
        """Return one Dodis record with a curated place and person."""
        return {
            "doc_id": "42",
            "meta": {
                "places": [{"qid": "Q39", "name": "Schweiz"}, {"qid": None, "name": "unknown"}],
                "persons": ["Walter Stampfli"],
            },
        }

    def test_a_place_with_a_qid_is_used_as_it_is(self):
        """No model, no inference: the archive already resolved it."""
        from osint_benchmark.link import curated

        rows = list(
            curated.link_records([self._record()], "dodis", "private", {"Q39"}, lambda n, k: {})
        )

        assert rows[0]["doc_id"] == "dodis:42"
        assert [e["qid"] for e in rows[0]["entities"]] == ["Q39"]

    def test_a_place_outside_the_public_entity_set_is_dropped(self):
        """It cannot bridge to a public corpus scoped to entities holding an article."""
        from osint_benchmark.link import curated

        rows = list(
            curated.link_records([self._record()], "dodis", "private", set(), lambda n, k: {})
        )

        assert rows[0]["entities"] == []

    def test_a_person_is_reconciled_by_name(self):
        """Curated links give people as names only, so they go through the same lookup."""
        from osint_benchmark.link import curated

        rows = list(
            curated.link_records(
                [self._record()],
                "dodis",
                "private",
                {"Q39", "Q123"},
                lambda names, kinds: {"Walter Stampfli": ["Q123"]},
            )
        )

        assert sorted(e["qid"] for e in rows[0]["entities"]) == ["Q123", "Q39"]

    def test_an_ambiguous_curated_name_is_still_ambiguous(self):
        """However carefully it was catalogued, two candidates are not a resolution."""
        from osint_benchmark.link import curated

        rows = list(
            curated.link_records(
                [self._record()],
                "dodis",
                "private",
                {"Q1", "Q2"},
                lambda names, kinds: {"Walter Stampfli": ["Q1", "Q2"]},
            )
        )

        assert rows[0]["entities"] == []
