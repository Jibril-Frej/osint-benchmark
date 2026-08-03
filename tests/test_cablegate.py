r"""Unit tests for the Cablegate parser.

The one that matters is the backslash-escapechar regression: the dump escapes in-body
quotes and apostrophes with a backslash (``\"``, ``\'``), not by doubling, and
``csv.reader`` must be given ``escapechar="\\"`` or bodies are silently truncated at the
first quoted phrase — which cost ~68% of the corpus text in the previous project and left
a file that looked perfectly well-formed.

Fixtures are small synthetic CSVs in ``tmp_path``; the real 1.7 GB dump is never touched.
"""

from __future__ import annotations

import json

from osint_benchmark.sources.cablegate import (
    _START,
    iso_date,
    iter_records,
    parse,
    parse_cable,
    to_document,
)


def _record_line(
    cid: str,
    date: str,
    reference: str,
    origin: str,
    classification: str,
    refs: str,
    header: str,
    body: str,
) -> str:
    """Build one raw CSV record line (all eight fields double-quoted)."""
    fields = [cid, date, reference, origin, classification, refs, header, body]
    return ",".join(f'"{field}"' for field in fields) + "\n"


def _write_dump(tmp_path, text: str):
    """Write a synthetic dump where the parser expects it, and return the raw root."""
    raw = tmp_path / "raw"
    (raw / "cablegate").mkdir(parents=True)
    (raw / "cablegate" / "cables.csv").write_text(text, encoding="utf-8")
    return raw


class TestParseCableEscaping:
    """parse_cable recovers backslash-escaped quotes and apostrophes in full."""

    def test_backslash_escaped_quote_survives_in_full(self):
        r"""A body with a backslash-escaped quoted phrase is recovered whole.

        Without escapechar="\\", the \" before "Sirvan Dam" flips the quote state and
        everything after it — including the trailing paragraphs and the signature — is
        dropped.
        """
        body = (
            'The delegation visited a hydropower dam (\\"Sirvan Dam\\") in Iran. '
            "This is a second paragraph describing the visit in detail. "
            "END OF MESSAGE. AMBASSADOR SMITH"
        )
        line = _record_line(
            "123456",
            "1/2/2010 8:00",
            "10BERN1",
            "Embassy Bern",
            "CONFIDENTIAL",
            "10BERN0",
            "VZCZ ROUTING HEADER",
            body,
        )
        cable = parse_cable(_START.match(line), line)

        assert '"Sirvan Dam"' in cable["body"]
        assert "This is a second paragraph describing the visit in detail." in cable["body"]
        assert cable["body"].endswith("AMBASSADOR SMITH")
        assert "\\" not in cable["body"]

    def test_backslash_escaped_apostrophe_unescapes(self):
        """A backslash-escaped apostrophe unescapes to a plain apostrophe."""
        body = "Iran\\'s nuclear program was the main topic of discussion."
        line = _record_line(
            "654321", "3/4/2009 12:30", "09BERN2", "Embassy Bern", "UNCLASSIFIED", "", "", body
        )
        cable = parse_cable(_START.match(line), line)

        assert cable["body"] == "Iran's nuclear program was the main topic of discussion."
        assert "\\" not in cable["body"]


class TestIterRecords:
    """iter_records segments a multi-record file with multi-line bodies."""

    def test_multi_record_multiline_body_segmentation(self, tmp_path):
        """A file with two records, each with a multi-line body, yields two records."""
        record_one = (
            '"1","1/1/2010 9:00","10BERN1","Embassy Bern","CONFIDENTIAL",'
            '"","header one",'
            '"first paragraph of cable one\n'
            "second paragraph of cable one\n"
            'END CABLE ONE"\n'
        )
        record_two = (
            '"2","1/2/2010 10:00","10BERN2","Embassy Bern","UNCLASSIFIED",'
            '"","header two",'
            '"first paragraph of cable two\n'
            'END CABLE TWO"\n'
        )
        csv_path = tmp_path / "cables.csv"
        csv_path.write_text(record_one + record_two, encoding="utf-8")

        records = list(iter_records(csv_path))

        assert len(records) == 2
        first_match, first_text = records[0]
        second_match, second_text = records[1]
        assert first_match.group(1) == "1"
        assert second_match.group(1) == "2"
        # The second paragraph of record one belongs to record one, not record two.
        assert "second paragraph of cable one" in first_text
        assert "second paragraph of cable one" not in second_text
        assert "first paragraph of cable two" in second_text


class TestToDocument:
    """The eight source fields become a Document without losing any of them."""

    def test_fields_land_where_the_projection_says(self):
        """Body becomes text, reference becomes the title, the rest go to meta."""
        line = _record_line(
            "1",
            "1/2/2010 8:00",
            "10BERN1",
            "Embassy Bern",
            "CONFIDENTIAL",
            "10BERN0",
            "VZCZ ROUTING",
            "body text",
        )
        document = to_document(parse_cable(_START.match(line), line))

        assert document.doc_id == "1"
        assert document.source == "cablegate"
        assert document.text == "body text"
        assert document.title == "10BERN1"
        assert document.meta["header"] == "VZCZ ROUTING"
        assert document.meta["classification"] == "CONFIDENTIAL"
        assert document.meta["origin"] == "Embassy Bern"

    def test_date_is_month_first_and_keeps_the_original(self):
        """1/2/2010 is 2 January, US-ordered, and the raw string survives in meta."""
        line = _record_line("1", "1/2/2010 8:00", "r", "o", "c", "", "", "b")
        document = to_document(parse_cable(_START.match(line), line))

        assert document.date == "2010-01-02T08:00:00"
        assert document.meta["date_raw"] == "1/2/2010 8:00"

    def test_unparseable_date_is_none_not_a_guess(self):
        """A date that will not parse yields None rather than an invented value."""
        assert iso_date("13/45/2010 99:99") is None


class TestParse:
    """parse() reads the dump into documents."""

    def test_every_origin_is_kept(self, tmp_path):
        """The private corpus is the whole of Cablegate, so nothing is filtered out.

        The previous project's builder kept only Embassy Bern. A retrieval corpus
        narrowed to the documents that produced questions is not a retrieval corpus, so
        origin filtering is a downstream concern.
        """
        raw = _write_dump(
            tmp_path,
            _record_line("1", "1/1/2010 9:00", "10BERN1", "Embassy Bern", "C", "", "h", "bern body")
            + _record_line(
                "2", "1/2/2010 10:00", "10PARIS1", "Embassy Paris", "U", "", "h", "paris body"
            ),
        )

        documents = list(parse(raw))

        assert [d["doc_id"] for d in documents] == ["1", "2"]
        assert {d["meta"]["origin"] for d in documents} == {"Embassy Bern", "Embassy Paris"}

    def test_malformed_record_costs_one_cable_and_no_more(self, tmp_path):
        """A record too malformed to parse yields empty content, and its neighbours survive."""
        malformed = '"2","1/2/2010 10:00","10BERN2","Embassy Bern","UNCLASSIFIED",\n'
        raw = _write_dump(
            tmp_path,
            _record_line("1", "1/1/2010 9:00", "10BERN1", "Embassy Bern", "C", "", "h1", "body one")
            + malformed
            + _record_line(
                "3", "1/3/2010 11:00", "10BERN3", "Embassy Bern", "U", "", "h3", "body three"
            ),
        )

        documents = list(parse(raw))

        assert [d["doc_id"] for d in documents] == ["1", "2", "3"]
        assert documents[0]["text"] == "body one"
        assert documents[1]["text"] == ""
        assert documents[1]["meta"]["header"] == ""
        assert documents[2]["text"] == "body three"

    def test_documents_serialise_to_the_on_disk_shape(self, tmp_path):
        """A parsed document round-trips through JSON with the expected keys."""
        raw = _write_dump(
            tmp_path,
            _record_line("1", "1/1/2010 9:00", "10BERN1", "Embassy Bern", "C", "", "h", "b"),
        )

        record = json.loads(json.dumps(next(iter(parse(raw)))))

        assert set(record) == {"doc_id", "source", "date", "lang", "title", "text", "meta"}
        assert record["lang"] == "en"
