"""Unit tests for the Curia Vista paging.

Two rules cost the previous project real data and are what these tests pin:

* paging is by keyset (``ID gt <last>``), because the service answers HTTP 500 once
  ``$skip`` gets deep;
* a **short** page is not the last page — the server returns short pages mid-stream, so
  only an empty one terminates. Treating a short page as the end truncates the fetch and
  leaves a file that looks complete.

Run against a local server that reproduces both behaviours rather than a mock, since the
bug being guarded is in how the client reacts to the server, not in its own logic.
"""

from __future__ import annotations

import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from osint_benchmark.sources.parliament import page_entity, parse, strip_odata

TOTAL = 25


class _Handler(BaseHTTPRequestHandler):
    """A minimal OData stand-in. ``guid_keys`` switches the ID type it serves."""

    def do_GET(self):  # noqa: N802 (the name is BaseHTTPRequestHandler's)
        """Answer one page: keyset by ID, or by $skip when the keys are GUIDs."""
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        filt = query.get("$filter", [""])[0]
        skip = int(query.get("$skip", ["0"])[0])
        after = int(filt.split("ID gt ")[1]) if "ID gt " in filt else 0

        if self.server.guid_keys:
            self.server.requests.append(skip)
            page = [f"guid-{i:04d}" for i in range(skip + 1, min(skip + 10, TOTAL) + 1)]
            rows = [{"ID": i, "Title": f"item {i}"} for i in page]
        else:
            self.server.requests.append(after)
            remaining = [i for i in range(1, TOTAL + 1) if i > after]
            # A deliberately SHORT page in the middle: two rows where ten were asked for.
            rows = [
                {"ID": i, "Title": f"item {i}"}
                for i in (remaining[:2] if not after else remaining[:10])
            ]

        body = json.dumps({"d": {"results": rows}})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        """Keep the test output quiet."""


@pytest.fixture
def server():
    """Run the OData stand-in and record the keyset offset of every request."""
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    httpd.requests = []
    httpd.guid_keys = False
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


class TestPaging:
    """Every row is fetched, however the server chooses to chunk them."""

    def test_a_short_page_does_not_end_the_fetch(self, server, tmp_path, monkeypatch):
        """The first page returns 2 of 25 rows; stopping there would lose 23."""
        monkeypatch.setattr(
            "osint_benchmark.sources.parliament.ODATA",
            f"http://127.0.0.1:{server.server_address[1]}",
        )
        out = tmp_path / "Business.jsonl"

        count = page_entity("Business", "Language eq 'DE'", out)

        assert count == TOTAL
        assert [json.loads(line)["ID"] for line in out.read_text().splitlines()] == list(
            range(1, TOTAL + 1)
        )

    def test_paging_is_by_keyset_not_offset(self, server, tmp_path, monkeypatch):
        """Each request asks for rows after the last ID seen, never for a deep $skip."""
        monkeypatch.setattr(
            "osint_benchmark.sources.parliament.ODATA",
            f"http://127.0.0.1:{server.server_address[1]}",
        )

        page_entity("Business", "Language eq 'DE'", tmp_path / "Business.jsonl")

        assert server.requests == [0, 2, 12, 22, 25]

    def test_guid_keyed_sets_page_by_offset_instead_of_being_dropped(
        self, server, tmp_path, monkeypatch
    ):
        """PersonInterest and PersonOccupation key on a GUID, so there is no ID gt cursor.

        Requiring an integer cursor to continue fetched their rows and then wrote none of
        them: both sets came back empty against the previous project's 1,979 and 240.
        """
        server.guid_keys = True
        monkeypatch.setattr(
            "osint_benchmark.sources.parliament.ODATA",
            f"http://127.0.0.1:{server.server_address[1]}",
        )
        out = tmp_path / "PersonInterest.jsonl"

        count = page_entity("PersonInterest", "Language eq 'DE'", out)

        assert count == TOTAL
        assert len(out.read_text().splitlines()) == TOTAL
        assert server.requests == [0, 10, 20, 25]

    def test_an_interrupted_fetch_resumes_from_the_last_id(self, server, tmp_path, monkeypatch):
        """Rows already on disk are not fetched again."""
        monkeypatch.setattr(
            "osint_benchmark.sources.parliament.ODATA",
            f"http://127.0.0.1:{server.server_address[1]}",
        )
        out = tmp_path / "Business.jsonl"
        out.write_text('{"ID": 1}\n{"ID": 2}\n', encoding="utf-8")

        count = page_entity("Business", "Language eq 'DE'", out)

        assert count == TOTAL
        assert server.requests[0] == 2


class TestStripOdata:
    """The OData envelope is not data."""

    def test_metadata_and_navigation_links_are_dropped(self):
        """__metadata and __deferred describe the API, not the record."""
        row = {"ID": 1, "__metadata": {"uri": "..."}, "Party": {"__deferred": {"uri": "..."}}}

        assert strip_odata(row) == {"ID": 1}

    def test_dotnet_date_stamps_become_iso_dates(self):
        """/Date(ms)/ is unreadable and unsortable; the ISO date is neither."""
        assert strip_odata({"SubmissionDate": "/Date(1262304000000)/"}) == {
            "SubmissionDate": "2010-01-01"
        }

    def test_a_timezone_suffix_is_handled(self):
        """Some stamps carry an offset, e.g. /Date(1262304000000+0060)/."""
        assert strip_odata({"D": "/Date(1262304000000+0060)/"}) == {"D": "2010-01-01"}


class TestParse:
    """Entity sets are told apart in the combined output."""

    def test_records_carry_their_entity_and_a_qualified_id(self, tmp_path):
        """IDs are unique only within an entity set, so doc_id is 'entity:ID'."""
        raw = tmp_path / "raw" / "parliament"
        raw.mkdir(parents=True)
        for entity in ("Person", "MemberCouncil", "PersonInterest", "PersonOccupation", "Party"):
            (raw / f"{entity}.jsonl").write_text('{"ID": 1}\n', encoding="utf-8")
        (raw / "Business.jsonl").write_text('{"ID": 1, "Title": "x"}\n', encoding="utf-8")

        records = list(parse(tmp_path / "raw"))

        assert {r["doc_id"] for r in records} == {
            "Person:1",
            "MemberCouncil:1",
            "PersonInterest:1",
            "PersonOccupation:1",
            "Party:1",
            "Business:1",
        }
        assert next(r for r in records if r["entity"] == "Business")["Title"] == "x"
