"""Unit tests for the shared fetch and verify steps.

These are shared rather than per-source on purpose: a copy per source would be a chance
per source for one to be weakened differently, which is how the previous project ended up
with a dependency gate that ran on one question stream and not the other.
"""

from __future__ import annotations

import pytest

from osint_benchmark.artifacts import read_jsonl
from osint_benchmark.schema import Document
from osint_benchmark.sources import base
from osint_benchmark.sources.base import Projection, Source, SourceUnavailable

PROJECTION = Projection(source="a test source", source_fields=("a",), kept={"a": "text"})

PINS = """
[demo]
kind = "private"

[[demo.origins]]
filename = "demo.csv"
url = ""
sha256 = "{sha}"
size = {size}
note = "Obtain demo.csv and place it under <raw>/demo/."
"""

CONTENT = "hello\n"
# sha256 of CONTENT, so the pin in these tests is a real one rather than a stub.
CONTENT_SHA = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"


def _source(documents=()):
    """Return a source whose parser yields fixed documents."""
    return Source(
        name="demo", kind="private", parse=lambda raw: iter(documents), projection=PROJECTION
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Point every root at tmp_path and write a pins file for the demo source."""
    (tmp_path / "pins").mkdir()
    (tmp_path / "pins" / "sources.toml").write_text(
        PINS.format(sha=CONTENT_SHA, size=len(CONTENT)), encoding="utf-8"
    )
    monkeypatch.setenv("OSINT_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("OSINT_PINS", str(tmp_path / "pins"))
    return tmp_path


class TestFetch:
    """fetch checks what is on disk and says how to get what is not."""

    def test_a_missing_unredistributable_file_reports_how_to_obtain_it(self, env):
        """The note from the pins file is what the user sees, not a bare traceback."""
        with pytest.raises(SourceUnavailable, match="Obtain demo.csv"):
            base.fetch(_source())

    def test_a_present_file_matching_its_pin_passes(self, env):
        """A file already on disk is accepted without being re-downloaded."""
        raw = env / "data" / "raw" / "demo"
        raw.mkdir(parents=True)
        (raw / "demo.csv").write_text(CONTENT, encoding="utf-8")

        assert base.fetch(_source()) == [raw / "demo.csv"]

    def test_a_wrong_size_is_refused_before_anything_is_parsed(self, env):
        """The cheap check runs first: a truncated download never reaches the parser."""
        raw = env / "data" / "raw" / "demo"
        raw.mkdir(parents=True)
        (raw / "demo.csv").write_text("wrong", encoding="utf-8")

        with pytest.raises(SourceUnavailable, match="pinned at"):
            base.fetch(_source())

    def test_a_wrong_checksum_is_refused(self, env):
        """Right size, wrong bytes — the case the size check cannot catch."""
        raw = env / "data" / "raw" / "demo"
        raw.mkdir(parents=True)
        (raw / "demo.csv").write_text("HELLO\n", encoding="utf-8")

        with pytest.raises(SourceUnavailable, match="hashes to"):
            base.fetch(_source())


class TestVerify:
    """verify is what stands between a parse regression and someone's wrong numbers."""

    def test_no_baseline_is_reported_but_is_not_a_failure(self, env):
        """Before the first release there is nothing to check against.

        The report says so rather than claiming a match, but a build is not wrong for
        being the first one — only a contradiction of published hashes is.
        """
        source = _source([Document(doc_id="1", source="demo", text="x")])
        base.parse(source)

        report = base.verify(source)

        assert report.baseline_missing
        assert "no published hashes" in report.summary()
        assert report.ok

    def test_a_published_baseline_then_verifies(self, env):
        """Recording a baseline and checking against it is a round trip."""
        source = _source([Document(doc_id="1", source="demo", text="x")])
        base.parse(source)

        base.write_hashes(source)
        report = base.verify(source)

        assert report.ok
        assert report.checked == 1
        assert [r["doc_id"] for r in read_jsonl(base.hashes_path("demo"))] == ["1"]

    def test_a_changed_document_is_caught(self, env):
        """The escapechar failure in miniature: same document id, different text."""
        base.parse(_source([Document(doc_id="1", source="demo", text="full text")]))
        base.write_hashes(_source())

        base.parse(_source([Document(doc_id="1", source="demo", text="trunc")]))
        report = base.verify(_source())

        assert report.changed == ("1",)
        assert not report.ok

    def test_a_missing_document_is_caught(self, env):
        """A cable that stopped being parsed is a loss, not an absence of news."""
        both = [
            Document(doc_id="1", source="demo", text="one"),
            Document(doc_id="2", source="demo", text="two"),
        ]
        base.parse(_source(both))
        base.write_hashes(_source())

        base.parse(_source(both[:1]))
        report = base.verify(_source())

        assert report.missing == ("2",)
        assert not report.ok
