"""Unit tests for the SECO sanctions parser.

The case that matters is the programme join: a target names only a sanctions-set id, and
the programme that listed them sits one level up, in the element that contains that set.
Getting it wrong loses the answer to "which ordinance listed this person?" without losing
the record, so nothing downstream notices.
"""

from __future__ import annotations

from osint_benchmark.sources.sanctions import FILENAME, parse

XML = """<?xml version="1.0" encoding="UTF-8"?>
<swiss-sanctions-list list-type="whole-list" date="2026-07-30">
  <sanctions-program ssid="20">
    <program-key lang="eng">Belarus</program-key>
    <program-name lang="eng">Ordinance on measures against Belarus</program-name>
    <program-name lang="ger">Verordnung</program-name>
    <origin>EU</origin>
    <sanctions-set ssid="200" lang="eng">Financial sanctions</sanctions-set>
  </sanctions-program>
  <target ssid="1">
    <individual>
      <identity>
        <name><name-part><value>Jane</value></name-part><value>Roe</value></name>
        <day-month-year year="1970"/>
        <address><address-details>12 Main St</address-details><country>CH</country></address>
        <justification>Listed for X.</justification>
        <other-information>Also known locally.</other-information>
      </identity>
      <sanctions-set-id>200</sanctions-set-id>
      <modification modification-type="listed" enactment-date="2022-02-25"
                    publication-date="2022-03-01" effective-date="2022-03-01"/>
      <modification modification-type="amended" enactment-date="2023-01-10"
                    publication-date="2023-01-11" effective-date="2023-01-11"/>
      <modification modification-type="de-listed" enactment-date="2024-09-10"
                    publication-date="2024-09-15" effective-date="2024-09-15">
        <removed>
          <target ssid="1">
            <individual>
              <identity><name><value>Jane Roe OLD ALIAS</value></name></identity>
              <address><address-details>99 Old Rd</address-details></address>
            </individual>
          </target>
        </removed>
      </modification>
    </individual>
  </target>
  <target ssid="2">
    <entity>
      <identity><name><value>Acme SA</value></name></identity>
      <sanctions-set-id>999</sanctions-set-id>
    </entity>
  </target>
  <target ssid="3">
    <individual><identity><day-month-year year="1980"/></identity></individual>
  </target>
</swiss-sanctions-list>
"""


def _raw(tmp_path):
    """Write the fixture where the parser expects it, and return the raw root."""
    raw = tmp_path / "raw"
    (raw / "sanctions").mkdir(parents=True)
    (raw / "sanctions" / FILENAME).write_text(XML, encoding="utf-8")
    return raw


class TestParse:
    """One record per named target, with the programme resolved onto it."""

    def test_the_programme_is_joined_through_the_sanctions_set(self, tmp_path):
        """The target names only set 200; the programme comes from the element holding it."""
        record = next(iter(parse(_raw(tmp_path))))

        assert record["sanctions_set_id"] == "200"
        assert record["programme"] == "Ordinance on measures against Belarus"
        assert record["programme_key"] == "Belarus"
        assert record["origin"] == "EU"
        assert record["measures"] == "Financial sanctions"

    def test_the_fields_that_were_once_dropped_are_kept(self, tmp_path):
        """Justification, addresses and the listing dates — the answerable ones."""
        record = next(iter(parse(_raw(tmp_path))))

        assert record["justification"] == "Listed for X."
        assert record["addresses"] == ["12 Main St CH"]
        assert record["other_information"] == ["Also known locally."]
        assert record["dobs"] == ["1970"]
        assert record["type"] == "individual"

    def test_modification_dates_use_the_real_attribute_names(self, tmp_path):
        """The attributes are modification-type and *-date, not type and date.

        The previous project read ``type``/``date``, which do not exist, so added and
        removed were silently None on every one of its 17,074 rows while its manifest
        recorded both as captured.
        """
        record = next(iter(parse(_raw(tmp_path))))

        assert record["added"] == "2022-03-01"
        assert record["removed"] == "2024-09-15"
        assert [m["type"] for m in record["modifications"]] == ["listed", "amended", "de-listed"]
        assert record["modifications"][0]["enactment_date"] == "2022-02-25"

    def test_a_modification_snapshot_is_not_folded_into_the_live_record(self, tmp_path):
        """<removed> embeds the target as it was; walking into it corrupts the current one."""
        records = list(parse(_raw(tmp_path)))

        assert records[0]["names"] == ["Jane Roe"]
        assert records[0]["addresses"] == ["12 Main St CH"]
        assert [r["doc_id"] for r in records] == ["1", "2"]

    def test_a_name_split_across_parts_is_rejoined(self, tmp_path):
        """A name is spread over <value> elements and has to be reassembled."""
        record = next(iter(parse(_raw(tmp_path))))

        assert record["names"] == ["Jane Roe"]

    def test_an_unmatched_sanctions_set_leaves_the_programme_empty(self, tmp_path):
        """Set 999 has no programme; that is blank, not an error and not a wrong join."""
        records = list(parse(_raw(tmp_path)))

        acme = next(r for r in records if r["doc_id"] == "2")
        assert acme["programme"] == ""
        assert acme["type"] == "entity"

    def test_a_nameless_target_is_skipped(self, tmp_path):
        """Target 3 has a birth year and no name, so there is nothing to identify."""
        assert [r["doc_id"] for r in parse(_raw(tmp_path))] == ["1", "2"]

    def test_every_record_carries_the_export_date(self, tmp_path):
        """The list is a live snapshot, so its own date is the only version marker."""
        assert all(r["list_date"] == "2026-07-30" for r in parse(_raw(tmp_path)))
