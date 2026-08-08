"""Unit tests for the entity-document graph and the document pairing."""

from __future__ import annotations

import pytest

from osint_benchmark.graph import bridge, entity_types
from osint_benchmark.pair import join


def _link(doc_id: str, side: str, *entities: tuple[str, str]):
    """Return one link row."""
    return {
        "doc_id": doc_id,
        "side": side,
        "entities": [{"qid": q, "surface_form": f} for q, f in entities],
    }


class TestBuild:
    """Inverting per-document links into per-entity bridges."""

    def test_an_entity_on_both_sides_bridges(self):
        """The whole point: one entity naming a private document and a public record."""
        rows = [
            _link("cable1", "private", ("Q1", "Iran")),
            _link("item1", "public", ("Q1", "Iran")),
        ]

        bridges = bridge.build(rows)

        assert bridges["Q1"].bridges
        assert bridges["Q1"].private == {"cable1"}
        assert bridges["Q1"].public == {"item1"}

    def test_an_entity_on_one_side_only_does_not_bridge(self):
        """Most entities are like this, and they are not candidates."""
        bridges = bridge.build([_link("cable1", "private", ("Q1", "Iran"))])

        assert not bridges["Q1"].bridges
        assert list(bridge.bridging(bridges)) == []

    def test_surface_forms_are_counted(self):
        """A bridge resting on one rare spelling is weaker than one resting on many."""
        rows = [
            _link("c1", "private", ("Q1", "Iran")),
            _link("c2", "private", ("Q1", "Iran")),
            _link("c3", "private", ("Q1", "IRAN")),
            _link("i1", "public", ("Q1", "Iran")),
        ]

        assert bridge.build(rows)["Q1"].surface_forms == {"IRAN": 1, "Iran": 3}

    def test_an_unknown_side_is_an_error(self):
        """Silently dropping it would quietly shrink one leg of the graph."""
        with pytest.raises(ValueError, match="side must be private or public"):
            bridge.build([_link("c1", "sideways", ("Q1", "x"))])

    def test_a_ubiquitous_entity_can_be_dropped(self):
        """An entity in hundreds of cables is a country, not a subject.

        It bridges everything and therefore distinguishes nothing; leaving it in floods
        the pairing step with pairs that share only a continent.
        """
        rows = [_link(f"c{i}", "private", ("Q1", "USA")) for i in range(10)]
        rows.append(_link("i1", "public", ("Q1", "USA")))
        rows += [_link("c99", "private", ("Q2", "Bern")), _link("i2", "public", ("Q2", "Bern"))]

        kept = [row["qid"] for row in bridge.bridging(bridge.build(rows), max_private=5)]

        assert kept == ["Q2"]

    def test_output_is_ordered_so_two_runs_agree(self):
        """Reproducibility is checked by fingerprint, so ordering has to be stable."""
        rows = [
            _link("c1", "private", ("Q3", "c"), ("Q1", "a")),
            _link("i1", "public", ("Q3", "c"), ("Q1", "a")),
        ]

        assert [r["qid"] for r in bridge.bridging(bridge.build(rows))] == ["Q1", "Q3"]


class TestInterval:
    """The signed gap between the two documents."""

    def test_the_interval_is_signed(self):
        """A public record predating the private report is a different situation."""
        assert join.interval("2010-01-01T00:00:00", "2010-01-11T00:00:00") == 10
        assert join.interval("2010-01-11T00:00:00", "2010-01-01T00:00:00") == -10

    def test_a_missing_date_yields_none_not_zero(self):
        """Zero days apart is a claim; not knowing is not."""
        assert join.interval(None, "2010-01-01T00:00:00") is None
        assert join.interval("2010-01-01T00:00:00", None) is None

    def test_an_unparseable_date_yields_none(self):
        """Sources carry malformed dates; a guess would be worse."""
        assert join.interval("not a date", "2010-01-01T00:00:00") is None


class TestPairDocuments:
    """One pair per (private document, public record, shared entity)."""

    def test_each_document_combination_becomes_a_pair(self):
        """Counts then mean documents rather than entities."""
        bridges = [{"qid": "Q1", "private": ["c1", "c2"], "public": ["i1"]}]

        pairs = list(join.pair_documents(bridges, {}, {}))

        assert {(p.private_id, p.public_id) for p in pairs} == {("c1", "i1"), ("c2", "i1")}

    def test_contemporaneity_is_recorded_not_assumed(self):
        """Dodis is archival: 90% of it predates the parliamentary record entirely.

        A pair outside the window is kept and marked, because it is still valid for the
        question types that do not depend on the two being about the same events.
        """
        bridges = [{"qid": "Q1", "private": ["c1"], "public": ["i1"]}]
        private = {"c1": "1950-01-01T00:00:00"}
        public = {"i1": "2006-01-01T00:00:00"}

        pair = next(join.pair_documents(bridges, private, public, window_days=90))

        assert not pair.same_period
        assert pair.days_apart == 20454

    def test_a_pair_inside_the_window_is_contemporaneous(self):
        """The case the interval and position-comparison types need."""
        bridges = [{"qid": "Q1", "private": ["c1"], "public": ["i1"]}]
        private = {"c1": "2006-01-01T00:00:00"}
        public = {"i1": "2006-02-01T00:00:00"}

        pair = next(join.pair_documents(bridges, private, public, window_days=90))

        assert pair.same_period
        assert pair.days_apart == 31

    def test_an_undated_pair_is_not_contemporaneous(self):
        """Unknown is not the same as close."""
        bridges = [{"qid": "Q1", "private": ["c1"], "public": ["i1"]}]

        pair = next(join.pair_documents(bridges, {"c1": None}, {"i1": None}))

        assert not pair.same_period
        assert pair.days_apart is None


class TestEntityTypes:
    """Which entities may anchor a bridge, and which may never."""

    def test_a_country_may_not_bridge(self):
        """The first real run bridged only on countries.

        Every question it produced asked what connects two countries, and the answer was
        that both documents mention international relations. A country co-occurs with
        everything, so it distinguishes nothing.
        """
        assert entity_types.classify({"instance_of": ["Q6256"]}) == "blocked"

    def test_a_person_or_an_organisation_may(self):
        """Specific enough that two documents naming one are plausibly about the same thing."""
        assert entity_types.classify({"instance_of": ["Q5"]}) == "bridgeable"
        assert entity_types.classify({"instance_of": ["Q43229"]}) == "bridgeable"

    def test_blocked_beats_bridgeable(self):
        """Wikidata models many states as organisations too; they are still countries."""
        assert entity_types.classify({"instance_of": ["Q43229", "Q3624078"]}) == "blocked"

    def test_a_category_is_not_a_thing_two_documents_can_share(self):
        """A generic noun resolves to a real entity, and co-occurs with everything.

        Each resolves to a Wikidata entity that is an instance of an organisation class,
        so a check on instance_of alone called them bridgeable -- and they co-occur with
        everything exactly as countries do. 896 of GDELT's actor names are these. What
        separates them from Associated Press is that other things are kinds of them.
        """
        police = {"instance_of": ["Q43229"], "subclass_of": ["Q1639780"]}

        assert entity_types.classify(police) == "blocked"

    def test_a_named_organisation_still_bridges(self):
        """The filter must not take the specific things with the generic ones."""
        associated_press = {"instance_of": ["Q43229", "Q192283"], "subclass_of": []}

        assert entity_types.classify(associated_press) == "bridgeable"

    def test_subclass_of_is_not_evidence_of_being_the_class(self):
        """It was read as such: the two properties were unioned before being checked.

        Under that reading "subclass of organisation" made something an organisation,
        which is how every generic noun became a bridge anchor.
        """
        assert entity_types.classify({"subclass_of": ["Q5"]}) == "blocked"

    def test_an_organisation_type_nobody_listed_still_bridges(self):
        """Associated Press is a news agency, a cooperative and a nonprofit.

        None of the three were in the allowlist, so a real named agency came out unknown
        and was dropped. An allowlist cannot anticipate the tail of organisation types.
        """
        assert entity_types.classify({"instance_of": ["Q192283", "Q163740"]}) == "bridgeable"

    def test_an_entity_with_no_type_at_all_is_excluded_by_default(self):
        """An entity nobody can vouch for is how the country problem returns by another route."""
        facts = [{"qid": "Q1", "statements": {}}]

        assert entity_types.bridgeable_qids(facts) == set()
        assert entity_types.bridgeable_qids(facts, keep_unknown=True) == {"Q1"}


class TestPairCap:
    """The combinatorics have to be bounded."""

    def test_pairs_per_entity_are_capped(self):
        """42 bridges once produced 30,416 pairs."""
        bridges = [
            {
                "qid": "Q1",
                "private": [f"c{i}" for i in range(20)],
                "public": [f"i{j}" for j in range(20)],
            }
        ]

        pairs = list(join.pair_documents(bridges, {}, {}, max_per_entity=25))

        assert len(pairs) == 25

    def test_no_cap_keeps_every_combination(self):
        """The cap is a choice, not a silent default deep in the code."""
        bridges = [{"qid": "Q1", "private": ["c1", "c2"], "public": ["i1", "i2"]}]

        assert len(list(join.pair_documents(bridges, {}, {}))) == 4
