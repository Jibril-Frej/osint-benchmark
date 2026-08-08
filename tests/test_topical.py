"""Unit tests for the cable-parliament join and the two types built on it.

Every condition here comes from a measurement in the previous project rather than from
taste, so each test names the failure it prevents.
"""

from __future__ import annotations

from datetime import date

from osint_benchmark.generate import chronology, events, typed
from osint_benchmark.pair import topical

LABELS = {"Q1": "Switzerland", "Q2": "Swissair", "Q3": "Iran", "Q4": "Micheline Calmy-Rey"}
PLACES = {"Q1", "Q3"}


def _cable(doc_id="cablegate:1", when=date(2005, 3, 1), tags=("ETRD",), qids=("Q2", "Q4")):
    """Return one linked cable in the shape the join reads."""
    return {
        "doc_id": doc_id,
        "date": when,
        "origin": "Embassy Bern",
        "subject": "AVIATION TALKS",
        "tags": set(tags),
        "qids": list(qids),
    }


def _item(doc_id="parliament:Business:9", when=date(2005, 4, 1), cats=("Wirtschaft",),
          qids=("Q2", "Q4"), title="Swissair und die Folgen"):  # fmt: skip
    """Return one parliamentary business item in the shape the join reads."""
    return {
        "doc_id": doc_id,
        "date": when,
        "title": title,
        "type": "Motion",
        "cats": set(cats),
        "has_response": True,
        "qids": list(qids),
    }


class TestTags:
    """What a cable declares its topic to be."""

    def test_the_tags_line_is_read_off_the_body(self):
        """They are in the text the parser already keeps, not a field it drops."""
        text = "VZCZC\nSUBJECT: AVIATION TALKS\nTAGS: ETRD, EAIR, PREL\n1. The talks..."

        assert topical.tags_in(text) == {"ETRD", "EAIR", "PREL"}

    def test_a_cable_with_no_tags_line_declares_nothing(self):
        """And is dropped rather than matched on entities alone."""
        assert topical.tags_in("SUBJECT: SOMETHING\n1. Text.") == set()

    def test_tags_map_onto_parliamentary_categories(self):
        """The crosswalk is what makes an English corpus joinable to a German one."""
        assert topical.topics({"ETRD", "PARM"}) == {"Wirtschaft", "Sicherheitspolitik"}


class TestJoin:
    """A pair needs a topic, a date and an entity — all three."""

    def _join(self, cables=None, business=None, **kw):
        """Run the join over one cable and one item unless told otherwise."""
        kw.setdefault("max_share", 1.0)
        return list(topical.join(cables or [_cable()], business or [_item()], LABELS, PLACES, **kw))

    def test_a_shared_topic_date_and_entity_make_a_pair(self):
        """The ordinary path."""
        pairs = self._join()

        assert len(pairs) == 1
        assert pairs[0]["shared_entities"] == ["Swissair", "Micheline Calmy-Rey"]

    def test_a_cable_whose_tags_map_to_nothing_is_dropped(self):
        """Entity overlap alone was measured to be too loose to use."""
        assert self._join(cables=[_cable(tags=("ZZZZ",))]) == []

    def test_a_different_subject_area_is_not_a_pair(self):
        """Two documents in the same month about different things are not related."""
        assert self._join(business=[_item(cats=("Gesundheit",))]) == []

    def test_an_item_outside_the_window_is_not_a_pair(self):
        """A cable reports on what is in front of it."""
        assert self._join(business=[_item(when=date(2006, 9, 1))]) == []

    def test_a_pair_with_no_shared_entity_is_dropped(self):
        """Topic alone puts a WTO cable beside every economic motion of the quarter."""
        assert self._join(business=[_item(qids=("Q7",))]) == []

    def test_an_entity_in_most_cables_carries_no_signal(self):
        """Switzerland links a branding postulate to every Swiss cable there is."""
        cables = [_cable(f"cablegate:{n}", qids=("Q1",)) for n in range(10)]

        assert self._join(cables=cables, business=[_item(qids=("Q1",))], max_share=0.15) == []


class TestFocused:
    """Which pairs are a topical link rather than a coincidence."""

    def test_an_entity_in_the_items_title_focuses_the_pair(self):
        """The item is about it, not merely adjacent to it."""
        pairs = list(
            topical.join(
                [_cable(qids=("Q2",))], [_item(qids=("Q2",))], LABELS, PLACES, max_share=1.0
            )
        )

        assert pairs[0]["shared_in_title"] == ["Swissair"]
        assert pairs[0]["focused"]

    def test_two_shared_countries_are_not_a_link(self):
        """Two documents sharing only countries share only a map reference."""
        pairs = list(
            topical.join(
                [_cable(qids=("Q1", "Q3"))],
                [_item(qids=("Q1", "Q3"), title="Aussenpolitik")],
                LABELS,
                {"Q1", "Q3"},
                max_share=1.0,
            )
        )

        assert not pairs[0]["focused"]

    def test_a_second_entity_that_is_not_a_place_focuses_it(self):
        """One of them has to be a thing rather than somewhere."""
        pairs = list(
            topical.join(
                [_cable(qids=("Q1", "Q4"))],
                [_item(qids=("Q1", "Q4"), title="Aussenpolitik")],
                LABELS,
                PLACES,
                max_share=1.0,
            )
        )

        assert pairs[0]["focused"]

    def test_a_title_match_is_on_whole_words(self):
        """The name Chad otherwise matches inside Schlechtwetterentschaedigung."""
        labels = {"Q9": "Chad"}
        pairs = list(
            topical.join(
                [_cable(qids=("Q9",))],
                [_item(qids=("Q9",), title="Schlechtwetterentschaedigung fuer Betriebe")],
                labels,
                set(),
                max_share=1.0,
            )
        )

        assert pairs[0]["shared_in_title"] == []


class TestChronology:
    """The interval type: gold is arithmetic on two metadata dates."""

    def _pair(self, private="2005-03-01", public="2005-04-01", focused=True):
        """Return one joined pair."""
        return {
            "private_id": "cablegate:1",
            "public_id": "parliament:Business:9",
            "private_date": private,
            "public_date": public,
            "private_subject": "AVIATION TALKS",
            "public_title": "Swissair und die Folgen",
            "shared_entities": ["Swissair"],
            "focused": focused,
        }

    def test_the_interval_is_the_answer(self):
        """31 days between the two, computed from metadata, no model involved."""
        items = list(chronology.build([self._pair()]))

        assert items[0].days == 31
        assert items[0].order == "private_first"

    def test_an_unfocused_pair_yields_no_interval(self):
        """An interval between unrelated events is arithmetic, not a question."""
        assert list(chronology.build([self._pair(focused=False)])) == []

    def test_a_same_day_pair_is_dropped(self):
        """Zero is a coincidence a solver can guess."""
        assert list(chronology.build([self._pair(public="2005-03-01")])) == []

    def test_a_gap_beyond_a_year_is_not_one_episode(self):
        """Whatever they share, they are no longer reporting on the same moment."""
        assert list(chronology.build([self._pair(public="2007-04-01")])) == []

    def test_the_answer_is_not_a_bare_number(self):
        """A one-digit answer trips the gate that rejects one-character parse failures."""
        texts = {"cablegate:1": "the talks", "parliament:Business:9": "die Motion"}
        items = list(chronology.build([self._pair(public="2005-03-06")]))

        candidate = next(iter(typed.from_chronology(items, texts)))

        assert candidate.answer == "5 days"


class TestPosture:
    """The one type whose gold a model decides."""

    def _candidate(self):
        """Return one posture candidate."""
        pair = {
            "private_id": "cablegate:1",
            "public_id": "parliament:Business:9",
            "private_date": "2005-03-01",
            "public_date": "2005-04-01",
            "public_title": "Swissair und die Folgen",
            "shared_entities": ["Swissair"],
            "shared_cats": ["Wirtschaft"],
            "focused": True,
        }
        texts = {"cablegate:1": "privately urged", "parliament:Business:9": "oeffentlich"}
        return next(iter(typed.from_posture([pair], texts)))

    def test_the_verdict_the_model_returns_becomes_the_answer(self):
        """Whether two positions agree cannot be settled before the question exists."""
        phraser = lambda p: '{"question": "Did it match?", "verdict": "Mixed", "rationale": "r"}'  # noqa: E731

        items = list(typed.phrase_candidates([self._candidate()], phraser, ("an analyst",)))

        assert items[0].answer == "Mixed"
        assert items[0].rationale == "r"

    def test_a_verdict_outside_the_labels_yields_no_item(self):
        """A gold the type never offered is not a gold."""
        phraser = lambda p: '{"question": "Did it match?", "verdict": "Probably"}'  # noqa: E731

        assert list(typed.phrase_candidates([self._candidate()], phraser, ("an analyst",))) == []

    def test_a_computed_type_ignores_any_verdict_it_is_sent(self):
        """Only a type that declares labels may have its gold decided by a model."""
        candidate = typed.Candidate(
            item_id="x",
            question_type="chronology",
            answer="31 days",
            gold_qid="",
            private_id="cablegate:1",
            public_id="parliament:Business:9",
            passage="",
            facts={
                "private_evidence": "the talks",
                "public_title": "die Motion",
                "public_type": "Motion",
            },
        )
        phraser = lambda p: '{"question": "How long?", "verdict": "Yes"}'  # noqa: E731

        items = list(typed.phrase_candidates([candidate], phraser, ("an analyst",)))

        assert items[0].answer == "31 days"


class TestEvents:
    """The type built on the two sources that bridge nothing."""

    def _document(self, when=date(2006, 4, 5), qids=("Q1",)):
        """Return one dated, linked confidential document."""
        return {
            "doc_id": "cablegate:1",
            "date": when,
            "entities": [{"qid": q} for q in qids],
        }

    def _event(self, doc_id="ucdp:9", when=date(2006, 4, 7), qids=("Q1",)):
        """Return one dated, linked public event record."""
        return {
            "doc_id": doc_id,
            "date": when,
            "qids": list(qids),
            "country": "Palestine",
            "side_a": "Hamas",
            "side_b": "Fatah",
            "best": "6",
            "date_start": when.isoformat(),
        }

    def test_an_event_in_the_window_is_matched(self):
        """Two days apart, same anchor: the comparison the type exists for."""
        found = list(
            events.build([self._document()], [self._event()], {"Q1": "Palestine"}, max_share=1.0)
        )

        assert len(found) == 1
        assert found[0].public_ids == ("ucdp:9",)
        assert "Hamas" in found[0].rendered

    def test_an_event_outside_the_window_is_not(self):
        """Time is what makes the match discriminative; without it the anchor is a country."""
        far = self._event(when=date(2006, 7, 7))

        assert (
            list(events.build([self._document()], [far], {"Q1": "Palestine"}, max_share=1.0)) == []
        )

    def test_an_anchor_named_by_most_documents_is_skipped(self):
        """It puts every document beside every event of the month."""
        documents = [{**self._document(), "doc_id": f"cablegate:{n}"} for n in range(10)]

        assert (
            list(events.build(documents, [self._event()], {"Q1": "Palestine"}, max_share=0.15))
            == []
        )

    def test_an_undated_document_cannot_be_matched(self):
        """The window is the whole signal, so a document without a date has none."""
        undated = {**self._document(), "date": None}

        assert (
            list(events.build([undated], [self._event()], {"Q1": "Palestine"}, max_share=1.0)) == []
        )

    def test_the_events_shown_are_capped(self):
        """Past a handful the prompt is a list rather than a comparison."""
        many = [self._event(doc_id=f"ucdp:{n}", when=date(2006, 4, 6)) for n in range(20)]

        found = list(
            events.build([self._document()], many, {"Q1": "Palestine"}, max_share=1.0, max_events=8)
        )

        assert len(found[0].public_ids) == 8

    def test_every_matched_event_is_cited_as_evidence(self):
        """The absence of a matching event among the ones there are is the evidence for No."""
        found = list(
            events.build(
                [self._document()],
                [self._event(), self._event(doc_id="ucdp:10")],
                {"Q1": "Palestine"},
                max_share=1.0,
            )
        )
        texts = {"cablegate:1": "clashes were reported"}

        candidate = next(iter(typed.from_events(found, texts)))
        item = typed.to_item(candidate, "Did clashes occur?", "an analyst", "stub")

        assert [e.doc_id for e in item.public_evidence] == ["ucdp:9", "ucdp:10"]
        assert item.two_sided
