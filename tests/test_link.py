"""Unit tests for entity linking and reconciliation.

The linker itself is injected, so these run with no model, no torch and no GPU — which is
the property that keeps every stage downstream of step 2 testable.
"""

from __future__ import annotations

import pytest

from osint_benchmark.link import reconcile
from osint_benchmark.link.refined import (
    Mention,
    keep,
    link_documents,
    narrative_body,
    per_document,
)

UNIVERSE = frozenset({"Q1", "Q2", "Q3"})


def _linker(*mentions: Mention):
    """Return a linker that always yields the given mentions."""
    return per_document(lambda text: list(mentions))


class TestKeep:
    """Both filters cost recall deliberately: a noisy bridge seeds a bad question."""

    def test_a_low_confidence_mention_is_dropped(self):
        """ReFinED will resolve almost anything if asked."""
        mentions = [Mention("Q1", "Bern", 0.2, "GPE"), Mention("Q2", "Iran", 0.95, "GPE")]

        assert [m.qid for m in keep(mentions, UNIVERSE)] == ["Q2"]

    def test_an_unrecognised_type_is_kept(self):
        """The type filter is a denylist because this model's types cannot be trusted.

        It labels countries ORG. An allowlist drops whatever it failed to anticipate, and
        what may anchor a bridge is decided from Wikidata in step 3, not from here.
        """
        assert len(keep([Mention("Q1", "x", 0.99, "SOMETHING_NEW")], UNIVERSE)) == 1

    def test_an_entity_outside_the_public_set_is_dropped(self):
        """An entity with no English article cannot bridge to the public corpus."""
        mentions = [Mention("Q99", "Obscure", 0.99, "ORG")]

        assert keep(mentions, UNIVERSE) == []

    def test_dates_and_quantities_are_not_entities_to_ask_about(self):
        """The model recognises them; nobody can build a bridge on them."""
        mentions = [Mention("Q1", "2006", 0.99, "DATE"), Mention("Q2", "Nestle", 0.99, "ORG")]

        assert [m.qid for m in keep(mentions, UNIVERSE)] == ["Q2"]

    def test_no_universe_skips_that_filter(self):
        """Only correct when the caller has already applied it."""
        assert len(keep([Mention("Q99", "x", 0.99, "ORG")], None)) == 1


class TestLinkDocuments:
    """Link rows are what the graph step reads."""

    def test_a_document_with_no_mentions_still_yields_a_row(self):
        """Absence of links is a fact about the document.

        Dropping it would make the linker's coverage impossible to measure afterwards.
        """
        rows = list(link_documents([{"doc_id": "1", "text": "x"}], _linker(), "private", UNIVERSE))

        assert rows == [{"doc_id": "1", "side": "private", "entities": []}]

    def test_repeated_entities_collapse_to_the_best_mention(self):
        """A cable naming Iran twelve times is one bridge, at its best confidence."""
        linker = _linker(Mention("Q1", "Iran", 0.91, "GPE"), Mention("Q1", "IRAN", 0.97, "GPE"))

        row = next(link_documents([{"doc_id": "1", "text": "x"}], linker, "private", UNIVERSE))

        assert len(row["entities"]) == 1
        assert row["entities"][0] == {"qid": "Q1", "surface_form": "IRAN", "confidence": 0.97}

    def test_entities_are_ordered_so_the_output_is_reproducible(self):
        """Two runs over the same documents must produce the same bytes."""
        linker = _linker(Mention("Q3", "c", 0.99, "ORG"), Mention("Q1", "a", 0.99, "ORG"))

        row = next(link_documents([{"doc_id": "1", "text": "x"}], linker, "private", UNIVERSE))

        assert [e["qid"] for e in row["entities"]] == ["Q1", "Q3"]


class TestNarrativeBody:
    """What the linker is shown decides what it can find; a cable is not all narrative."""

    def test_the_routing_preamble_is_dropped(self):
        """It is full of capitalised tokens and about nothing."""
        cable = "VZCZCXRO1234\nRR RUEHWEB\nSUBJECT: X\n\n1. (C) The minister said so.\nSMITH"

        assert narrative_body(cable) == "1. (C) The minister said so."

    def test_the_drafter_signature_is_dropped(self):
        """A cable's last line is a surname, and it is the corpus's favourite mislink."""
        assert narrative_body("\n1. (S) Text here.\nRICE").endswith("Text here.")

    def test_a_cable_with_no_numbered_paragraph_keeps_its_body(self):
        """Not every cable is formatted; losing them all would be worse than the noise."""
        assert narrative_body("Just some prose.") == "Just some prose."

    def test_a_long_last_line_is_narrative_not_a_signature(self):
        """Only a short trailing line is a name."""
        body = "\n1. (C) One.\nThis sentence is plainly not somebody's surname at all."

        assert narrative_body(body).endswith("surname at all.")


class TestBatching:
    """ReFinED is far faster per batch, so the interface is a batch one."""

    def test_documents_are_linked_in_batches_and_stay_in_order(self):
        """A reordered batch would attach one cable's entities to another cable."""
        seen = []

        def linker(texts):
            seen.append(len(texts))
            return [[Mention(f"Q{text}", text, 0.99, "ORG")] for text in texts]

        documents = [{"doc_id": str(i), "text": str(i)} for i in range(1, 6)]
        rows = list(link_documents(documents, linker, "private", None, batch_size=2))

        assert seen == [2, 2, 1]
        assert [row["doc_id"] for row in rows] == ["1", "2", "3", "4", "5"]
        assert [row["entities"][0]["qid"] for row in rows] == ["Q1", "Q2", "Q3", "Q4", "Q5"]

    def test_the_text_is_prepared_before_it_reaches_the_model(self):
        """Cables arrive ALL CAPS with a routing preamble; linking them raw finds little."""
        shown = []

        def linker(texts):
            shown.extend(texts)
            return [[] for _ in texts]

        list(
            link_documents(
                [{"doc_id": "1", "text": "RAW"}],
                linker,
                "private",
                None,
                prepare_text=str.lower,
            )
        )

        assert shown == ["raw"]


class TestReconcile:
    """Lookups for the sources that arrive with a name or a code, not a QID."""

    def test_a_code_resolves_exactly(self):
        """The code is the join key, so there is nothing to rank."""

        def query(text):
            assert "wdt:P901" in text
            return [{"s": {"value": "http://www.wikidata.org/entity/Q39"}, "c": {"value": "SZ"}}]

        assert reconcile.by_code("fips", ["SZ"], query) == {"SZ": "Q39"}

    def test_an_unknown_scheme_names_the_known_ones(self):
        """A typo must not silently resolve nothing."""
        with pytest.raises(KeyError, match="known: fips, gwno, iso3"):
            reconcile.by_code("nonsense", ["x"], lambda q: [])

    def test_empty_input_makes_no_query(self):
        """Asking for nothing should not send a query with an empty VALUES clause."""

        def query(text):
            raise AssertionError("should not have queried")

        assert reconcile.by_code("fips", [], query) == {}
        assert reconcile.by_label([], query=query) == {}

    def test_a_label_lookup_returns_candidates_not_an_answer(self):
        """A label match is ambiguous by construction; the caller decides."""

        def query(text):
            return [
                {"s": {"value": "http://www.wikidata.org/entity/Q1"}, "l": {"value": "Smith"}},
                {"s": {"value": "http://www.wikidata.org/entity/Q2"}, "l": {"value": "Smith"}},
            ]

        assert reconcile.by_label(["Smith"], query=query) == {"Smith": ["Q1", "Q2"]}

    def test_a_type_constraint_reaches_the_query(self):
        """What stops a person resolving to a company of the same name."""
        seen = []

        def query(text):
            seen.append(text)
            return []

        reconcile.by_label(["Acme"], kinds=("Q5",), query=query)

        assert "wdt:P31" in seen[0]
        assert "wd:Q5" in seen[0]

    def test_names_are_batched(self):
        """Larger batches drew closed connections against the previous endpoint."""
        seen = []

        def query(text):
            seen.append(text)
            return []

        reconcile.by_label([f"name{i}" for i in range(130)], query=query, batch=60)

        assert len(seen) == 3


class TestNameLength:
    """A single given name identifies nobody."""

    def test_one_word_person_names_are_not_resolved(self):
        """Single names like Muhammad and Khalid each resolved to one entity, and bridged.

        That is how a cable mentioning a common name gets paired with an unrelated
        listing.
        """
        from osint_benchmark.link.tabular import names_in

        record = {"names": ["Muhammad", "Ali Abdullah Saleh"]}

        assert names_in(record, ("names",), min_words=2) == ["Ali Abdullah Saleh"]

    def test_single_word_names_are_fine_for_other_sources(self):
        """Countries and organisations are legitimately one word."""
        from osint_benchmark.link.tabular import names_in

        assert names_in({"country": "Afghanistan"}, ("country",)) == ["Afghanistan"]
