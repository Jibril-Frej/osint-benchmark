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


class TestWithoutKwarg:
    """ReFinED cannot build a tokenizer under a current transformers without this."""

    def test_the_offending_keyword_is_dropped_and_the_rest_survive(self):
        """Everything else the caller passes still has to arrive."""
        from osint_benchmark.link.refined import without_kwarg

        def load(path, use_fast=False, add_special_tokens=None):
            return (path, use_fast, add_special_tokens)

        patched = without_kwarg(load, "add_special_tokens")

        loaded = patched("roberta", use_fast=True, add_special_tokens=False)

        assert loaded == ("roberta", True, None)

    def test_it_is_absent_rather_than_none(self):
        """Passing None would fail the same way: transformers rejects the key, not a value."""
        from osint_benchmark.link.refined import without_kwarg

        seen = {}

        def load(**kwargs):
            seen.update(kwargs)

        without_kwarg(load, "add_special_tokens")(add_special_tokens=False, use_fast=True)

        assert "add_special_tokens" not in seen
        assert seen == {"use_fast": True}

    def test_a_patched_function_is_recognisable_as_patched(self):
        """So that applying the patch twice is harmless."""
        from osint_benchmark.link.refined import PATCHED, without_kwarg

        assert getattr(without_kwarg(len, "x"), PATCHED, False)


class TestInstallHint:
    """Two unrelated failures look identical from the outside."""

    def test_a_missing_dependency_gets_the_install_line(self):
        """The expected case: it is optional because it pulls torch."""
        from osint_benchmark.link.refined import install_hint

        assert "uv sync --extra link" in install_hint(ImportError("No module named 'refined'"))

    def test_nltks_import_hook_is_not_reported_as_a_missing_dependency(self):
        """It is installed and refuses to load, and the traceback names neither cause.

        Reporting this as absent sends the reader to reinstall something already there.
        """
        from osint_benchmark.link.refined import install_hint

        hint = install_hint(ImportError("Blocked import of regex from current working directory"))

        assert "NLTK_DISABLE_IMPORT_SECURITY=1" in hint
        assert "not installed" not in hint


class TestNarrativeBody:
    """What the linker is shown decides what it can find; a cable is not all narrative."""

    def test_the_routing_preamble_is_dropped(self):
        """It is full of capitalised tokens and about nothing."""
        cable = "VZCZCXRO1234\nRR RUEHWEB\nSUBJECT: X\n\n1. (C) The minister said so.\nSMITH"

        assert narrative_body(cable) == "1.  The minister said so."

    def test_classification_markers_are_dropped(self):
        """(SBU) is a marking, not a word in the sentence -- and the linker resolves it.

        SBU came back as an entity 1,248 times in a 21k-cable run.
        """
        assert "SBU" not in narrative_body("\n1. (SBU) The minister said so.\nSMITH")

    def test_handling_markers_on_their_own_line_are_dropped(self):
        """SIPDIS sits inside the narrative, so stripping the preamble does not catch it.

        It was the second most frequent entity of a whole run, ahead of every country
        but one.
        """
        body = "\n1. (C) One.\nSIPDIS\nNOFORN\n2. (C) Two.\nSMITH"
        cleaned = narrative_body(body)

        assert "SIPDIS" not in cleaned
        assert "NOFORN" not in cleaned
        assert "One." in cleaned and "Two." in cleaned

    def test_a_word_that_merely_looks_like_a_marker_survives(self):
        """The markers are matched as whole lines and as parenthesised codes, not anywhere.

        A cable discussing a secret is not a cable marked SECRET.
        """
        body = "\n1. (C) They kept it secret from the Council.\nSMITH"

        assert "secret from the Council" in narrative_body(body)

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


class TestMerge:
    """The two linkers miss different entities, so the union is bigger than either."""

    def test_entities_from_both_linkers_survive(self):
        """22 bridges came only from the matcher and 15 only from ReFinED; both count."""
        from osint_benchmark.link.merge import merge_entities

        model = [{"qid": "Q1", "surface_form": "Siemens", "confidence": 0.9}]
        titles = [{"qid": "Q2", "surface_form": "Rafidain Bank", "confidence": 1.0}]

        assert [e["qid"] for e in merge_entities(model, titles)] == ["Q1", "Q2"]

    def test_the_stronger_reading_of_a_shared_entity_wins(self):
        """Both found it; the surface form kept should be the confident one's."""
        from osint_benchmark.link.merge import merge_entities

        weak = [{"qid": "Q1", "surface_form": "UAC", "confidence": 0.4}]
        strong = [{"qid": "Q1", "surface_form": "United Aircraft Corporation", "confidence": 0.9}]

        merged = merge_entities(weak, strong)

        assert len(merged) == 1
        assert merged[0]["surface_form"] == "United Aircraft Corporation"

    def test_the_result_is_ordered_so_two_runs_agree(self):
        """Link output is compared between runs; dict order is not a guarantee."""
        from osint_benchmark.link.merge import merge_entities

        a = [{"qid": "Q9", "surface_form": "b", "confidence": 1.0}]
        b = [{"qid": "Q3", "surface_form": "a", "confidence": 1.0}]

        assert [e["qid"] for e in merge_entities(a, b)] == ["Q3", "Q9"]

    def test_documents_the_other_linker_said_nothing_about_are_untouched(self):
        """The matcher finds something in 0.8% of cables; the rest must pass through."""
        from osint_benchmark.link.merge import merge_rows

        rows = [{"doc_id": "1", "side": "private", "entities": []}]

        assert list(merge_rows(rows, {})) == rows

    def test_merging_does_not_mutate_the_row_it_was_given(self):
        """The rows are streamed from one generator into another; aliasing would corrupt."""
        from osint_benchmark.link.merge import merge_rows

        original = {"doc_id": "1", "side": "private", "entities": []}
        extra = {"1": [{"qid": "Q1", "surface_form": "x", "confidence": 1.0}]}

        merged = next(iter(merge_rows([original], extra)))

        assert len(merged["entities"]) == 1
        assert original["entities"] == []


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
