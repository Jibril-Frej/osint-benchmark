"""Unit tests for search-based reconciliation.

Both Wikidata calls are injected, so these run offline. What is tested is the verification
— which is the whole reason searching is safe on a sanctions list.
"""

from __future__ import annotations

from osint_benchmark.link import search


def _searcher(*hits):
    """Return a search stand-in yielding the given candidates for any name."""
    return lambda name: [dict(hit) for hit in hits]


def _facts(**entities):
    """Return a facts stand-in for the given QIDs."""

    def fetch(qids):
        return {qid: entities[qid] for qid in qids if qid in entities}

    return fetch


def _person(label, births=(), classes=("Q5",)):
    """Return the facts of one human."""
    return {"label": label, "classes": set(classes), "births": set(births)}


class TestKey:
    """Comparison has to survive how the two sides write a name."""

    def test_word_order_does_not_matter(self):
        """A listing writes "Lukashenka Dzmitry"; Wikidata writes "Dmitry Lukashenka"."""
        assert search.key("Lukashenka Dmitry") == search.key("Dmitry Lukashenka")

    def test_accents_and_punctuation_do_not_matter(self):
        """Transliteration is not consistent between a legal instrument and an encyclopaedia."""
        assert search.key("Ali Abdullah-Saleh") == search.key("Ali Abdullah Saleh")

    def test_different_names_stay_different(self):
        """The normalisation must not collapse everything into agreement."""
        assert search.key("Ali Saleh") != search.key("Ali Ahmed")


class TestResolve:
    """Search finds candidates; the checks decide."""

    def test_a_matching_person_resolves(self):
        """The ordinary path."""
        qid = search.resolve(
            ["Ali Abdullah Saleh"],
            kind="individual",
            searcher=_searcher({"id": "Q1", "label": "Ali Abdullah Saleh"}),
            fetch=_facts(Q1=_person("Ali Abdullah Saleh")),
        )

        assert qid == "Q1"

    def test_a_single_token_name_is_never_resolved(self):
        """A bare given name is a collision waiting to happen, and once was one."""
        assert (
            search.resolve(
                ["Muhammad"],
                searcher=_searcher({"id": "Q1", "label": "Muhammad"}),
                fetch=_facts(Q1=_person("Muhammad")),
            )
            is None
        )

    def test_a_plausible_neighbour_is_rejected(self):
        """The search is fuzzy: it returns near misses, ranked, and they look convincing."""
        qid = search.resolve(
            ["Ali Abdullah Saleh"],
            kind="individual",
            searcher=_searcher({"id": "Q2", "label": "Ali Abdullah Salih Ahmed"}),
            fetch=_facts(Q2=_person("Ali Abdullah Salih Ahmed")),
        )

        assert qid is None

    def test_a_person_may_not_resolve_to_a_company(self):
        """A sanctions list is exactly where that costs something."""
        qid = search.resolve(
            ["Acme Trading"],
            kind="individual",
            searcher=_searcher({"id": "Q3", "label": "Acme Trading"}),
            fetch=_facts(Q3={"label": "Acme Trading", "classes": {"Q783794"}, "births": set()}),
        )

        assert qid is None

    def test_a_disagreeing_birth_year_is_a_different_person(self):
        """The check that makes searching safe rather than merely productive."""
        qid = search.resolve(
            ["Ali Abdullah Saleh"],
            kind="individual",
            years=["1942"],
            searcher=_searcher({"id": "Q1", "label": "Ali Abdullah Saleh"}),
            fetch=_facts(Q1=_person("Ali Abdullah Saleh", births=["1975"])),
        )

        assert qid is None

    def test_an_agreeing_birth_year_confirms(self):
        """Two independent agreements are what a listing deserves."""
        qid = search.resolve(
            ["Ali Abdullah Saleh"],
            kind="individual",
            years=["1942"],
            searcher=_searcher({"id": "Q1", "label": "Ali Abdullah Saleh"}),
            fetch=_facts(Q1=_person("Ali Abdullah Saleh", births=["1942"])),
        )

        assert qid == "Q1"

    def test_a_missing_birth_year_on_either_side_is_not_a_disagreement(self):
        """Most listings state none, and refusing them would undo the whole gain."""
        qid = search.resolve(
            ["Ali Abdullah Saleh"],
            kind="individual",
            years=["1942"],
            searcher=_searcher({"id": "Q1", "label": "Ali Abdullah Saleh"}),
            fetch=_facts(Q1=_person("Ali Abdullah Saleh")),
        )

        assert qid == "Q1"

    def test_every_variant_is_tried(self):
        """A listing carries transliterations and only one need match."""
        seen = []

        def searcher(name):
            seen.append(name)
            return [{"id": "Q1", "label": "Dmitry Lukashenka"}] if "Dmitry" in name else []

        qid = search.resolve(
            ["Lukashenka Dzmitry Alexandrovich", "Dmitry Lukashenka"],
            kind="individual",
            searcher=searcher,
            fetch=_facts(Q1=_person("Dmitry Lukashenka")),
        )

        assert len(seen) == 2
        assert qid == "Q1"

    def test_nothing_found_resolves_to_nothing(self):
        """Most sanctioned individuals genuinely have no Wikidata entity."""
        assert search.resolve(["A Person"], searcher=_searcher(), fetch=_facts()) is None
