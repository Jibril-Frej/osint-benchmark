"""Unit tests for the resolution question type.

Each condition removes a way the question could be answered without the private document.
A test that still passed with the condition removed would be testing nothing, so every one
below names what it protects.
"""

from __future__ import annotations

from osint_benchmark.generate import resolution

LABELS = {
    "Q1": "Hosni Mubarak",
    "Q2": "Alaa Mubarak",
    "Q3": "Gamal Mubarak",
    "Q9": "Solitary Name",
}
PEOPLE = {"Q1", "Q2", "Q3", "Q9"}
# Hosni is the famous one; the other two are minor figures.
ARTICLES = {"Q1": 90000, "Q2": 1200, "Q3": 2500, "Q9": 500}


def _links(*entities, doc_id="cablegate:1"):
    """Return one link row naming the given entities."""
    return [
        {
            "doc_id": doc_id,
            "entities": [
                {"qid": qid, "surface_form": surface, "confidence": conf}
                for qid, surface, conf in entities
            ],
        }
    ]


def _build(*entities, articles=None):
    """Run the builder over one document."""
    return list(resolution.build(_links(*entities), LABELS, PEOPLE, articles or ARTICLES))


class TestFamilyNames:
    """Ambiguity lives in the family name."""

    def test_people_are_indexed_by_family_name(self):
        """Three Mubaraks under one token is what makes the mention ambiguous."""
        index = resolution.family_names(LABELS)

        assert index["mubarak"] == {"Q1", "Q2", "Q3"}

    def test_a_one_word_label_indexes_nothing(self):
        """There is no family name to be ambiguous about."""
        assert "solitary" not in resolution.family_names({"Q9": "Solitary"})


class TestBuild:
    """Every condition closes a route to answering without the private document."""

    def test_a_bare_surname_naming_a_minor_bearer_is_a_candidate(self):
        """The ordinary path: the catalogue cannot choose, the document can."""
        items = _build(("Q2", "Mubarak", 0.99))

        assert len(items) == 1
        assert (items[0].qid, items[0].label) == ("Q2", "Alaa Mubarak")
        assert items[0].rank == 2

    def test_the_most_prominent_bearer_is_not_a_question(self):
        """ "Mubarak" collides with eight entities and every solver answers Hosni.

        Catalogue ambiguity is not difficulty when prior fame settles it.
        """
        assert _build(("Q1", "Mubarak", 0.99)) == []

    def test_a_famous_runner_up_is_still_famous(self):
        """The second most prominent Bush is still George W. Bush.

        Being less famous than a namesake is not the same as being obscure.
        """
        articles = {"Q1": 90000, "Q2": 60000, "Q3": 2500}

        assert _build(("Q2", "Mubarak", 0.99), articles=articles) == []

    def test_a_full_name_mention_needs_no_resolving(self):
        """The document that writes the whole name is not being ambiguous."""
        assert _build(("Q2", "Alaa Mubarak", 0.99)) == []

    def test_a_name_only_one_person_bears_is_not_ambiguous(self):
        """With a single bearer the public catalogue answers unaided."""
        labels = {"Q2": "Alaa Mubarak"}
        items = list(
            resolution.build(_links(("Q2", "Mubarak", 0.99)), labels, {"Q2"}, {"Q2": 1200})
        )

        assert items == []

    def test_an_entity_with_no_article_is_not_answerable(self):
        """A question whose answer nobody has written about has no public side."""
        assert _build(("Q2", "Mubarak", 0.99), articles={"Q1": 90000, "Q3": 2500}) == []

    def test_a_short_surface_is_noise_not_a_mention(self):
        """Three characters is an initialism or an OCR artefact."""
        assert _build(("Q2", "Mub", 0.99)) == []

    def test_an_unsure_mention_may_not_anchor_a_resolution(self):
        """This type asserts *which* person is meant, so a mislink makes the answer wrong."""
        assert _build(("Q2", "Mubarak", 0.5)) == []

    def test_the_candidates_are_ranked_by_prominence(self):
        """A reviewer needs to see what the solver would have had to choose between."""
        items = _build(("Q2", "Mubarak", 0.99))

        assert items[0].candidates == ("Q1", "Q3", "Q2")
