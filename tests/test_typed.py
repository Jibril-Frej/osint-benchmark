"""Unit tests for the typed question path: passages, the gold check, and item building.

The conditions tested here are the ones that replace the judge. A typed item's answer is
computed rather than written, so nothing downstream re-reads the documents to see whether it
is true — which makes each condition below the only thing standing between a wrong gold and
the release.
"""

from __future__ import annotations

from osint_benchmark.generate import association, passage, resolution, typed
from osint_benchmark.generate.association import Association
from osint_benchmark.generate.resolution import Resolution


class TestWrittenAs:
    """How a document writes a name, when it writes it at all."""

    def test_the_full_name_is_preferred_to_the_family_name(self):
        """Order decides whether an unambiguous mention looks ambiguous."""
        text = "A meeting with Max Petitpierre took place in Berne."

        assert passage.written_as("Max Petitpierre", text) == "Max Petitpierre"

    def test_the_family_name_is_found_when_the_full_name_is_not(self):
        """The case the resolution type exists for."""
        assert passage.written_as("Max Petitpierre", "Petitpierre replied the next day.") == (
            "Petitpierre"
        )

    def test_a_name_the_document_never_uses_is_not_a_mention(self):
        """A curated tag says a document concerns someone, not that it names them."""
        assert passage.written_as("Max Petitpierre", "The delegation left on Tuesday.") == ""


class TestLocate:
    """Curated links rewritten to the names the document actually uses."""

    def test_an_entity_absent_from_the_text_is_dropped(self):
        """Otherwise a question is built on a name the document does not use."""
        row = {
            "doc_id": "dodis:1",
            "entities": [
                {"qid": "Q1", "surface_form": "Max Petitpierre", "confidence": 1.0},
                {"qid": "Q2", "surface_form": "Walter Stucki", "confidence": 1.0},
            ],
        }

        located = passage.locate(row, "Petitpierre chaired the session.")

        assert [e["qid"] for e in located["entities"]] == ["Q1"]
        assert located["entities"][0]["surface_form"] == "Petitpierre"


class TestWindow:
    """The passage a phraser is shown."""

    def test_the_window_is_cut_around_the_mention(self):
        """The phraser writes about the situation, so it must be given the situation."""
        text = "x" * 1000 + "Petitpierre chaired it" + "y" * 1000

        around = passage.window(text, "Petitpierre", before=10, after=30)

        assert around.startswith("x" * 10 + "Petitpierre")
        assert len(around) == 40


class TestCheck:
    """Deciding which namesake a passage is about."""

    def _item(self, candidates=("Q1", "Q2")):
        """Return a resolution item over the given namesakes."""
        return Resolution(
            doc_id="cablegate:1",
            surface="Haraszti",
            qid="Q2",
            label="Emil Haraszti",
            candidates=candidates,
            rank=1,
            article_bytes=2986,
        )

    def test_the_namesake_whose_article_echoes_the_passage_wins(self):
        """Vocabulary decides identity where prominence cannot."""
        articles = {
            "Q1": "A musicologist who wrote about baroque opera and Hungarian folk melody.",
            "Q2": "A media-freedom envoy who reported on press intimidation and censorship.",
        }
        item = self._item()

        verdict = resolution.check(
            item, "raised press intimidation and censorship with the envoy", articles
        )

        assert verdict.verdict == "verified"

    def test_a_rival_that_fits_better_refutes_the_linker(self):
        """The failure this check exists for: the linker picked the obscure wrong one.

        The gold has to echo the passage a little, or the verdict is "unchecked" instead --
        a candidate whose article shares nothing with the passage is not evidence against
        the linker, it is an absence of evidence either way.
        """
        articles = {
            "Q1": "A media-freedom envoy who reported on press intimidation and censorship.",
            "Q2": "A musicologist who wrote about baroque opera and printing press ballads.",
        }
        item = self._item()

        verdict = resolution.check(
            item, "raised press intimidation and censorship with the envoy", articles
        )

        assert verdict.verdict == "refuted"
        assert verdict.rival == "Q1"

    def test_a_gold_with_no_article_is_unchecked_rather_than_believed(self):
        """An unchecked gold is what caused the problem; it must not pass by default."""
        item = self._item()

        verdict = resolution.check(item, "press intimidation", {"Q1": "A musicologist."})

        assert verdict.verdict == "unchecked"


class TestFromResolution:
    """The conditions between a resolution candidate and an item."""

    ARTICLES = {
        "Q1": "A musicologist who wrote about baroque opera.",
        "Q2": "A media-freedom envoy who reported on press intimidation and censorship.",
    }

    def _item(self):
        """Return one resolution candidate."""
        return Resolution(
            doc_id="cablegate:1",
            surface="Haraszti",
            qid="Q2",
            label="Emil Haraszti",
            candidates=("Q1", "Q2"),
            rank=1,
            article_bytes=2986,
        )

    def test_a_verified_mention_becomes_a_candidate(self):
        """The ordinary path."""
        texts = {"cablegate:1": "Haraszti raised press intimidation and censorship."}

        out = list(typed.from_resolution([self._item()], texts, self.ARTICLES))

        assert len(out) == 1
        assert out[0].answer == "Emil Haraszti"
        assert out[0].public_id == "enwiki:Q2"

    def test_a_document_that_spells_the_name_out_resolves_itself(self):
        """Then the public catalogue is not needed and the question needs one document."""
        texts = {"cablegate:1": "Emil Haraszti raised press intimidation and censorship."}

        assert list(typed.from_resolution([self._item()], texts, self.ARTICLES)) == []

    def test_a_refuted_gold_never_reaches_an_item(self):
        """Over 80% of raw candidates name the wrong person; this is the filter."""
        articles = {
            "Q1": "A media-freedom envoy who reported on press intimidation and censorship.",
            "Q2": "A musicologist who wrote about baroque opera.",
        }
        texts = {"cablegate:1": "Haraszti raised press intimidation and censorship."}

        assert list(typed.from_resolution([self._item()], texts, articles)) == []

    def test_the_phraser_is_never_shown_the_answer(self):
        """A model given the answer writes it into the question."""
        texts = {"cablegate:1": "Haraszti raised press intimidation and censorship."}
        candidate = next(iter(typed.from_resolution([self._item()], texts, self.ARTICLES)))

        assert "Emil" not in candidate.passage
        assert "Emil" not in str(candidate.facts)


class TestFromAssociation:
    """The conditions between an association candidate and an item."""

    LABELS = {"Q1": "Anna Meier", "Q2": "Beat Weber", "Q7": "Helvetic Society"}
    ARTICLES = {
        "Q1": "A Swiss trade negotiator.",
        "Q2": "A Swiss federal administrator.",
        "Q7": "A learned society founded in Schinznach.",
    }

    def _item(self):
        """Return one association candidate."""
        return Association(
            doc_id="cablegate:1",
            a="Q1",
            b="Q2",
            a_surface="Meier",
            b_surface="Weber",
            shared="Q7",
            predicate="member_of",
            degree=3,
        )

    def _texts(self, body="Meier and Weber agreed the wording over lunch."):
        """Return the private document text."""
        return {"cablegate:1": body}

    def test_a_pair_with_a_private_meeting_becomes_a_candidate(self):
        """The ordinary path."""
        out = list(
            typed.from_association([self._item()], self._texts(), self.LABELS, self.ARTICLES)
        )

        assert len(out) == 1
        assert (out[0].answer, out[0].public_id) == ("Helvetic Society", "enwiki:Q7")

    def test_a_document_naming_the_organisation_answers_itself(self):
        """Then the public record adds nothing and one document suffices."""
        texts = self._texts("Meier and Weber, both of the Helvetic Society, agreed the wording.")

        assert list(typed.from_association([self._item()], texts, self.LABELS, self.ARTICLES)) == []

    def test_a_pair_that_co_occurs_publicly_is_rejected(self):
        """If one article names the other person, the pair is public knowledge."""
        articles = {**self.ARTICLES, "Q1": "A Swiss trade negotiator who worked with Beat Weber."}

        assert (
            list(typed.from_association([self._item()], self._texts(), self.LABELS, articles)) == []
        )

    def test_a_pair_the_test_cannot_be_run_on_is_dropped_not_passed(self):
        """Passing by default would readmit exactly the publicly obvious pairs."""
        articles = {"Q7": self.ARTICLES["Q7"], "Q1": self.ARTICLES["Q1"]}

        assert (
            list(typed.from_association([self._item()], self._texts(), self.LABELS, articles)) == []
        )

    def test_an_answer_with_no_article_has_no_public_side(self):
        """The public evidence is the answer's article; without one there is none."""
        articles = {"Q1": self.ARTICLES["Q1"], "Q2": self.ARTICLES["Q2"]}

        assert (
            list(typed.from_association([self._item()], self._texts(), self.LABELS, articles)) == []
        )

    def test_the_phraser_is_never_shown_the_answer(self):
        """The names of the two people are shown; what they share is not."""
        candidate = next(
            iter(typed.from_association([self._item()], self._texts(), self.LABELS, self.ARTICLES))
        )

        assert "Helvetic" not in str(candidate.facts)
        assert "Helvetic" not in candidate.passage


class TestPhraseCandidates:
    """Turning candidates into items."""

    def _candidate(self):
        """Return one phrased-ready candidate."""
        return typed.Candidate(
            item_id="cablegate:1|Haraszti|Q2",
            question_type="resolution",
            answer="Emil Haraszti",
            gold_qid="Q2",
            private_id="cablegate:1",
            public_id="enwiki:Q2",
            passage="Haraszti raised press intimidation.",
            facts={"surface": "Haraszti", "bearers": "3"},
        )

    def test_an_item_cites_one_document_from_each_side(self):
        """Structural: an item citing one side cannot need both."""
        phraser = lambda prompt: '{"question": "Who raised press intimidation that week?"}'  # noqa: E731

        items = list(typed.phrase_candidates([self._candidate()], phraser, ("a desk officer",)))

        assert len(items) == 1
        assert items[0].two_sided
        assert items[0].answer == "Emil Haraszti"

    def test_the_computed_gold_survives_the_phrasing(self):
        """The model writes the question only; the answer comes from the graph."""
        phraser = lambda prompt: '{"question": "Who chaired it?", "answer": "somebody else"}'  # noqa: E731

        items = list(typed.phrase_candidates([self._candidate()], phraser, ("a desk officer",)))

        assert items[0].answer == "Emil Haraszti"

    def test_an_unparseable_reply_produces_no_item(self):
        """A malformed reply is a failure to write a question, not one to repair."""
        items = list(
            typed.phrase_candidates([self._candidate()], lambda p: "I cannot", ("an analyst",))
        )

        assert items == []


class TestRelationalPredicates:
    """The predicates the second Wikidata hop follows."""

    def test_employer_is_among_them(self):
        """The previous project named it and never fetched it, so it could not fire."""
        assert "employer" in association.RELATIONAL
