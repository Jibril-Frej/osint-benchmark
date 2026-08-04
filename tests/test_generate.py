"""Unit tests for question generation, its gates, and the model plumbing.

Every model here is a stub, which is the property that makes step 6 developable without a
GPU: the phraser and judge are injected, so what is tested is the pipeline's handling of
what a model returns rather than the model.
"""

from __future__ import annotations

from osint_benchmark.generate import emit, gates, phrase
from osint_benchmark.generate.item import Evidence, Item
from osint_benchmark.models.backend import agree, first_word, strip_reasoning


def _item(
    question="Which ministry did the delegation approach?",
    answer="the interior ministry",
    sides=("private", "public"),
    **kw,
):
    """Return an item with evidence on the given sides."""
    return Item(
        item_id=kw.pop("item_id", "c1|i1|Q1"),
        question_type="bridge",
        question=question,
        answer=answer,
        evidence=[Evidence(doc_id=f"d{i}", source="s", side=s) for i, s in enumerate(sides)],
        **kw,
    )


class TestGates:
    """Each gate exists because something got through without it."""

    def test_an_item_citing_one_side_cannot_need_both(self):
        """Structural: whatever the ablations later say."""
        assert not gates.two_sided(_item(sides=("private",)))
        assert gates.two_sided(_item())

    def test_a_question_containing_its_answer_measures_nothing(self):
        """The leak that makes a two-hop question a zero-hop one."""
        leaked = _item(question="Was the interior ministry the one approached?")

        assert not gates.answer_not_in_question(leaked)
        assert gates.answer_not_in_question(_item())

    def test_the_answer_check_ignores_punctuation_and_case(self):
        """A substring check missed an acronym against the name it stood for."""
        leaked = _item(question="Did they approach The Interior Ministry, or another?")

        assert not gates.answer_not_in_question(leaked)

    def test_source_attribution_turns_two_hops_into_one(self):
        """'The cable states' tells a solver exactly where to look."""
        assert not gates.no_source_attribution(_item(question="What does the cable say about X?"))
        assert gates.no_source_attribution(_item())

    def test_a_bare_attribute_is_trivia_not_analysis(self):
        """44 of the previous project's first 251 questions were attribute lookups."""
        assert not gates.not_a_bare_attribute(_item(question="What is the headquarters of X?"))
        assert gates.not_a_bare_attribute(_item())

    def test_an_empty_answer_is_a_parsing_failure(self):
        """Not an answer wearing an answer's clothes."""
        assert not gates.answer_is_substantive(_item(answer=" "))

    def test_run_returns_every_gate(self):
        """One suite, applied whole. A type cannot be given its own subset."""
        assert set(gates.run(_item())) == set(gates.GATES)


class TestEmit:
    """Candidate builders have no path to the file except through the gates."""

    def test_only_items_clearing_every_gate_are_accepted(self, tmp_path):
        """And the rejects are kept, so the accepted set is interpretable."""
        good = _item(item_id="good")
        bad = _item(item_id="bad", question="What does the cable say?")
        provenance = _provenance()

        accepted, rejected = emit.emit(
            [good, bad], tmp_path / "a.jsonl", tmp_path / "r.jsonl", provenance
        )

        assert (accepted, rejected) == (1, 1)
        assert "good" in (tmp_path / "a.jsonl").read_text()
        assert "bad" in (tmp_path / "r.jsonl").read_text()

    def test_gate_outcomes_are_recorded_on_the_item(self, tmp_path):
        """A reviewer needs to see which check failed, not just that one did."""
        bad = _item(question="What does the cable say?")

        emit.emit([bad], tmp_path / "a.jsonl", tmp_path / "r.jsonl", _provenance())

        assert bad.gates["no_source_attribution"] is False
        assert bad.gates["two_sided"] is True

    def test_a_template_that_survived_fails_the_run_not_the_item(self):
        """41 questions opening 'How many people were killed in' is a failed run."""
        items = [
            _item(question="How many people were killed in the raid?", item_id=f"i{i}")
            for i in range(10)
        ]

        alarms = emit.run_alarms(items)

        assert any("open with" in alarm for alarm in alarms)

    def test_a_varied_run_raises_no_opening_alarm(self):
        """The alarm has to be quiet when the run is fine."""
        items = [
            _item(question=f"Which body reviewed matter {i}?", item_id=f"i{i}") for i in range(30)
        ]

        assert not any("open with" in a for a in emit.run_alarms(items))


class TestModelPlumbing:
    """What the pipeline does with what a model returns."""

    def test_a_reasoning_trace_is_not_the_answer(self):
        """Reading it as one turns a verdict into a paragraph of deliberation."""
        assert strip_reasoning("<think>weighing it up</think>SUPPORTED") == "SUPPORTED"

    def test_an_unclosed_trace_is_a_truncation_not_an_answer(self):
        """The reply hit its ceiling mid-thought.

        The previous project scored these as failures to answer, which silently marked
        every question necessary.
        """
        assert strip_reasoning("<think>still weighing it up and the tokens ran") == ""

    def test_a_verdict_is_read_past_the_preamble(self):
        """Models add preamble however firmly they are told not to."""
        assert first_word("Sure! SUPPORTED, because...", ("SUPPORTED", "UNSUPPORTED")) == (
            "supported"
        )

    def test_a_word_that_is_not_a_permitted_verdict_is_a_non_answer(self):
        """Not a new verdict."""
        assert first_word("MAYBE", ("SUPPORTED", "UNSUPPORTED")) == ""

    def test_unanimous_replies_are_accepted(self):
        """Repeat-and-agree: n identical verdicts."""
        assert agree(lambda p: "SUPPORTED", "p", 3, lambda r: first_word(r, ("SUPPORTED",))) == (
            "supported"
        )

    def test_a_split_verdict_is_no_verdict(self):
        """A verdict nobody should act on."""
        replies = iter(["SUPPORTED", "UNSUPPORTED", "SUPPORTED"])

        result = agree(
            lambda p: next(replies), "p", 3, lambda r: first_word(r, ("SUPPORTED", "UNSUPPORTED"))
        )

        assert result == ""


class TestDraft:
    """A malformed reply is a failure to produce a question, not one to repair."""

    def test_json_is_extracted_from_a_chatty_reply(self):
        """Models wrap JSON in prose."""
        reply = 'Here you go:\n{"question": "Q?", "answer": "A", "reasoning": "R"}\nHope that helps'

        drafted = phrase.draft(
            {"private_id": "c", "public_id": "i", "qid": "Q1"},
            "priv",
            "pub",
            "Bridge",
            lambda p: reply,
        )

        assert drafted["question"] == "Q?"

    def test_a_reply_without_json_yields_nothing(self):
        """Rather than a question with empty fields."""
        assert (
            phrase.draft(
                {"private_id": "c", "public_id": "i", "qid": "Q1"},
                "priv",
                "pub",
                "B",
                lambda p: "I cannot do that",
            )
            == {}
        )

    def test_a_draft_missing_an_answer_yields_nothing(self):
        """Half a draft is not a question."""
        assert (
            phrase.draft(
                {"private_id": "c", "public_id": "i", "qid": "Q1"},
                "priv",
                "pub",
                "B",
                lambda p: '{"question": "Q?"}',
            )
            == {}
        )

    def test_the_asker_is_stable_for_an_item(self):
        """A rerun must not reword the question and change the fingerprint."""
        assert phrase.asker_for("c1|i1|Q1") == phrase.asker_for("c1|i1|Q1")

    def test_different_items_get_different_askers(self):
        """Varying the consumer is the defence against one template."""
        askers = {phrase.asker_for(f"c{i}|i|Q") for i in range(40)}

        assert len(askers) > 1


class TestBuildItems:
    """Drafting, verifying, and refusing to build from half a pair."""

    def test_a_pair_with_missing_evidence_is_skipped(self):
        """A question written from one document cannot need two."""
        pairs = [{"private_id": "c1", "public_id": "i1", "qid": "Q1"}]

        built = list(phrase.build_items(pairs, {"c1": "text"}, {}, lambda p: "", lambda p: ""))

        assert built == []

    def test_an_unverified_answer_is_not_kept(self):
        """What replaces the correctness check a computed answer got for free."""
        pairs = [{"private_id": "c1", "public_id": "i1", "qid": "Q1"}]
        texts = {"c1": "private text", "i1": "public text"}
        drafted = '{"question": "Which body?", "answer": "the council"}'

        built = list(
            phrase.build_items(pairs, texts, {}, lambda p: drafted, lambda p: "UNSUPPORTED", 1)
        )

        assert built == []

    def test_a_verified_answer_becomes_an_item(self):
        """The ordinary path."""
        pairs = [{"private_id": "c1", "public_id": "i1", "qid": "Q1"}]
        texts = {"c1": "private text", "i1": "public text"}
        drafted = '{"question": "Which body?", "answer": "the council", "reasoning": "both"}'

        built = list(
            phrase.build_items(
                pairs, texts, {"Q1": "Council"}, lambda p: drafted, lambda p: "SUPPORTED", 1
            )
        )

        assert len(built) == 1
        assert built[0].two_sided
        assert built[0].provenance["bridge_qid"] == "Q1"


def _provenance():
    """Return a minimal sound provenance."""
    from osint_benchmark.artifacts import Provenance

    return Provenance(source="test", source_fields=("a",), kept={"a": "b"}, kind="derived")


class TestSourceAttributionAfterTheRealRun:
    """The phrasings a real model actually produced."""

    def test_as_described_in_the_two_documents_is_caught(self):
        """This got through the first version of the denylist and was accepted."""
        leaked = _item(
            question="What is the connection between Cameroon and Canada as described in "
            "the two documents?"
        )

        assert not gates.no_source_attribution(leaked)

    def test_other_ways_of_pointing_at_the_evidence_are_caught(self):
        """The reference matters, not the particular wording."""
        for phrasing in (
            "What does the report say about X?",
            "According to the record, what happened?",
            "What is mentioned in both documents about X?",
            "As stated in the article, who resigned?",
        ):
            assert not gates.no_source_attribution(_item(question=phrasing)), phrasing

    def test_an_ordinary_question_still_passes(self):
        """The gate must not reject everything that mentions a noun."""
        assert gates.no_source_attribution(
            _item(question="Which ministry approved the transfer in March 2006?")
        )
