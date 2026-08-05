"""Unit tests for the ablation, the review page and the release freeze."""

from __future__ import annotations

import json

from osint_benchmark.generate.item import Evidence, Item, Necessity
from osint_benchmark.necessity import ablate
from osint_benchmark.release import freeze
from osint_benchmark.release.load import load_items
from osint_benchmark.review import page


def _item(item_id="c1|i1|Q1", **kw):
    """Return an item with one document on each side."""
    return Item(
        item_id=item_id,
        question_type="bridge",
        question=kw.pop("question", "Which body reviewed the matter?"),
        answer=kw.pop("answer", "the finance committee"),
        evidence=[
            Evidence(doc_id="c1", source="cablegate", side="private"),
            Evidence(doc_id="i1", source="parliament", side="public"),
        ],
        **kw,
    )


class TestSolved:
    """Whether the solver could answer under one condition."""

    def test_unanswerable_means_it_could_not(self):
        """The reply the prompt asks for when the evidence does not carry the answer."""
        assert not ablate.solved("q", "evidence", lambda p: "UNANSWERABLE")

    def test_an_empty_reply_counts_as_not_answered(self):
        """That is a truncated reasoning trace; scoring it as an answer scores deliberation."""
        assert not ablate.solved("q", "evidence", lambda p: "")

    def test_an_answer_counts_as_answered(self):
        """The condition succeeded, which means the question did not need what was withheld."""
        assert ablate.solved("q", "evidence", lambda p: "the finance committee")


class TestMeasure:
    """Three conditions, all recorded."""

    def test_a_question_needing_both_fails_every_condition(self):
        """The property the whole benchmark rests on."""
        necessity = ablate.measure(_item(), "priv", "pub", lambda p: "UNANSWERABLE")

        assert necessity.measured
        assert necessity.needs_both

    def test_a_closed_book_leak_is_visible_only_in_that_condition(self):
        """Seven of the previous project's questions passed both evidence conditions.

        "Who was Brazil's Foreign Minister during Lula's presidency" needs no evidence at
        all, and only the closed-book run saw it.
        """

        def solver(prompt):
            return "Celso Amorim" if ablate.NO_EVIDENCE in prompt else "UNANSWERABLE"

        necessity = ablate.measure(_item(), "priv", "pub", solver)

        assert necessity.closed_book
        assert not necessity.public_only
        assert not necessity.private_only
        assert not necessity.needs_both

    def test_necessity_is_recorded_never_used_to_drop(self):
        """A reviewer needs to see which condition succeeded."""
        items = list(ablate.measure_items([_item()], {}, lambda p: "an answer"))

        assert len(items) == 1
        assert not items[0].necessity.needs_both

    def test_the_control_catches_a_broken_solver(self):
        """A solver that cannot answer this makes every question look necessary.

        That is exactly what a 256-token ceiling did in the previous project.
        """
        assert not ablate.control(lambda p: "")
        assert ablate.control(lambda p: "green")


class TestMeasureItems:
    """What a run survives while measuring 135 questions."""

    def test_the_evidence_is_clipped_to_the_same_budget_as_step_six(self):
        """It was not, so step 7 sent whole cables and died on a 400 about token counts.

        Both stages read the same documents; disagreeing about how much of one fits means
        the second stage fails on exactly the documents the first one handled.
        """
        from osint_benchmark.generate.evidence import EVIDENCE_CHARS

        seen = []

        def solver(prompt):
            seen.append(len(prompt))
            return "UNANSWERABLE"

        ablate.solved("Which body?", "x" * (EVIDENCE_CHARS * 3), solver)

        assert max(seen) < EVIDENCE_CHARS * 2

    def test_one_unmeasurable_question_does_not_lose_the_others(self):
        """A single bad call threw away the measurement of 135 questions on job 13397."""
        from osint_benchmark.models.backend import ModelUnavailable

        calls = {"n": 0}

        def solver(prompt):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ModelUnavailable("400 Bad Request")
            return "UNANSWERABLE"

        items = [_item(item_id="a"), _item(item_id="b")]
        texts = {"d0": "private text", "d1": "public text"}

        measured = list(ablate.measure_items(items, texts, solver))

        assert len(measured) == 2
        assert not measured[0].necessity.measured
        assert measured[1].necessity.needs_both


class TestReviewPage:
    """One self-contained file: no server, no external asset."""

    def test_the_page_embeds_the_evidence_it_shows(self, tmp_path):
        """Nothing derived from the confidential corpora may leave the workstation."""
        html = page.render([_item()], {"c1": "the private text", "i1": "the public text"})

        assert "the private text" in html
        assert "the public text" in html
        assert "<script" in html and "src=" not in html
        assert "http://" not in html and "https://" not in html

    def test_evidence_is_escaped(self):
        """Corpus text is arbitrary and must not be able to close a tag."""
        html = page.render([_item()], {"c1": "<script>alert(1)</script>", "i1": "x"})

        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_necessity_flags_are_shown_when_measured(self):
        """Shown rather than filtered on: which condition succeeded is a judgement."""
        item = _item(necessity=Necessity(closed_book=True, public_only=False, private_only=False))

        assert "closed-book answerable" in page.render([item], {})

    def test_an_unmeasured_item_shows_no_necessity_flags(self):
        """Absent is not the same as passed."""
        assert "closed-book answerable" not in page.render([_item()], {})


class TestFreeze:
    """A release ships what the project authored and no corpus text."""

    def test_no_corpus_text_reaches_the_release(self, tmp_path):
        """Evidence is pointers, so nothing has to be redistributed."""
        freeze.freeze([_item()], {}, tmp_path)

        written = (tmp_path / "questions.jsonl").read_text()

        assert "the private text" not in written
        assert '"doc_id": "c1"' in written

    def test_the_datasheet_counts_what_is_there(self, tmp_path):
        """Written from the release, not by hand."""
        items = [
            _item(item_id="a", necessity=Necessity(False, False, False)),
            _item(item_id="b", necessity=Necessity(True, False, False)),
        ]

        sheet = freeze.freeze(items, {}, tmp_path)

        assert sheet["questions"] == 2
        assert sheet["necessity"]["measured"] == 2
        assert sheet["necessity"]["needs_both"] == 1
        assert sheet["necessity"]["answerable_closed_book"] == 1

    def test_the_release_carries_its_own_digest(self, tmp_path):
        """One hash over every question, so a copy can be checked."""
        sheet = freeze.freeze([_item()], {}, tmp_path)

        assert len(sheet["sha256"]) == 64
        assert json.loads((tmp_path / "datasheet.json").read_text())["sha256"] == sheet["sha256"]

    def test_items_round_trip_through_disk(self, tmp_path):
        """The review and release steps both read items back."""
        original = _item(necessity=Necessity(False, True, False), gates={"two_sided": True})
        freeze.freeze([original], {}, tmp_path)

        loaded = load_items(tmp_path / "questions.jsonl")

        assert len(loaded) == 1
        assert loaded[0].item_id == original.item_id
        assert loaded[0].necessity.public_only is True
        assert loaded[0].gates == {"two_sided": True}
        assert [e.side for e in loaded[0].evidence] == ["private", "public"]
