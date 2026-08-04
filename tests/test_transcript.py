"""Unit tests for the model-call transcript.

It exists because a run made 50 requests, accepted zero questions, and left no way to tell
whether the model returned unparseable JSON, refused, or answered fine and the parsing was
wrong.
"""

from __future__ import annotations

import json

from osint_benchmark.models import transcript


class TestTranscript:
    """Off unless asked for; complete when on."""

    def test_nothing_is_written_unless_it_is_switched_on(self, tmp_path, monkeypatch):
        """Prompts carry corpus text and replies carry gold answers.

        Neither should be written anywhere by accident, so this is opt-in.
        """
        monkeypatch.delenv("OSINT_TRANSCRIPT", raising=False)
        wrapped = transcript.transcribed(lambda p: "reply", "judge")

        assert wrapped("prompt") == "reply"
        assert list(tmp_path.iterdir()) == []

    def test_the_prompt_and_the_raw_reply_are_both_kept(self, tmp_path, monkeypatch):
        """Either alone is useless for working out why a run produced nothing."""
        path = tmp_path / "calls.jsonl"
        monkeypatch.setenv("OSINT_TRANSCRIPT", str(path))

        transcript.transcribed(lambda p: "SUPPORTED", "judge")("is it supported?")

        entry = json.loads(path.read_text().strip())
        assert entry["role"] == "judge"
        assert entry["prompt"] == "is it supported?"
        assert entry["reply"] == "SUPPORTED"

    def test_the_wrapper_returns_what_the_model_returned(self, tmp_path, monkeypatch):
        """Recording must not change behaviour."""
        monkeypatch.setenv("OSINT_TRANSCRIPT", str(tmp_path / "c.jsonl"))

        assert transcript.transcribed(lambda p: "x", "solver")("p") == "x"

    def test_calls_accumulate_rather_than_overwrite(self, tmp_path, monkeypatch):
        """A run that dies mid-way should leave the calls it managed to make."""
        path = tmp_path / "calls.jsonl"
        monkeypatch.setenv("OSINT_TRANSCRIPT", str(path))
        wrapped = transcript.transcribed(lambda p: p.upper(), "phraser")

        wrapped("one")
        wrapped("two")

        assert [json.loads(x)["reply"] for x in path.read_text().splitlines()] == ["ONE", "TWO"]

    def test_an_empty_reply_is_recorded_not_skipped(self, tmp_path, monkeypatch):
        """The most informative case: the model was cut off mid-thought.

        The caller will treat it as a failure to answer, and a transcript that omitted it
        would make that look like a call that never happened.
        """
        path = tmp_path / "calls.jsonl"
        monkeypatch.setenv("OSINT_TRANSCRIPT", str(path))

        transcript.transcribed(lambda p: "", "solver")("p")

        assert json.loads(path.read_text().strip())["reply_chars"] == 0
