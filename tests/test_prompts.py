"""Unit tests for prompt loading and model settings.

The rule these enforce is that every prompt lives in ``prompts/`` and every model
parameter in ``config/models.toml``, so both can be reviewed as a set. The moment a prompt
is inlined in a ``.py`` file it stops being reviewable, which is the failure mode this
guards.
"""

from __future__ import annotations

import pytest

from osint_benchmark import paths
from osint_benchmark.models import prompts, settings


class TestRender:
    """Placeholders and supplied values must correspond exactly."""

    def test_a_prompt_renders_with_its_declared_values(self, tmp_path):
        """The ordinary path: every slot filled."""
        (tmp_path / "p.md").write_text("Question: {question}\nEvidence: {evidence}")

        out = prompts.render("p", tmp_path, question="who?", evidence="a document")

        assert out == "Question: who?\nEvidence: a document"

    def test_a_missing_value_is_an_error_not_a_literal_brace(self, tmp_path):
        """Unfilled, `{evidence}` would be sent to the model as those nine characters."""
        (tmp_path / "p.md").write_text("{question} {evidence}")

        with pytest.raises(KeyError, match="missing evidence"):
            prompts.render("p", tmp_path, question="who?")

    def test_a_surplus_value_is_an_error(self, tmp_path):
        """The caller believes it is supplying context the prompt never asks for."""
        (tmp_path / "p.md").write_text("{question}")

        with pytest.raises(KeyError, match="unexpected evidence"):
            prompts.render("p", tmp_path, question="who?", evidence="ignored")

    def test_doubled_braces_reach_the_model_literally(self, tmp_path):
        """JSON examples in a prompt need literal braces."""
        (tmp_path / "p.md").write_text('Reply as {{"answer": "..."}} about {question}')

        assert prompts.render("p", tmp_path, question="x") == 'Reply as {"answer": "..."} about x'

    def test_a_missing_prompt_names_the_directory(self, tmp_path):
        """A typo should say where it looked."""
        with pytest.raises(FileNotFoundError, match="no prompt 'nope'"):
            prompts.read("nope", tmp_path)


class TestTheCommittedPrompts:
    """The real prompt files, as they will be sent."""

    def test_every_prompt_loads_and_declares_placeholders(self):
        """A prompt with no slots is either a mistake or not really a prompt."""
        names = prompts.names()

        assert names, "no prompts found"
        for name in names:
            assert prompts.placeholders(name), f"{name} declares no placeholders"

    def test_no_prompt_text_is_inlined_in_python(self):
        """Prompt text in a .py file is text nobody will review with the others.

        Checks for the instruction-shaped phrasing the prompts use, not for prose in
        general: docstrings legitimately quote and explain them.
        """
        offenders = []
        for path in (paths.ROOT / "osint_benchmark").rglob("*.py"):
            body = path.read_text(encoding="utf-8")
            for marker in ("Reply with exactly one word", "Reply as JSON with exactly"):
                if marker in body:
                    offenders.append(f"{path.name}: {marker!r}")
        assert not offenders, "prompt text found in Python: " + "; ".join(offenders)


class TestSettings:
    """Parameters come from config/, and a missing one is reported."""

    def test_every_role_the_prompts_imply_is_configured(self):
        """phraser, judge and solver each need settings to run at all."""
        for role in ("phraser", "judge", "solver"):
            assert settings.load(role).model

    def test_repeat_and_agree_needs_sampling_on(self):
        """The judge asks n times and keeps only a unanimous verdict.

        At temperature 0 the repeats are identical, so unanimity is guaranteed and the
        check measures nothing.
        """
        judge = settings.load("judge")

        assert judge.samples > 1
        assert judge.temperature > 0

    def test_reasoning_models_have_room_for_their_trace(self):
        """QwQ opens with a <think> block.

        At the judge's usual 256-token ceiling every reply truncates mid-reasoning and
        scores as a failure to answer, which silently marked every question necessary in
        the previous project.
        """
        for role in ("phraser", "solver"):
            assert settings.load(role).max_tokens >= 2048

    def test_the_judge_is_a_different_family_from_the_phraser(self):
        """Scoring your own output is self-enhancement bias, not evaluation."""
        assert settings.load("judge").model != settings.load("phraser").model

    def test_an_unknown_role_lists_the_known_ones(self):
        """A typo should not read as an unconfigured model."""
        with pytest.raises(KeyError, match="known: judge, phraser, solver"):
            settings.load("noone")

    def test_a_missing_setting_is_reported_not_defaulted(self, tmp_path):
        """A plausible default silently changes what a run did."""
        (tmp_path / "models.toml").write_text('[judge]\nmodel = "x"\nendpoint = ""\n')

        with pytest.raises(KeyError, match="missing: max_tokens, samples, temperature"):
            settings.load("judge", tmp_path / "models.toml")

    def test_the_environment_can_override_the_endpoint(self, monkeypatch):
        """A cluster job points at whatever it just started serving."""
        monkeypatch.setenv("OSINT_MODEL_ENDPOINT", "http://127.0.0.1:8080")

        assert settings.load("judge").endpoint == "http://127.0.0.1:8080"
