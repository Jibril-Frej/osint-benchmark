"""A scripted model, for running the pipeline without serving anything.

Not a model and not pretending to be one: it returns fixed, obviously-synthetic replies so
the wiring of steps 6 and 7 can be exercised on a machine with no GPU. Anything built with
it is labelled ``stub`` in the item's provenance and in the release datasheet, because a
question written by this is not a question.

Its purpose is to answer "does the chain run and produce output at every step?" -- which is
a different question from "are the questions any good", and one worth being able to answer
without a GPU.
"""

from __future__ import annotations

import hashlib

from osint_benchmark.models.backend import Complete

MARKER = "STUB"


def phraser() -> Complete:
    """Return a stand-in phraser that drafts a syntactically valid question."""

    def complete(prompt: str) -> str:
        """Return a draft whose wording varies with the prompt but says nothing."""
        seed = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        return (
            '{"question": "Which public body was named in connection with this matter '
            f'({seed})?", "answer": "{MARKER} answer {seed}", '
            '"reasoning": "STUB: no model was used to write this."}'
        )

    return complete


def judge(verdict: str = "SUPPORTED") -> Complete:
    """Return a stand-in judge that always returns the same verdict."""
    return lambda prompt: verdict


def solver(answerable: bool = False) -> Complete:
    """Return a stand-in solver.

    Defaults to answering nothing, so a stub run reports every question as needing both
    sides. That is an artefact of the stub, not a measurement, which is why the datasheet
    records the model that produced it.
    """
    return lambda prompt: "a stub answer" if answerable else "UNANSWERABLE"
