"""Measure whether a question really needs both sides.

Re-solve each finished question three ways — closed-book, public evidence only, private
evidence only — and a question that survives is one no single source could answer.

A separate step because it re-reads finished questions and changes nothing about them, so
it never has to run in the same job that wrote them.

All three conditions are kept, not the two the necessity claim strictly needs. The previous
project's numbers are the argument: seven of its questions were answerable closed-book —
"Who was Brazil's Foreign Minister during Lula's presidency" — and all seven passed *both*
evidence conditions. Only the closed-book run saw them.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from osint_benchmark.generate.item import Item, Necessity
from osint_benchmark.models import prompts
from osint_benchmark.models.backend import Complete

UNANSWERABLE = "unanswerable"
NO_EVIDENCE = "(no evidence provided)"


def solved(question: str, evidence: str, solver: Complete) -> bool:
    """Return whether the solver answered from this evidence alone.

    An empty reply counts as *not* answered — that is a truncated reasoning trace, and the
    alternative is to score deliberation as an answer.
    """
    reply = solver(prompts.render("necessity_solve", question=question, evidence=evidence))
    stripped = reply.strip().lower()
    return bool(stripped) and not stripped.startswith(UNANSWERABLE)


def measure(item: Item, private_text: str, public_text: str, solver: Complete) -> Necessity:
    """Return the three ablation outcomes for one item.

    Each is True when the solver *could* answer under that condition — that is, when the
    question fails to need what was withheld.
    """
    return Necessity(
        closed_book=solved(item.question, NO_EVIDENCE, solver),
        public_only=solved(item.question, public_text, solver),
        private_only=solved(item.question, private_text, solver),
    )


def control(solver: Complete) -> bool:
    """Return whether the solver can answer a question whose evidence plainly contains it.

    Run this before trusting any necessity number. A solver that cannot answer *this* is
    broken, and a broken solver makes every question look perfectly necessary — which is
    precisely the failure a token ceiling caused in the previous project.
    """
    return solved(
        "What colour is the sky described as in the evidence?",
        "The report notes that the sky was recorded as green throughout the observation.",
        solver,
    )


def measure_items(items: Iterable[Item], texts: dict[str, str], solver: Complete) -> Iterator[Item]:
    """Yield each item with its necessity measured.

    The outcome is recorded, never used to drop the item. Which condition succeeded says
    something different in each case, and a reviewer needs to see it — the previous project
    kept 52 of 404 questions that failed at least one condition, deliberately.
    """
    for item in items:
        private_text = " ".join(texts.get(e.doc_id, "") for e in item.private_evidence)
        public_text = " ".join(texts.get(e.doc_id, "") for e in item.public_evidence)
        item.necessity = measure(item, private_text, public_text, solver)
        yield item
