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

import sys
from collections.abc import Iterable, Iterator

from osint_benchmark.generate.evidence import clip
from osint_benchmark.generate.item import Item, Necessity
from osint_benchmark.models import prompts
from osint_benchmark.models.backend import Complete, ModelUnavailable, agree, first_word

UNANSWERABLE = "unanswerable"
NO_EVIDENCE = "(no evidence provided)"


def answered(question: str, evidence: str, solver: Complete) -> str:
    """Return the solver's answer from this evidence alone, or empty if it gave none.

    An empty reply counts as no answer — that is a truncated reasoning trace, and the
    alternative is to score deliberation as an answer.
    """
    # Clipped to the same budget step 6 uses. It was not, and step 7 therefore sent whole
    # cables: a run wrote 135 questions and then died measuring the first one whose cable
    # was long, on an HTTP 400 about token counts. The two stages read the same documents
    # and must agree about how much of one fits.
    reply = solver(prompts.render("necessity_solve", question=question, evidence=clip(evidence)))
    stripped = reply.strip()
    return "" if not stripped or stripped.lower().startswith(UNANSWERABLE) else stripped


def solved(
    question: str,
    gold: str,
    evidence: str,
    solver: Complete,
    judge: Complete | None = None,
    samples: int = 1,
) -> bool:
    """Return whether this evidence alone yields the *right* answer.

    Producing an answer is not the same as knowing one, and measuring the first instead of
    the second is what made every necessity figure this project has reported wrong — in
    whichever direction the solver's temperament pointed.

    A cautious prompt had the solver refuse 79% of everything, so 41% of questions looked
    to need both documents; a human check found four of six of those were answerable from
    one side. Rewriting the prompt to make it try produced the mirror image: it answered
    97% and *nothing* looked necessary. Neither number described the questions. Both
    described the solver.

    So the answer is compared against the one the question was built with. An adversarial
    solver is now the right kind — let it try its hardest, then check whether it was right.
    Without a ``judge`` this falls back to the old behaviour and any answer counts, which
    is only correct for the stub.
    """
    candidate = answered(question, evidence, solver)
    if not candidate or judge is None:
        return bool(candidate)
    prompt = prompts.render(
        "necessity_equivalent", question=question, gold=gold, candidate=clip(candidate, 2000)
    )
    return agree(judge, prompt, samples, lambda r: first_word(r, ("MATCH", "DIFFERENT"))) == "match"


def measure(
    item: Item,
    private_text: str,
    public_text: str,
    solver: Complete,
    judge: Complete | None = None,
    samples: int = 1,
) -> Necessity:
    """Return the three ablation outcomes for one item.

    Each is True when that condition *did* produce the right answer — that is, when the
    question fails to need what was withheld.
    """
    return Necessity(
        closed_book=solved(item.question, item.answer, NO_EVIDENCE, solver, judge, samples),
        public_only=solved(item.question, item.answer, public_text, solver, judge, samples),
        private_only=solved(item.question, item.answer, private_text, solver, judge, samples),
    )


def control(solver: Complete) -> bool:
    """Return whether the solver can answer a question whose evidence plainly contains it.

    Run this before trusting any necessity number. A solver that cannot answer *this* is
    broken, and a broken solver makes every question look perfectly necessary — which is
    precisely the failure a token ceiling caused in the previous project.
    """
    return bool(
        answered(
            "What colour is the sky described as in the evidence?",
            "The report notes that the sky was recorded as green throughout the observation.",
            solver,
        )
    )


def measure_items(
    items: Iterable[Item],
    texts: dict[str, str],
    solver: Complete,
    judge: Complete | None = None,
    samples: int = 1,
) -> Iterator[Item]:
    """Yield each item with its necessity measured.

    The outcome is recorded, never used to drop the item. Which condition succeeded says
    something different in each case, and a reviewer needs to see it — the previous project
    kept 52 of 404 questions that failed at least one condition, deliberately.
    """
    for item in items:
        private_text = " ".join(texts.get(e.doc_id, "") for e in item.private_evidence)
        public_text = " ".join(texts.get(e.doc_id, "") for e in item.public_evidence)
        # One unmeasurable question costs its own measurement and nothing else. Step 6
        # already worked this way; step 7 did not, so a single bad call threw away the
        # measurement of 135 questions that had taken four hours to write.
        try:
            item.necessity = measure(item, private_text, public_text, solver, judge, samples)
        except ModelUnavailable as exc:
            print(f"  unmeasured {item.item_id}: {exc}", file=sys.stderr)
        yield item
