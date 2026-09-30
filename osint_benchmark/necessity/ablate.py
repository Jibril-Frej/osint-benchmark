"""Measure whether a question really needs both sides.

Re-solve each finished question three ways — closed-book, public evidence only, private
evidence only — and a question that survives is one no single source could answer.

A separate step because it re-reads finished questions and changes nothing about them, so
it never has to run in the same job that wrote them.

All three conditions are kept, not the two the necessity claim strictly needs. The previous
project's numbers are the argument: seven of its questions were answerable closed-book —
"Who was Brazil's Foreign Minister during Lula's presidency" — and all seven passed *both*
evidence conditions. Only the closed-book run saw them.

A question asked over a page scan gets one more, **OCR text only**: the page's transcript in
place of its image, beside the public side. It answers a question the other three cannot --
whether the picture is doing any work, or whether a transcript would have served as well --
and like the rest it is recorded, not used to drop anything. Its private-only condition is
the image alone when a vision model is given, and the transcript alone when not; the item
cannot tell which afterwards, so a run says which in its provenance note.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from osint_benchmark.generate.evidence import clip
from osint_benchmark.generate.item import Item, Necessity
from osint_benchmark.models import prompts
from osint_benchmark.models.backend import Complete, Look, ModelUnavailable, agree, first_word

UNANSWERABLE = "unanswerable"
NO_EVIDENCE = "(no evidence provided)"
# What the solve prompt's evidence slot says when the evidence is the attached image.
IMAGE_EVIDENCE = "(the attached page image; nothing else)"


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
    return right(question, gold, answered(question, evidence, solver), judge, samples)


def right(
    question: str, gold: str, candidate: str, judge: Complete | None, samples: int = 1
) -> bool:
    """Return whether an ablation's answer is the gold one; any answer counts without a judge."""
    if not candidate or judge is None:
        return bool(candidate)
    prompt = prompts.render(
        "necessity_equivalent", question=question, gold=gold, candidate=clip(candidate, 2000)
    )
    return agree(judge, prompt, samples, lambda r: first_word(r, ("MATCH", "DIFFERENT"))) == "match"


def looked(question: str, image: Path, look: Look) -> str:
    """Return a vision model's answer from the page image alone, or empty if it gave none."""
    reply = look(
        prompts.render("necessity_solve", question=question, evidence=IMAGE_EVIDENCE), image
    )
    stripped = reply.strip()
    return "" if not stripped or stripped.lower().startswith(UNANSWERABLE) else stripped


def measure(
    item: Item,
    private_text: str,
    public_text: str,
    solver: Complete,
    judge: Complete | None = None,
    samples: int = 1,
    look: Look | None = None,
    image: Path | None = None,
) -> Necessity:
    """Return the ablation outcomes for one item.

    Each is True when that condition *did* produce the right answer — that is, when the
    question fails to need what was withheld. ``image`` is the item's page scan, when it
    has one; ``look`` a vision model to show it to.
    """
    question, gold = item.question, item.answer
    if image is not None and look is not None:
        private_only = right(question, gold, looked(question, image, look), judge, samples)
    else:
        private_only = solved(question, gold, private_text, solver, judge, samples)
    return Necessity(
        closed_book=solved(question, gold, NO_EVIDENCE, solver, judge, samples),
        public_only=solved(question, gold, public_text, solver, judge, samples),
        private_only=private_only,
        ocr_only=(
            solved(question, gold, f"{private_text}\n\n{public_text}", solver, judge, samples)
            if item.image is not None
            else None
        ),
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
    workers: int = 1,
    look: Look | None = None,
    image_root: Path | None = None,
) -> Iterator[Item]:
    """Yield each item with its necessity measured.

    The outcome is recorded, never used to drop the item. Which condition succeeded says
    something different in each case, and a reviewer needs to see it — the previous project
    kept 52 of 404 questions that failed at least one condition, deliberately.

    ``workers`` items are measured at once. This is the most model calls any step makes —
    three ablations, each answered and then checked against the gold — and every one of them
    is independent of every other, so nothing about the result depends on the order they are
    issued in. Items are still yielded in order.

    An item's image path is relative to ``image_root``, the directory its item file is in.
    """

    def measured(item: Item) -> Item:
        """Measure one item, in place, and return it."""
        private_text = " ".join(texts.get(e.doc_id, "") for e in item.private_evidence)
        public_text = " ".join(texts.get(e.doc_id, "") for e in item.public_evidence)
        # One unmeasurable question costs its own measurement and nothing else. Step 6
        # already worked this way; step 7 did not, so a single bad call threw away the
        # measurement of 135 questions that had taken four hours to write.
        image = None
        if item.image is not None:
            image = Path(item.image["path"])
            image = image if image.is_absolute() or image_root is None else image_root / image
        try:
            item.necessity = measure(
                item, private_text, public_text, solver, judge, samples, look, image
            )
        except ModelUnavailable as exc:
            print(f"  unmeasured {item.item_id}: {exc}", file=sys.stderr)
        return item

    if workers <= 1:
        yield from (measured(item) for item in items)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(measured, items)
