"""A private document names someone by family name alone; which bearer does it mean?

The previous project's second-largest type: 72 of its 404. Ported with every condition,
because the conditions *are* the question — remove any one and what remains is a trivia
lookup rather than a two-document problem.

**Why both documents are needed, by construction.** The public catalogue holds several
people with that family name and nothing to choose between them, so a solver with only
public sources cannot pick; the detail that identifies which one — what they were doing,
where, with whom — exists only in the private document. And the answer is a public fact
about a person the private document never names in full.

**Catalogue ambiguity alone is not difficulty.** "Mubarak" collides with eight other
entities and every solver answers Hosni from prior fame. The item is real only when the
referent is *not* the most prominent bearer, so the private context has to do the work.
Prominence is measured by Wikipedia article length rather than claim count, because a
twenty-predicate slice gives even world figures only a handful of claims.

And the gold must be obscure in absolute terms, not merely less famous than a namesake:
the second most prominent Bush is still George W. Bush.

The previous project's warning, kept because it governs how this must be used: **over 80%
of raw candidates name the wrong person.** The type keeps mentions the linker resolved to a
non-prominent bearer, which is largely the same set as the ones it got wrong — so a
verification pass over the gold is not optional, and :func:`build` yields candidates rather
than questions.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

# The linker's confidence floor. A resolution item asserts *which* person a mention means,
# so a mislinked mention does not cost a pair, it makes the question's answer wrong.
MIN_CONFIDENCE = 0.90

# A surface this short is an initialism or OCR noise, not a usable person mention.
MIN_SURFACE = 4

# Wikipedia article length above which a person is too well known for the private context
# to be doing the work. Not "less famous than the top bearer" -- famous in absolute terms.
MAX_GOLD_ARTICLE_CHARS = 4000

WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Resolution:
    """One ambiguous family-name mention and the bearer the linker chose.

    Attributes:
        doc_id: The private document containing the mention.
        surface: How the document wrote the name.
        qid: The bearer the linker resolved it to — the proposed gold, not yet verified.
        label: That person's full name, which is the answer.
        candidates: Every catalogue entity sharing the family name.
        rank: Where the gold sits in prominence order; 0 would be the obvious answer.
        article_chars: The gold's article length, the obscurity measure.
    """

    doc_id: str
    surface: str
    qid: str
    label: str
    candidates: tuple[str, ...]
    rank: int
    article_chars: int


def normalise(text: str) -> str:
    """Return a surface form as its lowercase alphanumeric tokens."""
    return " ".join(WORD.findall(text.lower()))


def family_names(labels: dict[str, str]) -> dict[str, set[str]]:
    """Return ``family name -> the QIDs of everyone catalogued under it``.

    The given name is skipped deliberately: ambiguity lives in the family name, and that is
    what a document writing one bare token is being ambiguous about.
    """
    index: dict[str, set[str]] = {}
    for qid, label in labels.items():
        tokens = normalise(label).split()
        if len(tokens) < 2:
            continue
        for token in tokens[1:]:
            if len(token) >= MIN_SURFACE:
                index.setdefault(token, set()).add(qid)
    return index


def build(
    links: Iterable[dict],
    labels: dict[str, str],
    people: set[str],
    article_chars: dict[str, int],
    max_gold_chars: int = MAX_GOLD_ARTICLE_CHARS,
) -> Iterator[Resolution]:
    """Yield one candidate per bare-family-name mention of a non-obvious bearer.

    ``article_chars`` maps QID to Wikipedia article length. An entity absent from it has no
    article and is skipped: a question whose answer nobody has written about is not
    answerable from public sources at all.

    These are *candidates*. Most of them name the wrong person, and verifying the gold
    against the private context is a separate step.
    """
    index = family_names(labels)
    for row in links:
        for entity in row.get("entities", []):
            qid = entity["qid"]
            if entity.get("confidence", 1.0) < MIN_CONFIDENCE:
                continue
            if qid not in people or qid not in labels:
                continue
            surface = normalise(entity.get("surface_form", ""))
            # The mention must be a *bare* family name. A document writing the full name is
            # not being ambiguous and needs no resolving.
            if not surface or " " in surface or len(surface) < MIN_SURFACE:
                continue
            candidates = index.get(surface, set())
            # Two bearers at least, and the linker's choice must be among them.
            if len(candidates) < 2 or qid not in candidates:
                continue
            if qid not in article_chars:
                continue
            ranked = sorted(candidates, key=lambda c: (-article_chars.get(c, 0), c))
            rank = ranked.index(qid)
            # Rank 0 is the answer prior fame gives, so the private context did no work.
            # The absolute ceiling is the separate guard: the second most prominent Bush is
            # still George W. Bush.
            if rank == 0 or article_chars[qid] > max_gold_chars:
                continue
            yield Resolution(
                doc_id=row["doc_id"],
                surface=entity.get("surface_form", ""),
                qid=qid,
                label=labels[qid],
                candidates=tuple(ranked),
                rank=rank,
                article_chars=article_chars[qid],
            )
