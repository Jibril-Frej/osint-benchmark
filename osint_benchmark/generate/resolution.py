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
Prominence is measured by how long the person's Wikipedia article is, rather than by how
many claims they have, because a twenty-predicate slice gives even world figures only a
handful of claims.

And the gold must be obscure in absolute terms, not merely less famous than a namesake:
the second most prominent Bush is still George W. Bush.

The previous project's warning, kept because it governs how this must be used: **over 80%
of raw candidates name the wrong person.** The type keeps mentions the linker resolved to a
non-prominent bearer, which is largely the same set as the ones it got wrong — so a
verification pass over the gold is not optional, and :func:`build` yields candidates rather
than questions.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

# The linker's confidence floor. A resolution item asserts *which* person a mention means,
# so a mislinked mention does not cost a pair, it makes the question's answer wrong.
MIN_CONFIDENCE = 0.90

# A surface this short is an initialism or OCR noise, not a usable person mention.
MIN_SURFACE = 4

# Whole-article size above which a person is too well known for the private context to be
# doing the work. Not "less famous than the top bearer" -- famous in absolute terms.
#
# The previous project measured this as 4,000 characters of plain text, read from a full-text
# dump it had on disk. This repo asks the API, which gives an article's size in bytes for free
# and charges a request per article for plain text, so the threshold had to be restated. On 25
# biographies spanning world figures to near-stubs, the articles under 4,000 plain characters
# reached at most 9,330 bytes and those over it began at 8,690: the two overlap only in that
# band, and 9,000 sits inside it. The lead section, which is the text this project actually
# stores, is no use for this at all -- it is a median 11% of the article and a world figure's
# lead is no longer than a minor official's.
MAX_GOLD_ARTICLE_BYTES = 9000

WORD = re.compile(r"[a-z0-9]+")

# Words long enough to carry meaning, for the gold check below.
CONTENT_WORD = re.compile(r"[a-z][a-z'-]{2,}")

# How much the gold must beat its best rival by before the linker's choice is believed.
MARGIN = 1.25

# Vocabulary shared by every dispatch and every biography, which is evidence of nothing.
COMMON = frozenset(
    {
        "the", "and", "for", "that", "with", "was", "were", "has", "have", "had", "not",
        "but", "his", "her", "its", "their", "who", "which", "from", "this", "there",
        "been", "also", "than", "then", "government", "minister", "president", "state",
        "national", "official", "officials", "public", "political", "party", "united",
        "states", "international", "world", "new", "first", "former", "during", "after",
        "before", "would", "could", "said", "says", "one", "two", "years", "year",
    }
)  # fmt: skip


@dataclass(frozen=True)
class Check:
    """What comparing the passage against each namesake's article concluded.

    Attributes:
        verdict: ``verified``, ``refuted``, or ``unchecked``.
        gold_score: How much of the gold's article vocabulary the passage echoes.
        rival_score: The best any other bearer of the name scored.
        rival: Which bearer that was, so a reviewer can see who the linker may have meant.
    """

    verdict: str
    gold_score: float
    rival_score: float
    rival: str = ""


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
        article_bytes: The whole article's size, the obscurity measure.
    """

    doc_id: str
    surface: str
    qid: str
    label: str
    candidates: tuple[str, ...]
    rank: int
    article_bytes: int


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
    article_bytes: dict[str, int],
    max_gold_bytes: int = MAX_GOLD_ARTICLE_BYTES,
) -> Iterator[Resolution]:
    """Yield one candidate per bare-family-name mention of a non-obvious bearer.

    ``article_bytes`` maps QID to its whole article's size in bytes. An entity absent from it
    has no article and is skipped: a question whose answer nobody has written about is not
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
            if qid not in article_bytes:
                continue
            ranked = sorted(candidates, key=lambda c: (-article_bytes.get(c, 0), c))
            rank = ranked.index(qid)
            # Rank 0 is the answer prior fame gives, so the private context did no work.
            # The absolute ceiling is the separate guard: the second most prominent Bush is
            # still George W. Bush.
            if rank == 0 or article_bytes[qid] > max_gold_bytes:
                continue
            yield Resolution(
                doc_id=row["doc_id"],
                surface=entity.get("surface_form", ""),
                qid=qid,
                label=labels[qid],
                candidates=tuple(ranked),
                rank=rank,
                article_bytes=article_bytes[qid],
            )


def content_words(text: str) -> Counter:
    """Return a text's content words, minus the vocabulary every document shares."""
    return Counter(w for w in CONTENT_WORD.findall((text or "").lower()) if w not in COMMON)


def echo(passage: Counter, article: Counter) -> float:
    """Return how much of an article's vocabulary the passage echoes.

    Rare words weigh more, which is the whole signal: two texts both saying "OSCE" is
    evidence that they are about the same person, both saying "meeting" is not. Weighting is
    by word length rather than a corpus frequency table, because a table would have to be
    built and pinned for a signal that only has to rank four or five candidates.
    """
    if not passage or not article:
        return 0.0
    return sum(
        math.log1p(count) * math.log1p(passage[word]) / math.log1p(len(word))
        for word, count in article.items()
        if word in passage
    )


def check(
    item: Resolution, passage: str, articles: dict[str, str], margin: float = MARGIN
) -> Check:
    """Return whether the passage bears out the bearer the linker chose.

    The type keeps mentions resolved to a *non-prominent* bearer, and "the linker chose the
    obscure one" and "the linker got it wrong" are largely the same set — a bare family name
    is the hardest case an entity linker faces, so the filter selects for its errors. In the
    previous project over 80% of raw candidates named the wrong person: the OSCE
    representative on freedom of the media was Miklos Haraszti and the item asked about Emil,
    the Solana in a diplomatic cable was Javier and not Fernando.

    Prominence cannot settle this, so vocabulary does. Each bearer has an article; the right
    one talks about the same things as the passage, and a musicologist does not. An item
    survives only when the gold beats every rival by ``margin``.

    A candidate with no article returns ``unchecked`` rather than passing: this cannot make
    the linker right, it can only remove the items where it is demonstrably wrong, and an
    unchecked gold is what caused the problem.
    """
    words = content_words(passage)
    scores = {qid: echo(words, content_words(articles.get(qid, ""))) for qid in item.candidates}
    gold = scores.get(item.qid, 0.0)
    rivals = {qid: score for qid, score in scores.items() if qid != item.qid}
    rival, rival_score = max(rivals.items(), key=lambda kv: kv[1], default=("", 0.0))
    if not articles.get(item.qid) or gold <= 0:
        verdict = "unchecked"
    elif gold >= margin * max(rival_score, 1e-9):
        verdict = "verified"
    else:
        verdict = "refuted"
    return Check(
        verdict=verdict,
        gold_score=round(gold, 2),
        rival_score=round(rival_score, 2),
        rival=rival if verdict == "refuted" else "",
    )
