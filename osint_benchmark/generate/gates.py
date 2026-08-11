"""The checks every question passes, whatever type it is.

These are the ones computable without a model. Necessity needs a solver and lives in step 7;
everything here runs offline and is what :func:`~osint_benchmark.generate.emit.emit` applies
before an item can be written.

Each gate exists because something got through without it:

* **two_sided** — an item citing one side cannot need both, whatever the ablations say.
* **answer_not_in_question** — the question that contains its own answer measures nothing.
* **no_source_attribution** — "the cable states" tells a solver where to look and turns a
  two-hop question into a one-hop one. It also reads as a quiz about documents rather than
  an intelligence requirement.
* **not_a_bare_attribute** — birthplace, founding year, capital. 44 of the previous
  project's first 251 questions were two-hop attribute trivia; asking for an encyclopaedia
  field makes the private document a lookup key rather than evidence. An item that withholds
  its subjects is exempt, because there is then nothing to look the field off.
* **answer_is_substantive** — a one-character or empty answer is a parsing failure wearing
  an answer's clothes.
* **no_foreign_script** — QwQ drops into Chinese mid-sentence; three of 515 questions in one
  run carried a clause of it.
* **not_referential** — the looser form of source attribution, matched by pattern rather
  than by phrase, because the literal list missed "as noted in diplomatic records".
* **polar_when_the_answer_is_a_verdict** — a yes/no answer needs a question a yes settles.
* **withholds_what_it_was_told_to** — some questions are only necessary because they
  describe their subjects instead of naming them. An association asks what two people both
  belong to, and a solver given their names can look the pair up in public sources without
  the confidential document at all. The prompt says not to name them; this checks.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from osint_benchmark.generate.item import Item

# Predicates that may never be the asked-for value. These are the encyclopaedia lookups
# that made the previous project's first generation read as trivia rather than analysis.
ATTRIBUTE_DENYLIST = (
    "birthplace",
    "place of birth",
    "born in",
    "date of birth",
    "founding year",
    "founded in",
    "year was",
    "inception",
    "capital of",
    "headquarters",
    "headquartered",
    "political party",
    "population of",
)

# Phrases that point at the evidence instead of asking about the world. A real run got
# "What is the connection between Cameroon and Canada as described in the two documents?"
# past the first version of this list, which only knew about "the document states" and
# "according to the cable" -- so the phrasing matters less than the *reference*, and what
# is matched now is any mention of the documents themselves.
SOURCE_ATTRIBUTION = (
    "the cable",
    "the cables",
    "the telegram",
    "the dispatch",
    "the document",
    "the documents",
    "both documents",
    "two documents",
    "the report",
    "the record",
    "the records",
    # Plurals and qualifiers, because the singular list let through "Did the public records
    # from Iraq show increased resistance activity" -- which points a solver at the evidence
    # exactly as squarely as "the record" does, and is what the event prompt's own wording
    # invites.
    "public record",
    "public records",
    "the public record",
    "the event record",
    "event records",
    "the article",
    "the passage",
    "the text",
    "the excerpt",
    "as reported in",
    "as described in",
    "as stated in",
    "as mentioned in",
    "according to the",
    "in the source",
)

# The literal list above only catches a question that names the evidence outright. A
# generator reaches for the same move in looser words -- "as described in US diplomatic
# records" -- which points a solver at where to look just as squarely. Only source
# ATTRIBUTION is caught, never any mention of diplomacy: "the diplomatic exchange on Iran"
# names an event a question is legitimately about.
REFERENTIAL = re.compile(
    r"\b(?:(?:as\s+)?(?:described|noted|detailed|reported|recorded|mentioned|discussed|stated)"
    r"\s+in\s+[\w\s.'\u2019-]{0,40}?(?:diplomatic|embassy|cable|dispatch|archive)"
    r"|diplomatic\s+(?:record|archive)s?"
    r"|embassy\s+(?:report|reporting|cable))",
    re.I,
)

# QwQ is a Chinese-developed model and drops into Chinese mid-sentence -- the previous
# project caught a question ending in a clause of Chinese after "Shah Deniz". German and
# French accents are Latin-1 and stay welcome; a script the corpora never use is a
# generation fault rather than content. Three of 515 questions carried one.
FOREIGN_SCRIPT = re.compile(
    r"[\u0400-\u04ff\u0590-\u05ff\u0600-\u06ff\u3000-\u30ff\u4e00-\u9fff\uff00-\uffef]"
)

# A four-way verdict answers a polar question. "To what extent did ..." is an essay prompt
# no exact-match label can grade, so a verdict item must open with one of these.
POLAR_OPENERS = frozenset(
    {"did", "do", "does", "is", "was", "were", "are", "has", "have", "had",
     "will", "would", "can", "could", "should"}
)  # fmt: skip

# Above this trigram overlap with an already-accepted question, it is the same question.
MAX_DUPLICATE = 0.6

MIN_ANSWER_CHARS = 2

# The labels an adjudicated answer may take, so the polar check knows when it applies.
VERDICTS = frozenset({"Yes", "No", "Mixed", "Not enough evidence"})

_WORD = re.compile(r"[a-z0-9]+")

Gate = Callable[[Item], bool]


def _words(text: str) -> list[str]:
    """Return the lowercase word tokens of a string."""
    return _WORD.findall(text.lower())


def two_sided(item: Item) -> bool:
    """The item cites at least one document from each side of the boundary."""
    return item.two_sided


def answer_not_in_question(item: Item) -> bool:
    """The question does not give its own answer away.

    Compared on word sequences rather than raw substrings so that punctuation and casing
    do not hide a leak — the previous project's substring check missed an acronym against
    the spelled-out name it stood for.
    """
    answer = _words(item.answer)
    if not answer:
        return False
    question = _words(item.question)
    span = len(answer)
    return not any(question[i : i + span] == answer for i in range(len(question) - span + 1))


def no_source_attribution(item: Item) -> bool:
    """The question asks about the world, not about the documents."""
    lowered = item.question.lower()
    return not any(phrase in lowered for phrase in SOURCE_ATTRIBUTION)


def not_a_bare_attribute(item: Item) -> bool:
    """The question does not ask for an encyclopaedia field off a subject it names.

    Both halves matter. The failure this catches is the *lookup key*: the confidential
    document identifies an entity, and the answer is then a field anyone could read off it —
    44 of the previous project's first 251 questions were that shape. What makes it a defect
    is that the question names, or lets a solver name, the thing to look up.

    A question that withholds its subjects cannot have that shape, because there is nothing
    to look the attribute off. So an item declaring ``withheld`` — and a separate gate checks
    it kept that promise — is exempt. This is deliberately a property of the item rather than
    of its type: per-type gates are what let the previous project's dependency check run on
    one stream and not the other, and 82% of the resulting set had a decorative private hop.

    Measured on the run that prompted this: 102 association questions whose answer was a
    political party passed, and 19 identical ones were rejected for saying "political party"
    rather than "organization". That is a filter on synonyms, not on quality.
    """
    if item.provenance.get("withheld"):
        return True
    lowered = item.question.lower()
    return not any(phrase in lowered for phrase in ATTRIBUTE_DENYLIST)


def answer_is_substantive(item: Item) -> bool:
    """The answer is long enough to be one."""
    return len(item.answer.strip()) >= MIN_ANSWER_CHARS


def no_foreign_script(item: Item) -> bool:
    """The question is in the language the corpora are written in."""
    return FOREIGN_SCRIPT.search(item.question) is None


def not_referential(item: Item) -> bool:
    """The question does not attribute its content to the sources in looser words."""
    return REFERENTIAL.search(item.question) is None


def polar_when_the_answer_is_a_verdict(item: Item) -> bool:
    """A yes/no/mixed answer needs a question a yes settles.

    Only checked when the answer *is* one of the verdict labels: every other type answers
    with a name or a number, and "Which body..." is right for those.
    """
    if item.answer not in VERDICTS:
        return True
    words = _words(item.question)
    return bool(words) and words[0] in POLAR_OPENERS


def withholds_what_it_was_told_to(item: Item) -> bool:
    """The question does not name the subjects it was built to describe instead.

    ``provenance["withheld"]`` is a semicolon-separated list set by the question type that
    knows what naming would give away. An item that sets nothing passes: most types have
    nothing to withhold beyond their answer, which a separate gate already covers.
    """
    withheld = [name.strip() for name in item.provenance.get("withheld", "").split(";")]
    question = _words(item.question)
    for name in withheld:
        tokens = _words(name)
        if not tokens:
            continue
        span = len(tokens)
        if any(question[i : i + span] == tokens for i in range(len(question) - span + 1)):
            return False
    return True


GATES: dict[str, Gate] = {
    "two_sided": two_sided,
    "answer_not_in_question": answer_not_in_question,
    "no_source_attribution": no_source_attribution,
    "not_a_bare_attribute": not_a_bare_attribute,
    "answer_is_substantive": answer_is_substantive,
    "withholds_what_it_was_told_to": withholds_what_it_was_told_to,
    "no_foreign_script": no_foreign_script,
    "not_referential": not_referential,
    "polar_when_the_answer_is_a_verdict": polar_when_the_answer_is_a_verdict,
}


def run(item: Item) -> dict[str, bool]:
    """Return each gate's outcome for one item."""
    return {name: gate(item) for name, gate in GATES.items()}


def passes(item: Item) -> bool:
    """True when the item clears every gate."""
    return all(run(item).values())


def opening_share(items: Iterable[Item], words: int = 5) -> dict[str, float]:
    """Return the share of questions starting with each opening phrase.

    A run-level alarm rather than a per-item gate. Forty-one questions opening "How many
    people were killed in" is a failed run, not forty-one bad questions, and it has to fail
    the run rather than reach the gold set.
    """
    openings = [" ".join(_words(item.question)[:words]) for item in items]
    if not openings:
        return {}
    counts: dict[str, int] = {}
    for opening in openings:
        counts[opening] = counts.get(opening, 0) + 1
    return {
        opening: count / len(openings)
        for opening, count in sorted(counts.items(), key=lambda kv: -kv[1])
    }
