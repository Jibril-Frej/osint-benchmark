"""A linker that matches article titles in text, with no model.

Real linking, not a stand-in: it resolves genuine mentions to genuine QIDs. It is simply
much worse than ReFinED — it cannot disambiguate (every "Washington" resolves the same
way), it has no alias knowledge, and it only finds titles spelled exactly as Wikipedia
spells them.

It exists so the pipeline can be run end to end without a GPU, a 2 GB download or the
ReFinED install. Anything built with it is labelled as such in the provenance, because a
question resting on a dictionary match is not a question resting on a linker.

Longest-match-first, so "United States Department of State" wins over "United States".
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

from osint_benchmark.link.refined import Mention, per_document

# Single-word titles are unusable here. Wikipedia has articles called "Told", "Right",
# "Last" and "Claim", and the cables are ALL CAPS -- which destroys the capitalisation that
# would otherwise separate a name from an ordinary word. Matching case-insensitively on
# single words turned 200 cables into 51,499 mentions of nothing. Requiring two words costs
# enormous recall and is the only thing that makes this usable without a model.
MIN_TITLE_WORDS = 2
MIN_TITLE_CHARS = 6

# Two words is not enough on its own. Wikipedia has articles titled "Has Been", "To Have",
# "There Is" and "The Same" -- songs and albums, mostly -- and against 7.5M titles almost
# any English phrase is somebody's title. Requiring at least one word that is not a common
# function word is what separates "United States" from "AND THAT".
STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before
    being below between both but by can did do does doing down during each few for from
    further had has have having he her here hers him his how i if in into is it its itself
    just me more most my no nor not now of off on once only or other our out over own same
    she should so some such than that the their them then there these they this those
    through to too under until up very was we were what when where which while who whom
    why will with would you your
    """.split()
)

# A parenthetical disambiguator is not part of the name as it appears in prose.
DISAMBIGUATOR = re.compile(r"\s*\([^)]*\)$")


def build_dictionary(
    index: Iterable[dict],
    min_chars: int = MIN_TITLE_CHARS,
    min_words: int = MIN_TITLE_WORDS,
) -> dict[str, str]:
    """Return ``lowercased title -> QID`` from the entity index.

    Titles that collide keep the first QID seen. That is a real limitation, not a
    resolution: the dictionary linker cannot disambiguate, which is the main reason it is
    not the linker the benchmark is built with.
    """
    dictionary: dict[str, str] = {}
    for row in index:
        title = DISAMBIGUATOR.sub("", row.get("title", "")).strip()
        words = title.split()
        if len(title) < min_chars or len(words) < min_words:
            continue
        if all(word.lower().strip(".,'-") in STOPWORDS for word in words):
            continue
        dictionary.setdefault(title.lower(), row["qid"])
    return dictionary


def linker(dictionary: dict[str, str], max_words: int = 6):
    """Return a :data:`~osint_benchmark.link.refined.Linker` over a title dictionary.

    Batched only to satisfy the interface: there is nothing to gain from a batch here.
    """
    tokens = re.compile(r"\w[\w'-]*")

    def link(text: str) -> Iterator[Mention]:
        """Yield each title found in the text, longest match first."""
        matches = list(tokens.finditer(text or ""))
        taken = [False] * len(matches)
        for span in range(max_words, 0, -1):
            for start in range(len(matches) - span + 1):
                if any(taken[start : start + span]):
                    continue
                first, last = matches[start], matches[start + span - 1]
                phrase = text[first.start() : last.end()]
                qid = dictionary.get(phrase.lower())
                if qid is None:
                    continue
                for i in range(start, start + span):
                    taken[i] = True
                # Confidence is 1.0 because an exact title match is exact. It says nothing
                # about whether the right entity was chosen -- that is what it cannot do.
                yield Mention(qid=qid, surface_form=phrase, confidence=1.0, entity_type="ORG")

    return per_document(link)
