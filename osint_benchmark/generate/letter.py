"""Read who sent a Dodis letter and to whom, off the OCR of its first page.

Both visual types rest on the parts of a letter a reader sees before the first sentence: the
letterhead says which Swiss mission wrote it and where that mission sat, and the address
block says whom it was for. Both are parsed from the OCR rather than from the archive's
metadata, because the metadata records which people and places a document *concerns*, and a
letter concerns many people it is not addressed to.

What is parsed is only a key into public data -- a place name to look a country up by, a
family name to find among the archive's own list of the document's people. The answer is
never read here; it is computed from Wikidata afterwards.

Everything here is conservative. An OCR line that half-matches is skipped rather than
repaired, because a wrong key produces a confident wrong gold, and a missed letter only
produces one question fewer.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Words that open a Swiss mission's letterhead, in the archive's two main languages. The
# match is on the folded form (no accents, lower case).
MISSION = re.compile(
    r"\b(gesandtschaft|botschaft|generalkonsulat|konsulat|legation|ambassade|"
    r"consulat general|consulat|mission)\b"
)

# How far past the mission word the place may be written. The letterhead is a few short
# lines, and the dateline beside it usually names the city; further down is the letter.
LETTERHEAD_CHARS = 260

# How much of the first page counts as its head. A letter's opening block is well inside
# this; the rest of the page is prose, where a country name means nothing about the sender.
HEAD_CHARS = 700

# The German and French forms of address that open an address block. "Herrn" is the dative
# of an address line; "Herr" alone opens the salutation, which names nobody.
ADDRESS = re.compile(r"^\s*(?:an\s+)?(herrn|monsieur|a monsieur|madame|frau)\b(.*)$", re.I)

# Titles and honorifics between the form of address and the name.
TITLES = frozenset(
    {
        "minister", "ministre", "bundesrat", "bundesprasident", "bundespraesident", "dr",
        "prof", "professor", "botschafter", "ambassadeur", "legationsrat", "legationssekretar",
        "direktor", "directeur", "conseiller", "federal", "le", "la", "president", "chef",
        "delegue", "general", "oberst", "colonel", "generalkonsul", "konsul", "consul",
        "charge", "affaires", "von", "de", "herrn", "monsieur", "nationalrat",
        "standerat", "regierungsrat", "staatssekretar", "vizedirektor", "sous",
    }
)  # fmt: skip

# The most words a name in an address line runs to: two given names, a particle, a family
# name.
MAX_NAME_TOKENS = 4

# Particles that belong to a family name rather than to the title before it.
PARTICLES = frozenset({"von", "de", "van", "di", "du"})

WORD = re.compile(r"[a-z][a-z'-]+")


def fold(text: str) -> str:
    """Return text lower-cased with its accents removed, for matching only."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def unspaced(line: str) -> str:
    """Return a line with letter-spaced words closed up.

    Typists emphasised a name by spacing it out -- "Z e h n d e r" -- and OCR keeps the
    spaces, so the family name arrives as six one-letter words.
    """
    return re.sub(r"\b(?:[^\W\d_] ){2,}[^\W\d_]\b", lambda m: m.group(0).replace(" ", ""), line)


def first_page(text: str, doc_id: str) -> str:
    """Return the OCR text of page one.

    The archive's watermark ``dodis.ch/<id>`` is printed at the top of every page, and the
    OCR kept it, so it marks where each page begins.
    """
    mark = f"dodis.ch/{doc_id}"
    starts = [m.start() for m in re.finditer(re.escape(mark), text)]
    if len(starts) >= 2:
        return text[starts[0] : starts[1]]
    return text


@dataclass(frozen=True)
class Letterhead:
    """Which kind of mission sent a letter, and where the letterhead says it sat.

    Attributes:
        mission: The mission word as folded, e.g. ``gesandtschaft``.
        window: The folded letterhead text in which the place is to be found.
    """

    mission: str
    window: str


def letterhead(page: str) -> Letterhead | None:
    """Return the letterhead of a page, or None when it has none in its head.

    Only the first mission word counts. A letter from the Political Department *to* a
    legation names the legation in its address block, and that is further down.
    """
    head = fold(page[:HEAD_CHARS])
    match = MISSION.search(head)
    if not match:
        return None
    # "Schweizerische" or "de Suisse" must be close: a letter from the French embassy in
    # Bern also opens with "Ambassade", and its host country is Switzerland.
    near = head[max(0, match.start() - 40) : match.end() + 40]
    if "schweiz" not in near and "suisse" not in near:
        return None
    return Letterhead(
        mission=match.group(1), window=head[match.start() : match.end() + LETTERHEAD_CHARS]
    )


def places_in(window: str, gazetteer: dict[str, str]) -> set[str]:
    """Return the country QIDs whose names, or whose capitals' names, the window uses.

    ``gazetteer`` maps a folded place name to its country's QID. Matched on whole words, so
    that "Iran" is not found in "Tirana".
    """
    found = set()
    for name, qid in gazetteer.items():
        if re.search(rf"(?<![a-z]){re.escape(name)}(?![a-z])", window):
            found.add(qid)
    return found


@dataclass(frozen=True)
class Addressee:
    """The person an address block names, as the letter writes them.

    Attributes:
        line: The address line, as OCR read it, for a reviewer.
        family: The family name, folded.
        initials: The given-name initials written before it, folded, possibly empty.
        given_written: True when a given name is written out in full, not as an initial.
    """

    line: str
    family: str
    initials: tuple[str, ...]
    given_written: bool


def addressee(page: str) -> Addressee | None:
    """Return the addressee of a page's address block, or None when there is none.

    The address block is the first line opening "Herrn" or "Monsieur" that goes on to name
    someone. "Monsieur le Conseiller fédéral," is a salutation, not an address, and names
    nobody -- it returns None rather than the last title as a family name.
    """
    for raw in page.splitlines():
        match = ADDRESS.match(raw)
        if not match:
            continue
        rest = fold(unspaced(match.group(2)))
        rest = rest.split(",")[0]
        tokens = re.findall(r"[a-z][a-z'-]*\.?", rest)
        names: list[str] = []
        for token in tokens:
            bare = token.rstrip(".")
            if bare in TITLES and bare not in PARTICLES and not names:
                continue
            names.append(token)
        # An address names one person in a few words. A longer run is a sentence that
        # happens to open with "Herrn", and its last word is not anyone's name.
        if not names or len(names) > MAX_NAME_TOKENS:
            continue
        initials = tuple(t.rstrip(".") for t in names[:-1] if len(t.rstrip(".")) == 1)
        written = [t for t in names[:-1] if len(t.rstrip(".")) > 1 and t not in PARTICLES]
        family = names[-1].rstrip(".")
        if len(family) < 3:
            continue
        # A trailing particle is part of the family name ("von Graffenried").
        if len(names) >= 2 and names[-2] in PARTICLES:
            family = f"{names[-2]} {family}"
            written = [t for t in written if t != names[-2]]
        return Addressee(
            line=raw.strip(),
            family=family,
            initials=initials,
            given_written=bool(written),
        )
    return None


def family_of(full_name: str) -> str:
    """Return the folded family name of a full name, keeping a particle before it."""
    tokens = WORD.findall(fold(full_name))
    if not tokens:
        return ""
    if len(tokens) >= 2 and tokens[-2] in PARTICLES:
        return f"{tokens[-2]} {tokens[-1]}"
    return tokens[-1]


def initials_agree(written: tuple[str, ...], full_name: str) -> bool:
    """Return whether initials written on a letter fit a full name's given names."""
    if not written:
        return True
    given = [t[0] for t in WORD.findall(fold(full_name))[:-1]]
    return all(initial in given for initial in written)
