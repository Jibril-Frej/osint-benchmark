"""The benchmark item: one question, its answer, and the evidence it rests on.

One schema for every question type. The previous project gave each type its own shape and
its own gate, so the dependency check — the one that verifies the private side is actually
needed — ran on one stream and not the other, and 82% of the resulting set had a decorative
private hop. The bug was specified, not accidental. One schema and one gate suite is what
makes that unavailable.

Evidence is carried as identifiers and offsets, never as text. Two reasons: the release
then contains no corpus text, so nothing has to be redistributed; and an item cannot drift
from the documents it claims to rest on, because it does not hold a copy of them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum


class Verdict(StrEnum):
    """What a review pass concluded about an item."""

    PASS = "pass"
    ADJUST = "adjust"
    REJECT = "reject"


@dataclass(frozen=True)
class Evidence:
    """One document an item rests on.

    Attributes:
        doc_id: Its id in the corpus it came from.
        source: Which corpus, so the text can be found again.
        side: ``private`` or ``public``.
        revision: For live sources, the revision the text was read at. None for the
            corpora, whose whole file is pinned instead.
        offsets: ``(start, end)`` into the document text, when the item rests on a span
            rather than the whole document.
    """

    doc_id: str
    source: str
    side: str
    revision: int | None = None
    offsets: tuple[int, int] | None = None


@dataclass
class Necessity:
    """Whether each ablation could answer the question without the other side.

    Recorded as three outcomes rather than one pass/fail, because which condition succeeds
    says something different in each case. The previous project measured 52 of 404
    questions failing at least one condition, and the pattern mattered: the two types whose
    gold was a register field were also the two most answerable from the public side alone,
    while seven closed-book leaks were invisible in *both* evidence conditions and only the
    closed-book run caught them.
    """

    closed_book: bool | None = None
    public_only: bool | None = None
    private_only: bool | None = None

    @property
    def measured(self) -> bool:
        """True once all three ablations have run."""
        return None not in (self.closed_book, self.public_only, self.private_only)

    @property
    def needs_both(self) -> bool:
        """True when no single condition could answer it."""
        return self.measured and not any((self.closed_book, self.public_only, self.private_only))


@dataclass
class Item:
    """One benchmark question.

    Attributes:
        item_id: Stable identifier, derived from the pair it was built from.
        question_type: Which family it belongs to.
        question: The question as asked.
        answer: The answer, written from both documents.
        rationale: Why both documents are needed, in the writer's words.
        evidence: The documents it rests on, at least one per side.
        necessity: The ablation outcomes, once measured.
        gates: Which computable checks it passed, and which it failed.
        provenance: How it was made -- models, prompts, revisions.
    """

    item_id: str
    question_type: str
    question: str
    answer: str
    rationale: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    necessity: Necessity = field(default_factory=Necessity)
    gates: dict[str, bool] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)

    @property
    def private_evidence(self) -> list[Evidence]:
        """The confidential documents this item rests on."""
        return [e for e in self.evidence if e.side == "private"]

    @property
    def public_evidence(self) -> list[Evidence]:
        """The public records this item rests on."""
        return [e for e in self.evidence if e.side == "public"]

    @property
    def two_sided(self) -> bool:
        """True when the item cites at least one document from each side.

        Structural, not a measurement: an item citing one side cannot possibly need both,
        whatever the ablations later say.
        """
        return bool(self.private_evidence and self.public_evidence)

    def to_json(self) -> dict:
        """Return the on-disk form."""
        record = asdict(self)
        record["doc_id"] = self.item_id
        record["evidence"] = [
            {**asdict(e), "offsets": list(e.offsets) if e.offsets else None} for e in self.evidence
        ]
        return record
