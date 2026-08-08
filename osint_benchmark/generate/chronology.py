"""How long passed between a private exchange and the public proceeding on the same subject.

The previous project's cleanest type, and the one this repo should have ported first: 32 of
its 32 questions needed all three conditions — the only type at 100%. Nothing else it built
came close.

The reason is structural rather than lucky. **Gold is an integer computed from two dates**,
both of them exact metadata, so no model touches the answer and there is nothing for one to
get wrong. And neither corpus holds both dates: the cable knows when it was written and
nothing about the motion, the parliamentary record knows when the item was submitted and
nothing about the cable. A solver holding one document cannot subtract.

That is necessity by construction in its strongest available form — not "the answer happens
not to be in the other document", but "the answer is a function of both documents and of
nothing else".

The pairing is :mod:`osint_benchmark.pair.topical`, and only its focused pairs are used: an
interval between two documents that merely share a country is an interval between unrelated
events, and the number would be arithmetic rather than a question.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date, datetime

# A year. Beyond it the two documents are not reporting on the same moment, whatever they
# share -- and the interval stops being a fact about one episode.
MAX_DAYS = 365


@dataclass(frozen=True)
class Chronology:
    """One interval between a confidential document and a public proceeding.

    Attributes:
        private_id: The confidential document, namespaced.
        public_id: The parliamentary item, namespaced.
        days: The interval, which is the answer.
        order: Which came first, so the question can ask in the right direction.
        subject: The cable's subject line, for a reviewer.
        title: The item's title, for a reviewer.
        shared: The entities that link them.
    """

    private_id: str
    public_id: str
    days: int
    order: str
    subject: str
    title: str
    shared: tuple[str, ...]


def parse(value: str) -> date | None:
    """Return an ISO date, or None when it cannot be read."""
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def build(pairs: Iterable[dict], max_days: int = MAX_DAYS) -> Iterator[Chronology]:
    """Yield one item per focused pair whose two dates are a usable interval apart.

    A zero-day interval is dropped: same-day is a coincidence a solver can guess, and the
    question would have no work in it.
    """
    for pair in pairs:
        if not pair.get("focused"):
            continue
        private, public = parse(pair["private_date"]), parse(pair["public_date"])
        if not private or not public:
            continue
        days = abs((public - private).days)
        if days == 0 or days > max_days:
            continue
        yield Chronology(
            private_id=pair["private_id"],
            public_id=pair["public_id"],
            days=days,
            order="private_first" if private < public else "public_first",
            subject=pair.get("private_subject", ""),
            title=pair.get("public_title", ""),
            shared=tuple(pair.get("shared_entities", ()))[:4],
        )
