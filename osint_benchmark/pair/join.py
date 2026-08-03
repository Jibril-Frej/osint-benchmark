"""Pair one confidential document with one public record over a shared entity.

Run identically for every private corpus so the counts mean the same thing on both sides.
The previous project measured Cablegate through per-type joins built at different times and
Dodis not at all, which made the two impossible to compare.

Two properties are recorded on every pair, because the question builders need them and
rediscovering them per type is how they drift apart:

* ``same_period`` — whether the two are close enough in time to be about the same events.
  This is not a formality. Dodis is archival: 90% of its dated documents fall between 1890
  and 1979, against Cablegate's 2003–2010 and a parliamentary record starting in 1978. A
  Dodis pair is normally *not* contemporaneous, so the interval and position-comparison
  question types simply do not apply to it.
* ``days_apart`` — the signed interval, which is the whole answer for the chronology type
  and free here.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date, datetime

DEFAULT_WINDOW_DAYS = 90


@dataclass(frozen=True)
class Pair:
    """One confidential document beside one public record, and why they belong together."""

    private_id: str
    public_id: str
    qid: str
    private_date: str | None
    public_date: str | None
    days_apart: int | None
    same_period: bool

    def to_json(self) -> dict:
        """Return the on-disk form."""
        return {
            "doc_id": f"{self.private_id}|{self.public_id}|{self.qid}",
            "private_id": self.private_id,
            "public_id": self.public_id,
            "qid": self.qid,
            "private_date": self.private_date,
            "public_date": self.public_date,
            "days_apart": self.days_apart,
            "same_period": self.same_period,
        }


def as_date(value: str | None) -> date | None:
    """Return the date part of an ISO timestamp, or None if it is absent or unparseable."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return None


def interval(private_date: str | None, public_date: str | None) -> int | None:
    """Return public minus private in days, or None when either date is missing.

    Signed on purpose: a public record *predating* the private report is a different
    situation from one following it, and a question about the interval needs to know which.
    """
    first, second = as_date(private_date), as_date(public_date)
    if first is None or second is None:
        return None
    return (second - first).days


def pair_documents(
    bridges: Iterable[dict],
    private_dates: dict[str, str | None],
    public_dates: dict[str, str | None],
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> Iterator[Pair]:
    """Yield one pair per (private document, public record, shared entity).

    One row per triple rather than per entity, so counts mean documents rather than
    entities, and a document naming three shared entities produces three pairs that can be
    judged separately.
    """
    for bridge in bridges:
        qid = bridge["qid"]
        for private_id in bridge["private"]:
            for public_id in bridge["public"]:
                private_date = private_dates.get(private_id)
                public_date = public_dates.get(public_id)
                days = interval(private_date, public_date)
                yield Pair(
                    private_id=private_id,
                    public_id=public_id,
                    qid=qid,
                    private_date=private_date,
                    public_date=public_date,
                    days_apart=days,
                    same_period=days is not None and abs(days) <= window_days,
                )
