"""Does the public event record bear out the incident a confidential document describes?

The type built on the two sources that bridge nothing. A country co-occurs with everything,
so it is a poor *bridge* anchor and step 5 excludes it — but an anchor plus a three-week
window around the document's date is specific, and that is what the previous project matched
on. Its own matcher skipped the giant countries for exactly the reason the bridge type
excludes them, so the two are not in tension: what makes an event match discriminative is
time, not the entity.

**The public side is structured data, not prose.** A UCDP record is a date, a place, the
parties and a death count; it asserts that something happened and carries no narrative. So
the question is a comparison — the private document describes an incident, the public record
either bears it out or does not — and the answer is a four-way verdict decided by the model
that writes the question, because whether two accounts describe the same event cannot be
settled before knowing what is being asked.

**Why both documents are needed.** The public record has no account to compare against: it
says an event occurred, not what anyone reported about it. The private document has no
independent confirmation. Neither alone answers "is this borne out".
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date

# A document reports on what happened recently, and a public event record is dated to the
# day. Three weeks either side is the previous project's window.
WINDOW_DAYS = 21

# Events shown for one anchor. Past this the prompt is a list rather than a comparison.
MAX_EVENTS = 8

# An anchor named by more than this share of the confidential corpus is a country in the
# sense that makes it useless: it puts every document beside every event of the month.
MAX_ANCHOR_SHARE = 0.15

# Fields of a public event record worth showing, in the order they read naturally.
SHOWN = (
    ("date_start", "date"),
    ("country", "country"),
    ("where_description", "place"),
    ("side_a", "one side"),
    ("side_b", "the other side"),
    ("conflict_name", "conflict"),
    ("best", "deaths recorded"),
    ("source_headline", "reported as"),
)


@dataclass(frozen=True)
class Match:
    """One confidential document and the public events that might bear it out.

    Attributes:
        private_id: The confidential document, namespaced.
        anchor_qid: The entity both sides name.
        anchor_label: Its name, for the prompt.
        public_ids: The event records, namespaced.
        rendered: Those events as lines a model can read.
        window: How many days either side were searched.
    """

    private_id: str
    anchor_qid: str
    anchor_label: str
    public_ids: tuple[str, ...]
    rendered: str
    window: int


def render(event: dict) -> str:
    """Return one public event record as a line of text.

    Field by field rather than through a sentence template. The fields are the corpus's own
    considered vocabulary, and a template per source is a thing to keep in step with parsers
    that change.
    """
    parts = [f"{label} {event[key]}" for key, label in SHOWN if event.get(key)]
    return "  - " + ", ".join(parts) if parts else ""


def anchors(rows: Iterable[dict], share: float, total: int) -> set[str]:
    """Return the QIDs named too widely in the private corpus to anchor anything."""
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update({e["qid"] for e in row.get("entities", ())})
    return {qid for qid, seen in counts.items() if seen > share * total} if total else set()


def by_day(events: Iterable[dict]) -> dict[tuple[str, int, int], list[dict]]:
    """Bucket events by anchor and month, so the scan does not go quadratic."""
    buckets: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for event in events:
        when: date = event["date"]
        for qid in event["qids"]:
            buckets[(qid, when.year, when.month)].append(event)
    return buckets


def build(
    documents: list[dict],
    events: list[dict],
    labels: dict[str, str],
    window: int = WINDOW_DAYS,
    max_share: float = MAX_ANCHOR_SHARE,
    max_events: int = MAX_EVENTS,
    outcomes: Counter | None = None,
) -> Iterator[Match]:
    """Yield one match per (document, anchor) with at least one event in the window.

    ``documents`` are the confidential documents with a ``date`` and their linked ``qids``;
    ``events`` the public records with a ``date`` and theirs.
    """
    if outcomes is None:
        outcomes = Counter()
    too_common = anchors(documents, max_share, len(documents))
    outcomes["anchors_too_common"] = len(too_common)
    buckets = by_day(events)

    for document in documents:
        when = document.get("date")
        if not when:
            outcomes["undated_document"] += 1
            continue
        for qid in dict.fromkeys(e["qid"] for e in document.get("entities", ())):
            if qid in too_common or qid not in labels:
                continue
            found: list[dict] = []
            for offset in (-1, 0, 1):
                total = when.year * 12 + (when.month - 1) + offset
                year, month = divmod(total, 12)
                found += [
                    event
                    for event in buckets.get((qid, year, month + 1), [])
                    if abs((event["date"] - when).days) <= window
                ]
            if not found:
                continue
            found.sort(key=lambda e: e["date"])
            chosen = found[:max_events]
            outcomes["match"] += 1
            yield Match(
                private_id=document["doc_id"],
                anchor_qid=qid,
                anchor_label=labels[qid],
                public_ids=tuple(event["doc_id"] for event in chosen),
                rendered="\n".join(line for event in chosen if (line := render(event))),
                window=window,
            )
