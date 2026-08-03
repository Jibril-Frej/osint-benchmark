"""The only way an item reaches disk.

Candidate builders return :class:`~osint_benchmark.generate.item.Item` objects. This runs
the gate suite over each one, records the outcomes on the item, and writes only those that
clear every gate. Rejects are written too, to a separate file, because a question set is
only interpretable next to what was thrown away and why.

The point is that there is no other path. A question type cannot skip a gate by writing its
own output, because writing its own output is not something it can do.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from osint_benchmark.artifacts import Provenance, write_records
from osint_benchmark.generate import gates
from osint_benchmark.generate.item import Item

ALARM_SHARE = 0.10


def judged(items: Iterable[Item]) -> Iterator[Item]:
    """Yield each item with its gate outcomes recorded."""
    for item in items:
        item.gates = gates.run(item)
        yield item


def emit(
    items: Iterable[Item],
    accepted_path: Path,
    rejected_path: Path,
    provenance: Provenance,
) -> tuple[int, int]:
    """Write the items that clear every gate, and separately those that do not.

    Returns:
        ``(accepted, rejected)`` counts.
    """
    accepted: list[Item] = []
    rejected: list[Item] = []
    for item in judged(items):
        (accepted if all(item.gates.values()) else rejected).append(item)

    total = len(accepted) + len(rejected)
    write_records(accepted_path, (i.to_json() for i in accepted), provenance, rows_in=total)
    write_records(
        rejected_path,
        (i.to_json() for i in rejected),
        Provenance(
            source=provenance.source,
            source_fields=provenance.source_fields,
            kept=provenance.kept,
            dropped=provenance.dropped,
            kind="derived",
            note="Items that failed at least one gate. Kept so the accepted set is interpretable.",
        ),
        rows_in=total,
    )
    return len(accepted), len(rejected)


def run_alarms(items: list[Item], share: float = ALARM_SHARE) -> list[str]:
    """Return the run-level problems in a finished set.

    Not per-item gates: these are properties of the *set*. A tenth of a run opening with the
    same five words is a template that survived, and it should fail the run rather than
    reach the gold set one question at a time.
    """
    problems = []
    for opening, fraction in gates.opening_share(items).items():
        if fraction >= share:
            problems.append(f"{fraction:.0%} of questions open with {opening!r}")
    types = {item.question_type for item in items}
    if len(items) >= 20 and len(types) == 1:
        problems.append(f"every question is of one type ({types.pop()})")
    return problems
