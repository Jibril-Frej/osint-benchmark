"""Write derived files, and record what each one kept from its source.

Two corpora were quietly reduced to a fraction of their source in the previous project
and nothing downstream could tell: the event store kept seven join keys and lost
casualties and party names; the sanctions parse kept four fields and lost the programme,
the justification and the listing dates. In both cases the projection was right for the
job it was written for and wrong for the job it was later used for, and the loss was
invisible because nothing recorded it.

So every derived file gets a sidecar naming its source, the fields it kept and the fields
it dropped **with a reason**. :func:`check_provenance` then fails on any source field that
is neither kept nor explained. The writer is the only public way to emit documents, which
is what makes this hold: per-file discipline does not, a check applied to everything does.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SIDECAR_SUFFIX = ".provenance.json"

# What a derived file is. The name should say which, so an index is never mistaken for a
# corpus again.
KINDS = ("raw", "corpus", "index", "projection", "derived")


@dataclass(frozen=True)
class Provenance:
    """What a derived file took from its source.

    Attributes:
        source: Where the data came from, specifically enough to find it again.
        source_fields: Every field the source offers, whether or not it was used.
        kept: Source field -> what it became here. A rename is explicit rather than
            looking like a loss.
        dropped: Source field -> why it was omitted. An empty reason is an error.
        kind: One of :data:`KINDS`.
        note: Anything a reader of the output would otherwise have to guess.
    """

    source: str
    source_fields: tuple[str, ...]
    kept: dict[str, str]
    dropped: dict[str, str] = field(default_factory=dict)
    kind: str = "corpus"
    note: str = ""


def sidecar_path(output: Path) -> Path:
    """Return the provenance sidecar path for a derived file."""
    return output.with_name(output.name + SIDECAR_SUFFIX)


def write_provenance(
    output: Path,
    provenance: Provenance,
    *,
    rows_in: int | None = None,
    rows_out: int | None = None,
) -> Path:
    """Write the sidecar beside a derived file and return its path.

    Raises:
        ValueError: If the provenance would not pass :func:`check_provenance` — a bad
            record is worse than none, because it reads as a guarantee.
    """
    data = {
        "output": str(output),
        "kind": provenance.kind,
        "source": provenance.source,
        "source_fields": sorted(provenance.source_fields),
        "kept": dict(sorted(provenance.kept.items())),
        "dropped": dict(sorted(provenance.dropped.items())),
        "rows_in": rows_in,
        "rows_out": rows_out,
        "note": provenance.note,
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    problems = check_provenance(data)
    if problems:
        raise ValueError(f"provenance for {output}: " + "; ".join(problems))
    path = sidecar_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def check_provenance(data: dict[str, Any]) -> list[str]:
    """Return the problems in one provenance record; an empty list means it is sound."""
    problems: list[str] = []
    for required in ("output", "kind", "source", "source_fields", "kept"):
        if required not in data:
            problems.append(f"missing field {required!r}")
    if problems:
        return problems

    if data["kind"] not in KINDS:
        problems.append(f"unknown kind {data['kind']!r}")

    source_fields = set(data["source_fields"])
    kept = set(data["kept"])  # keys are SOURCE field names; values are ours
    dropped = set(data.get("dropped") or {})

    unexplained = source_fields - kept - dropped
    if unexplained:
        problems.append(
            "source fields neither kept nor explained: " + ", ".join(sorted(unexplained))
        )

    # A kept key that is not a source field means the mapping is the wrong way round: the
    # keys must be the source's names, the values ours.
    invented = kept - source_fields
    if invented:
        problems.append(
            "kept keys that are not source fields (keys must be source names): "
            + ", ".join(sorted(invented))
        )

    for name, reason in (data.get("dropped") or {}).items():
        if not str(reason).strip():
            problems.append(f"field {name!r} dropped with no reason given")

    rows_in, rows_out = data.get("rows_in"), data.get("rows_out")
    if isinstance(rows_in, int) and isinstance(rows_out, int) and rows_out > rows_in:
        problems.append(f"rows_out ({rows_out}) exceeds rows_in ({rows_in})")

    return problems


def canonical(record: dict[str, Any]) -> str:
    """Return the one-line JSON form used both on disk and for hashing."""
    return json.dumps(record, ensure_ascii=False, sort_keys=True)


def record_hash(record: dict[str, Any]) -> str:
    """Return the SHA-256 of a record's canonical form.

    The whole record is hashed, not just its text, so a change to a date or a
    classification is caught too. A schema change therefore invalidates the pins, which
    is correct: it is a different artefact.
    """
    return hashlib.sha256(canonical(record).encode("utf-8")).hexdigest()


def write_records(
    output: Path,
    records: Iterable[dict[str, Any]],
    provenance: Provenance,
    *,
    rows_in: int | None = None,
) -> int:
    """Write records as JSONL with their sidecar, and return how many were written.

    Every record carries a ``doc_id`` unique within its source — that is what ``verify``
    keys on. Prose sources emit :meth:`~osint_benchmark.schema.Document.to_json`; tabular
    ones emit their own shape, since forcing a sanctions listing into a Document would
    invent a document that does not exist.

    The sidecar is written last, so a file without one is a run that died rather than a
    projection nobody documented.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical(record) + "\n")
            count += 1
    write_provenance(output, provenance, rows_in=rows_in, rows_out=count)
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield each record of a JSONL file."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)
