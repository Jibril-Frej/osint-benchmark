"""Read items back from disk.

Shared by the review and release steps: both read a finished item file, and neither should
own the schema knowledge.
"""

from __future__ import annotations

from pathlib import Path

from osint_benchmark.artifacts import read_jsonl
from osint_benchmark.generate.item import Evidence, Item, Necessity


def load_items(path: Path) -> list[Item]:
    """Return the items in a written item file."""
    items = []
    for row in read_jsonl(path):
        items.append(
            Item(
                item_id=row["item_id"],
                question_type=row["question_type"],
                question=row["question"],
                answer=row["answer"],
                rationale=row.get("rationale", ""),
                evidence=[
                    Evidence(
                        doc_id=e["doc_id"],
                        source=e["source"],
                        side=e["side"],
                        revision=e.get("revision"),
                        offsets=tuple(e["offsets"]) if e.get("offsets") else None,
                    )
                    for e in row.get("evidence", [])
                ],
                necessity=Necessity(**(row.get("necessity") or {})),
                gates=row.get("gates", {}),
                provenance=row.get("provenance", {}),
            )
        )
    return items
