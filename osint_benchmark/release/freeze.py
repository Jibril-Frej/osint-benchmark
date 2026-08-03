"""Freeze a version of the benchmark.

What ships is what this project authored: questions, answers, rationales, gate outcomes,
necessity flags and evidence *pointers*. Never corpus text -- evidence is carried as
``(doc_id, offsets)``, so a release contains nothing that has to be redistributed and a
user rebuilds the corpora from ``pins/`` instead.

The datasheet is written from the release, not by hand. A hand-written one describes the
set somebody meant to build.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from osint_benchmark.artifacts import canonical
from osint_benchmark.generate.item import Item


def digest(items: Iterable[dict]) -> str:
    """Return one sha256 over every released item, in order."""
    running = hashlib.sha256()
    for item in items:
        running.update(canonical(item).encode("utf-8"))
        running.update(b"\n")
    return running.hexdigest()


def datasheet(items: list[Item], corpora: dict[str, dict]) -> dict:
    """Return the release's datasheet: what is in it, and what is known about it."""
    measured = [i for i in items if i.necessity.measured]
    return {
        "questions": len(items),
        "types": {
            kind: sum(1 for i in items if i.question_type == kind)
            for kind in sorted({i.question_type for i in items})
        },
        "necessity": {
            "measured": len(measured),
            "needs_both": sum(1 for i in measured if i.necessity.needs_both),
            "answerable_closed_book": sum(1 for i in measured if i.necessity.closed_book),
            "answerable_public_only": sum(1 for i in measured if i.necessity.public_only),
            "answerable_private_only": sum(1 for i in measured if i.necessity.private_only),
        },
        "gates": {
            name: sum(1 for i in items if i.gates.get(name))
            for name in sorted({g for i in items for g in i.gates})
        },
        "corpora": corpora,
        "evidence": (
            "Carried as (doc_id, offsets). No corpus text is included; rebuild the corpora "
            "from pins/ with pipeline/01_sources.py."
        ),
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def freeze(items: list[Item], corpora: dict[str, dict], out_dir: Path) -> dict:
    """Write the release and return its datasheet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    records = [item.to_json() for item in items]

    questions = out_dir / "questions.jsonl"
    with questions.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical(record) + "\n")

    sheet = datasheet(items, corpora)
    sheet["sha256"] = digest(records)
    (out_dir / "datasheet.json").write_text(
        json.dumps(sheet, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return sheet
