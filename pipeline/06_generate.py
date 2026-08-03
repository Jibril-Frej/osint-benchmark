"""Step 6: write a question and its answer for each pair. Needs the cluster and a GPU.

The model writes both. Every draft is verified against the source documents before it is
kept, and only unanimous judge verdicts count -- that is what replaces the correctness
check a computed answer would have had for free.

Usage::

    OSINT_MODEL_ENDPOINT=http://127.0.0.1:8080 uv run python pipeline/06_generate.py --limit 20

Serve the models named in config/models.toml first. Without an endpoint this reports what
to serve rather than failing per-question after an hour of work.
"""

from __future__ import annotations

import argparse
import sys

from osint_benchmark import paths
from osint_benchmark.artifacts import Provenance, read_jsonl
from osint_benchmark.generate import emit, phrase
from osint_benchmark.generate.evidence import entity_labels, evidence_texts
from osint_benchmark.models import settings, stub
from osint_benchmark.models.backend import ModelUnavailable, llama_server


def main(argv: list[str] | None = None) -> int:
    """Draft, verify and emit questions."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="draft from only the first N pairs")
    parser.add_argument(
        "--stub",
        action="store_true",
        help="run with a scripted stand-in instead of a served model, to exercise the wiring",
    )
    args = parser.parse_args(argv)

    pairs_path = paths.data_dir() / "pairs" / "pairs.jsonl"
    if not pairs_path.exists():
        raise SystemExit(f"{pairs_path} is missing: run pipeline/05_pair.py first")

    phraser_settings = settings.load("phraser")
    judge_settings = settings.load("judge")
    if args.stub:
        phraser, judge = stub.phraser(), stub.judge()
        model_note = "STUB: no model was used. These questions are placeholders."
        print(model_note, file=sys.stderr)
    else:
        try:
            phraser = llama_server(phraser_settings)
            judge = llama_server(judge_settings)
        except ModelUnavailable as exc:
            print(exc, file=sys.stderr)
            return 1
        model_note = (
            f"Questions and answers written by {phraser_settings.model}, verified against "
            f"both documents by {judge_settings.model} at {judge_settings.samples} "
            "samples, unanimous only."
        )

    pairs = list(read_jsonl(pairs_path))
    if args.limit:
        pairs = pairs[: args.limit]

    items = list(
        phrase.build_items(
            pairs,
            evidence_texts(),
            entity_labels(),
            phraser,
            judge,
            judge_samples=1 if args.stub else judge_settings.samples,
        )
    )

    out = paths.data_dir() / "items"
    accepted, rejected = emit.emit(
        items,
        out / "accepted.jsonl",
        out / "rejected.jsonl",
        Provenance(
            source=f"pairs from {pairs_path}",
            source_fields=("private_id", "public_id", "qid"),
            kept={
                "private_id": "evidence (private side)",
                "public_id": "evidence (public side)",
                "qid": "provenance.bridge_qid",
            },
            kind="derived",
            note=model_note,
        ),
    )
    print(f"{accepted} accepted, {rejected} rejected of {len(pairs)} pairs -> {out}")

    for alarm in emit.run_alarms(items):
        print(f"RUN ALARM: {alarm}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
