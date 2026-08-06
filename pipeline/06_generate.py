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
from collections import Counter

from osint_benchmark import paths
from osint_benchmark.artifacts import Provenance, read_jsonl, write_records
from osint_benchmark.generate import emit, phrase
from osint_benchmark.generate.evidence import entity_labels, evidence_texts, sources_for
from osint_benchmark.models import settings, stub, transcript
from osint_benchmark.models.backend import ModelUnavailable, vllm
from osint_benchmark.release.load import load_items


def main(argv: list[str] | None = None) -> int:
    """Draft, verify and emit questions."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="draft from only the first N pairs")
    parser.add_argument(
        "--stub",
        action="store_true",
        help="run with a scripted stand-in instead of a served model, to exercise the wiring",
    )
    parser.add_argument(
        "--draft",
        action="store_true",
        help=(
            "write unverified drafts and stop. Use with --verify as a second pass against "
            "a different served model: a judge from the phraser's own family is scoring "
            "its own output, and the two models do not fit on the GPUs together"
        ),
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="judge and gate the drafts from --draft, rather than writing new ones",
    )
    args = parser.parse_args(argv)
    if args.draft and args.verify:
        raise SystemExit("--draft and --verify are the two passes; run them one at a time")

    out = paths.data_dir() / "items"
    drafts_path = out / "drafted.jsonl"
    pairs_path = paths.data_dir() / "pairs" / "pairs.jsonl"
    if args.verify:
        if not drafts_path.exists():
            raise SystemExit(f"{drafts_path} is missing: run with --draft first")
    elif not pairs_path.exists():
        raise SystemExit(f"{pairs_path} is missing: run pipeline/05_pair.py first")

    phraser_settings = settings.load("phraser")
    judge_settings = settings.load("judge")
    if args.stub:
        phraser, judge = stub.phraser(), stub.judge()
        model_note = "STUB: no model was used. These questions are placeholders."
        print(model_note, file=sys.stderr)
    else:
        try:
            # Only the model this pass needs. In two-pass mode the other one is not
            # served -- it is 54 GB that has been torn down, or has not started yet.
            phraser = None if args.verify else vllm(phraser_settings)
            judge = None if args.draft else vllm(judge_settings)
        except ModelUnavailable as exc:
            print(exc, file=sys.stderr)
            return 1
        model_note = (
            f"Questions and answers written by {phraser_settings.model}, verified against "
            f"both documents by {judge_settings.model} at {judge_settings.samples} "
            "samples, unanimous only."
        )

    if transcript.transcript_path():
        print(f"transcribing model calls to {transcript.transcript_path()}")
    outcomes: Counter = Counter()
    samples = 1 if args.stub else judge_settings.samples

    if args.verify:
        drafts = load_items(drafts_path)
        judge = transcript.transcribed(judge, "judge")
        texts = evidence_texts(sources_for(drafts))
        items = list(phrase.verify_items(drafts, texts, judge, samples, outcomes))
        started, source = len(drafts), f"drafts from {drafts_path}"
    else:
        pairs = list(read_jsonl(pairs_path))
        if args.limit:
            pairs = pairs[: args.limit]
        phraser = transcript.transcribed(phraser, "phraser")
        texts, labels = evidence_texts(), entity_labels()
        if args.draft:
            items = list(phrase.draft_items(pairs, texts, labels, phraser, outcomes=outcomes))
        else:
            judge = transcript.transcribed(judge, "judge")
            items = list(
                phrase.build_items(pairs, texts, labels, phraser, judge, samples, outcomes=outcomes)
            )
        started, source = len(pairs), f"pairs from {pairs_path}"

    kept = outcomes["drafted"] if args.draft else outcomes["verified"]
    if started - kept:
        print(f"{started - kept} of {started} produced no question: {dict(outcomes)}")

    if args.draft:
        written = write_records(
            drafts_path,
            (item.to_json() for item in items),
            Provenance(
                source=source,
                source_fields=("private_id", "public_id", "qid"),
                kept={"private_id": "evidence", "public_id": "evidence", "qid": "bridge_qid"},
                kind="derived",
                note=(
                    f"Unverified drafts by {phraser_settings.model}. No judge has seen "
                    "these and no gate has been applied: run --verify next."
                ),
            ),
            rows_in=started,
        )
        print(f"{written} drafted of {started} pairs -> {drafts_path}")
        return 0

    accepted, rejected = emit.emit(
        items,
        out / "accepted.jsonl",
        out / "rejected.jsonl",
        Provenance(
            source=source,
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
    print(f"{accepted} accepted, {rejected} rejected of {started} -> {out}")

    for alarm in emit.run_alarms(items):
        print(f"RUN ALARM: {alarm}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
