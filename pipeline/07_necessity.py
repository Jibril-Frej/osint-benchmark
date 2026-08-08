"""Step 7: measure whether each question needs both sides. Needs the cluster and a GPU.

Re-solves finished questions three ways and records the outcomes. It changes nothing about
the questions, which is why it is a separate step from writing them.

Usage::

    OSINT_MODEL_ENDPOINT=http://127.0.0.1:8080 uv run python pipeline/07_necessity.py
    OSINT_MODEL_ENDPOINT=... uv run python pipeline/07_necessity.py --control

Run --control first. It feeds the solver evidence that plainly contains the answer; a
solver that cannot answer THAT is broken, and a broken solver makes every question look
perfectly necessary.
"""

from __future__ import annotations

import argparse
import sys

from osint_benchmark import paths
from osint_benchmark.artifacts import Provenance, write_records
from osint_benchmark.generate.evidence import evidence_texts, sources_for
from osint_benchmark.models import settings, stub, transcript
from osint_benchmark.models.backend import ModelUnavailable, vllm
from osint_benchmark.necessity import ablate
from osint_benchmark.release.load import load_items


def main(argv: list[str] | None = None) -> int:
    """Measure necessity over the accepted questions."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--control", action="store_true", help="check the solver is not broken")
    parser.add_argument(
        "--stub",
        action="store_true",
        help="run with a scripted stand-in instead of a served model, to exercise the wiring",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "how many questions to measure at once. This step makes more model calls than "
            "any other and every one is independent, so one at a time wastes the server's "
            "batching"
        ),
    )
    args = parser.parse_args(argv)

    solver_settings = settings.load("solver")
    judge_settings = settings.load("judge")
    if args.stub:
        solver, judge = stub.solver(), None
        model_note = "STUB: no model was used. These flags are placeholders."
        print(model_note, file=sys.stderr)
    else:
        try:
            solver = vllm(solver_settings)
            # The same served model in both roles here, unlike step 6: this judge is not
            # scoring anybody's output, only deciding whether two answers say the same
            # thing, so there is no self-enhancement to avoid.
            judge = vllm(judge_settings) if judge_settings.model == solver_settings.model else None
        except ModelUnavailable as exc:
            print(exc, file=sys.stderr)
            return 1
        if judge is None:
            print(
                f"WARNING: judge is {judge_settings.model} but {solver_settings.model} is "
                "served; ablation answers will not be checked for correctness",
                file=sys.stderr,
            )
        model_note = (
            f"Necessity measured by {solver_settings.model}: closed-book, public-only and "
            "private-only, each ablation answer checked against the gold answer. "
            "Recorded, never used to drop an item."
        )

    solver = transcript.transcribed(solver, "solver")
    if judge is not None:
        judge = transcript.transcribed(judge, "equivalence")
    if transcript.transcript_path():
        print(f"transcribing model calls to {transcript.transcript_path()}")

    if args.control:
        ok = ablate.control(solver)
        print(
            f"control: solver {'answered' if ok else 'FAILED to answer'} a question whose "
            "evidence plainly contains the answer"
        )
        return 0 if ok else 1

    accepted = paths.data_dir() / "items" / "accepted.jsonl"
    if not accepted.exists():
        raise SystemExit(f"{accepted} is missing: run pipeline/06_generate.py first")

    items = load_items(accepted)
    # The corpora are read off the items rather than off whatever link files are lying
    # about, so this step needs only step 1 and an accepted.jsonl. That is what lets a run
    # that wrote its questions and then failed here be resumed for the cost of the
    # measurement, instead of repeating four hours of linking and drafting.
    samples = 1 if args.stub else judge_settings.samples
    sources = sources_for(items)
    print(f"evidence from: {', '.join(sources) or 'nothing cited'}")
    measured = list(
        ablate.measure_items(
            items, evidence_texts(sources), solver, judge, samples, workers=args.workers
        )
    )
    output = paths.data_dir() / "items" / "measured.jsonl"
    write_records(
        output,
        (item.to_json() for item in measured),
        Provenance(
            source=str(accepted),
            source_fields=("item_id", "question", "answer", "evidence"),
            kept={
                "item_id": "item_id",
                "question": "question",
                "answer": "answer",
                "evidence": "evidence",
            },
            kind="derived",
            note=model_note,
        ),
    )
    needs_both = sum(1 for i in measured if i.necessity.needs_both)
    print(f"{len(measured)} measured, {needs_both} need both sides -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
