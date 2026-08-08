"""Step 6: write the questions. Needs the cluster and a GPU.

Two paths, because the benchmark has two kinds of question and they need opposite things
from the model.

The **bridge** path (the default) hands the model two documents and asks it for a question
*and* its answer. Every draft is then verified against the source documents, and only
unanimous judge verdicts count -- that is what replaces the correctness check a computed
answer would have had for free.

The **typed** path computes the answer from the joint graph first: the organisation two
people both belong to, the full name of a person a document calls by family name alone. The
model is shown the situation and never the answer, and writes only the question. There is no
judge, because there is no model-written answer to check; what stands in its place is a set
of conditions computed from the documents, applied in
:mod:`osint_benchmark.generate.typed`.

Usage::

    OSINT_MODEL_ENDPOINT=http://127.0.0.1:8080 uv run python pipeline/06_generate.py --limit 20
    OSINT_MODEL_ENDPOINT=http://127.0.0.1:8080 uv run python pipeline/06_generate.py \
        --types association,resolution --per-type 120

Serve the models named in config/models.toml first. Without an endpoint this reports what
to serve rather than failing per-question after an hour of work.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter

from osint_benchmark import paths
from osint_benchmark.artifacts import Provenance, read_jsonl, write_records
from osint_benchmark.generate import (
    association,
    chronology,
    emit,
    events,
    phrase,
    resolution,
    typed,
)
from osint_benchmark.generate.evidence import (
    entity_labels,
    evidence_texts,
    linked_sources,
    sources_for,
    sources_for_pairs,
)
from osint_benchmark.graph import entity_types
from osint_benchmark.models import settings, stub, transcript
from osint_benchmark.models.backend import ModelUnavailable, vllm
from osint_benchmark.pair import topical
from osint_benchmark.release.load import load_items
from osint_benchmark.sources import base, get_source, refs

TYPES = ("association", "resolution", "chronology", "posture", "event")

# The two types built on the cable-parliament join rather than on the Wikidata slice.
JOINED = ("chronology", "posture")


def links_for(sources: tuple[str, ...]) -> list[dict]:
    """Return the link rows of the named sources, if they were linked at all."""
    links = paths.data_dir() / "links"
    return [
        row
        for name in sources
        if (links / f"{name}.jsonl").exists()
        for row in read_jsonl(links / f"{name}.jsonl")
    ]


def private_links() -> list[dict]:
    """Return every link row from the confidential corpora.

    Read off the rows' own ``side`` rather than from a list of corpus names, so a corpus
    added later needs nothing changed here.
    """
    links = paths.data_dir() / "links"
    if not links.is_dir():
        raise SystemExit(f"{links} is missing: run pipeline/02_link.py first")
    return [
        row
        for path in sorted(links.glob("*.jsonl"))
        for row in read_jsonl(path)
        if row.get("side") == "private"
    ]


def sided(rows: list[dict], side: str) -> dict[str, list[str]]:
    """Return ``doc_id -> the QIDs it names``, for the link rows on one side."""
    return {
        row["doc_id"]: [e["qid"] for e in row.get("entities", ())]
        for row in rows
        if row.get("side") == side
    }


def topical_pairs(links: list[dict], facts: list[dict], outcomes: Counter) -> list[dict]:
    """Return the cable-parliament join the chronology and posture types rest on.

    Built here rather than in step 5 because it is a *topical* join and step 5 is an entity
    one: this needs the cables' State Department tags, the items' German subject categories
    and a date window, none of which a bridge map carries.
    """
    labels = typed.labels_in(facts)
    # Which entities are places, resolved the way step 3 does it: an entity is not an
    # instance of "geographic location" itself, it is an instance of "city" or "canton", so
    # the *classes* present are asked about once and every entity checked against those.
    # Naming the ancestor directly finds almost nothing, which is a silent way to make every
    # pair look focused.
    place_classes = entity_types.descendants_of(entity_types.classes_in(facts), entity_types.PLACE)
    places = {
        row["qid"]
        for row in facts
        if set((row.get("statements") or {}).get("instance_of", ())) & place_classes
    }
    print(f"{len(places)} of {len(facts)} entities are places, from {len(place_classes)} classes")

    public_qids = sided(links, "public")
    documents = {name: {} for name in ("cablegate", "parliament")}
    for name in documents:
        output = base.output_path(get_source(name))
        if output.exists():
            for row in read_jsonl(output):
                documents[name][refs.ref(name, row["doc_id"])] = row

    cables = []
    for row in links:
        if row.get("side") != "private" or not row["doc_id"].startswith("cablegate:"):
            continue
        record = documents["cablegate"].get(row["doc_id"])
        when = topical.parse_date(record.get("date") if record else None)
        if not record or not when:
            continue
        cables.append(
            {
                "doc_id": row["doc_id"],
                "date": when,
                "origin": (record.get("meta") or {}).get("origin", ""),
                "subject": topical.subject_of(record.get("text", "")),
                "tags": topical.tags_in(record.get("text", "")),
                "qids": [e["qid"] for e in row.get("entities", ())],
            }
        )

    business = []
    for doc_id, record in documents["parliament"].items():
        if record.get("entity") != "Business":
            continue
        when = topical.parse_date(record.get("SubmissionDate"))
        cats = {c.strip() for c in str(record.get("TagNames") or "").split("|") if c.strip()}
        if not when or not cats:
            continue
        business.append(
            {
                "doc_id": doc_id,
                "date": when,
                "title": record.get("Title") or "",
                "type": record.get("BusinessTypeName") or "",
                "cats": cats,
                "has_response": bool(record.get("FederalCouncilResponseText")),
                "qids": public_qids.get(doc_id, []),
            }
        )

    print(f"{len(cables)} dated cables, {len(business)} dated and categorised business items")
    pairs = list(topical.join(cables, business, labels, places, outcomes=outcomes))
    print(
        f"topical join: {len(pairs)} pairs, {sum(p['focused'] for p in pairs)} focused "
        f"({outcomes['no_mapped_topic']} cables had no mapped topic)"
    )
    return pairs


def dated_events(links: list[dict], sources: tuple[str, ...] = ("ucdp", "gdelt")) -> list[dict]:
    """Return the public event records that carry both a date and a linked entity.

    Read from the corpora rather than from a bridge map, because an event is matched on when
    and where it happened rather than on being named twice.
    """
    qids = {
        row["doc_id"]: [e["qid"] for e in row.get("entities", ())]
        for row in links
        if row.get("entities")
    }
    out = []
    for name in sources:
        source = get_source(name)
        output = base.output_path(source)
        if not output.exists():
            continue
        kept = 0
        for row in read_jsonl(output):
            doc_id = refs.ref(name, row["doc_id"])
            linked = qids.get(doc_id)
            when = topical.parse_date(refs.record_date(name, row))
            if not linked or not when:
                continue
            kept += 1
            out.append({**row, "doc_id": doc_id, "date": when, "qids": linked})
        print(f"{name}: {kept} dated and linked event records")
    return out


def typed_candidates(
    wanted: tuple[str, ...],
    per_type: int,
    seed: int,
    outcomes: Counter,
) -> list[typed.Candidate]:
    """Return the typed candidates, answers already computed, ready to be phrased.

    Sampled with a seeded shuffle rather than taken from the front. The corpora are ordered,
    so the first N cables are all from one decade and the first N Dodis documents from one
    volume; the previous project also found that reading the corpora as one stream let
    Cablegate, forty times larger and first, fill the cap on its own so Dodis was never
    reached. Each type is capped separately for the same reason.
    """
    facts_path = paths.data_dir() / "facts" / "wikidata.jsonl"
    articles_path = paths.data_dir() / "facts" / "articles.jsonl"
    if not facts_path.exists():
        raise SystemExit(f"{facts_path} is missing: run pipeline/04_public.py --scope linked")
    facts = list(read_jsonl(facts_path))
    articles, sizes = typed.articles_in(read_jsonl(articles_path) if articles_path.exists() else [])
    labels, people = typed.labels_in(facts), typed.people_in(facts)
    print(f"{len(facts)} entities, {len(people)} of them people, {len(articles)} with an article")

    texts = evidence_texts(linked_sources())
    rows = typed.as_written(private_links(), texts, outcomes)
    print(
        f"{len(rows)} private documents, {outcomes['mentions_located']} mentions found in "
        f"the text as written ({outcomes['mentions_not_in_text']} were not)"
    )

    rng = random.Random(seed)
    candidates: list[typed.Candidate] = []
    pairs = topical_pairs(rows, facts, outcomes) if set(wanted) & set(JOINED) else []
    if "chronology" in wanted:
        built = list(chronology.build(pairs))
        rng.shuffle(built)
        kept = list(typed.from_chronology(built, texts, outcomes))
        print(f"chronology: {len(built)} intervals, {len(kept)} with both documents")
        candidates += kept[:per_type]
    if "event" in wanted:
        dated = [
            {**row, "date": topical.parse_date(refs.record_date(refs.split(row["doc_id"])[0], row))}
            for row in rows
        ]
        matches = list(
            events.build(
                [row for row in dated if row["date"]],
                dated_events(links_for(("ucdp", "gdelt"))),
                labels,
                outcomes=outcomes,
            )
        )
        rng.shuffle(matches)
        kept = list(typed.from_events(matches, texts, outcomes))
        print(f"event: {len(matches)} matched anchors, {len(kept)} with evidence")
        candidates += kept[:per_type]
    if "posture" in wanted:
        adjudicable = list(typed.from_posture(pairs, texts, outcomes))
        rng.shuffle(adjudicable)
        print(f"posture: {len(adjudicable)} focused pairs to adjudicate")
        candidates += adjudicable[:per_type]
    if "association" in wanted:
        graph = association.relations(facts)
        built = list(association.build(rows, graph, people))
        rng.shuffle(built)
        kept = list(typed.from_association(built, texts, labels, articles, outcomes))
        print(f"association: {len(built)} raw, {len(kept)} survive the conditions")
        candidates += kept[:per_type]
    if "resolution" in wanted:
        built = list(resolution.build(rows, labels, people, sizes))
        rng.shuffle(built)
        kept = list(typed.from_resolution(built, texts, articles, labels, outcomes=outcomes))
        print(f"resolution: {len(built)} raw, {len(kept)} survive the conditions")
        candidates += kept[:per_type]
    return candidates


def run_typed(args: argparse.Namespace, wanted: tuple[str, ...]) -> int:
    """Build, phrase, gate and write the typed questions; return a process exit code."""
    phraser_settings = settings.load("phraser")
    if args.stub:
        phraser = stub.phraser()
        model_note = "STUB: no model was used. These questions are placeholders."
        print(model_note, file=sys.stderr)
    else:
        try:
            phraser = vllm(phraser_settings)
        except ModelUnavailable as exc:
            print(exc, file=sys.stderr)
            return 1
        model_note = (
            f"Answers computed from the Wikidata slice and the link files; no model wrote "
            f"one. Questions phrased by {phraser_settings.model}, which was never shown the "
            "answer. Necessity is measured in step 7 and is not assumed here."
        )

    outcomes: Counter = Counter()
    candidates = typed_candidates(wanted, args.per_type, args.seed, outcomes)
    print(f"{len(candidates)} candidates to phrase: {dict(outcomes)}")
    if not candidates:
        print("no candidates: nothing to phrase", file=sys.stderr)
        return 1

    if transcript.transcript_path():
        print(f"transcribing model calls to {transcript.transcript_path()}")
    model = "stub" if args.stub else phraser_settings.model
    items = list(
        typed.phrase_candidates(
            candidates,
            transcript.transcribed(phraser, "phraser"),
            phrase.ASKERS,
            model=model,
            outcomes=outcomes,
            workers=args.workers,
        )
    )

    out = paths.data_dir() / "items"
    accepted, rejected = emit.emit(
        items,
        out / "accepted.jsonl",
        out / "rejected.jsonl",
        Provenance(
            source=f"link files and the Wikidata slice, types {', '.join(wanted)}",
            source_fields=("doc_id", "entities", "statements", "label"),
            kept={
                "doc_id": "evidence (private side)",
                "entities": "the mentions a question is built on",
                "statements": "the shared affiliation, which is the answer",
                "label": "the answer's wording",
            },
            kind="derived",
            note=model_note,
        ),
    )
    print(f"{accepted} accepted, {rejected} rejected of {len(candidates)} -> {out}")
    print(f"outcomes: {dict(outcomes)}")
    for alarm in emit.run_alarms(items):
        print(f"RUN ALARM: {alarm}", file=sys.stderr)
    return 0


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
    parser.add_argument(
        "--types",
        help=(
            "build typed questions instead of bridge ones, from a comma-separated list of "
            f"{', '.join(TYPES)}. Their answers are computed from the graph, so there is no "
            "judge and no draft/verify split"
        ),
    )
    parser.add_argument(
        "--per-type",
        type=int,
        default=200,
        help="with --types, how many candidates of each type to phrase",
    )
    parser.add_argument("--seed", type=int, default=0, help="which sample of candidates to take")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "with --types, how many phrasing requests to have in flight. The server "
            "batches continuously, so one at a time leaves the GPUs mostly idle"
        ),
    )
    args = parser.parse_args(argv)
    if args.draft and args.verify:
        raise SystemExit("--draft and --verify are the two passes; run them one at a time")
    wanted = tuple(name.strip() for name in (args.types or "").split(",") if name.strip())
    if unknown := [name for name in wanted if name not in TYPES]:
        raise SystemExit(f"unknown question type {', '.join(unknown)}; known: {', '.join(TYPES)}")
    if wanted and (args.draft or args.verify):
        raise SystemExit("--types computes its answers, so there is nothing to draft or verify")
    if wanted:
        return run_typed(args, wanted)

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
        # From the pairs, not from data/links: a run reusing a saved pair set has no
        # link files, and globbing for them loaded nothing at all -- 100 of 100 pairs
        # reported as having no evidence, on a job that had already served the model.
        sources = sources_for_pairs(pairs)
        print(f"evidence from: {', '.join(sources) or 'nothing cited'}")
        texts, labels = evidence_texts(sources), entity_labels()
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
