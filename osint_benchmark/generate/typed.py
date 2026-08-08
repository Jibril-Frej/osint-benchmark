"""Turn a typed candidate into a benchmark item, without a model ever seeing the answer.

The generic ``bridge`` type asks a model to write a question *and* its answer from two
documents, and then spends a judge, a gate suite and an ablation finding out whether the
answer is true and whether both documents were needed. The typed builders invert that: the
answer is computed from the joint graph — a shared affiliation, a person's full name — so no
model is in the gold path, and the model's only job is to phrase the situation.

That changes what the phraser may be told. It is given the passage and, for an association,
the two people's names; it is never given the answer, because a model handed the answer
writes it into the question. This is also why typed items do not go through
:func:`~osint_benchmark.generate.phrase.verify`: there is no model-written answer to check
against the documents, and asking a judge whether an organisation's article "supports" the
claim that two people both belong to it would reject nearly everything for the wrong reason.

**What replaces the judge here** is a set of conditions computed from the documents
themselves, ported from the previous project:

* the answer must not be named in the private document, or the public half is decoration;
* the two people must not appear in each other's article, or the pair is public knowledge;
* an entity with no article drops the item rather than passing it, because there the
  condition cannot be tested at all and a silent pass readmits exactly what it excludes;
* the resolution gold must survive :func:`~osint_benchmark.generate.resolution.check`.

**One thing the ablation cannot measure for these types, stated plainly.** The public
evidence cited is the *answer entity's* article. The documents that actually state an
association — the two people's own Wikidata records — are the ones you can only look up once
you know who the two people are, which is the private document's whole contribution; putting
them in the public-only ablation would hand that contribution to the solver and measure
nothing. The previous project ran the public-only ablation for these two types with no
public evidence at all, so its 171-of-180 "public-only fails" was a second closed-book run
rather than a measurement. This is the same limitation, named rather than reported as a
result.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from osint_benchmark.generate import passage, resolution
from osint_benchmark.generate.association import Association
from osint_benchmark.generate.chronology import Chronology
from osint_benchmark.generate.item import Evidence, Item
from osint_benchmark.generate.resolution import Resolution
from osint_benchmark.models import prompts
from osint_benchmark.models.backend import Complete, ModelUnavailable
from osint_benchmark.sources import refs

# How much of the passage the phraser is shown. The previous project's figure: enough for a
# meeting and who was in it, short enough that the model writes about the situation.
PASSAGE_CHARS = 1400

# Openings already used, shown back to the phraser so it does not write the same question
# forty times. Only this many, because the list is prompt text and grows with the run.
OPENINGS_SHOWN = 12

# Wikidata's class for a human being.
HUMAN = "Q5"

# How many candidates share one view of the openings already used. Fixed, so that the same
# recipe produces the same questions whatever it is run on -- see phrase_candidates.
CHUNK = 8


@dataclass(frozen=True)
class Candidate:
    """One typed question with its answer already computed, waiting for its wording.

    Attributes:
        item_id: Stable identifier, derived from the document and the entities involved.
        question_type: ``association`` or ``resolution``.
        answer: The gold, computed from the graph rather than written by a model.
        gold_qid: The entity the answer names.
        private_id: The confidential document, namespaced.
        public_id: The public record the answer is read from, namespaced.
        passage: The few hundred words around the mention, which is what the phraser sees.
        offsets: Where that passage sits in the private document, so the item can cite it
            without carrying a copy of it.
        facts: What the type's prompt needs, beyond the passage.
        verdicts: When the answer is a judgement rather than a computed value, the labels it
            may take. The model then decides the gold in the same call that writes the
            question, because whether a private position matches a public one cannot be
            settled before knowing what is being asked.
        provenance: How the candidate was arrived at, for a reviewer to check.
    """

    item_id: str
    question_type: str
    answer: str
    gold_qid: str
    private_id: str
    public_id: str
    passage: str
    offsets: tuple[int, int] = (0, 0)
    facts: dict = field(default_factory=dict)
    verdicts: tuple[str, ...] = ()
    provenance: dict = field(default_factory=dict)


def article_ref(qid: str) -> str:
    """Return the namespaced reference to an entity's Wikipedia article."""
    return f"enwiki:{qid}"


def people_in(facts: Iterable[dict]) -> set[str]:
    """Return the QIDs of the entities Wikidata calls human.

    Both types are about people: an organisation pair shares a trade body by coincidence
    rather than affiliation, and only a person is referred to by family name alone.
    """
    return {
        record["qid"]
        for record in facts
        if HUMAN in (record.get("statements") or {}).get("instance_of", ())
    }


def labels_in(facts: Iterable[dict]) -> dict[str, str]:
    """Return ``QID -> label`` for the entities that have one."""
    return {record["qid"]: record["label"] for record in facts if record.get("label")}


def articles_in(rows: Iterable[dict]) -> tuple[dict[str, str], dict[str, int]]:
    """Return ``QID -> lead text`` and ``QID -> whole-article size in bytes``."""
    text: dict[str, str] = {}
    size: dict[str, int] = {}
    for row in rows:
        qid = row.get("qid")
        if not qid:
            continue
        text[qid] = row.get("text", "")
        size[qid] = int(row.get("article_bytes") or 0)
    return text, size


def as_written(rows: Iterable[dict], texts: dict[str, str], outcomes: Counter) -> list[dict]:
    """Return the link rows with every mention rewritten to the document's own spelling.

    Applied to every private corpus rather than only the catalogued ones. Dodis records
    *which* entities a document concerns and not how it writes them, so without this its
    surfaces are the archivists' index form — "Petitpierre, Max" — which is never a bare
    family name and never appears in the text. But the check is worth running everywhere:
    it is the difference between a question about what a document says and a question about
    what a catalogue says it is about, and the cables are upper-case, so the form a question
    should quote is the one the document used.
    """
    out = []
    for row in rows:
        text = texts.get(row["doc_id"], "")
        if not text:
            outcomes["no_document_text"] += 1
            continue
        located = passage.locate(row, text)
        outcomes["mentions_located"] += len(located["entities"])
        outcomes["mentions_not_in_text"] += len(row.get("entities", [])) - len(located["entities"])
        if located["entities"]:
            out.append(located)
    return out


def names(text: str, label: str) -> bool:
    """Return whether a text uses a label, compared on normalised tokens."""
    return bool(label) and passage.normalise(label) in passage.normalise(text)


def remembering(source: dict[str, str]) -> Callable[[str], str]:
    """Return a lookup that normalises each text once and keeps the result.

    A document naming six people yields fifteen pairs and each condition below re-reads the
    same two articles and the same cable. Without this, normalising runs over gigabytes of
    text that has already been normalised.
    """
    seen: dict[str, str] = {}

    def normalised(key: str) -> str:
        if key not in seen:
            seen[key] = passage.normalise(source.get(key, ""))
        return seen[key]

    return normalised


def from_association(
    items: Iterable[Association],
    texts: dict[str, str],
    labels: dict[str, str],
    articles: dict[str, str],
    outcomes: Counter | None = None,
) -> Iterator[Candidate]:
    """Yield the associations whose necessity conditions hold, as candidates.

    ``texts`` maps a namespaced document reference to its text, ``labels`` a QID to its name
    and ``articles`` a QID to its article lead. Every drop is counted rather than silent: a
    builder that turns 4,000 associations into 12 items should say which condition did it.
    """
    if outcomes is None:
        outcomes = Counter()
    read_article = remembering(articles)
    read_document = remembering(texts)
    for item in items:
        private = texts.get(item.doc_id, "")
        if not private:
            outcomes["no_private_text"] += 1
            continue
        answer = labels.get(item.shared, "")
        if not answer or not articles.get(item.shared):
            # No article means no public evidence to cite and no public document the answer
            # can be read from, so there is nothing for the public half of the question.
            outcomes["gold_has_no_article"] += 1
            continue
        if not articles.get(item.a) or not articles.get(item.b):
            # The public co-occurrence test below cannot be run. Dropped rather than passed
            # by default, which would readmit the publicly obvious pairs this type excludes.
            outcomes["pair_untestable"] += 1
            continue
        if passage.normalise(labels.get(item.b, "")) in read_article(item.a) or passage.normalise(
            labels.get(item.a, "")
        ) in read_article(item.b):
            outcomes["pair_co_occurs_publicly"] += 1
            continue
        if passage.normalise(answer) in read_document(item.doc_id):
            # The private document names the affiliation itself, so the public record adds
            # nothing and a solver holding one document can answer.
            outcomes["gold_named_privately"] += 1
            continue
        outcomes["candidate"] += 1
        start, end = passage.span(private, item.a_surface)
        yield Candidate(
            item_id=f"{item.doc_id}|{item.a}|{item.b}|{item.shared}",
            question_type="association",
            answer=answer,
            gold_qid=item.shared,
            private_id=item.doc_id,
            public_id=article_ref(item.shared),
            passage=private[start:end],
            offsets=(start, end),
            facts={"a_label": labels.get(item.a, ""), "b_label": labels.get(item.b, "")},
            provenance={
                "a_qid": item.a,
                "b_qid": item.b,
                "predicate": item.predicate,
                "shared_degree": str(item.degree),
                # Naming either of them lets a public-only solver look the pair up, and the
                # question stops needing the confidential document. The prompt says not to;
                # the gate suite checks.
                "withheld": f"{labels.get(item.a, '')}; {labels.get(item.b, '')}",
            },
        )


def from_resolution(
    items: Iterable[Resolution],
    texts: dict[str, str],
    articles: dict[str, str],
    labels: dict[str, str] | None = None,
    margin: float = resolution.MARGIN,
    outcomes: Counter | None = None,
) -> Iterator[Candidate]:
    """Yield the resolutions whose gold survives the namesake check, as candidates.

    ``labels`` is only for the reviewer: the namesakes are recorded by name as well as by
    QID, because deciding whether the linker picked the right bearer means reading who the
    others were, and "Q1053159, Q2432, Q60849" is not something a person can check.
    """
    if outcomes is None:
        outcomes = Counter()
    for item in items:
        private = texts.get(item.doc_id, "")
        if not private:
            outcomes["no_private_text"] += 1
            continue
        if names(private, item.label):
            # The document spells the full name out somewhere, so the mention resolves
            # itself and the public catalogue is not needed.
            outcomes["full_name_in_document"] += 1
            continue
        if not articles.get(item.qid):
            outcomes["gold_has_no_article"] += 1
            continue
        start, end = passage.span(private, item.surface)
        around = private[start:end]
        verdict = resolution.check(item, around, articles, margin)
        if verdict.verdict != "verified":
            outcomes[f"gold_{verdict.verdict}"] += 1
            continue
        outcomes["candidate"] += 1
        yield Candidate(
            item_id=f"{item.doc_id}|{item.surface}|{item.qid}",
            question_type="resolution",
            answer=item.label,
            gold_qid=item.qid,
            private_id=item.doc_id,
            public_id=article_ref(item.qid),
            passage=around,
            offsets=(start, end),
            facts={"surface": item.surface, "bearers": str(len(item.candidates))},
            provenance={
                "namesakes": "; ".join(
                    (labels or {}).get(qid, qid) for qid in item.candidates if qid != item.qid
                ),
                "candidate_qids": " ".join(item.candidates),
                "prominence_rank": str(item.rank),
                "gold_article_bytes": str(item.article_bytes),
                "gold_score": str(verdict.gold_score),
                "rival_score": str(verdict.rival_score),
                "closest_rival": (labels or {}).get(verdict.rival, verdict.rival),
            },
        )


# What a posture question may be answered with. Four labels rather than two because a
# private position and a public one can genuinely half-agree, and because "the two documents
# are not about the same thing" is a correct answer rather than a failure to find one.
POSTURE_VERDICTS = ("Yes", "No", "Mixed", "Not enough evidence")

# How much of each document the phraser is shown. The previous project's figures: the
# confidential side is read closely, the parliamentary side is skimmed for its position.
PRIVATE_CHARS, PUBLIC_CHARS = 5000, 4000

# A chronology question shows only the opening of the cable: the interval is the answer, and
# the model is being asked to describe the episode, not to analyse it.
CHRONOLOGY_CHARS = 900


def from_chronology(
    items: Iterable[Chronology],
    texts: dict[str, str],
    outcomes: Counter | None = None,
) -> Iterator[Candidate]:
    """Yield one candidate per interval, with the number of days as its answer.

    The answer is written "N days" rather than as a bare number. A bare one-digit answer
    fails the gate that rejects one-character answers as parse failures, and "12" is not what
    a colleague would say back to you anyway.
    """
    if outcomes is None:
        outcomes = Counter()
    for item in items:
        private = texts.get(item.private_id, "")
        public = texts.get(item.public_id, "")
        if not private or not public:
            outcomes["no_evidence"] += 1
            continue
        outcomes["candidate"] += 1
        yield Candidate(
            item_id=f"{item.private_id}|{item.public_id}|days",
            question_type="chronology",
            answer=f"{item.days} days",
            gold_qid="",
            private_id=item.private_id,
            public_id=item.public_id,
            passage="",
            facts={
                "private_evidence": private[:CHRONOLOGY_CHARS],
                "public_title": item.title,
                "public_type": "parliamentary proceeding",
            },
            provenance={
                "days": str(item.days),
                "order": item.order,
                "shared_entities": "; ".join(item.shared),
                "private_subject": item.subject,
                "gold": "computed from two metadata dates; no model wrote it",
            },
        )


def from_posture(
    pairs: Iterable[dict],
    texts: dict[str, str],
    outcomes: Counter | None = None,
) -> Iterator[Candidate]:
    """Yield one candidate per focused pair, for the model to adjudicate.

    The only type here whose gold a model decides. Whether a position taken privately matches
    the one taken publicly cannot be settled before knowing what is being asked, so the
    question and the verdict are written in the same call — which is why this one keeps the
    rationale the model gives, and the computed types do not.
    """
    if outcomes is None:
        outcomes = Counter()
    for pair in pairs:
        if not pair.get("focused"):
            continue
        private = texts.get(pair["private_id"], "")
        public = texts.get(pair["public_id"], "")
        if not private or not public:
            outcomes["no_evidence"] += 1
            continue
        outcomes["candidate"] += 1
        yield Candidate(
            item_id=f"{pair['private_id']}|{pair['public_id']}|posture",
            question_type="posture",
            answer="",
            gold_qid="",
            private_id=pair["private_id"],
            public_id=pair["public_id"],
            passage="",
            facts={
                "private_evidence": private[:PRIVATE_CHARS],
                "private_date": pair["private_date"],
                "private_origin": pair.get("private_origin", ""),
                "public_evidence": public[:PUBLIC_CHARS],
                "public_date": pair["public_date"],
                "public_title": pair.get("public_title", ""),
                "public_type": pair.get("public_type", "parliamentary proceeding"),
            },
            verdicts=POSTURE_VERDICTS,
            provenance={
                "shared_entities": "; ".join(pair.get("shared_entities", ())),
                "shared_categories": "; ".join(pair.get("shared_cats", ())),
                "days_apart": str(pair.get("days_apart", "")),
                "gold": "adjudicated by the model that wrote the question",
            },
        )


def ask(candidate: Candidate, asker: str, openings: list[str], phraser: Complete) -> dict:
    """Return the phraser's reply for one candidate, parsed; empty when it wrote nothing.

    The prompt is chosen by type. ``openings`` is the list of first words already used in
    this run, shown back so the run does not collapse into one template — the previous
    project's defence against forty-one questions beginning the same way.

    Which values reach the prompt is decided by the prompt itself: everything the candidate
    can offer is assembled, then narrowed to what the file declares. That is what lets four
    types with quite different inputs share one call, while a prompt asking for something no
    type provides still fails loudly rather than rendering a literal brace.
    """
    name = f"phrase_{candidate.question_type}"
    offered = {
        "asker": asker,
        "used": ", ".join(sorted(set(openings))[:OPENINGS_SHOWN]) or "none yet",
        "passage": candidate.passage[:PASSAGE_CHARS],
        **candidate.facts,
    }
    declared = prompts.placeholders(name)
    reply = phraser(prompts.render(name, **{k: v for k, v in offered.items() if k in declared}))
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        written = json.loads(reply[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return written if isinstance(written, dict) else {}


def answer_for(candidate: Candidate, written: dict) -> str:
    """Return the item's gold: computed for most types, adjudicated for the rest.

    A type declaring ``verdicts`` is one whose answer is a judgement about two documents —
    whether a private position matches a public one — and that cannot be settled before
    knowing what the question is, so the model decides it in the same call. A reply naming
    no allowed label yields nothing rather than a guess.
    """
    if not candidate.verdicts:
        return candidate.answer
    verdict = str(written.get("verdict", "")).strip()
    match = [label for label in candidate.verdicts if label.lower() == verdict.lower()]
    return match[0] if match else ""


def to_item(candidate: Candidate, question: str, asker: str, model: str) -> Item:
    """Return the benchmark item for a phrased candidate."""
    source = refs.split(candidate.private_id)[0]
    return Item(
        item_id=candidate.item_id,
        question_type=candidate.question_type,
        question=question,
        answer=candidate.answer,
        rationale="",
        evidence=[
            Evidence(
                doc_id=candidate.private_id,
                source=source,
                side="private",
                offsets=candidate.offsets,
            ),
            Evidence(doc_id=candidate.public_id, source="wikipedia", side="public"),
        ],
        provenance={
            **candidate.provenance,
            "gold_qid": candidate.gold_qid,
            "asker": asker,
            "phraser": model,
            "gold": "computed from the joint graph; no model wrote it",
        },
    )


def phrase_candidates(
    candidates: Iterable[Candidate],
    phraser: Complete,
    askers: tuple[str, ...],
    model: str = "",
    outcomes: Counter | None = None,
    workers: int = 1,
) -> Iterator[Item]:
    """Yield one item per candidate the phraser could write a question for.

    The asker is chosen by position rather than at random, so a rerun of the same candidates
    produces the same questions and the release fingerprint does not move for no reason.

    ``workers`` requests are in flight at once. A reasoning model spends most of a minute
    per question and the server batches continuously, so issuing one request at a time
    leaves the GPUs mostly idle — the difference between an afternoon and an hour.

    The openings list is prompt text, so letting workers append to it as replies land would
    make the wording depend on which request finished first. Candidates are therefore taken
    in fixed groups of :data:`CHUNK`: everyone in a group sees the same openings, and the
    group's results extend the list in order afterwards. The group size is fixed rather than
    equal to ``workers`` on purpose — tied to ``workers`` it would make the questions depend
    on how much hardware the run had, and the same recipe would produce a different
    benchmark on a bigger machine.
    """
    if outcomes is None:
        outcomes = Counter()
    ordered = list(candidates)
    openings: list[str] = []
    for start in range(0, len(ordered), CHUNK):
        chunk = list(enumerate(ordered[start : start + CHUNK], start))
        seen = list(openings)

        def one(pair: tuple[int, Candidate], used: list[str] = seen) -> dict | None:
            """Return the phraser's reply for one candidate, or None if it failed."""
            index, candidate = pair
            try:
                return ask(candidate, askers[index % len(askers)], used, phraser)
            except ModelUnavailable as exc:
                print(f"  skipped {candidate.item_id}: {exc}", file=sys.stderr)
                return None

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                written = list(pool.map(one, chunk))
        else:
            written = [one(pair) for pair in chunk]

        for (index, candidate), reply in zip(chunk, written, strict=True):
            if reply is None:
                outcomes["phraser_error"] += 1
                continue
            question = str(reply.get("question", "")).strip() if reply else ""
            if not question:
                outcomes["unparseable_reply"] += 1
                continue
            answer = answer_for(candidate, reply)
            if not answer:
                # Only an adjudicated type reaches this: the model wrote a question and then
                # a verdict outside the labels it was given, so there is no gold to keep.
                outcomes["no_usable_verdict"] += 1
                continue
            openings.append(" ".join(question.split()[:4]))
            outcomes["phrased"] += 1
            item = to_item(candidate, question, askers[index % len(askers)], model)
            item.answer = answer
            item.rationale = str(reply.get("rationale", "")).strip()
            yield item
