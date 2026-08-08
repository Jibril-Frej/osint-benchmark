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
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from osint_benchmark.generate import passage, resolution
from osint_benchmark.generate.association import Association
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
        facts: What the type's prompt needs, beyond the passage.
        provenance: How the candidate was arrived at, for a reviewer to check.
    """

    item_id: str
    question_type: str
    answer: str
    gold_qid: str
    private_id: str
    public_id: str
    passage: str
    facts: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)


def article_ref(qid: str) -> str:
    """Return the namespaced reference to an entity's Wikipedia article."""
    return f"enwiki:{qid}"


def names(text: str, label: str) -> bool:
    """Return whether a text uses a label, compared on normalised tokens."""
    return bool(label) and passage.normalise(label) in passage.normalise(text)


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
        if names(articles[item.a], labels.get(item.b, "")) or names(
            articles[item.b], labels.get(item.a, "")
        ):
            outcomes["pair_co_occurs_publicly"] += 1
            continue
        if names(private, answer):
            # The private document names the affiliation itself, so the public record adds
            # nothing and a solver holding one document can answer.
            outcomes["gold_named_privately"] += 1
            continue
        outcomes["candidate"] += 1
        yield Candidate(
            item_id=f"{item.doc_id}|{item.a}|{item.b}|{item.shared}",
            question_type="association",
            answer=answer,
            gold_qid=item.shared,
            private_id=item.doc_id,
            public_id=article_ref(item.shared),
            passage=passage.window(private, item.a_surface),
            facts={"a_label": labels.get(item.a, ""), "b_label": labels.get(item.b, "")},
            provenance={
                "a_qid": item.a,
                "b_qid": item.b,
                "predicate": item.predicate,
                "shared_degree": str(item.degree),
            },
        )


def from_resolution(
    items: Iterable[Resolution],
    texts: dict[str, str],
    articles: dict[str, str],
    margin: float = resolution.MARGIN,
    outcomes: Counter | None = None,
) -> Iterator[Candidate]:
    """Yield the resolutions whose gold survives the namesake check, as candidates."""
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
        around = passage.window(private, item.surface)
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
            facts={"surface": item.surface, "bearers": str(len(item.candidates))},
            provenance={
                "candidate_qids": " ".join(item.candidates),
                "prominence_rank": str(item.rank),
                "gold_article_bytes": str(item.article_bytes),
                "gold_score": str(verdict.gold_score),
                "rival_score": str(verdict.rival_score),
            },
        )


def ask(candidate: Candidate, asker: str, openings: list[str], phraser: Complete) -> str:
    """Return the phraser's question for one candidate, or empty if it wrote none.

    The prompt is chosen by type and never carries the answer. ``openings`` is the list of
    first words already used in this run, shown back so the run does not collapse into one
    template — the previous project's defence against forty-one questions beginning the
    same way.
    """
    used = ", ".join(sorted(set(openings))[:OPENINGS_SHOWN]) or "none yet"
    reply = phraser(
        prompts.render(
            f"phrase_{candidate.question_type}",
            passage=candidate.passage[:PASSAGE_CHARS],
            asker=asker,
            used=used,
            **candidate.facts,
        )
    )
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end <= start:
        return ""
    try:
        written = json.loads(reply[start : end + 1])
    except json.JSONDecodeError:
        return ""
    return str(written.get("question", "")).strip() if isinstance(written, dict) else ""


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
            Evidence(doc_id=candidate.private_id, source=source, side="private"),
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
) -> Iterator[Item]:
    """Yield one item per candidate the phraser could write a question for.

    The asker is chosen by position rather than at random, so a rerun of the same candidates
    produces the same questions and the release fingerprint does not move for no reason.
    """
    if outcomes is None:
        outcomes = Counter()
    openings: list[str] = []
    for index, candidate in enumerate(candidates):
        asker = askers[index % len(askers)]
        try:
            question = ask(candidate, asker, openings, phraser)
        except ModelUnavailable as exc:
            outcomes["phraser_error"] += 1
            print(f"  skipped {candidate.item_id}: {exc}", file=sys.stderr)
            continue
        if not question:
            outcomes["unparseable_reply"] += 1
            continue
        openings.append(" ".join(question.split()[:4]))
        outcomes["phrased"] += 1
        yield to_item(candidate, question, asker, model)
