"""Write a question and its answer from one pair of documents.

The model does both. That is the decision this project made over computing the answer from
the public record: a computed answer cannot be hallucinated, but it makes the private
document a lookup key rather than evidence — and the two previous types whose gold was a
register field were also the two most answerable from the public side alone, at 10 of 34
and 6 of 10.

What a model-written answer loses is the free correctness check, so three things stand in
for it: :func:`verify` re-reads the source documents and rejects an answer they do not
support, the judge is asked repeatedly and only a unanimous verdict counts, and step 7
measures necessity rather than assuming it.

The asker is varied on purpose. An intelligence requirement is posed by a consumer with a
decision to make, and varying that consumer varies register and phrasing — which is the
main defence against a run collapsing into one template.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable, Iterator

from osint_benchmark.generate.evidence import clip
from osint_benchmark.generate.item import Evidence, Item
from osint_benchmark.models import prompts
from osint_benchmark.models.backend import (
    Complete,
    ModelUnavailable,
    agree,
    first_word,
)

ASKERS = (
    "a desk officer preparing a country brief",
    "an analyst supporting a sanctions review",
    "a policy adviser drafting talking points for a bilateral meeting",
    "a researcher compiling a subject profile",
    "an editor fact-checking a story about the subject",
)


def asker_for(item_id: str) -> str:
    """Return the asker for an item, chosen deterministically from its id.

    Deterministic so a rerun produces the same question rather than a differently-phrased
    one, which would change the corpus fingerprint for no reason.
    """
    digest = hashlib.sha256(item_id.encode()).digest()
    return ASKERS[digest[0] % len(ASKERS)]


def item_id(pair: dict) -> str:
    """Return a stable id for the item built from one pair."""
    return f"{pair['private_id']}|{pair['public_id']}|{pair['qid']}"


def draft(pair: dict, private_text: str, public_text: str, bridge: str, phraser: Complete) -> dict:
    """Ask the model for one question and its answer, as JSON.

    Returns an empty dict when the reply is not usable — truncated, or not JSON. A
    malformed reply is a failure to produce a question, not a question to repair.
    """
    prompt = prompts.render(
        "generate_question",
        private_evidence=clip(private_text),
        public_evidence=clip(public_text),
        bridge=bridge,
        asker=asker_for(item_id(pair)),
    )
    reply = phraser(prompt)
    if not reply:
        return {}
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        drafted = json.loads(reply[start : end + 1])
    except json.JSONDecodeError:
        return {}
    if not isinstance(drafted, dict) or not drafted.get("question") or not drafted.get("answer"):
        return {}
    return drafted


def verify(item: Item, private_text: str, public_text: str, judge: Complete, samples: int) -> bool:
    """Return whether the documents actually support the answer.

    This is what replaces the correctness check a computed answer got for free. The
    previous project's generator, without it, gave a Federal Councillor who left office in
    1893 an exit year of 1943.
    """
    prompt = prompts.render(
        "verify_answer",
        private_evidence=clip(private_text),
        public_evidence=clip(public_text),
        question=item.question,
        answer=item.answer,
    )
    verdict = agree(judge, prompt, samples, lambda r: first_word(r, ("SUPPORTED", "UNSUPPORTED")))
    return verdict == "supported"


def draft_items(
    pairs: list[dict],
    texts: dict[str, str],
    labels: dict[str, str],
    phraser: Complete,
    question_type: str = "bridge",
    outcomes: Counter | None = None,
) -> Iterator[Item]:
    """Yield one unverified item per pair the phraser could write a question from.

    Separate from verification so the two can run against *different models*, which they
    must: a judge from the phraser's own family is scoring its own output, and 62 GB of
    phraser plus 54 GB of judge does not fit in 80 GB of GPU. Drafting and judging in one
    loop forced one model to do both.

    Pairs whose evidence is missing are skipped rather than drafted from half a pair: a
    question written from one document cannot need two.

    ``outcomes`` is counted into, and the caller should report it. Every way of losing a
    pair here used to be a bare ``continue``, so a run that turned 80 pairs into one
    question said only "1 accepted, 0 rejected" and gave no hint which of four quite
    different things had happened — the answer took a model transcript that is off by
    default. A count is not a diagnosis, but it names the stage.
    """
    if outcomes is None:
        outcomes = Counter()
    for pair in pairs:
        private_text = texts.get(pair["private_id"], "")
        public_text = texts.get(pair["public_id"], "")
        if not private_text or not public_text:
            outcomes["no_evidence"] += 1
            continue
        if private_text == public_text:
            # Both sides resolved to the same document. A question cannot need two
            # documents when it has one, and the pipeline used to build them anyway: bare
            # doc_ids collide across corpora, so the "public record" for sanctions target
            # 47703 was cable 47703. Refuse rather than draft from it.
            outcomes["same_document"] += 1
            continue

        bridge = labels.get(pair["qid"], pair["qid"])
        # One unusable pair must not end the run. A model that refuses, times out or is
        # handed something it cannot process costs that question, not the other 24.
        try:
            drafted = draft(pair, private_text, public_text, bridge, phraser)
        except ModelUnavailable as exc:
            outcomes["phraser_error"] += 1
            print(f"  skipped {item_id(pair)}: {exc}", file=sys.stderr)
            continue
        if not drafted:
            # The commonest loss by far, and the least obvious: a reasoning model given
            # too small a token budget spends all of it inside its <think> block and is
            # truncated before it ever emits the JSON. The reply is not empty, so nothing
            # upstream looks wrong. 79 of 80 pairs went this way on job 13349.
            outcomes["unparseable_draft"] += 1
            continue

        item = Item(
            item_id=item_id(pair),
            question_type=question_type,
            question=str(drafted["question"]).strip(),
            answer=str(drafted["answer"]).strip(),
            rationale=str(drafted.get("reasoning", "")).strip(),
            evidence=[
                Evidence(doc_id=pair["private_id"], source="private", side="private"),
                Evidence(doc_id=pair["public_id"], source="public", side="public"),
            ],
            provenance={"bridge_qid": pair["qid"], "asker": asker_for(item_id(pair))},
        )
        outcomes["drafted"] += 1
        yield item


def verify_items(
    items: Iterable[Item],
    texts: dict[str, str],
    judge: Complete,
    judge_samples: int = 3,
    outcomes: Counter | None = None,
) -> Iterator[Item]:
    """Yield the drafted items the judge can verify against their own documents.

    Reads the evidence back off each item, so this runs from a written draft file and a
    served judge and needs nothing else — which is what lets the judge be a different
    model, loaded after the phraser has been torn down.
    """
    if outcomes is None:
        outcomes = Counter()
    for item in items:
        private_text = " ".join(texts.get(e.doc_id, "") for e in item.private_evidence)
        public_text = " ".join(texts.get(e.doc_id, "") for e in item.public_evidence)
        if not private_text or not public_text:
            outcomes["no_evidence"] += 1
            continue
        try:
            verified = verify(item, private_text, public_text, judge, judge_samples)
        except ModelUnavailable as exc:
            outcomes["judge_error"] += 1
            print(f"  skipped {item.item_id}: {exc}", file=sys.stderr)
            continue
        if verified:
            outcomes["verified"] += 1
            yield item
        else:
            # Counted, because a rejected draft reaches neither accepted.jsonl nor
            # rejected.jsonl: those hold what passed and failed the *gates*, and this
            # never gets that far. Silently, it looked like the pair had never existed.
            outcomes["judge_rejected"] += 1


def build_items(
    pairs: list[dict],
    texts: dict[str, str],
    labels: dict[str, str],
    phraser: Complete,
    judge: Complete,
    judge_samples: int = 3,
    question_type: str = "bridge",
    outcomes: Counter | None = None,
) -> Iterator[Item]:
    """Draft and verify in one pass, for when one served model does both.

    The single-model path: the stub runs, the smoke run, and any profile where phraser and
    judge are the same. A release cannot be built this way — scoring your own output is
    self-enhancement bias, not evaluation — so the two-pass path exists alongside it.
    """
    if outcomes is None:
        outcomes = Counter()
    drafted = draft_items(pairs, texts, labels, phraser, question_type, outcomes)
    yield from verify_items(drafted, texts, judge, judge_samples, outcomes)
