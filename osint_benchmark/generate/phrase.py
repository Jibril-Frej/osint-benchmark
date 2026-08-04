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
from collections.abc import Iterator

from osint_benchmark.generate.item import Evidence, Item
from osint_benchmark.models import prompts
from osint_benchmark.models.backend import (
    Complete,
    ModelUnavailable,
    agree,
    first_word,
)

# Evidence is truncated to fit the context window. A cable can run to thousands of words
# and a served model answers a prompt longer than its window with a bare 400, killing the
# run. Truncating loses the tail of a long document; failing loses the whole run.
EVIDENCE_CHARS = 6000

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


def clip(text: str, limit: int = EVIDENCE_CHARS) -> str:
    """Return text bounded to a character budget, marked where it was cut."""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n[... truncated]"


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


def build_items(
    pairs: list[dict],
    texts: dict[str, str],
    labels: dict[str, str],
    phraser: Complete,
    judge: Complete,
    judge_samples: int = 3,
    question_type: str = "bridge",
) -> Iterator[Item]:
    """Yield one item per pair whose draft the judge can verify.

    Pairs whose evidence is missing are skipped rather than drafted from half a pair: a
    question written from one document cannot need two.
    """
    for pair in pairs:
        private_text = texts.get(pair["private_id"], "")
        public_text = texts.get(pair["public_id"], "")
        if not private_text or not public_text:
            continue

        bridge = labels.get(pair["qid"], pair["qid"])
        # One unusable pair must not end the run. A model that refuses, times out or is
        # handed something it cannot process costs that question, not the other 24.
        try:
            drafted = draft(pair, private_text, public_text, bridge, phraser)
        except ModelUnavailable as exc:
            print(f"  skipped {item_id(pair)}: {exc}", file=sys.stderr)
            continue
        if not drafted:
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
        try:
            verified = verify(item, private_text, public_text, judge, judge_samples)
        except ModelUnavailable as exc:
            print(f"  skipped {item.item_id}: {exc}", file=sys.stderr)
            continue
        if verified:
            yield item
