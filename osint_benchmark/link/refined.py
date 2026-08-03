"""Link English prose to Wikidata entities with ReFinED.

ReFinED reads a document, finds mentions and resolves each to a QID in one pass. It
replaced spaCy plus a separate entity linker in the previous project on speed, recall and
mislink rate; installing it there needed transformers 4.46 plus two source patches, which
is why it is imported lazily and behind an interface.

**The model is injected.** A :data:`Linker` is anything that turns text into mentions, so
every stage downstream of this — the graph, the pairing, the gates — is testable with a
stub and needs no GPU, no 2 GB of weights and no install. :func:`load` is the only place
that touches ReFinED itself.

Two filters are applied to everything the model returns, and both cost recall on purpose,
because a noisy bridge seeds a bad question:

* a confidence floor, since ReFinED will resolve almost anything if asked;
* membership of the public entity set — an entity with no English Wikipedia article cannot
  bridge to a public corpus scoped to entities that have one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

DEFAULT_CONFIDENCE = 0.90

# Entity classes worth keeping as bridge candidates. Dates, quantities and ordinals are
# recognised by the model and are not entities anyone can ask a question about.
KEEP_TYPES = frozenset({"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "EVENT", "WORK_OF_ART"})


@dataclass(frozen=True)
class Mention:
    """One recognised entity mention.

    Attributes:
        qid: The resolved Wikidata entity.
        surface_form: The text as it appeared, before normalisation.
        confidence: The model's score for the resolution.
        entity_type: The coarse class, used to drop dates and quantities.
    """

    qid: str
    surface_form: str
    confidence: float
    entity_type: str = ""


# Text -> mentions. Injected so nothing downstream needs the model.
Linker = Callable[[str], Iterable[Mention]]


def keep(
    mentions: Iterable[Mention],
    entity_set: frozenset[str] | set[str] | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
) -> list[Mention]:
    """Return the mentions worth treating as bridge candidates.

    ``entity_set`` is the public entity universe from ``wikipedia_index``. Passing None
    skips that filter, which is only right when the caller has already applied it.
    """
    kept = []
    for mention in mentions:
        if mention.confidence < confidence:
            continue
        if mention.entity_type and mention.entity_type not in KEEP_TYPES:
            continue
        if entity_set is not None and mention.qid not in entity_set:
            continue
        kept.append(mention)
    return kept


def link_documents(
    documents: Iterable[dict],
    linker: Linker,
    side: str,
    entity_set: frozenset[str] | set[str] | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    text_field: str = "text",
) -> Iterator[dict]:
    """Yield one link row per document, in the shape the graph step reads.

    A document with no surviving mentions still yields a row with an empty list. The
    absence of links is a fact about the document; dropping it would make the coverage of
    the linker impossible to measure afterwards.
    """
    for document in documents:
        mentions = keep(linker(document.get(text_field) or ""), entity_set, confidence)
        deduplicated: dict[str, Mention] = {}
        for mention in mentions:
            best = deduplicated.get(mention.qid)
            if best is None or mention.confidence > best.confidence:
                deduplicated[mention.qid] = mention
        yield {
            "doc_id": document["doc_id"],
            "side": side,
            "entities": [
                {
                    "qid": m.qid,
                    "surface_form": m.surface_form,
                    "confidence": round(m.confidence, 4),
                }
                for m in sorted(deduplicated.values(), key=lambda m: m.qid)
            ],
        }


def load(model_name: str = "wikipedia_model_with_numbers", device: str = "cpu") -> Linker:
    """Return a :data:`Linker` backed by ReFinED.

    Imported here rather than at module scope so every test, and every stage that only
    needs the *shape* of a link, runs without the dependency installed.

    Raises:
        ImportError: With the install line, since this is the dependency that historically
            cost the most time to get working.
    """
    try:
        from refined.inference.processor import Refined  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "ReFinED is not installed. It is an optional dependency because it pulls "
            "torch and transformers: uv sync --extra link"
        ) from exc

    model = Refined.from_pretrained(model_name=model_name, entity_set="wikidata", device=device)

    def linker(text: str) -> Iterator[Mention]:
        """Resolve one document's mentions."""
        for span in model.process_text(text):
            entity = getattr(span, "predicted_entity", None)
            qid = getattr(entity, "wikidata_entity_id", None)
            if not qid:
                continue
            yield Mention(
                qid=qid,
                surface_form=span.text,
                confidence=float(getattr(span, "entity_linking_model_confidence_score", 0.0) or 0),
                entity_type=(getattr(span, "coarse_mention_type", "") or "").upper(),
            )

    return linker
