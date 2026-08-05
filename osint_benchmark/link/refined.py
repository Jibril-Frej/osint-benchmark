"""Link English prose to Wikidata entities with ReFinED.

ReFinED reads a document, finds mentions and resolves each to a QID in one pass. It
replaced spaCy plus a separate entity linker in the previous project on speed, recall and
mislink rate; installing it there needed transformers 4.46 plus a CUDA torch, which is why
it is imported lazily and behind an interface.

**The model is injected.** A :data:`Linker` is anything that turns a batch of texts into
mentions, so every stage downstream of this — the graph, the pairing, the gates — is
testable with a stub and needs no GPU, no 2 GB of weights and no install. :func:`load` is
the only place that touches ReFinED itself.

Three things about the *input* matter more than any setting here, and all three were paid
for in the previous project:

* **Cables are ALL CAPS.** A linker trained on cased English reads ``SECRETARY RICE`` as
  noise. Every text is truecased before it is linked (:func:`prepare`).
* **A cable does not begin with its narrative.** It begins with routing and
  classification lines, and ends with a drafter's signature. Linking those produces
  mentions of communications infrastructure (:func:`narrative_body`).
* **Batching is not an optimisation, it is the difference between four hours and four
  days** over 251k cables, so the interface is a batch one.

Two filters are applied to what the model returns:

* a confidence floor, since ReFinED will resolve almost anything if asked;
* membership of the public entity set — an entity with no English Wikipedia article cannot
  bridge to a public corpus scoped to entities that have one.

What is deliberately *not* filtered here is the kind of thing an entity is. ReFinED's own
coarse types are unreliable — it labels countries ORG — so the decision about what may
anchor a bridge is taken from Wikidata in :mod:`osint_benchmark.graph.entity_types`, where
it can be read against the actual class hierarchy.

**This step is not bit-reproducible.** Two runs over identical input — same corpus, same
checkpoint, same settings, same GPU — gave 296,570 and 296,636 mentions over 45,296 and
45,300 entities. About 0.02%, consistent with floating-point non-determinism moving a
handful of scores across the confidence floor, though that is inference and not something
measured. The bridges and pairs were identical both times, so nothing downstream moved.

It matters only for what may be pinned: a fingerprint over link output would not match
between runs. Nothing does that. ``pipeline/09_release.py`` fingerprints the *parsed
corpora*, which are a deterministic function of bytes whose checksums are in ``pins/``, and
freezes the items themselves — so a rebuilder confirms they have the same corpora and the
same questions, not that their linker landed on the same 45,300 entities.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass

from osint_benchmark.sources import refs

# ReFinED resolves confidently and wrongly about as often as it resolves confidently and
# rightly, so a high floor buys less than it costs. The previous project ran at 0.5 and
# filtered afterwards on Wikidata types, which is the filter that actually discriminates.
DEFAULT_CONFIDENCE = 0.5
DEFAULT_BATCH = 64

# The linker reads a fixed-length window, and a cable's first few thousand characters carry
# its subject. Beyond this the marginal mention is a boilerplate footer.
MAX_CHARS = 4000

# Spans the model recognises that nobody can build a question on. A denylist rather than an
# allowlist on purpose: an allowlist silently drops every type it failed to anticipate, and
# this model's type labels are not trustworthy enough to be used that way.
DROP_TYPES = frozenset({"DATE", "TIME", "CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "MONEY"})

# Where a cable's narrative begins: the first numbered paragraph.
NARRATIVE_START = re.compile(r"\n\s*1\.\s*\(")

# Per-paragraph classification markers: "1. (SBU) The ambassador said". They are not part
# of the sentence, and the linker resolves them: SBU came back as an entity 1,248 times in
# 21k cables, and it is a marking, not a thing anyone can ask about.
CLASSIFICATION = re.compile(r"\((?:TS|S|C|U|SBU|LOU|SI|NF)(?://?[A-Z]+)*\)")

# Distribution and handling markers that sit on their own line inside the narrative.
# SIPDIS is the worst of them -- 1,946 mentions, more than any real entity except the
# United States -- because it survives the preamble strip by appearing further down.
MARKER_LINE = re.compile(
    r"^\s*(?:SIPDIS|NOFORN|NODIS|EXDIS|STADIS|CONFIDENTIAL|SECRET|UNCLAS(?:SIFIED)?"
    r"|LIMITED OFFICIAL USE|E\.O\.\s*\d+.*)\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Mention:
    """One recognised entity mention.

    Attributes:
        qid: The resolved Wikidata entity.
        surface_form: The text as it appeared, before normalisation.
        confidence: The model's score for the resolution.
        entity_type: The coarse class, used only to drop dates and quantities.
    """

    qid: str
    surface_form: str
    confidence: float
    entity_type: str = ""


# A batch of texts -> the mentions of each, in order. Injected so nothing downstream needs
# the model; batched because the model is far faster that way and the interface should not
# hide it.
Linker = Callable[[Sequence[str]], list[list[Mention]]]

# Text -> the text actually given to the linker. See :func:`prepare`.
Prepare = Callable[[str], str]


def per_document(link: Callable[[str], Iterable[Mention]]) -> Linker:
    """Turn a one-document-at-a-time linker into a :data:`Linker`.

    For linkers with nothing to gain from a batch — the dictionary matcher, any stub.
    """

    def batch(texts: Sequence[str]) -> list[list[Mention]]:
        """Link each text independently."""
        return [list(link(text)) for text in texts]

    return batch


def narrative_body(body: str) -> str:
    """Drop a cable's routing preamble and its drafter's signature.

    The narrative starts at the first numbered paragraph, ``1. (``. Before it are the
    classification, the addressee list and the reference lines — real text, full of
    capitalised tokens, none of it about anything. A short trailing line is the drafter's
    name and goes too: it is the single most reliably mislinked span in the corpus.

    A cable with no numbered paragraph keeps its whole body, minus that signature.

    Classification and distribution markings go too, wherever they appear. They read as
    ordinary capitalised tokens and the linker duly resolves them: ``SIPDIS`` was the
    second most frequent "entity" in a 21k-cable run, ahead of every country but one.
    """
    start = NARRATIVE_START.search(body)
    text = body[start.start() :] if start else body
    text = MARKER_LINE.sub("", CLASSIFICATION.sub("", text))
    lines = [line for line in text.strip().splitlines() if line.strip()]
    # Only when something is left afterwards. A one-line body is short enough to look like
    # a signature, and dropping it leaves the linker nothing at all to read.
    if len(lines) > 1:
        tail = lines[-1].strip()
        if tail and len(tail) <= 30 and len(tail.split()) <= 3:
            lines = lines[:-1]
    return "\n".join(lines)


def allow_nltk_imports() -> None:
    """Stop NLTK's import hook blocking its own dependencies.

    NLTK installs a meta-path finder that refuses any import resolving *inside the working
    directory or below it*. uv puts the virtualenv at ``.venv`` inside the project, so from
    the repository root every module NLTK needs is "in the working directory" and NLTK
    cannot import at all — taking ReFinED and truecase down with it, with a traceback that
    names neither.

    ``PYTHONSAFEPATH`` does not help despite what the error says: the finder inspects
    resolved paths directly and never consults ``sys.path``. Nor does running from
    elsewhere — from ``/`` it blocks the standard library. The documented off-switch is the
    only remedy, and what it guards against, a hostile module dropped in the working
    directory, is not a threat model that applies to a checked-out repository whose
    virtualenv we created.

    ``setdefault``, so an operator who has decided otherwise keeps their decision.
    """
    os.environ.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")


def truecase(text: str) -> str:
    """Restore sentence casing to text that has lost it.

    Cablegate is transmitted in upper case. A linker trained on cased English gets almost
    nothing from it, and this one step is worth more than every threshold in this module.

    Returns the text unchanged if ``truecase`` is not installed, so a run without it
    degrades rather than fails — but a full run should have it.
    """
    allow_nltk_imports()
    try:
        import truecase as truecase_lib  # noqa: PLC0415
    except ImportError:  # pragma: no cover - depends on the environment
        return text
    return truecase_lib.get_true_case(text)


def prepare(text: str, max_chars: int = MAX_CHARS) -> str:
    """Return the text a cable should be linked on: narrative only, cased, bounded."""
    return truecase(narrative_body(text or "")[:max_chars])


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
        if mention.entity_type in DROP_TYPES:
            continue
        if entity_set is not None and mention.qid not in entity_set:
            continue
        kept.append(mention)
    return kept


def _chunks(items: Iterable[dict], size: int) -> Iterator[list[dict]]:
    """Yield consecutive lists of at most ``size`` items."""
    chunk: list[dict] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def link_documents(
    documents: Iterable[dict],
    linker: Linker,
    side: str,
    entity_set: frozenset[str] | set[str] | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    text_field: str = "text",
    batch_size: int = DEFAULT_BATCH,
    prepare_text: Prepare | None = None,
    source: str = "",
) -> Iterator[dict]:
    """Yield one link row per document, in the shape the graph step reads.

    A document with no surviving mentions still yields a row with an empty list. The
    absence of links is a fact about the document; dropping it would make the coverage of
    the linker impossible to measure afterwards.

    ``prepare_text`` is applied before linking and is not recorded: the row refers to the
    document, and what the linker was shown is a property of the run, held in the
    provenance sidecar.

    ``source`` namespaces the emitted ``doc_id``. Everything downstream mixes corpora, and
    a bare id is only unique within one — see :mod:`osint_benchmark.sources.refs`.
    """
    for chunk in _chunks(documents, batch_size):
        texts = [document.get(text_field) or "" for document in chunk]
        if prepare_text is not None:
            texts = [prepare_text(text) for text in texts]
        for document, mentions in zip(chunk, linker(texts), strict=True):
            deduplicated: dict[str, Mention] = {}
            for mention in keep(mentions, entity_set, confidence):
                best = deduplicated.get(mention.qid)
                if best is None or mention.confidence > best.confidence:
                    deduplicated[mention.qid] = mention
            yield {
                "doc_id": refs.ref(source, document["doc_id"]) if source else document["doc_id"],
                "source": source,
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


PATCHED = "_osint_drops_add_special_tokens"


def without_kwarg(function: Callable, name: str) -> Callable:
    """Return ``function`` with one keyword argument silently dropped."""

    def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        """Call the wrapped function without the offending keyword."""
        kwargs.pop(name, None)
        return function(*args, **kwargs)

    setattr(wrapper, PATCHED, True)
    return wrapper


def patch_tokenizer_loading() -> bool:
    """Let ReFinED's tokenizers load under a current transformers.

    ReFinED passes ``add_special_tokens`` to ``AutoTokenizer.from_pretrained``, which
    forwards unrecognised keywords to the tokenizer's constructor; transformers refuses any
    that collide with a method name, and ``add_special_tokens`` is a method on every
    tokenizer. The model therefore cannot load at all, and the ``AttributeError`` arrives
    twelve frames from anything of ours.

    Dropping the argument is not a workaround. ``add_special_tokens`` governs *encoding*,
    not construction — passing it here never did anything.

    The patch goes on ``AutoTokenizer.from_pretrained`` rather than on ReFinED's
    ``get_tokenizer``, because there are two call sites and only one of them goes through
    that function: ``data_lookups.py`` calls the tokenizer loader directly. Patching the
    one cost a cluster job that failed exactly as it had before.

    Returns False if the patch was already applied, so applying it twice is harmless.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415

    if getattr(AutoTokenizer.from_pretrained, PATCHED, False):
        return False
    AutoTokenizer.from_pretrained = without_kwarg(
        AutoTokenizer.from_pretrained, "add_special_tokens"
    )
    return True


def install_hint(exc: ImportError) -> str:
    """Return what to do about a failed ReFinED import.

    Two failures look identical from the outside and have nothing to do with each other.
    The dependency may be absent — expected, it is optional because it pulls torch. Or it
    may be installed and refuse to load, because ReFinED imports NLTK first and NLTK blocks
    its own dependencies (see :func:`allow_nltk_imports`). That one reports a missing
    ``regex``, names neither ReFinED nor the fix, and cost two cluster jobs.
    """
    if "current working directory" in str(exc):
        return (
            f"ReFinED is installed but will not import: {exc}\n"
            "This is NLTK's import hook, not a missing dependency. Set "
            "NLTK_DISABLE_IMPORT_SECURITY=1. Note that PYTHONSAFEPATH, which the message "
            "above recommends, has no effect on it."
        )
    return (
        "ReFinED is not installed. It is an optional dependency because it pulls "
        f"torch and transformers: uv sync --extra link ({exc})"
    )


def load(
    model_name: str = "wikipedia_model_with_numbers",
    device: str = "cpu",
    entity_set: str = "wikipedia",
) -> Linker:
    """Return a :data:`Linker` backed by ReFinED.

    Imported here rather than at module scope so every test, and every stage that only
    needs the *shape* of a link, runs without the dependency installed.

    ``entity_set`` is ``wikipedia`` rather than ``wikidata``: the ~6.3M entities holding an
    English article are exactly the benchmark's public pool, and resolving into the other
    93M can only produce mentions that no public document can match.

    Raises:
        ImportError: With the install line, since this is the dependency that historically
            cost the most time to get working.
    """
    allow_nltk_imports()
    try:
        from refined.inference.processor import Refined  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(install_hint(exc)) from exc

    patch_tokenizer_loading()
    model = Refined.from_pretrained(model_name=model_name, entity_set=entity_set, device=device)

    def batch(texts: Sequence[str]) -> list[list[Mention]]:
        """Link one batch in a single forward pass."""
        results = []
        for document in model.process_text_batch(list(texts)):
            spans = document.spans if hasattr(document, "spans") else document
            results.append([m for m in (_mention(span) for span in spans or []) if m])
        return results

    return batch


def _mention(span) -> Mention | None:  # noqa: ANN001 - a ReFinED span, typed by duck
    """Read one ReFinED span, or None if it resolved to nothing."""
    entity = getattr(span, "predicted_entity", None)
    qid = getattr(entity, "wikidata_entity_id", None) if entity else None
    if not qid:
        return None
    return Mention(
        qid=qid,
        surface_form=span.text,
        confidence=float(getattr(span, "entity_linking_model_confidence_score", 0.0) or 0.0),
        entity_type=(getattr(span, "coarse_mention_type", "") or "").upper(),
    )
