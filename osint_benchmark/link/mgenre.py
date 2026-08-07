"""Link German and French prose to Wikidata with WikiNeural NER and mGENRE.

ReFinED is English-only, so the parliamentary record — and any other German or French
corpus — needs a different linker. This is the stack the previous project validated on
Swiss archival text: WikiNeural finds the mention spans, mGENRE generates a Wikipedia
title for each, and a (language, title) → QID mapping resolves it.

**Expect partial recall.** On error-free text that stack recovered roughly 40% of people
and places. The previous project's own conclusion was that the linker is the ceiling, not
the input — so a parliamentary index built this way is a floor to improve on, not a
measurement of how much the corpus contains.

Three things here are not obvious and all were paid for:

* **The mapping is ~90 GB in RAM** from a 3.7 GB pickle, and its key shape is not
  documented. :func:`to_qid` tries the three shapes it has been seen to take rather than
  assuming one, because guessing wrong yields a silent zero-recall run.
* **A leaked WordPiece continuation is not a mention.** ``##fristen`` and two-letter
  fragments link confidently to nonsense — "Je" resolved to the Polish article for German.
* **Results are cached on the surface form alone.** Context-sensitive linking is more
  correct in principle, but parliamentary German repeats a small set of proper nouns so
  often that re-deriving them dominates the runtime, and a proper noun's sense is stable.

As with ReFinED, the model is injected: everything here except :func:`load` is testable
with no GPU, no 3.7 GB pickle and no 90 GB of RAM.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from osint_benchmark.link.refined import Mention

# Characters of context each side of a mention. mGENRE reads the marked span in context,
# and too little of it turns a name into a guess.
CONTEXT_CHARS = 200

# Window and overlap for chunking a long item. The overlap exists so a mention on a
# boundary is seen whole at least once; mentions found twice are deduplicated by offset.
WINDOW_CHARS = 1500
OVERLAP_CHARS = 250

# A mention shorter than this is a fragment, not a name.
MIN_SURFACE = 3
MIN_SCORE = 0.85

# What mGENRE emits: "Title >> lang".
GENERATED = re.compile(r"(.*?)\s*>>\s*(\w+)\s*$")

# Entity classes worth keeping. LOC and PER are what this corpus is about; ORG is kept
# because a federal office is a legitimate bridge, MISC dropped because in this NER's
# vocabulary it is mostly adjectives of nationality.
KEEP_GROUPS = frozenset({"PER", "LOC", "ORG"})


def mapping_path() -> Path:
    """Return where the (language, title) → QID pickle lives."""
    return Path(os.environ.get("OSINT_MGENRE_MAPPING", Path.home() / "lang_title2wikidataID.pkl"))


def chunks(text: str, window: int = WINDOW_CHARS, overlap: int = OVERLAP_CHARS):
    """Yield ``(offset, window)`` over a long text, overlapping so no mention is split."""
    if not text:
        return
    step = max(window - overlap, 1)
    for start in range(0, len(text), step):
        piece = text[start : start + window]
        if piece:
            yield start, piece
        if start + window >= len(text):
            return


def to_qid(mapping: dict, title: str, lang: str | None) -> str | None:
    """Return the QID for a generated title, tolerating the mapping's key shape.

    The pickle has been seen keyed by ``(lang, title)``, by ``(title, lang)`` and by title
    alone, and the value is sometimes a bare id and sometimes a collection. Assuming one
    shape produces a run that links nothing and reports no error.
    """
    for key in ((lang, title), (title, lang), title):
        try:
            value = mapping.get(key)
        except TypeError:  # an unhashable key shape
            continue
        if not value:
            continue
        qid = next(iter(value)) if isinstance(value, list | set | tuple) else value
        text = str(qid)
        return text if text.startswith("Q") else f"Q{text}"
    return None


def usable(surface: str, group: str, score: float) -> bool:
    """Return whether a recognised span is worth trying to link."""
    if "##" in surface or len(surface.strip()) < MIN_SURFACE:
        return False
    if group not in KEEP_GROUPS:
        return False
    return score >= MIN_SCORE


def marked(window: str, start: int, end: int, context: int = CONTEXT_CHARS) -> str:
    """Return the mention wrapped in the markers mGENRE expects, inside its context."""
    return (
        window[max(0, start - context) : start]
        + " [START] "
        + window[start:end]
        + " [END] "
        + window[end : end + context]
    )


def spans(text: str, ner: Callable[[str], Iterable[dict]]) -> list[tuple[str, str]]:
    """Return ``(surface form, marked context)`` for every usable mention, deduplicated.

    Deduplicated by ``(absolute offset, surface)``: the overlap between windows means a
    mention near a boundary is genuinely seen twice, and linking it twice costs a forward
    pass for a result already held.
    """
    seen: set[tuple[int, str]] = set()
    found: list[tuple[str, str]] = []
    for offset, window in chunks(text):
        for entity in ner(window) or []:
            surface = str(entity.get("word", ""))
            if not usable(surface, entity.get("entity_group", ""), float(entity.get("score", 1.0))):
                continue
            key = (offset + int(entity["start"]), surface)
            if key in seen:
                continue
            seen.add(key)
            found.append((surface, marked(window, int(entity["start"]), int(entity["end"]))))
    return found


def parse_generated(sequence: str) -> tuple[str, str | None]:
    """Return ``(title, language)`` from one mGENRE output."""
    match = GENERATED.match(sequence)
    if match:
        return match.group(1).strip(), match.group(2)
    return sequence.strip(), None


def link_text(
    text: str,
    ner: Callable[[str], Iterable[dict]],
    generate: Callable[[list[str]], list[str]],
    mapping: dict,
    cache: dict[str, str | None] | None = None,
) -> Iterator[Mention]:
    """Yield one mention per resolved span, as the shape the graph step reads.

    ``cache`` maps surface form to QID across documents and is the difference between a
    run that finishes and one that does not: parliamentary German repeats its proper nouns
    relentlessly.
    """
    if cache is None:
        cache = {}
    found = spans(text, ner)
    todo = [(surface, context) for surface, context in found if surface not in cache]
    if todo:
        generated = generate([context for _, context in todo])
        for (surface, _), sequence in zip(todo, generated, strict=True):
            title, lang = parse_generated(sequence)
            cache[surface] = to_qid(mapping, title, lang)
    for surface, _ in found:
        qid = cache.get(surface)
        if qid:
            # Confidence is the NER's, not the linker's: mGENRE generates rather than
            # scores, so there is no calibrated number to report and inventing one would
            # let a threshold downstream pretend to a precision nobody measured.
            yield Mention(qid=qid, surface_form=surface, confidence=1.0, entity_type="")


def load(device: str = "cuda", batch: int = 16, beams: int = 5):
    """Return ``(ner, generate, mapping)`` for :func:`link_text`.

    Imported lazily, like ReFinED, so everything above is testable without the stack.

    The mapping is a 3.7 GB pickle that occupies roughly 90 GB once loaded, so this needs
    a job that asked for the memory. A self-test runs before any corpus is touched: a
    mapping whose key shape is not what :func:`to_qid` tries produces a run that links
    nothing and reports no error, and discovering that after an hour is the expensive way.

    Raises:
        ImportError: With the install line, since this pulls torch and transformers.
        SystemExit: If the mapping is absent, or present and unusable.
    """
    import pickle  # noqa: PLC0415

    try:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("mGENRE needs torch and transformers: uv sync --extra link") from exc

    path = mapping_path()
    if not path.exists():
        raise SystemExit(
            f"{path} is missing: mGENRE needs the (language, title) -> QID mapping. "
            "Set OSINT_MGENRE_MAPPING."
        )

    recogniser = pipeline(
        "ner",
        model="Babelscape/wikineural-multilingual-ner",
        aggregation_strategy="simple",
        device=0 if device == "cuda" else -1,
    )
    tokenizer = AutoTokenizer.from_pretrained("facebook/mgenre-wiki", use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mgenre-wiki").to(device)
    model.eval()

    print(f"loading {path} (~90 GB in memory) ...", flush=True)
    with path.open("rb") as handle:
        mapping = pickle.load(handle)  # noqa: S301 - our own file, not untrusted input
    print(f"  {len(mapping)} entries", flush=True)

    # Prove the key shape before spending a run on it.
    probes = [("Ungarn", "de"), ("Hongrie", "fr"), ("Hungary", "en")]
    resolved = {t: to_qid(mapping, t, lang) for t, lang in probes}
    print(f"  self-test: {resolved}", flush=True)
    if not any(resolved.values()):
        raise SystemExit(
            f"{path} resolved none of {probes}. Its key shape is not one to_qid tries, "
            "and a run against it would link nothing and report no error."
        )

    @torch.no_grad()
    def generate(contexts: list[str]) -> list[str]:
        """Generate one Wikipedia title per marked context."""
        out: list[str] = []
        for start in range(0, len(contexts), batch):
            encoded = tokenizer(
                contexts[start : start + batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(device)
            produced = model.generate(
                **encoded, num_beams=beams, num_return_sequences=1, max_new_tokens=25
            )
            out.extend(tokenizer.batch_decode(produced, skip_special_tokens=True))
        return out

    return recogniser, generate, mapping
