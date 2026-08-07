"""Gather the text a question can rest on, and the labels that name its bridge.

Shared by the generation and necessity steps. It lives in the package rather than in a
numbered pipeline file because both steps need it and a file whose name starts with a digit
cannot be imported.

Only *linked* corpora are read. Nothing can be evidence unless it was linked, so reading
the rest is wasted -- and one of them is GDELT, whose 91.6M events take long enough that
the step never finishes.
"""

from __future__ import annotations

from collections.abc import Iterable

from osint_benchmark import paths
from osint_benchmark.artifacts import read_jsonl
from osint_benchmark.sources import ALL, base, get_source, refs


def linked_sources() -> list[str]:
    """Return the sources that were linked, and can therefore appear in a pair."""
    links = paths.data_dir() / "links"
    return [path.stem for path in sorted(links.glob("*.jsonl")) if path.stem in ALL]


# How much of a document any prompt may carry. A cable can run to tens of thousands of
# characters and the served context is finite: step 6 clipped and step 7 did not, so a run
# wrote 135 questions and then died measuring the first one whose cable was long, on an
# HTTP 400 that named token counts and nothing about which stage or which document.
EVIDENCE_CHARS = 6000


def clip(text: str, limit: int = EVIDENCE_CHARS) -> str:
    """Return text bounded to a character budget, marked where it was cut."""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n[... truncated]"


# Fields that identify the record rather than say anything about its subject. A question
# built on one of these is a question about the filing system.
NOT_EVIDENCE = frozenset({"doc_id", "sanctions_set_id", "programme_key", "lang", "source"})


def record_text(row: dict) -> str:
    """Return the evidence text of one record, rendering it if it has no prose.

    The prose corpora carry ``text``. The tabular ones do not: a sanctions listing is a
    name, a programme, a set of measures and a justification, held in separate fields, and
    the evidence lookup simply skipped any record without ``text``. That left the public
    side of every pair with no evidence at all — invisible until the id collision that was
    substituting a cable for it got fixed.

    Rendered as ``field: value`` lines rather than a prose template. The fields differ per
    source and a template per source is a thing to keep in step with six parsers; the field
    names are already the parser's considered vocabulary.
    """
    if row.get("text"):
        return str(row["text"])
    lines = []
    for key, value in row.items():
        if key in NOT_EVIDENCE or not value:
            continue
        if isinstance(value, list):
            parts = [str(v) for v in value if v and not isinstance(v, dict)]
            if not parts:
                continue
            value = "; ".join(parts)
        elif isinstance(value, dict):
            continue
        lines.append(f"{key.replace('_', ' ')}: {value}")
    return "\n".join(lines)


def sources_in(doc_ids: Iterable[str]) -> list[str]:
    """Return the corpora named by a set of document references.

    What lets every step after linking work out which corpora to load from its own input,
    rather than from whichever link files happen to be lying about. The alternative reads
    ``data/links/*.jsonl``, which is wrong twice over: a step resuming from a saved pairs
    or items file has no link files at all and silently loads nothing, and a rerun with
    different sources loads the wrong ones. Both have happened.
    """
    named = {refs.split(doc_id)[0] for doc_id in doc_ids}
    return sorted(name for name in named if name in ALL)


def sources_for(items: Iterable) -> list[str]:
    """Return the corpora a set of items cites, read off the items themselves."""
    return sources_in(evidence.doc_id for item in items for evidence in item.evidence)


def sources_for_pairs(pairs: Iterable[dict]) -> list[str]:
    """Return the corpora a set of pairs cites, read off the pairs themselves."""
    return sources_in(
        doc_id for pair in pairs for doc_id in (pair["private_id"], pair["public_id"])
    )


def evidence_texts(sources: list[str] | None = None) -> dict[str, str]:
    """Return ``source:doc_id -> text`` for the linked corpora and the fetched article leads.

    Namespaced, and this is the lookup that made the case for namespacing: keyed by bare
    ``doc_id`` it returned cable 47703 as the "public record" for sanctions target 47703,
    and every pair became a cable beside an unrelated cable.
    """
    texts: dict[str, str] = {}
    for name in sources if sources is not None else linked_sources():
        output = base.output_path(get_source(name))
        if not output.exists():
            continue
        for row in read_jsonl(output):
            if rendered := record_text(row):
                texts[refs.ref(name, row["doc_id"])] = rendered
    articles = paths.data_dir() / "facts" / "articles.jsonl"
    if articles.exists():
        for row in read_jsonl(articles):
            texts[row["doc_id"]] = row["text"]
    return texts


def entity_labels() -> dict[str, str]:
    """Return ``QID -> label`` so a bridge can be named rather than numbered."""
    path = paths.data_dir() / "facts" / "wikidata.jsonl"
    if not path.exists():
        return {}
    return {row["qid"]: row["label"] for row in read_jsonl(path) if row.get("label")}
