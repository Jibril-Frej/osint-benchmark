"""Gather the text a question can rest on, and the labels that name its bridge.

Shared by the generation and necessity steps. It lives in the package rather than in a
numbered pipeline file because both steps need it and a file whose name starts with a digit
cannot be imported.

Only *linked* corpora are read. Nothing can be evidence unless it was linked, so reading
the rest is wasted -- and one of them is GDELT, whose 91.6M events take long enough that
the step never finishes.
"""

from __future__ import annotations

from osint_benchmark import paths
from osint_benchmark.artifacts import read_jsonl
from osint_benchmark.sources import ALL, base, get_source


def linked_sources() -> list[str]:
    """Return the sources that were linked, and can therefore appear in a pair."""
    links = paths.data_dir() / "links"
    return [path.stem for path in sorted(links.glob("*.jsonl")) if path.stem in ALL]


def evidence_texts(sources: list[str] | None = None) -> dict[str, str]:
    """Return ``doc_id -> text`` for the linked corpora and the fetched article leads."""
    texts: dict[str, str] = {}
    for name in sources if sources is not None else linked_sources():
        output = base.output_path(get_source(name))
        if not output.exists():
            continue
        for row in read_jsonl(output):
            if row.get("text"):
                texts[row["doc_id"]] = row["text"]
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
