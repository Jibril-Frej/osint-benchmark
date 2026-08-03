"""Gather the text a question can rest on, and the labels that name its bridge.

Shared by the generation and necessity steps. It lives in the package rather than in a
numbered pipeline file because both steps need it and a file whose name starts with a digit
cannot be imported.
"""

from __future__ import annotations

from osint_benchmark import paths
from osint_benchmark.artifacts import read_jsonl
from osint_benchmark.sources import ALL, base, get_source


def evidence_texts() -> dict[str, str]:
    """Return ``doc_id -> text`` across every built corpus and the fetched article leads."""
    texts: dict[str, str] = {}
    for name in ALL:
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
