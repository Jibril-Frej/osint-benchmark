"""The public entity set: which Wikidata entities have an English Wikipedia article.

This index is the universe both public sources are scoped to. An entity bridges the trust
boundary only if it appears here, and the Wikidata slice is filtered to the same set — two
public sources over one entity universe is easier to defend than either choice alone.

Built by joining two MediaWiki SQL dumps: ``page_props`` gives ``page_id -> QID`` for pages
carrying a ``wikibase_item`` property, and ``page`` gives ``page_id -> title`` for
namespace 0 (articles, not talk pages or templates).

Both are read as text with errors replaced and matched by shape rather than parsed as SQL.
A dump is one enormous ``INSERT`` line per table, so a real parser would have to hold it
all; matching the tuple shape streams instead, and a stray byte elsewhere in the dump
cannot abort the pass.
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Iterator
from pathlib import Path

from osint_benchmark.sources.base import Projection, Source

WIKI = "enwiki"
DATE = "20260601"

# (page_id,'wikibase_item','Qnnn') -- the only page_props shape we want.
WIKIBASE_ROW = re.compile(r"\((\d+),'wikibase_item','(Q\d+)'")
# (page_id,0,'title' -- namespace 0 only. The title may contain escaped quotes.
NS0_ROW = re.compile(r"\((\d+),0,'((?:[^'\\]|\\.)*)'")

PROJECTION = Projection(
    source=f"MediaWiki {WIKI}-{DATE} page and page_props SQL dumps",
    source_fields=("page_id", "page_namespace", "page_title", "pp_page", "pp_propname", "pp_value"),
    kept={
        "pp_value": "doc_id and qid (the wikibase_item value)",
        "pp_page": "page_id",
        "page_id": "page_id",
        "page_title": "title",
    },
    dropped={
        "page_namespace": "used as a filter, not carried: only namespace 0 is kept",
        "pp_propname": "used as a filter, not carried: only wikibase_item rows are kept",
    },
    kind="index",
    note=(
        "One record per entity that has an article in this wiki. Titles have underscores "
        "restored to spaces and escaped quotes unescaped, as MediaWiki stores them."
    ),
)


def page_qids(path: Path) -> dict[int, str]:
    """Return ``page_id -> QID`` for every page with a ``wikibase_item`` property."""
    found: dict[int, str] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "wikibase_item" not in line:
                continue
            for page_id, qid in WIKIBASE_ROW.findall(line):
                found[int(page_id)] = qid
    return found


def article_titles(path: Path) -> Iterator[tuple[int, str]]:
    """Yield ``(page_id, title)`` for namespace-0 rows of a ``page`` dump."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "INSERT INTO" not in line:
                continue
            for page_id, title in NS0_ROW.findall(line):
                yield int(page_id), title.replace("\\'", "'").replace("_", " ")


def parse(raw_dir: Path) -> Iterator[dict]:
    """Yield one record per entity holding an article in this wiki.

    Only the ``page_id -> QID`` map is held; the page table is streamed and each row
    emitted as it is read. Holding both maps needs several GB for 7.5M entries, which is
    fine on a workstation and is not fine on a shared login node -- the first cluster run
    was killed here, leaving output with no provenance sidecar beside it.

    Output order therefore follows the page dump rather than QID order. Still
    deterministic, since it is the same file every time.
    """
    directory = raw_dir / "wikipedia_index"
    qids = page_qids(directory / f"{WIKI}-{DATE}-page_props.sql.gz")
    for page_id, title in article_titles(directory / f"{WIKI}-{DATE}-page.sql.gz"):
        qid = qids.get(page_id)
        if qid is not None:
            yield {"doc_id": qid, "qid": qid, "page_id": page_id, "title": title}


SOURCE = Source(name="wikipedia_index", kind="public", parse=parse, projection=PROJECTION)
