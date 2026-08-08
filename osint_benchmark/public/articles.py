"""Fetch Wikipedia article text for the bridge entities, pinned to the revision read.

The public half of a question's evidence. Only bridge entities are fetched -- a few
thousand articles rather than seven million -- because this is the text a question is
*written from*. The full corpus a system later searches is a different artefact and is
specified rather than built here; a retrieval corpus narrowed to the answer key is not a
retrieval corpus.

The lead section is taken rather than the whole article: it is what states who or what the
subject is, and a full article brings in navigation, tables and trivia that make a question
harder to ground without making it better.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Iterator

from osint_benchmark import public

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "osint-benchmark/0.1 (research; https://github.com/Jibril-Frej/osint-benchmark)"
BATCH = 20

Fetch = Callable[[list[str]], dict]


def fetch_titles(titles: list[str], timeout: float = 60.0) -> dict:
    """Return the API payload for a batch of article titles."""
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        # ``info`` costs nothing extra and carries ``length``, the whole article's size in
        # bytes. That is the prominence signal: the lead this request returns is a poor one,
        # because a world figure's lead and a minor official's are both a few paragraphs.
        # Full plain text would be the direct measure, but the API serves it one article per
        # request, which is 62,000 requests rather than 3,100.
        "prop": "extracts|revisions|info",
        "exintro": "1",
        "explaintext": "1",
        "rvprop": "ids|timestamp",
        "titles": "|".join(titles),
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    def read() -> dict:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read())

    return public.patiently(read)


def fetch_articles(
    entities: Iterable[tuple[str, str]],
    fetch: Fetch = fetch_titles,
    pause: float = 0.1,
    on_error: Callable[[str, Exception], None] | None = None,
    workers: int = 1,
) -> Iterator[dict]:
    """Yield one record per (QID, title), skipping and reporting failures.

    A missing or empty article is skipped rather than yielded empty: a question cannot be
    built on evidence that is not there, and an empty extract would look like one that can.
    """
    pairs = list(dict.fromkeys(entities))
    by_title = {title: qid for qid, title in pairs}
    chunks = [
        [title for _, title in pairs[start : start + BATCH]]
        for start in range(0, len(pairs), BATCH)
    ]

    def attempt(titles: list[str]) -> dict | Exception:
        """Return a batch's payload, or the failure to be reported in order below."""
        try:
            return fetch(titles)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return exc

    for titles, payload in zip(chunks, public.in_flight(attempt, chunks, workers), strict=True):
        if isinstance(payload, Exception):
            if on_error is not None:
                on_error(", ".join(titles), payload)
            continue
        for page in payload.get("query", {}).get("pages", []):
            title, extract = page.get("title", ""), (page.get("extract") or "").strip()
            if page.get("missing") or not extract:
                continue
            revisions = page.get("revisions") or [{}]
            yield {
                "doc_id": f"enwiki:{by_title.get(title, title)}",
                "qid": by_title.get(title, ""),
                "title": title,
                "page_id": page.get("pageid"),
                "revision": revisions[0].get("revid"),
                "revision_date": revisions[0].get("timestamp"),
                "article_bytes": page.get("length", 0),
                "text": extract,
            }
        if pause:
            time.sleep(pause)
