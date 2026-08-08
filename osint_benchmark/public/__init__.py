"""Step 4: the public sources that can only be fetched once the entities are known.

The bulk sources in step 1 depend on nothing. These depend on the graph: article text and
Wikidata statements are fetched for bridge entities, and the commercial register is queried
for the firms the reporting actually names -- it holds 2.8M publications, and mirroring it
wholesale would be pointless when only a few hundred companies matter.

Everything here records the revision it read, because these sources are live and a gold
answer taken from one is only correct against a particular version of it.
"""

from __future__ import annotations  # noqa: E402  (the module docstring comes first)

import time  # noqa: E402
import urllib.error  # noqa: E402
from collections.abc import Callable, Iterator  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

# What a Wikimedia server says when it wants us to slow down, or is briefly unavailable.
# Retried rather than reported: a slice is tens of thousands of requests and meets one of
# these sooner or later, and giving up would report fifty entities as unknown data because
# the server was busy for a second.
BUSY = (429, 503)


def patiently(read: Callable[[], dict], attempts: int = 5, backoff: float = 5.0) -> dict:
    """Return a request's result, waiting and retrying while the server says it is busy.

    Only for the codes in :data:`BUSY`. Any other error is a real failure and is raised so
    the caller can report it: a run that retries a 404 forever is worse than one that says
    what is missing.
    """
    for attempt in range(attempts):
        try:
            return read()
        except urllib.error.HTTPError as exc:
            if exc.code not in BUSY or attempt == attempts - 1:
                raise
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError("unreachable: the loop above either returns or raises")


def in_flight(work: Callable[[list], object], chunks: list, workers: int) -> Iterator:
    """Yield each chunk's result, running ``workers`` of them at a time.

    Measured against the live APIs, four at a time, each on titles the other run had not
    touched: the Wikidata slice went 1.45x faster and the article leads 1.6x. Both are far
    short of four, because the limit is the far end rather than the round trip -- 50
    entities are 16 MB, since ``wbgetentities`` returns every identifier and reference and
    there is no way to ask it not to.

    Worth measuring twice: run back to back over the same titles, the second one looked 36x
    faster, which was Wikipedia's cache and not concurrency at all.

    Results come back in order either way, so nothing downstream can tell how many were in
    flight.
    """
    if workers <= 1:
        yield from (work(chunk) for chunk in chunks)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(work, chunks)


def reporting(what: str, total: int, every: int = 25) -> Callable[[int], None]:
    """Return a callback that prints how far along a long fetch is.

    Written after watching a fetch stage sit silent for forty-eight minutes with no way to
    tell work from a hang -- the failures were being collected and printed at the end, so
    silence meant either. A line every few hundred entities costs nothing and answers it.
    """
    start = time.monotonic()

    def progress(done: int) -> None:
        if done % every and done != total:
            return
        elapsed = time.monotonic() - start
        rate = done / elapsed if elapsed else 0.0
        left = (total - done) / rate if rate else 0.0
        print(
            f"  {what}: {done}/{total} in {elapsed / 60:.0f}m"
            f" ({rate * 60:.0f}/min, about {left / 60:.0f}m left)",
            flush=True,
        )

    return progress
