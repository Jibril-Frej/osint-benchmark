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
from collections.abc import Callable  # noqa: E402

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
