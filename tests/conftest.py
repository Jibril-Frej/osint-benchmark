"""Test fixtures shared by the whole suite.

The one thing here exists because the suite is run inside cluster jobs that legitimately
set ``OSINT_*`` in their environment, and a test that reads an ambient root is a test whose
result depends on who ran it.
"""

from __future__ import annotations

import os

import pytest

# Every root the pipeline resolves from the environment. A test that wants one sets it;
# no test should inherit one.
ROOTS = ("OSINT_DATA", "OSINT_RAW", "OSINT_DOCS", "OSINT_PINS", "OSINT_CONFIG")


@pytest.fixture(autouse=True)
def isolated_roots(monkeypatch):
    """Clear the path environment variables before every test.

    A job that exports ``OSINT_RAW`` to reuse its downloads made three tests fail on the
    cluster and nowhere else: their fixture set ``OSINT_DATA`` and left ``OSINT_RAW``
    pointing at the caller's real corpora. Clearing the lot here means a fixture only has
    to set what it uses, and the suite gives the same answer wherever it runs.
    """
    for name in ROOTS:
        monkeypatch.delenv(name, raising=False)
    assert not any(name in os.environ for name in ROOTS)
