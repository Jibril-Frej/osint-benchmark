"""Unit tests for the resuming downloader.

Resume is the one piece of fetching that can produce a *plausible* wrong file rather than
an error: append to the wrong partial and you get a seam of two downloads, at the right
sort of size, that parses fine. So these run against a real HTTP server rather than a
mocked one, and cover the three ways a server can answer a ``Range`` request.

The checksum in :func:`~osint_benchmark.sources.base.fetch` is the backstop for all of
this; these tests are what keep it from having to be.
"""

from __future__ import annotations

import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from osint_benchmark.sources.base import _download

BODY = b"".join(f"line {i:04d}\n".encode() for i in range(500))


class _Handler(BaseHTTPRequestHandler):
    """Serve one fixed body, honouring Range or not depending on the server's mode."""

    def do_GET(self):  # noqa: N802 (the name is BaseHTTPRequestHandler's)
        """Answer with the whole body, a partial one, or 416, recording which."""
        self.server.agents.append(self.headers.get("User-Agent", ""))
        if getattr(self.server, "always_404", False):
            self.server.served.append((404, 0))
            self.send_error(404)
            return
        if getattr(self.server, "fail_times", 0) > 0:
            self.server.fail_times -= 1
            self.server.served.append((500, 0))
            self.send_error(500)
            return
        rng = self.headers.get("Range")
        if rng and self.server.honour_range:
            start = int(rng.removeprefix("bytes=").split("-")[0])
            if start >= len(BODY):
                self.server.served.append((416, 0))
                self.send_error(416)
                return
            self.server.served.append((206, len(BODY) - start))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(BODY) - 1}/{len(BODY)}")
            self.end_headers()
            self.wfile.write(BODY[start:])
            return
        self.server.served.append((200, len(BODY)))
        self.send_response(200)
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()
        self.wfile.write(BODY)

    def log_message(self, *args):
        """Keep the test output quiet."""


@pytest.fixture
def server():
    """Run a local HTTP server; set ``honour_range`` to choose how it answers."""
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    httpd.honour_range = True
    httpd.served = []
    httpd.agents = []
    httpd.fail_times = 0
    httpd.always_404 = False  # the User-Agent of every request, so a test can prove we identify
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def _url(server) -> str:
    """Return the served URL."""
    return f"http://127.0.0.1:{server.server_address[1]}/file"


class TestDownload:
    """A download lands complete, or not at all."""

    def test_a_fresh_download_writes_the_whole_body(self, server, tmp_path):
        """Nothing on disk: the file arrives whole and the partial is gone."""
        dest = tmp_path / "file"

        _download(_url(server), dest, len(BODY))

        assert dest.read_bytes() == BODY
        assert not dest.with_name("file.part").exists()
        assert server.served == [(200, len(BODY))]

    def test_an_interrupted_download_resumes_from_where_it_stopped(self, server, tmp_path):
        """A partial file becomes a Range request, and the result is the whole body.

        This is the case the enwiki dump needs: a connection dropped near the end of a
        25 GB transfer must not cost the whole transfer. Asserting the result alone would
        pass without any resume at all, so what is checked is that only the missing bytes
        crossed the wire.
        """
        dest = tmp_path / "file"
        dest.with_name("file.part").write_bytes(BODY[:1000])

        _download(_url(server), dest, len(BODY))

        assert dest.read_bytes() == BODY
        assert server.served == [(206, len(BODY) - 1000)]

    def test_a_server_ignoring_range_restarts_instead_of_splicing(self, server, tmp_path):
        """Answering 200 with the whole body must not be appended to the partial.

        Appending would produce a file of the right sort of size, made of one partial
        download followed by a complete one — wrong in a way only a checksum catches.
        """
        server.honour_range = False
        dest = tmp_path / "file"
        dest.with_name("file.part").write_bytes(BODY[:1000])

        _download(_url(server), dest, len(BODY))

        assert dest.read_bytes() == BODY
        assert len(dest.read_bytes()) == len(BODY)  # not 1000 + len(BODY)

    def test_a_partial_at_or_past_the_pinned_length_is_discarded(self, server, tmp_path):
        """A leftover from a different file is not a resumable download."""
        dest = tmp_path / "file"
        dest.with_name("file.part").write_bytes(b"x" * (len(BODY) + 50))

        _download(_url(server), dest, len(BODY))

        assert dest.read_bytes() == BODY
        # No Range was sent at all: the pinned size settled it before asking.
        assert server.served == [(200, len(BODY))]

    def test_an_unsatisfiable_range_falls_back_to_a_full_download(self, server, tmp_path):
        """With no pinned size to check against, a 416 is the only signal, and it is used."""
        dest = tmp_path / "file"
        dest.with_name("file.part").write_bytes(b"x" * (len(BODY) + 50))

        _download(_url(server), dest, expected_size=None)

        assert dest.read_bytes() == BODY
        assert server.served == [(416, 0), (200, len(BODY))]


class TestUserAgent:
    """Wikimedia refuses urllib's default agent."""

    def test_downloads_identify_the_project(self, server, tmp_path):
        """A 403 from the dumps server was invisible locally, where the files were present.

        Only a machine that had to fetch them saw it, which is the argument for building
        from scratch somewhere else.
        """
        from osint_benchmark.sources.base import USER_AGENT

        assert "osint-benchmark" in USER_AGENT
        _download(_url(server), tmp_path / "file", len(BODY))

        assert server.agents and all("osint-benchmark" in a for a in server.agents)


class TestRetry:
    """A flaky answer must not kill an unattended job."""

    def test_a_transient_failure_is_retried(self, server, tmp_path, monkeypatch):
        """archive.org redirects to a storage node that returns 500 often enough to matter.

        The first batch job died 13 seconds in for exactly this.
        """
        monkeypatch.setattr("osint_benchmark.sources.base.BACKOFF_SECONDS", 0)
        server.fail_times = 2
        dest = tmp_path / "file"

        _download(_url(server), dest, len(BODY))

        assert dest.read_bytes() == BODY
        assert [s for s, _ in server.served if s == 500] == [500, 500]

    def test_a_permanent_failure_is_not_retried(self, server, tmp_path, monkeypatch):
        """Retrying a 404 only wastes time."""
        monkeypatch.setattr("osint_benchmark.sources.base.BACKOFF_SECONDS", 0)
        server.always_404 = True

        with pytest.raises(urllib.error.HTTPError):
            _download(_url(server), tmp_path / "file", len(BODY))

        assert len([s for s, _ in server.served if s == 404]) == 1


class TestStall:
    """A dead connection must fail, not hang."""

    def test_a_stalled_connection_times_out(self, tmp_path, monkeypatch):
        """A batch job sat in step 1 for 38 minutes on a download that takes seven.

        Every other fetcher in the project passes a timeout; the one that moves gigabytes
        did not, so a stalled socket was bounded only by the job's walltime.
        """
        import socket as socket_module

        monkeypatch.setattr("osint_benchmark.sources.base.BACKOFF_SECONDS", 0)
        monkeypatch.setattr("osint_benchmark.sources.base.READ_TIMEOUT", 0.3)
        monkeypatch.setattr("osint_benchmark.sources.base.RETRIES", 2)

        listener = socket_module.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            with pytest.raises((TimeoutError, urllib.error.URLError, OSError)):
                _download(f"http://127.0.0.1:{port}/file", tmp_path / "file", 10)
        finally:
            listener.close()
