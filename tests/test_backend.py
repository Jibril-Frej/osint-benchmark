"""Unit tests for the vLLM client.

Run against a local server speaking the OpenAI chat API rather than a mock, because the
bug worth catching is a mismatch between what this sends and what a server expects — which
is exactly what the previous client got wrong, posting llama.cpp's native shape.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from osint_benchmark.models.backend import ModelUnavailable, vllm
from osint_benchmark.models.settings import Settings


class _Handler(BaseHTTPRequestHandler):
    """Answer /v1/chat/completions the way vLLM does."""

    def do_POST(self):  # noqa: N802 (the name is BaseHTTPRequestHandler's)
        """Record the request and reply with the server's canned content."""
        length = int(self.headers.get("Content-Length", 0))
        self.server.requests.append(
            {"path": self.path, "body": json.loads(self.rfile.read(length))}
        )
        if self.server.status != 200:
            self.send_error(self.server.status)
            return
        body = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": self.server.content}}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        """Keep the test output quiet."""


@pytest.fixture
def server():
    """Run a stand-in vLLM server."""
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    httpd.requests = []
    httpd.content = "SUPPORTED"
    httpd.status = 200
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def _settings(server, **kw):
    """Return settings pointing at the stand-in server."""
    return Settings(
        role=kw.get("role", "judge"),
        model=kw.get("model", "Qwen/QwQ-32B"),
        endpoint=f"http://127.0.0.1:{server.server_address[1]}",
        temperature=kw.get("temperature", 0.7),
        max_tokens=kw.get("max_tokens", 1024),
        samples=kw.get("samples", 3),
    )


class TestVllm:
    """What the client puts on the wire, and what it makes of the reply."""

    def test_it_posts_the_openai_chat_shape(self, server):
        """The previous client posted llama.cpp's native /completion shape instead.

        Path, message list and parameter names all differ between the two, so a client
        written for one silently fails against the other.
        """
        vllm(_settings(server))("say something")

        request = server.requests[0]
        assert request["path"] == "/v1/chat/completions"
        assert request["body"]["messages"] == [{"role": "user", "content": "say something"}]
        assert request["body"]["model"] == "Qwen/QwQ-32B"

    def test_the_configured_parameters_reach_the_server(self, server):
        """Temperature and the token ceiling are load-bearing, not decoration."""
        vllm(_settings(server, temperature=0.3, max_tokens=4096))("x")

        body = server.requests[0]["body"]
        assert body["temperature"] == 0.3
        assert body["max_tokens"] == 4096

    def test_the_reply_content_is_returned(self, server):
        """The answer lives at choices[0].message.content."""
        server.content = "  the finance committee  "

        assert vllm(_settings(server))("x") == "the finance committee"

    def test_a_reasoning_trace_is_stripped(self, server):
        """QwQ opens with one, and it is not the answer."""
        server.content = "<think>weighing it up</think>SUPPORTED"

        assert vllm(_settings(server))("x") == "SUPPORTED"

    def test_an_empty_choices_list_is_not_an_answer(self, server):
        """A server can return 200 with nothing usable."""
        server.content = ""

        assert vllm(_settings(server))("x") == ""

    def test_a_server_error_names_the_endpoint(self, server):
        """A failure at question 400 should say what was unreachable."""
        server.status = 500

        with pytest.raises(ModelUnavailable, match="v1/chat/completions"):
            vllm(_settings(server))("x")

    def test_no_endpoint_fails_before_any_work(self, server):
        """Failing here beats failing per-question after an hour."""
        settings = Settings(
            role="judge", model="m", endpoint="", temperature=0.7, max_tokens=10, samples=1
        )

        with pytest.raises(ModelUnavailable, match="no endpoint"):
            vllm(settings)
