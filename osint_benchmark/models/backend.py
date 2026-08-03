"""Talk to a served model.

One protocol — :data:`Complete`, text in and text out — so every stage that needs a model
takes one as an argument and a test passes a stub. Nothing above this module knows whether
there is a GPU anywhere.

The backend is a llama.cpp ``llama-server`` speaking the OpenAI completions shape, which is
what the previous project served QwQ-32B and Llama-3.3-70B through on the cluster. Swapping
it for anything else means writing one function of the same shape.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence

from osint_benchmark.models.settings import Settings

# Prompt in, completion out.
Complete = Callable[[str], str]

# QwQ and its family open a reply with a reasoning trace. It is not the answer, and reading
# it as one is how a solver's verdict becomes a paragraph of deliberation.
THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


class ModelUnavailable(RuntimeError):
    """No endpoint is configured, or the configured one did not answer."""


def strip_reasoning(reply: str) -> str:
    """Return a reply with any reasoning trace removed.

    An *unclosed* trace means the reply hit its token ceiling mid-thought. That is a
    truncation, not an answer, and it returns empty so the caller treats it as a failure to
    answer rather than scoring the deliberation. In the previous project the opposite
    happened: every truncated reply scored as a failure to answer, which silently marked
    every question necessary.
    """
    if "<think>" in reply.lower() and "</think>" not in reply.lower():
        return ""
    return THINK.sub("", reply).strip()


def llama_server(settings: Settings, timeout: float = 600.0) -> Complete:
    """Return a :data:`Complete` backed by a running llama-server.

    Raises:
        ModelUnavailable: If the role has no endpoint configured. Failing here beats
            failing per-question after an hour of work.
    """
    if not settings.endpoint:
        raise ModelUnavailable(
            f"[{settings.role}] has no endpoint: serve {settings.model} and set it in "
            "config/models.toml, or export OSINT_MODEL_ENDPOINT"
        )

    def complete(prompt: str) -> str:
        """Send one prompt and return the reply, reasoning stripped."""
        payload = json.dumps(
            {
                "prompt": prompt,
                "temperature": settings.temperature,
                "n_predict": settings.max_tokens,
                "stream": False,
            }
        ).encode()
        request = urllib.request.Request(
            f"{settings.endpoint.rstrip('/')}/completion",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                body = json.loads(response.read())
        except (urllib.error.URLError, OSError) as exc:
            raise ModelUnavailable(f"{settings.endpoint}: {exc}") from exc
        return strip_reasoning(body.get("content", ""))

    return complete


def agree(complete: Complete, prompt: str, samples: int, extract: Callable[[str], str]) -> str:
    """Ask the same prompt several times and return the answer only if they agree.

    Repeat-and-agree: a split verdict is a verdict nobody should act on, so it returns
    empty rather than picking a side. This is only meaningful with sampling on — at
    temperature 0 the repeats are identical and unanimity is guaranteed, which is why
    :mod:`osint_benchmark.models.settings` is tested for it.
    """
    replies = [extract(complete(prompt)) for _ in range(max(1, samples))]
    first = replies[0]
    return first if all(reply == first for reply in replies) else ""


def first_word(reply: str, allowed: Sequence[str]) -> str:
    """Return the reply's leading verdict word, lowercased, or empty if it is not allowed.

    Models add preamble however firmly they are told not to. Anything that is not one of the
    permitted verdicts is a non-answer, not a new verdict.
    """
    for token in re.findall(r"[A-Za-z_]+", reply):
        lowered = token.lower()
        if lowered in {word.lower() for word in allowed}:
            return lowered
    return ""
