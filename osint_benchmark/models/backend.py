"""Talk to a served model.

One protocol — :data:`Complete`, text in and text out — so every stage that needs a model
takes one as an argument and a test passes a stub. Nothing above this module knows whether
there is a GPU anywhere.

**vLLM is the serving stack**, chosen over the llama.cpp server the previous project used.
The workload decides it: each question costs about seven calls -- one draft, three judge
samples for repeat-and-agree, three solver calls for the ablation -- so a few hundred
questions is several thousand generations, each with two documents in the prompt and a
reasoning model emitting thousands of tokens. That is what continuous batching is for.
llama.cpp's advantage is CPU inference, which buys nothing here: steps 6 and 7 are build
steps only the benchmark's authors run, and ``models.stub`` already covers exercising the
chain without a GPU.

The wire format is the OpenAI chat API, which llama-server also speaks, so the decision is
about serving infrastructure rather than about this file.
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

    Three shapes, and only the first is the obvious one:

    * **Matched tags.** The trace is between them and comes out.
    * **An opening tag with no closing one.** The reply hit its token ceiling mid-thought.
      That is a truncation, not an answer, so it returns empty and the caller treats it as
      a failure to answer rather than scoring the deliberation. In the previous project the
      opposite happened, which silently marked every question necessary.
    * **A closing tag with no opening one.** The serving template opened the block itself.
      vLLM's chat template for QwQ appends ``<think>`` to the *prompt*, so the model's
      completion begins inside the trace and only ever emits ``</think>``. Nothing matches,
      the whole reasoning trace survives as the "answer", and a caller looking for JSON
      finds a brace somewhere in the deliberation and fails to parse. This accounted for 80
      of 80 discarded drafts on job 13350, and for the 79 of 80 on job 13349 that I had
      blamed on the token budget.
    """
    lowered = reply.lower()
    opened, closed = "<think>" in lowered, "</think>" in lowered
    if opened and not closed:
        return ""
    if closed and not opened:
        return reply[lowered.rfind("</think>") + len("</think>") :].strip()
    return THINK.sub("", reply).strip()


def vllm(settings: Settings, timeout: float = 600.0) -> Complete:
    """Return a :data:`Complete` backed by a vLLM server.

    Raises:
        ModelUnavailable: If the role has no endpoint configured. Failing here beats
            failing per-question after an hour of work.
    """
    if not settings.endpoint:
        raise ModelUnavailable(
            f"[{settings.role}] has no endpoint: serve {settings.model} with vLLM and set "
            "it in config/models.toml, or export OSINT_MODEL_ENDPOINT"
        )
    url = f"{settings.endpoint.rstrip('/')}/v1/chat/completions"

    def complete(prompt: str) -> str:
        """Send one prompt and return the reply, reasoning stripped."""
        payload = json.dumps(
            {
                "model": settings.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": settings.temperature,
                "max_tokens": settings.max_tokens,
            }
        ).encode()
        request = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                body = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            # The reason is in the body, not the status line. Discarding it turns "your
            # prompt is longer than the context window" into a bare 400.
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except Exception:  # noqa: BLE001 - a body we cannot read is not a new failure
                pass
            raise ModelUnavailable(f"{url}: {exc} {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise ModelUnavailable(f"{url}: {exc}") from exc
        choices = body.get("choices") or []
        if not choices:
            return ""
        return strip_reasoning(choices[0].get("message", {}).get("content") or "")

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
