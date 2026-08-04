"""Record what was sent to a model and what came back.

A run can produce nothing and give no reason. The first real generation attempt made 50
requests and accepted zero questions, and there was no way to tell whether the model
returned unparseable JSON, refused, or answered fine and the parsing was wrong — because
nothing kept the replies.

So every call can be transcribed: prompt, raw reply, what the caller made of it. Off by
default because the prompts carry corpus text and the replies carry gold answers, neither
of which should be written anywhere by accident. ``OSINT_TRANSCRIPT`` turns it on and says
where the file goes.

This is a debugging record, not a provenance record. It is not part of a release and it is
not what a rebuilder checks against.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

from osint_benchmark.models.backend import Complete

_LOCK = threading.Lock()


def transcript_path() -> Path | None:
    """Return where calls should be transcribed, or None when transcription is off."""
    configured = os.environ.get("OSINT_TRANSCRIPT")
    return Path(configured) if configured else None


def record(role: str, prompt: str, reply: str, path: Path | None = None) -> None:
    """Append one call to the transcript, if transcription is on.

    Appends rather than buffers, and locks, so a run that dies mid-way still leaves the
    calls it managed to make — which is the case worth debugging.
    """
    destination = path or transcript_path()
    if destination is None:
        return
    entry = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "role": role,
        "prompt": prompt,
        "reply": reply,
        "reply_chars": len(reply),
    }
    with _LOCK:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def transcribed(complete: Complete, role: str, path: Path | None = None) -> Complete:
    """Wrap a completion function so every call is recorded.

    The reply recorded is what the caller receives — after the reasoning trace is
    stripped. An empty reply here is the most informative case of all: it means the model
    was cut off mid-thought and the caller will treat it as a failure to answer.
    """

    def wrapped(prompt: str) -> str:
        reply = complete(prompt)
        record(role, prompt, reply, path)
        return reply

    return wrapped
