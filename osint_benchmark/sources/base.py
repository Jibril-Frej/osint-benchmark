"""The contract every bulk source implements, and the parts that are shared.

The split is not fetch/parse/verify per source, because those are not equally
source-specific. **verify** is identical everywhere — hash the built documents, compare
against the committed pins — and a copy per source would be a chance per source for one
to be weakened differently. **fetch** varies only in its file list and, for a paged API,
its pagination. **parse** is where everything source-specific lives.

So a source module declares two things — a :class:`Projection` saying what its parser
keeps and drops, and a ``parse`` function — and the fetching and verifying happen here.

Where the files come from is not in the Python at all: URLs, sizes and checksums live in
``pins/sources.toml``, so re-pinning a dump date is a data edit rather than a code change.
"""

from __future__ import annotations

import hashlib
import shutil
import tomllib
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from osint_benchmark import config
from osint_benchmark.artifacts import (
    Provenance,
    canonical,
    read_jsonl,
    record_hash,
    write_documents,
)
from osint_benchmark.schema import Document

HASH_CHUNK = 1 << 20


class SourceUnavailable(RuntimeError):
    """A raw file is missing and cannot be fetched automatically."""


@dataclass(frozen=True)
class Origin:
    """One raw file a source is built from, as pinned in ``pins/sources.toml``.

    Attributes:
        filename: The name the file takes under ``<raw>/<source>/``.
        url: Where to fetch it. Empty when the file cannot be redistributed or
            downloaded unattended, in which case ``note`` must say how to obtain it.
        sha256: The expected checksum, or empty when not yet pinned.
        size: The expected size in bytes, or None when not yet pinned.
        note: How to obtain the file, and anything else a rebuilder needs.
    """

    filename: str
    url: str = ""
    sha256: str = ""
    size: int | None = None
    note: str = ""


@dataclass(frozen=True)
class Projection:
    """What a source's parser keeps from that source's fields, and what it drops."""

    source: str
    source_fields: tuple[str, ...]
    kept: dict[str, str]
    dropped: dict[str, str] = field(default_factory=dict)
    kind: str = "corpus"
    note: str = ""

    def to_provenance(self) -> Provenance:
        """Return the provenance record written beside the parsed output."""
        return Provenance(
            source=self.source,
            source_fields=self.source_fields,
            kept=self.kept,
            dropped=self.dropped,
            kind=self.kind,
            note=self.note,
        )


@dataclass(frozen=True)
class Source:
    """One bulk corpus: where its raw files land, and how they become documents.

    Attributes:
        name: The key used everywhere — on the command line, in ``pins/sources.toml``,
            and as the parsed output's filename.
        kind: ``"private"`` or ``"public"``. Which side of the membrane it sits on.
        parse: Raw directory -> documents. Pure: no network, no writing.
        projection: What ``parse`` keeps and drops.
    """

    name: str
    kind: str
    parse: Callable[[Path], Iterator[Document]]
    projection: Projection


@dataclass(frozen=True)
class VerifyReport:
    """The outcome of checking built documents against the committed hashes."""

    source: str
    checked: int = 0
    changed: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()
    baseline_missing: bool = False

    @property
    def ok(self) -> bool:
        """True when every pinned document is present and unchanged."""
        return not (self.changed or self.missing or self.unexpected or self.baseline_missing)

    def summary(self) -> str:
        """Return a one-line human summary."""
        if self.baseline_missing:
            return f"{self.source}: no pinned hashes; run with --write-pins to record a baseline"
        if self.ok:
            return f"{self.source}: {self.checked} documents match the pinned hashes"
        return (
            f"{self.source}: {len(self.changed)} changed, {len(self.missing)} missing, "
            f"{len(self.unexpected)} unexpected (of {self.checked} pinned)"
        )


def load_origins(name: str, pins_file: Path | None = None) -> tuple[Origin, ...]:
    """Return a source's pinned raw files.

    Raises:
        KeyError: If the source has no entry in ``pins/sources.toml``.
    """
    path = pins_file or (config.pins_dir() / "sources.toml")
    pins = tomllib.loads(path.read_text(encoding="utf-8"))
    if name not in pins:
        raise KeyError(f"{name!r} has no entry in {path}")
    return tuple(Origin(**entry) for entry in pins[name].get("origins", []))


def file_hash(path: Path) -> str:
    """Return the SHA-256 of a file, read in chunks so a 25 GB dump does not load."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def raw_path(source: Source, origin: Origin, raw_dir: Path | None = None) -> Path:
    """Return where one of a source's raw files belongs on disk."""
    return (raw_dir or config.raw_dir()) / source.name / origin.filename


def fetch(source: Source, raw_dir: Path | None = None, *, check_hash: bool = True) -> list[Path]:
    """Ensure every raw file for a source is present and matches its pin.

    A file already on disk is never re-downloaded, so pointing ``OSINT_RAW`` at an
    existing copy of the corpora costs nothing. A source whose pin carries no URL — one
    that cannot be redistributed — reports how to obtain the file instead of failing
    silently.

    Raises:
        SourceUnavailable: If a file is absent and has no URL, or if it is present but
            does not match its pinned size or checksum.
    """
    paths = []
    for origin in load_origins(source.name):
        dest = raw_path(source, origin, raw_dir)
        if not dest.exists():
            if not origin.url:
                raise SourceUnavailable(
                    f"{source.name}: {dest} is missing and is not redistributed.\n{origin.note}"
                )
            dest.parent.mkdir(parents=True, exist_ok=True)
            _download(origin.url, dest)
        _check_raw(source, origin, dest, check_hash=check_hash)
        paths.append(dest)
    return paths


def parse(source: Source, raw_dir: Path | None = None, docs_dir: Path | None = None) -> Path:
    """Parse a source's raw files into ``<docs>/<name>.jsonl`` and return that path."""
    output = (docs_dir or config.docs_dir()) / f"{source.name}.jsonl"
    write_documents(
        output,
        source.parse(raw_dir or config.raw_dir()),
        source.projection.to_provenance(),
    )
    return output


def hashes_path(name: str, pins_dir: Path | None = None) -> Path:
    """Return the committed per-document hash file for a source."""
    return (pins_dir or config.pins_dir()) / "hashes" / f"{name}.jsonl"


def verify(
    source: Source,
    docs_dir: Path | None = None,
    pins_dir: Path | None = None,
    *,
    write_pins: bool = False,
) -> VerifyReport:
    """Check built documents against the committed hashes.

    This exists because users build the corpora themselves, so a parse regression becomes
    *their* wrong numbers rather than a crash. The failure it is here to catch is real: a
    missing ``escapechar`` in the Cablegate parse cost 68% of the corpus text and produced
    a perfectly well-formed file.
    """
    output = (docs_dir or config.docs_dir()) / f"{source.name}.jsonl"
    built = {record["doc_id"]: record_hash(record) for record in read_jsonl(output)}

    pins_file = hashes_path(source.name, pins_dir)
    if write_pins:
        pins_file.parent.mkdir(parents=True, exist_ok=True)
        with pins_file.open("w", encoding="utf-8") as handle:
            for doc_id, digest in sorted(built.items()):
                handle.write(canonical({"doc_id": doc_id, "sha256": digest}) + "\n")
        return VerifyReport(source=source.name, checked=len(built))

    if not pins_file.exists():
        return VerifyReport(source=source.name, checked=len(built), baseline_missing=True)

    pinned = {record["doc_id"]: record["sha256"] for record in read_jsonl(pins_file)}
    changed = tuple(sorted(k for k, v in pinned.items() if k in built and built[k] != v))
    missing = tuple(sorted(set(pinned) - set(built)))
    unexpected = tuple(sorted(set(built) - set(pinned)))
    return VerifyReport(
        source=source.name,
        checked=len(pinned),
        changed=changed,
        missing=missing,
        unexpected=unexpected,
    )


def _download(url: str, dest: Path) -> None:
    """Stream a URL to disk, via a partial file so an interrupted run leaves no half-file.

    There is no resume yet. The first source that needs one is Wikipedia, whose dump is
    25 GB; adding it before then would be untested code.
    """
    partial = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url) as response, partial.open("wb") as handle:  # noqa: S310
        shutil.copyfileobj(response, handle)
    partial.replace(dest)


def _check_raw(source: Source, origin: Origin, path: Path, *, check_hash: bool) -> None:
    """Check one raw file against its pinned size and checksum."""
    if origin.size is not None and path.stat().st_size != origin.size:
        raise SourceUnavailable(
            f"{source.name}: {path} is {path.stat().st_size} bytes, pinned at {origin.size}"
        )
    if check_hash and origin.sha256:
        actual = file_hash(path)
        if actual != origin.sha256:
            raise SourceUnavailable(
                f"{source.name}: {path} hashes to {actual}, pinned as {origin.sha256}"
            )
