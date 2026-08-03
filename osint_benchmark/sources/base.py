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
import urllib.error
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
    write_records,
)

HASH_CHUNK = 1 << 20
HTTP_PARTIAL_CONTENT = 206
HTTP_RANGE_NOT_SATISFIABLE = 416


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
    """One bulk corpus: where its raw files land, and how they become records.

    Attributes:
        name: The key used everywhere — on the command line, in ``pins/sources.toml``,
            and as the parsed output's filename.
        kind: ``"private"`` or ``"public"``. Which side of the membrane it sits on.
        parse: Raw directory -> records, each carrying a ``doc_id``. Pure: no network,
            no writing.
        projection: What ``parse`` keeps and drops.
        acquire: Set only when the source is not a set of files to download — the
            parliamentary record is a paged API, so it fetches itself. Returns the raw
            paths it wrote.
    """

    name: str
    kind: str
    parse: Callable[[Path], Iterator[dict]]
    projection: Projection
    acquire: Callable[[Path], list[Path]] | None = None


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
        """True when nothing contradicts the published hashes."""
        return not (self.changed or self.missing or self.unexpected)

    def summary(self) -> str:
        """Return a one-line human summary."""
        if self.baseline_missing:
            return f"{self.source}: {self.checked} documents built, no published hashes to check"
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


def fetch(source: Source, raw_dir: Path | None = None) -> list[Path]:
    """Ensure every raw file for a source is present and matches its pin.

    A file already on disk is never re-downloaded, so pointing ``OSINT_RAW`` at an
    existing copy of the corpora costs nothing, and an interrupted download resumes from
    where it stopped rather than starting over. A source whose pin carries no URL — one
    that cannot be fetched unattended — reports how to obtain the file instead of failing
    silently.

    Raises:
        SourceUnavailable: If a file is absent and has no URL, or if it is present but
            does not match its pinned size or checksum.
    """
    if source.acquire is not None:
        return source.acquire((raw_dir or config.raw_dir()) / source.name)

    paths = []
    for origin in load_origins(source.name):
        dest = raw_path(source, origin, raw_dir)
        if not dest.exists():
            if not origin.url:
                raise SourceUnavailable(
                    f"{source.name}: {dest} is missing and cannot be fetched.\n{origin.note}"
                )
            dest.parent.mkdir(parents=True, exist_ok=True)
            _download(origin.url, dest, origin.size)
        _check_raw(source, origin, dest)
        paths.append(dest)
    return paths


def parse(source: Source, raw_dir: Path | None = None, docs_dir: Path | None = None) -> Path:
    """Parse a source's raw files into ``<docs>/<name>.jsonl`` and return that path."""
    output = (docs_dir or config.docs_dir()) / f"{source.name}.jsonl"
    write_records(
        output,
        source.parse(raw_dir or config.raw_dir()),
        source.projection.to_provenance(),
    )
    return output


def hashes_path(name: str, pins_dir: Path | None = None) -> Path:
    """Return the committed per-document hash file for a source."""
    return (pins_dir or config.pins_dir()) / "hashes" / f"{name}.jsonl"


def document_hashes(source: Source, docs_dir: Path | None = None) -> dict[str, str]:
    """Return ``doc_id -> sha256`` for a source's built documents.

    Raises:
        ValueError: If two records share a ``doc_id``. Keying on it means duplicates
            would collapse into one entry: the count would understate the corpus, and a
            later run could lose every copy but one and still verify clean. That is not
            hypothetical — it hid 8,470 spurious sanctions rows until the counts were
            compared against the source by hand.
    """
    output = (docs_dir or config.docs_dir()) / f"{source.name}.jsonl"
    hashes: dict[str, str] = {}
    duplicates: list[str] = []
    for record in read_jsonl(output):
        doc_id = record["doc_id"]
        if doc_id in hashes:
            duplicates.append(doc_id)
        hashes[doc_id] = record_hash(record)
    if duplicates:
        shown = ", ".join(sorted(set(duplicates))[:5])
        raise ValueError(
            f"{source.name}: {len(duplicates)} records repeat a doc_id ({shown}); "
            "doc_id must be unique within a source"
        )
    return hashes


def write_hashes(
    source: Source, docs_dir: Path | None = None, pins_dir: Path | None = None
) -> Path:
    """Record the built documents' hashes as the baseline everyone else checks against.

    Freezing a baseline is part of publishing a release, so this belongs to step 9 and is
    deliberately not something a run of step 1 can do — the pins a rebuilder verifies
    against must come from the release, not from their own rebuild.
    """
    pins_file = hashes_path(source.name, pins_dir)
    pins_file.parent.mkdir(parents=True, exist_ok=True)
    with pins_file.open("w", encoding="utf-8") as handle:
        for doc_id, digest in sorted(document_hashes(source, docs_dir).items()):
            handle.write(canonical({"doc_id": doc_id, "sha256": digest}) + "\n")
    return pins_file


def verify(
    source: Source, docs_dir: Path | None = None, pins_dir: Path | None = None
) -> VerifyReport:
    """Check built documents against the hashes published with the benchmark.

    This exists because users build the corpora themselves, so a parse regression becomes
    *their* wrong numbers rather than a crash. The failure it is here to catch is real: a
    missing ``escapechar`` in the Cablegate parse cost 68% of the corpus text and produced
    a perfectly well-formed file.

    No published baseline is reported, not treated as a failure: before the first release
    there is nothing to check against, and a build is not wrong for being the first.
    """
    built = document_hashes(source, docs_dir)
    pins_file = hashes_path(source.name, pins_dir)
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


def _download(url: str, dest: Path, expected_size: int | None = None) -> None:
    """Stream a URL into ``<dest>.part`` and rename it only once the transfer completes.

    An interrupted download is resumed from where it stopped rather than restarted: the
    partial file's size becomes a ``Range`` request. This matters because the corpora are
    large — the enwiki dump is 25 GB, forty minutes of transfer on a good link — and a
    connection dropped near the end would otherwise cost the whole thing.

    Resuming is only safe because :func:`_check_raw` hashes the result afterwards. Two
    ways a resumed file can be wrong are caught there and nowhere else: a server that
    ignores ``Range`` and replies 200 with the whole body (handled here by restarting
    rather than appending), and a stale partial left over from a different version of the
    file, which would resume into a seam of two different downloads.
    """
    partial = dest.with_name(dest.name + ".part")
    have = partial.stat().st_size if partial.exists() else 0
    # A partial already at or past the pinned length is not a resumable download, it is a
    # leftover from a different file. Asking for bytes past the end would earn a 416.
    if expected_size is not None and have >= expected_size:
        have = 0

    request = urllib.request.Request(url)
    if have:
        request.add_header("Range", f"bytes={have}-")

    try:
        response = urllib.request.urlopen(request)  # noqa: S310
    except urllib.error.HTTPError as exc:
        if have and exc.code == HTTP_RANGE_NOT_SATISFIABLE:
            # The partial does not belong to this file. Drop it and start over.
            partial.unlink(missing_ok=True)
            _download(url, dest, expected_size)
            return
        raise

    with response:
        # 206 means the server honoured the range and is sending only the remainder.
        # Anything else is the whole file, so appending would splice two copies together.
        resuming = have > 0 and response.status == HTTP_PARTIAL_CONTENT
        with partial.open("ab" if resuming else "wb") as handle:
            shutil.copyfileobj(response, handle)
    partial.replace(dest)


def _check_raw(source: Source, origin: Origin, path: Path) -> None:
    """Check one raw file against its pinned size and checksum."""
    if origin.size is not None and path.stat().st_size != origin.size:
        raise SourceUnavailable(
            f"{source.name}: {path} is {path.stat().st_size} bytes, pinned at {origin.size}"
        )
    if origin.sha256:
        actual = file_hash(path)
        if actual != origin.sha256:
            raise SourceUnavailable(
                f"{source.name}: {path} hashes to {actual}, pinned as {origin.sha256}"
            )
