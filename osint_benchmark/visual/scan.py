"""Cut the bare scan out of a Dodis PDF, without the archive's own marks on it.

Every Dodis PDF carries three things the diplomat did not write: a ``dodis.ch/<id>``
watermark in the top-right corner, a QR code encoding the same address, and the Dodis logo.
Any one of them names the document, and a question whose image names its source hands a
solver the archive record, its summary and its catalogue of persons -- the private half of
the question, looked up rather than read.

**They are not masked here, because they are not in the scan.** A Dodis PDF is a page of
layered objects: the scan is one raster the size of the page, and the watermark is a text
object, the QR code a 200x200 bilevel image and the logo a 300x85 one, drawn over it.
Rendering the page would paint all four together and leave a masking step guessing where the
marks landed; taking the scan raster alone never paints them at all. Measured over forty
PDFs, every one had that layout.

What this can refuse, it does. A page whose scan is not a single raster covering the page --
a few documents were stored as a stack of thin strips -- is reported rather than stitched,
because a strip that happens to be the right size is indistinguishable from an overlay.

Uses poppler's ``pdfimages`` and ``pdfinfo`` rather than a Python PDF library, so the
project keeps its empty dependency list; both come with any poppler install.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# How much of the page the scan must cover. A scan is the page; anything well short of it
# is an overlay or a strip, and choosing it would put the wrong picture in the benchmark.
MIN_COVERAGE = 0.85

# The archive's two raster marks, by pixel size. Checked by name as well as by coverage so
# that a PDF laid out differently fails loudly instead of yielding its QR code as "the scan".
APPARATUS_SIZES = frozenset({(200, 200), (300, 85)})

POINTS_PER_INCH = 72.0

Run = Callable[[list[str]], str]


class ScanUnavailable(RuntimeError):
    """Raised when a page has no single raster that can be taken as its scan."""


@dataclass(frozen=True)
class Raster:
    """One image object on a PDF page, as ``pdfimages -list`` reports it.

    Attributes:
        num: The image's index in the listing, which is also its extraction index.
        kind: ``image``, ``smask`` or ``stencil``; only ``image`` is a picture.
        width: Width in pixels.
        height: Height in pixels.
        x_ppi: Horizontal resolution at which it is placed on the page.
        y_ppi: Vertical resolution.
    """

    num: int
    kind: str
    width: int
    height: int
    x_ppi: float
    y_ppi: float

    def coverage(self, page_width: float, page_height: float) -> float:
        """Return the share of the page, in points, this image is drawn over."""
        if not self.x_ppi or not self.y_ppi or not page_width or not page_height:
            return 0.0
        drawn_w = self.width / self.x_ppi * POINTS_PER_INCH
        drawn_h = self.height / self.y_ppi * POINTS_PER_INCH
        return (drawn_w * drawn_h) / (page_width * page_height)


@dataclass(frozen=True)
class Scan:
    """The bare scan of one page, written to disk.

    Attributes:
        path: Where the PNG was written.
        sha256: Its digest, so a release can prove which bytes a question was asked over.
        page: The 1-based page it came from.
        width: Width in pixels.
        height: Height in pixels.
    """

    path: Path
    sha256: str
    page: int
    width: int
    height: int

    def to_json(self, relative_to: Path | None = None) -> dict:
        """Return the on-item form, with the path relative to the item file when given."""
        path = self.path.relative_to(relative_to) if relative_to else self.path
        return {
            "path": str(path),
            "sha256": self.sha256,
            "page": self.page,
            "width": self.width,
            "height": self.height,
            # The scan raster was taken on its own, so the watermark, QR code and logo were
            # never painted into it. Recorded so the gate can refuse an image made any
            # other way.
            "bare": True,
        }


def run(command: list[str]) -> str:
    """Run a poppler tool and return its standard output."""
    if shutil.which(command[0]) is None:
        raise ScanUnavailable(f"{command[0]} is not installed; it comes with poppler-utils")
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout


def page_size(pdf: Path, page: int = 1, runner: Run = run) -> tuple[float, float]:
    """Return a page's width and height in points."""
    out = runner(["pdfinfo", "-f", str(page), "-l", str(page), str(pdf)])
    match = re.search(rf"Page\s+{page} size:\s+([\d.]+) x ([\d.]+)", out) or re.search(
        r"Page size:\s+([\d.]+) x ([\d.]+)", out
    )
    if not match:
        raise ScanUnavailable(f"{pdf.name}: pdfinfo reported no size for page {page}")
    return float(match.group(1)), float(match.group(2))


def rasters(pdf: Path, page: int = 1, runner: Run = run) -> list[Raster]:
    """Return the image objects on one page."""
    out = runner(["pdfimages", "-list", "-f", str(page), "-l", str(page), str(pdf)])
    found = []
    for line in out.splitlines()[2:]:
        fields = line.split()
        if len(fields) < 14 or not fields[1].isdigit():
            continue
        try:
            found.append(
                Raster(
                    num=int(fields[1]),
                    kind=fields[2],
                    width=int(fields[3]),
                    height=int(fields[4]),
                    x_ppi=float(fields[12]),
                    y_ppi=float(fields[13]),
                )
            )
        except ValueError:
            continue
    return found


def choose(images: list[Raster], page_width: float, page_height: float) -> Raster:
    """Return the one raster that is the page's scan.

    Raises:
        ScanUnavailable: When no image covers the page, or more than one does. Either way
            there is no telling which picture the diplomat's page is.
    """
    pictures = [
        image
        for image in images
        if image.kind == "image" and (image.width, image.height) not in APPARATUS_SIZES
    ]
    covering = [
        image for image in pictures if image.coverage(page_width, page_height) >= MIN_COVERAGE
    ]
    if len(covering) != 1:
        raise ScanUnavailable(
            f"{len(covering)} rasters cover the page ({len(pictures)} pictures in all); "
            "a scan stored as strips or overlays is not taken"
        )
    return covering[0]


def digest(path: Path) -> str:
    """Return the SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(pdf: Path, out: Path, page: int = 1, runner: Run = run) -> Scan:
    """Write the bare scan of one page to ``out`` as PNG and return where it went.

    Raises:
        ScanUnavailable: When the page has no single scan raster, or poppler is missing.
    """
    width, height = page_size(pdf, page, runner)
    chosen = choose(rasters(pdf, page, runner), width, height)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        prefix = Path(scratch) / "img"
        runner(["pdfimages", "-png", "-f", str(page), "-l", str(page), str(pdf), str(prefix)])
        # pdfimages numbers its output files by the same index as its listing.
        written = Path(scratch) / f"img-{chosen.num:03d}.png"
        if not written.exists():
            raise ScanUnavailable(f"{pdf.name}: pdfimages wrote no image {chosen.num}")
        shutil.move(written, out)
    return Scan(path=out, sha256=digest(out), page=page, width=chosen.width, height=chosen.height)


def scans_dir() -> Path:
    """Return where the Dodis PDFs live, one ``dodis-<id>.pdf`` per document."""
    return Path(os.environ.get("OSINT_DODIS_SCANS", Path.home() / "dodis_scans"))


def pdf_for(doc_id: str) -> Path:
    """Return the PDF of one Dodis document, by its bare id."""
    return scans_dir() / f"dodis-{doc_id}.pdf"
