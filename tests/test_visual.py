"""Unit tests for the page-scan question types: the scan, the letter, the public record.

Nothing here touches a PDF, poppler, QLever or a model. The scan is cut by a stand-in for
poppler's output, the letters are written for the test, and Wikidata answers from a table,
because what is worth pinning down is what each step refuses -- every refusal is a wrong
gold, or a leaked source, that did not reach the release.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from osint_benchmark.generate import gates, letter, visual
from osint_benchmark.generate.item import Evidence, Item, Necessity
from osint_benchmark.generate.typed import to_item
from osint_benchmark.models import prompts
from osint_benchmark.necessity import ablate
from osint_benchmark.public import offices
from osint_benchmark.public.offices import Term
from osint_benchmark.release.load import load_items
from osint_benchmark.visual import scan

# -- the scan ------------------------------------------------------------------------------

LIST_HEADER = (
    "page   num  type   width height color comp bpc  enc interp  object ID x-ppi y-ppi "
    "size ratio\n" + "-" * 90 + "\n"
)


def _row(num: int, width: int, height: int, ppi: int = 300, kind: str = "image") -> str:
    """Return one line of ``pdfimages -list`` output."""
    return (
        f"   1 {num:5d} {kind:6s} {width:5d} {height:5d}  gray    1   8  image  no "
        f"{num + 10:5d}  0 {ppi:5d} {ppi:5d}  100K 1.0%\n"
    )


# An A4 page in points, and a 300 ppi scan of it.
A4 = "Page size:      595.276 x 841.89 pts (A4)\n"
PAGE_SCAN = (2480, 3508)


def _poppler(listing: str, info: str = A4, written: tuple[int, ...] | None = None):
    """Return a runner standing in for pdfinfo and pdfimages."""

    def runner(command: list[str]) -> str:
        if command[0] == "pdfinfo":
            return info
        if "-list" in command:
            return listing
        prefix = Path(command[-1])
        for num in written if written is not None else range(3):
            (prefix.parent / f"img-{num:03d}.png").write_bytes(f"picture {num}".encode())
        return ""

    return runner


class TestScan:
    """Which raster is taken as the page, and what is refused."""

    def test_the_page_raster_is_taken_and_the_marks_are_not(self, tmp_path):
        """The QR code and the logo are separate rasters, and never the one chosen."""
        listing = LIST_HEADER + _row(0, *PAGE_SCAN) + _row(1, 200, 200) + _row(2, 300, 85)
        out = tmp_path / "scans" / "p1.png"

        taken = scan.extract(tmp_path / "d.pdf", out, runner=_poppler(listing))

        assert out.read_bytes() == b"picture 0"
        assert (taken.width, taken.height, taken.page) == (*PAGE_SCAN, 1)
        assert taken.sha256 == scan.digest(out)

    def test_the_on_item_form_is_relative_and_marked_bare(self, tmp_path):
        """The gate reads ``bare``; the release reads a path next to the item file."""
        listing = LIST_HEADER + _row(0, *PAGE_SCAN)
        out = tmp_path / "images" / "p1.png"

        form = scan.extract(tmp_path / "d.pdf", out, runner=_poppler(listing)).to_json(tmp_path)

        assert form["path"] == "images/p1.png"
        assert form["bare"] is True

    def test_a_scan_stored_as_strips_is_refused(self, tmp_path):
        """No strip covers the page, so there is nothing to take."""
        listing = LIST_HEADER + "".join(_row(n, 2480, 350) for n in range(10))

        with pytest.raises(scan.ScanUnavailable, match="0 rasters"):
            scan.extract(tmp_path / "d.pdf", tmp_path / "o.png", runner=_poppler(listing))

    def test_two_covering_rasters_are_refused(self, tmp_path):
        """Two full-page pictures leave no telling which is the page."""
        listing = LIST_HEADER + _row(0, *PAGE_SCAN) + _row(1, *PAGE_SCAN)

        with pytest.raises(scan.ScanUnavailable, match="2 rasters"):
            scan.extract(tmp_path / "d.pdf", tmp_path / "o.png", runner=_poppler(listing))

    def test_a_mask_is_not_a_picture(self):
        """A soft mask the size of the page is not the scan."""
        images = [scan.Raster(0, "smask", *PAGE_SCAN, 300, 300)]

        with pytest.raises(scan.ScanUnavailable):
            scan.choose(images, 595.276, 841.89)

    def test_a_page_without_a_size_is_refused(self, tmp_path):
        """Coverage cannot be judged without one."""
        with pytest.raises(scan.ScanUnavailable, match="no size"):
            scan.page_size(tmp_path / "d.pdf", runner=_poppler("", info="Pages: 1\n"))

    def test_a_raster_pdfimages_did_not_write_is_refused(self, tmp_path):
        """A listing that promises an image the extraction does not produce."""
        listing = LIST_HEADER + _row(0, *PAGE_SCAN)

        with pytest.raises(scan.ScanUnavailable, match="wrote no image"):
            scan.extract(
                tmp_path / "d.pdf", tmp_path / "o.png", runner=_poppler(listing, written=())
            )

    def test_the_pdf_is_found_under_the_configured_directory(self, tmp_path, monkeypatch):
        """One ``dodis-<id>.pdf`` per document."""
        monkeypatch.setenv("OSINT_DODIS_SCANS", str(tmp_path))

        assert scan.pdf_for("123") == tmp_path / "dodis-123.pdf"


# -- the letter ----------------------------------------------------------------------------

LETTER = """dodis.ch/123
SCHWEIZERISCHE GESANDTSCHAFT
IN OESTERREICH
Wien, den 3. Mai 1950
Herrn Minister A. Z e h n d e r
Chef der Abteilung für Politische Angelegenheiten
Bern
Herr Minister,
Ich beehre mich, Ihnen über die Lage zu berichten. Petitpierre war anwesend.
dodis.ch/123
Seite zwei. Paris wird erwähnt.
"""


class TestLetter:
    """What the head of a letter says, read conservatively."""

    def test_page_one_ends_at_the_second_watermark(self):
        """Paris on page two is not in the letterhead of page one."""
        page = letter.first_page(LETTER, "123")

        assert "Wien" in page
        assert "Paris" not in page

    def test_a_swiss_letterhead_is_found_with_its_place(self):
        """The mission word, and the window that carries the city."""
        head = letter.letterhead(letter.first_page(LETTER, "123"))

        assert head is not None
        assert head.mission == "gesandtschaft"
        assert "wien" in head.window

    def test_a_foreign_embassy_is_not_a_swiss_mission(self):
        """The French embassy in Bern is hosted by Switzerland, not posted from it."""
        assert letter.letterhead("AMBASSADE DE FRANCE\nBerne, le 3 mai 1950\n") is None

    def test_a_place_is_matched_on_whole_words(self):
        """Iran is not in Tirana."""
        assert letter.places_in("legation a tirana", {"iran": "Q794", "tirana": "Q222"}) == {"Q222"}

    def test_the_address_line_gives_initials_and_a_spaced_out_family_name(self):
        """The title is skipped and the letter-spaced name closed up."""
        who = letter.addressee(letter.first_page(LETTER, "123"))

        assert who is not None
        assert (who.family, who.initials, who.given_written) == ("zehnder", ("a",), False)

    def test_a_particle_belongs_to_the_family_name(self):
        """A particle is kept: von Graffenried, not Graffenried."""
        who = letter.addressee("Herrn Dr. von Graffenried\n")

        assert who is not None
        assert who.family == "von graffenried"

    def test_a_given_name_written_out_is_noticed(self):
        """Then the full name is on the page, and there is nothing to look up."""
        who = letter.addressee("Monsieur Max Petitpierre\n")

        assert who is not None
        assert who.given_written

    def test_a_salutation_names_nobody(self):
        """A form of address followed only by titles names nobody."""
        assert letter.addressee("Monsieur le Conseiller fédéral,\n") is None

    def test_a_sentence_opening_with_herrn_is_not_an_address(self):
        """Its last word is not anyone's name."""
        assert (
            letter.addressee("Herrn Müller wurde gestern von uns mitgeteilt dass es regne\n")
            is None
        )

    def test_a_short_sentence_is_still_not_a_usable_address(self):
        """Up to its comma it is short enough, but its words read as a written given name."""
        who = letter.addressee("Herrn Müller wurde mitgeteilt, dass es regnet\n")

        assert who is None or who.given_written

    def test_initials_must_fit_the_given_names(self):
        """A. fits Alfred and not Max."""
        assert letter.initials_agree(("a",), "Alfred Zehnder")
        assert not letter.initials_agree(("a",), "Max Zehnder")
        assert letter.initials_agree((), "Max Zehnder")


# -- the public record ---------------------------------------------------------------------


def _uri(qid: str) -> dict:
    return {"value": f"http://www.wikidata.org/entity/{qid}"}


def _lit(value: str) -> dict:
    return {"value": value}


class TestOffices:
    """What is asked of Wikidata, and what is made of the answer."""

    def test_switzerland_is_never_a_host(self):
        """Every letterhead names it; it is the sender."""
        rows = [{"c": _uri("Q40")}, {"c": _uri("Q39")}, {"c": _uri("Q40")}]

        assert offices.governed(lambda q: rows) == ["Q40"]

    def test_a_name_two_states_share_is_dropped(self):
        """Berlin names two German states; Wien names one."""
        rows = [
            {"c": _uri("Q40"), "l": _lit("Wien")},
            {"c": _uri("Q40"), "l": _lit("Österreich")},
            {"c": _uri("Q183"), "l": _lit("Berlin")},
            {"c": _uri("Q16957"), "l": _lit("Berlin")},
            {"c": _uri("Q145"), "l": _lit("UK")},
        ]

        places = offices.gazetteer(["Q40", "Q183", "Q16957", "Q39"], lambda q: rows)

        assert places == {"wien": "Q40", "osterreich": "Q40"}

    def test_the_states_are_asked_by_key_not_by_scan(self):
        """A label scan across every state times out on the endpoint."""
        asked: list[str] = []

        offices.gazetteer(["Q40", "Q39"], lambda q: asked.append(q) or [])

        assert "VALUES ?c { wd:Q40 }" in asked[0]

    def test_a_term_is_kept_open_ended_and_an_unlabelled_one_skipped(self):
        """An incumbent has no end; a holder with no English name has no answer."""
        rows = [
            {
                "c": _uri("Q40"),
                "h": _uri("Q1"),
                "hl": _lit("Leopold Figl"),
                "s": _lit("1945-12-20T00:00:00Z"),
                "e": _lit("1953-04-02T00:00:00Z"),
            },
            {"c": _uri("Q40"), "h": _uri("Q2"), "hl": _lit("Now"), "s": _lit("2020-01-01")},
            {"c": _uri("Q40"), "h": _uri("Q3"), "s": _lit("1930-01-01")},
        ]

        held = offices.terms(["Q40"], lambda q: rows)

        assert held == [
            Term("Q40", "Q1", "Leopold Figl", "1945-12-20", "1953-04-02"),
            Term("Q40", "Q2", "Now", "2020-01-01", ""),
        ]

    def test_namesakes_are_all_returned_for_the_caller_to_drop(self):
        """Two Swiss people under one name is an ambiguity, not a pick."""
        rows = [
            {"s": _uri("Q10"), "l": _lit("Alfred Zehnder")},
            {"s": _uri("Q11"), "l": _lit("Alfred Zehnder")},
        ]

        assert offices.swiss_people(["Alfred Zehnder"], lambda q: rows) == {
            "Alfred Zehnder": ["Q10", "Q11"]
        }

    def test_a_person_record_gathers_positions_once(self):
        """One row per position; one record per person."""
        base = {"s": _uri("Q10"), "l": _lit("Alfred Zehnder"), "b": _lit("1900-06-01")}
        rows = [
            {**base, "d": _lit("Swiss diplomat"), "pl": _lit("ambassador")},
            {**base, "pl": _lit("ambassador")},
            {**base, "pl": _lit("head of division")},
        ]

        record = offices.person_records(["Q10"], lambda q: rows)["Q10"]

        assert record == {
            "label": "Alfred Zehnder",
            "description": "Swiss diplomat",
            "born": "1900",
            "positions": ["ambassador", "head of division"],
        }
        assert offices.render_person(record) == (
            "Alfred Zehnder (born 1900): Swiss diplomat.\n"
            "Positions held: ambassador; head of division."
        )

    def test_terms_render_in_date_order(self):
        """The evidence a public-only solver is shown."""
        held = [Term("Q40", "Q2", "B", "1953-04-02", ""), Term("Q40", "Q1", "A", "1945-12-20", "x")]

        assert offices.render_terms("Austria", held) == (
            "Heads of government of Austria:\n- A: 1945-12-20 to x\n- B: 1953-04-02 to present"
        )


# -- the builders --------------------------------------------------------------------------

IMAGE = {"path": "images/dodis-123-p1.png", "sha256": "ab" * 32, "bare": True}
FIGL = Term("Q40", "Q1", "Leopold Figl", "1945-12-20", "1953-04-02")


def _document(text: str = LETTER, date_raw: str = "3.5.1950", **meta) -> dict:
    return {
        "doc_id": "123",
        "text": text,
        "date": "1950-05-03",
        "meta": {
            "classification": "Vertraulich",
            "persons": ["Zehnder, Alfred", "Petitpierre, Max"],
            "date_raw": date_raw,
            **meta,
        },
    }


def _offices(documents, terms=None, places=None, scanner=lambda d: dict(IMAGE)):
    outcomes: Counter = Counter()
    built = list(
        visual.from_office_holder(
            documents,
            places if places is not None else {"wien": "Q40"},
            terms if terms is not None else {"Q40": [FIGL]},
            {"Q40": "Austria"},
            scanner,
            outcomes,
        )
    )
    return built, outcomes


def _unavailable(doc_id: str) -> dict:
    raise scan.ScanUnavailable("strips")


class TestOfficeHolder:
    """A letter, its host country and its day, to one head of government."""

    def test_a_certain_letter_becomes_a_candidate(self):
        """The gold is Wikidata's; the private side only says where and when."""
        (candidate,), outcomes = _offices([_document()])

        assert candidate.answer == "Leopold Figl"
        assert candidate.public_id == "offices:Q40"
        assert candidate.private_id == "dodis:123"
        assert candidate.image == IMAGE
        assert candidate.facts == {"mission": "gesandtschaft"}
        assert candidate.provenance["withheld"] == "Austria; 1950"
        assert candidate.provenance["classification"] == "Vertraulich"
        assert candidate.provenance["names_on_scan"] == "Alfred Zehnder; Max Petitpierre"
        assert outcomes == Counter(candidate=1)

    @pytest.mark.parametrize(
        ("document", "kwargs", "reason"),
        [
            (_document(text="Bern, 3. Mai 1950\nLieber Freund"), {}, "no_letterhead"),
            (_document(date_raw="1950"), {}, "date_not_to_the_day"),
            (_document(), {"places": {}}, "host_country_not_named"),
            (
                _document(),
                {"places": {"wien": "Q40", "oesterreich": "Q41"}},
                "host_country_unclear",
            ),
            (_document(), {"terms": {"Q40": []}}, "no_term_on_date"),
            (
                _document(),
                {"terms": {"Q40": [FIGL, Term("Q40", "Q9", "Karl Renner", "1950-05-03", "")]}},
                "term_ambiguous",
            ),
            (_document(text=LETTER + "Figl sprach. Leopold Figl"), {}, "gold_named_privately"),
            (_document(), {"scanner": _unavailable}, "no_scan"),
        ],
    )
    def test_every_uncertain_letter_is_dropped_and_counted(self, document, kwargs, reason):
        """A missed letter is a question fewer; a wrong key is a wrong gold."""
        built, outcomes = _offices([document], **kwargs)

        assert built == []
        assert outcomes == Counter({reason: 1})

    def test_one_holder_answers_at_most_per_answer_questions(self):
        """Three letters from Vienna in Figl's term give two questions, not three."""
        built, outcomes = _offices([_document()] * 3)

        assert len(built) == visual.PER_ANSWER == 2
        assert outcomes == Counter(candidate=2, answer_repeated=1)

    def test_the_same_holder_recorded_twice_is_not_ambiguous(self):
        """Two overlapping statements naming one person still name one person."""
        (candidate,), _ = _offices([_document()], terms={"Q40": [FIGL, FIGL]})

        assert candidate.gold_qid == "Q1"


ZEHNDER = {"label": "Alfred Zehnder", "description": "Swiss diplomat", "positions": []}


def _addressees(documents, swiss=None, records=None, scanner=lambda d: dict(IMAGE)):
    outcomes: Counter = Counter()
    built = list(
        visual.from_addressee(
            documents,
            swiss if swiss is not None else {"Alfred Zehnder": ["Q10"]},
            records if records is not None else {"Q10": ZEHNDER},
            scanner,
            outcomes,
        )
    )
    return built, outcomes


class TestAddressee:
    """An abbreviated address line, to one Swiss person."""

    def test_a_resolvable_addressee_becomes_a_candidate(self):
        """The gold is the archive's full name, confirmed on Wikidata."""
        (candidate,), outcomes = _addressees([_document()])

        assert candidate.answer == "Alfred Zehnder"
        assert candidate.public_id == "people:Q10"
        assert candidate.provenance["withheld"] == "zehnder"
        assert candidate.provenance["archive_name"] == "Alfred Zehnder"
        assert candidate.image == IMAGE
        assert outcomes == Counter(candidate=1)

    @pytest.mark.parametrize(
        ("document", "kwargs", "reason"),
        [
            (_document(text="dodis.ch/123\nBericht\n"), {}, "no_address_line"),
            (_document(text="Monsieur Alfred Zehnder\n"), {}, "given_name_on_letter"),
            (_document(persons=["Petitpierre, Max"]), {}, "addressee_not_catalogued"),
            (
                _document(persons=["Zehnder, Alfred", "Zehnder, Anton"]),
                {},
                "addressee_ambiguous",
            ),
            (_document(text=LETTER + "\nAlfred Zehnder"), {}, "full_name_in_document"),
            (_document(), {"swiss": {}}, "not_on_wikidata"),
            (_document(), {"swiss": {"Alfred Zehnder": ["Q10", "Q11"]}}, "namesakes_on_wikidata"),
            (_document(), {"records": {}}, "no_public_record"),
            (_document(), {"scanner": _unavailable}, "no_scan"),
        ],
    )
    def test_every_unresolved_addressee_is_dropped_and_counted(self, document, kwargs, reason):
        """Each condition under its own name, so a run says where its letters went."""
        built, outcomes = _addressees([document], **kwargs)

        assert built == []
        assert outcomes == Counter({reason: 1})


class TestAddresseeCap:
    """The head of the Political Department received most letters."""

    def test_one_addressee_answers_at_most_per_answer_questions(self):
        """The rest are counted, not silently lost."""
        built, outcomes = _addressees([_document()] * 4)

        assert len(built) == 2
        assert outcomes == Counter(candidate=2, answer_repeated=2)


class TestNamesOnScan:
    """The people a name-redacting strategy has to hide."""

    def test_archive_names_are_read_in_reading_order(self):
        """The archive writes Family, Given."""
        assert visual.display_name("Petitpierre, Max") == "Max Petitpierre"
        assert visual.display_name("Max Petitpierre") == "Max Petitpierre"

    def test_only_names_legible_on_the_page_are_listed(self):
        """A person the archive tags but the page never names is not on the scan."""
        page = "Herrn A. Z e h n d e r\nBericht"

        assert visual.names_on_scan(page, ["Zehnder, Alfred", "Petitpierre, Max"]) == [
            "Alfred Zehnder"
        ]


# -- the item, the gate, the release -------------------------------------------------------


def _item(image: dict | None = IMAGE, ocr_only: bool | None = None) -> Item:
    return Item(
        item_id="dodis:123|addressee|Q10",
        question_type="addressee",
        question="Who is this letter addressed to, in full?",
        answer="Alfred Zehnder",
        rationale="",
        evidence=[
            Evidence(doc_id="dodis:123", source="dodis", side="private"),
            Evidence(doc_id="people:Q10", source="people", side="public"),
        ],
        necessity=Necessity(ocr_only=ocr_only),
        image=dict(image) if image else None,
    )


class TestImageOnTheItem:
    """The image and the fourth condition, present only where they mean something."""

    def test_a_text_item_carries_neither_field(self):
        """So the text types' files are unchanged by images existing."""
        record = _item(image=None).to_json()

        assert "image" not in record
        assert "ocr_only" not in record["necessity"]

    def test_an_image_item_round_trips_through_the_item_file(self, tmp_path):
        """What step 6 writes, step 7 and the release read back."""
        path = tmp_path / "accepted.jsonl"
        path.write_text(json.dumps(_item(ocr_only=False).to_json()) + "\n")

        (loaded,) = load_items(path)

        assert loaded.image == IMAGE
        assert loaded.necessity.ocr_only is False

    def test_a_candidate_hands_its_image_to_the_item(self):
        """And a text candidate hands none."""
        (candidate,), _ = _addressees([_document()])

        assert to_item(candidate, "q?", "an analyst", "m").image == IMAGE

    def test_the_gate_takes_a_bare_scan_with_a_digest(self):
        """Passes a text item too: it has no image to be wrong about."""
        assert gates.image_is_bare_scan(_item())
        assert gates.image_is_bare_scan(_item(image=None))

    @pytest.mark.parametrize(
        "image",
        [
            {**IMAGE, "bare": False},
            {k: v for k, v in IMAGE.items() if k != "bare"},
            {**IMAGE, "sha256": ""},
            {**IMAGE, "path": ""},
        ],
    )
    def test_the_gate_refuses_any_other_image(self, image):
        """A rendered page carries the watermark; an undigested one cannot be audited."""
        assert not gates.image_is_bare_scan(_item(image=image))

    def test_the_gate_is_registered(self):
        """A gate outside the registry is never run."""
        assert gates.GATES["image_is_bare_scan"] is gates.image_is_bare_scan


# -- necessity -----------------------------------------------------------------------------


def _solver(answers: dict[str, str]):
    """Return a solver that answers by what its evidence contains."""

    def solve(prompt: str) -> str:
        for marker, answer in answers.items():
            if marker in prompt:
                return answer
        return "unanswerable"

    return solve


class TestNecessity:
    """The OCR-only condition, and the image shown to a vision model."""

    def test_an_image_item_is_measured_on_the_transcript_too(self):
        """Transcript beside the public record: does the picture add anything?"""
        solver = _solver({"PRIVATE\n\nPUBLIC": "Alfred Zehnder"})

        measured = ablate.measure(_item(), "PRIVATE", "PUBLIC", solver)

        assert (measured.private_only, measured.public_only, measured.ocr_only) == (
            False,
            False,
            True,
        )

    def test_a_text_item_has_no_ocr_condition(self):
        """There is no image for a transcript to stand in for."""
        measured = ablate.measure(_item(image=None), "PRIVATE", "PUBLIC", _solver({}))

        assert measured.ocr_only is None

    def test_a_vision_model_is_shown_the_image_for_private_only(self, tmp_path):
        """The text solver is not asked the private-only condition at all."""
        page = tmp_path / "p.png"
        page.write_bytes(b"png")
        shown: list[Path] = []

        def look(prompt: str, image: Path) -> str:
            shown.append(image)
            assert ablate.IMAGE_EVIDENCE in prompt
            return "Alfred Zehnder"

        measured = ablate.measure(_item(), "PRIVATE", "PUBLIC", _solver({}), look=look, image=page)

        assert measured.private_only is True
        assert shown == [page]

    def test_the_image_path_is_resolved_beside_the_item_file(self, tmp_path):
        """Items record it relative, so the release can move as one directory."""
        shown: list[Path] = []

        def look(prompt: str, image: Path) -> str:
            shown.append(image)
            return "unanswerable"

        list(ablate.measure_items([_item()], {}, _solver({}), look=look, image_root=tmp_path))

        assert shown == [tmp_path / IMAGE["path"]]


# -- the prompts ---------------------------------------------------------------------------


class TestPrompts:
    """Both phrasing prompts render with what their candidates offer."""

    def test_the_office_holder_prompt_renders(self):
        """The mission word, and nothing that names the country or the year."""
        text = prompts.render("phrase_office_holder", mission="gesandtschaft", asker="x", used="y")

        assert "gesandtschaft" in text
        assert "{" in text  # the JSON reply shape survives

    def test_the_addressee_prompt_renders(self):
        """Nothing about the person reaches the phraser at all."""
        assert prompts.placeholders("phrase_addressee") == {"asker", "used"}


class TestReviewAndRelease:
    """The scan reaches the reviewer; only its pointer reaches the release."""

    def test_the_review_page_links_the_scan_and_shows_the_ocr_flag(self):
        """Linked beside data/, not inlined."""
        from osint_benchmark.review import page

        rendered = page.render([_item(ocr_only=True)], {}, image_base="items/")

        assert 'src="items/images/dodis-123-p1.png"' in rendered
        assert "OCR-only: ANSWERED it" in rendered

    def test_a_text_item_has_no_scan_on_the_page(self):
        """No empty image frame for questions that have none."""
        from osint_benchmark.review import page

        assert "<img" not in page.render([_item(image=None)], {})

    def test_the_release_carries_the_digest_not_the_image(self, tmp_path):
        """A release holds nothing that has to be redistributed."""
        from osint_benchmark.release import freeze

        freeze.freeze([_item(ocr_only=True)], {}, tmp_path)

        (record,) = [json.loads(line) for line in (tmp_path / "questions.jsonl").open()]
        assert record["image"]["sha256"] == IMAGE["sha256"]
        assert not list(tmp_path.glob("**/*.png"))
