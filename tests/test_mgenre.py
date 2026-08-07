"""Unit tests for the German/French linker.

Both models are injected, so these run with no GPU, no 3.7 GB pickle and no 90 GB of RAM —
which is the only way this stack is developable at all.
"""

from __future__ import annotations

from osint_benchmark.link import mgenre


def _ner(*entities):
    """Return an NER stand-in yielding the given spans for any window."""
    return lambda window: list(entities)


def _entity(word, start, end, group="PER", score=0.99):
    """Return one recognised span in the shape the pipeline emits."""
    return {"word": word, "start": start, "end": end, "entity_group": group, "score": score}


class TestToQid:
    """The mapping's key shape is undocumented and getting it wrong links nothing."""

    def test_a_language_and_title_key(self):
        """The shape the previous project found first."""
        assert mgenre.to_qid({("de", "Ungarn"): "Q28"}, "Ungarn", "de") == "Q28"

    def test_the_reversed_key(self):
        """It has also been seen this way round."""
        assert mgenre.to_qid({("Ungarn", "de"): "Q28"}, "Ungarn", "de") == "Q28"

    def test_a_bare_title_key(self):
        """And keyed by title alone."""
        assert mgenre.to_qid({"Ungarn": "Q28"}, "Ungarn", "de") == "Q28"

    def test_a_collection_value_takes_the_first(self):
        """Some entries hold a set of candidates rather than one id."""
        assert mgenre.to_qid({("de", "Ungarn"): ["Q28", "Q99"]}, "Ungarn", "de") == "Q28"

    def test_a_bare_number_is_given_its_q(self):
        """Some values are ids without the prefix."""
        assert mgenre.to_qid({("de", "Ungarn"): 28}, "Ungarn", "de") == "Q28"

    def test_an_unknown_title_resolves_to_nothing(self):
        """Rather than to a plausible neighbour."""
        assert mgenre.to_qid({}, "Nirgendwo", "de") is None


class TestUsable:
    """Two filters that exist because of what they let through."""

    def test_a_wordpiece_continuation_is_not_a_mention(self):
        """The tokenizer leaks them and they link confidently to nonsense."""
        assert not mgenre.usable("##fristen", "PER", 0.99)

    def test_a_two_letter_fragment_is_not_a_name(self):
        """ "Je" resolved to the Polish article for German."""
        assert not mgenre.usable("Je", "PER", 0.99)

    def test_a_low_confidence_span_is_dropped(self):
        assert not mgenre.usable("Ungarn", "LOC", 0.5)

    def test_nationality_adjectives_are_not_entities(self):
        """MISC in this vocabulary is mostly "schweizerisch" and its friends."""
        assert not mgenre.usable("schweizerisch", "MISC", 0.99)

    def test_a_person_a_place_and_an_office_are_all_kept(self):
        """A federal office is a legitimate thing for two documents to share."""
        for group in ("PER", "LOC", "ORG"):
            assert mgenre.usable("Bundesrat", group, 0.99)


class TestChunks:
    """A long item is windowed, and the windows overlap for a reason."""

    def test_the_whole_text_is_covered(self):
        text = "x" * 4000
        covered = set()
        for offset, window in mgenre.chunks(text, window=1500, overlap=250):
            covered.update(range(offset, offset + len(window)))

        assert covered == set(range(4000))

    def test_windows_overlap_so_a_boundary_mention_is_seen_whole(self):
        offsets = [offset for offset, _ in mgenre.chunks("x" * 4000, window=1500, overlap=250)]

        assert offsets[1] < 1500

    def test_an_empty_text_yields_nothing(self):
        assert list(mgenre.chunks("")) == []


class TestLinkText:
    """What the graph step reads."""

    def test_a_resolved_mention_becomes_a_link(self):
        """The ordinary path."""
        mentions = list(
            mgenre.link_text(
                "Der Bundesrat sprach über Ungarn.",
                _ner(_entity("Ungarn", 26, 32, group="LOC")),
                lambda contexts: ["Ungarn >> de"],
                {("de", "Ungarn"): "Q28"},
            )
        )

        assert [(m.qid, m.surface_form) for m in mentions] == [("Q28", "Ungarn")]

    def test_a_title_the_mapping_does_not_know_yields_nothing(self):
        """A generated title is a guess until the mapping confirms it."""
        mentions = list(
            mgenre.link_text(
                "Ungarn",
                _ner(_entity("Ungarn", 0, 6, group="LOC")),
                lambda contexts: ["Nirgendwo >> de"],
                {},
            )
        )

        assert mentions == []

    def test_a_repeated_surface_form_is_generated_once(self):
        """Parliamentary German repeats its proper nouns relentlessly.

        Re-deriving them dominates the runtime, and for a proper noun the sense is stable.
        """
        calls = []

        def generate(contexts):
            calls.append(len(contexts))
            return ["Ungarn >> de"] * len(contexts)

        cache: dict[str, str | None] = {}
        for _ in range(3):
            list(
                mgenre.link_text(
                    "Ungarn",
                    _ner(_entity("Ungarn", 0, 6, group="LOC")),
                    generate,
                    {("de", "Ungarn"): "Q28"},
                    cache,
                )
            )

        assert calls == [1]

    def test_the_mention_carries_its_context_to_the_model(self):
        """mGENRE reads the marked span in context; without it a name is a guess."""
        seen = []

        def generate(contexts):
            seen.extend(contexts)
            return ["Ungarn >> de"] * len(contexts)

        list(
            mgenre.link_text(
                "Der Bundesrat sprach über Ungarn im Jahr 1956.",
                _ner(_entity("Ungarn", 26, 32, group="LOC")),
                generate,
                {("de", "Ungarn"): "Q28"},
            )
        )

        assert "[START] Ungarn [END]" in seen[0]
        assert "Bundesrat" in seen[0]


class TestParseGenerated:
    """mGENRE emits "Title >> lang"."""

    def test_a_title_and_language_are_split(self):
        assert mgenre.parse_generated("Ungarn >> de") == ("Ungarn", "de")

    def test_a_reply_without_a_language_is_still_a_title(self):
        """Rather than discarded: the mapping can be keyed by title alone."""
        assert mgenre.parse_generated("Ungarn") == ("Ungarn", None)
