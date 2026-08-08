"""Unit tests for the entity-driven public fetches.

The network calls are injected, so these run offline. What they pin is the handling of the
shapes the live APIs actually return — including the one that cost a debugging round.
"""

from __future__ import annotations

import urllib.error

from osint_benchmark.public import articles, wikidata


def _entity(qid="Q42", labels=None, claims=None, revid=7):
    """Return a Special:EntityData payload."""
    return {
        "entities": {
            qid: {
                "lastrevid": revid,
                "labels": labels if labels is not None else {"en": {"value": "Douglas Adams"}},
                "descriptions": {"en": {"value": "author"}},
                "claims": claims or {},
            }
        }
    }


class TestEnglishTerm:
    """Reading only `en` silently loses a large share of entity names."""

    def test_the_english_label_is_preferred(self):
        """The ordinary case."""
        assert wikidata.english({"en": {"value": "Switzerland"}, "mul": {"value": "Suisse"}}) == (
            "Switzerland"
        )

    def test_a_mul_label_is_used_when_english_is_absent(self):
        """Wikidata moves labels identical across languages to `mul` and drops `en`.

        Q42 has no `en` label at all. Reading only `en` returns empty, which looks like an
        entity with no name rather than a lookup in the wrong place.
        """
        assert wikidata.english({"mul": {"value": "Douglas Adams"}}) == "Douglas Adams"

    def test_neither_yields_empty_not_an_error(self):
        """Some entities genuinely have no English name."""
        assert wikidata.english({"de": {"value": "Bern"}}) == ""


class TestStatements:
    """Only the shapes a question can use are unpacked."""

    def test_each_datatype_becomes_a_readable_string(self):
        """A stringified dict would let plumbing be mistaken for an answer."""
        claims = {
            "P31": [
                {"mainsnak": {"datavalue": {"type": "wikibase-entityid", "value": {"id": "Q5"}}}}
            ],
            "P569": [
                {"mainsnak": {"datavalue": {"type": "time", "value": {"time": "+1952-03-11"}}}}
            ],
        }

        out = wikidata.statements({"claims": claims})

        assert out["instance_of"] == ["Q5"]
        assert out["birth_date"] == ["1952-03-11"]

    def test_a_deprecated_statement_is_not_an_answer(self):
        """Wikidata marks superseded values rather than deleting them."""
        claims = {
            "P159": [
                {
                    "rank": "deprecated",
                    "mainsnak": {"datavalue": {"type": "wikibase-entityid", "value": {"id": "Q1"}}},
                }
            ]
        }

        assert "headquarters" not in wikidata.statements({"claims": claims})

    def test_an_unhandled_datatype_is_dropped_not_stringified(self):
        """Globe coordinates and the like are not answers to any question here."""
        claims = {"P159": [{"mainsnak": {"datavalue": {"type": "globecoordinate", "value": {}}}}]}

        assert wikidata.statements({"claims": claims}) == {}


class TestFetchEntities:
    """A failed fetch is unknown data, not absent data."""

    def test_the_revision_read_is_recorded(self):
        """A gold answer from a live source is only correct against one revision."""
        record = next(wikidata.fetch_entities(["Q42"], lambda q: _entity(revid=99), pause=0))

        assert record["revision"] == 99

    def test_a_failure_is_reported_not_yielded_as_empty(self):
        """Silently turning failures into empty records is how a third of a set vanished."""
        seen = []

        def fetch(qid):
            raise urllib.error.URLError("down")

        records = list(
            wikidata.fetch_entities(["Q1"], fetch, pause=0, on_error=lambda q, e: seen.append(q))
        )

        assert records == []
        assert seen == ["Q1"]

    def test_repeated_qids_are_fetched_once(self):
        """A bridge list can name the same entity from several documents."""
        calls = []

        def fetch(qid):
            calls.append(qid)
            return _entity(qid)

        list(wikidata.fetch_entities(["Q42", "Q42"], fetch, pause=0))

        assert calls == ["Q42"]


class TestFetchArticles:
    """Lead sections for bridge entities, pinned to the revision read."""

    def test_a_record_carries_its_revision_and_qid(self):
        """The public half of a question's evidence has to be re-findable."""
        payload = {
            "query": {
                "pages": [
                    {
                        "title": "Douglas Adams",
                        "pageid": 8091,
                        "extract": "An English author.",
                        "revisions": [{"revid": 42, "timestamp": "2026-01-01T00:00:00Z"}],
                    }
                ]
            }
        }

        record = next(articles.fetch_articles([("Q42", "Douglas Adams")], lambda t: payload, 0))

        assert record["doc_id"] == "enwiki:Q42"
        assert record["qid"] == "Q42"
        assert record["revision"] == 42
        assert record["text"] == "An English author."

    def test_a_missing_article_yields_nothing(self):
        """A question cannot be built on evidence that is not there."""
        payload = {"query": {"pages": [{"title": "Nope", "missing": True}]}}

        assert list(articles.fetch_articles([("Q1", "Nope")], lambda t: payload, 0)) == []

    def test_an_empty_extract_yields_nothing(self):
        """An empty extract would look like evidence that exists."""
        payload = {"query": {"pages": [{"title": "Stub", "extract": "   ", "pageid": 1}]}}

        assert list(articles.fetch_articles([("Q1", "Stub")], lambda t: payload, 0)) == []

    def test_titles_are_batched(self):
        """One request per article would be twenty times the traffic."""
        calls = []

        def fetch(titles):
            calls.append(len(titles))
            return {"query": {"pages": []}}

        list(articles.fetch_articles([(f"Q{i}", f"T{i}") for i in range(45)], fetch, 0))

        assert calls == [20, 20, 5]


def _step4():
    """Import the numbered pipeline file, whose name cannot be imported normally."""
    import importlib.util

    from osint_benchmark import paths

    spec = importlib.util.spec_from_file_location("step4", paths.ROOT / "pipeline" / "04_public.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestLinkedEntities:
    """The typed builders need every linked entity, not only the bridges."""

    def test_every_linked_entity_is_returned(self, tmp_path, monkeypatch):
        """An association is two people in one document; neither need be a bridge.

        Scoping the fetch to bridges would starve exactly the type that does not use them.
        """
        import json

        links = tmp_path / "links"
        links.mkdir(parents=True)
        (links / "cablegate.jsonl").write_text(
            json.dumps(
                {
                    "doc_id": "cablegate:1",
                    "entities": [
                        {"qid": "Q1", "confidence": 0.99},
                        {"qid": "Q2", "confidence": 0.5},
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("OSINT_DATA", str(tmp_path))

        assert _step4().linked_entities() == ["Q1", "Q2"]

    def test_an_unsure_mention_can_be_excluded(self, tmp_path, monkeypatch):
        """A question built on a mislink has a false premise, so the floor is higher here."""
        import json

        links = tmp_path / "links"
        links.mkdir(parents=True)
        (links / "cablegate.jsonl").write_text(
            json.dumps(
                {
                    "doc_id": "cablegate:1",
                    "entities": [
                        {"qid": "Q1", "confidence": 0.99},
                        {"qid": "Q2", "confidence": 0.5},
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("OSINT_DATA", str(tmp_path))

        assert _step4().linked_entities(confidence=0.9) == ["Q1"]


class TestNeighbours:
    """The second hop the association type cannot work without."""

    def test_the_entities_pointed_at_are_returned(self):
        """The organisation two people share may itself appear in no document."""
        records = [
            {"qid": "Q1", "statements": {"member_of": ["Q7"], "instance_of": ["Q5"]}},
            {"qid": "Q2", "statements": {"member_of": ["Q7"], "employer": ["Q8"]}},
        ]

        assert _step4().neighbours_of(records, frozenset({"member_of", "employer"})) == ["Q7", "Q8"]

    def test_an_entity_already_fetched_is_not_fetched_twice(self):
        """The first hop covers it, and a second request for it is wasted."""
        records = [
            {"qid": "Q1", "statements": {"member_of": ["Q2"]}},
            {"qid": "Q2", "statements": {}},
        ]

        assert _step4().neighbours_of(records, frozenset({"member_of"})) == []

    def test_a_taxonomic_predicate_is_not_followed(self):
        """Fetching "human" for every person is 60,000 requests for one entity."""
        records = [{"qid": "Q1", "statements": {"instance_of": ["Q5"]}}]

        assert _step4().neighbours_of(records, frozenset({"member_of"})) == []
