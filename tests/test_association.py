"""Unit tests for the association question type.

Every condition here is a *necessity* condition — the reason both documents are required —
so a test that passes when one is removed would be testing nothing. Each one below states
which side of the ablation it protects.
"""

from __future__ import annotations

from osint_benchmark.generate import association


def _facts(**edges):
    """Return Wikidata records from ``QID=[(predicate, target), ...]``."""
    out = []
    for qid, pairs in edges.items():
        statements: dict[str, list[str]] = {}
        for predicate, target in pairs:
            statements.setdefault(predicate, []).append(target)
        out.append({"qid": qid, "statements": statements})
    return out


def _links(doc_id="cablegate:1", *entities):
    """Return one link row naming the given entities."""
    return [
        {
            "doc_id": doc_id,
            "entities": [
                {"qid": qid, "surface_form": surface, "confidence": conf}
                for qid, surface, conf in entities
            ],
        }
    ]


PEOPLE = {"Q1", "Q2", "Q3"}


class TestRelations:
    """Only predicates that carry a real affiliation."""

    def test_a_taxonomic_predicate_is_not_an_affiliation(self):
        """instance_of connects any two people through "human"; a true and worthless edge."""
        graph = association.relations(_facts(Q1=[("instance_of", "Q5")]))

        assert graph == {}

    def test_position_held_is_excluded_although_relational(self):
        """Two leaders sharing an office are usually successors: public, obvious, dull."""
        graph = association.relations(_facts(Q1=[("position_held", "Q100")]))

        assert graph == {}

    def test_party_and_employer_are_affiliations(self):
        """What a shared membership actually looks like."""
        graph = association.relations(_facts(Q1=[("party", "Q7"), ("employer", "Q8")]))

        assert graph["Q1"] == {"party": {"Q7"}, "employer": {"Q8"}}


class TestBuild:
    """Each condition protects one side of the ablation."""

    def test_two_people_sharing_an_affiliation_become_an_item(self):
        """The ordinary path: the pair is private, the affiliation is public."""
        facts = _facts(Q1=[("party", "Q7")], Q2=[("party", "Q7")])
        graph = association.relations(facts)

        items = list(
            association.build(
                _links("cablegate:1", ("Q1", "Meier", 0.99), ("Q2", "Weber", 0.99)),
                graph,
                PEOPLE,
            )
        )

        assert len(items) == 1
        assert (items[0].a, items[0].b, items[0].shared) == ("Q1", "Q2", "Q7")

    def test_a_pair_already_linked_publicly_is_rejected(self):
        """The public-only failure condition.

        If the catalogue already connects them, the private document adds nothing and the
        question is answerable without it.
        """
        facts = _facts(Q1=[("party", "Q7"), ("member_of", "Q2")], Q2=[("party", "Q7")])
        graph = association.relations(facts)

        items = list(
            association.build(
                _links("cablegate:1", ("Q1", "Meier", 0.99), ("Q2", "Weber", 0.99)),
                graph,
                PEOPLE,
            )
        )

        assert items == []

    def test_a_neighbour_everything_points_at_is_a_category(self):
        """Sharing "politician" is not a connection, however true the edge is."""
        edges = {f"Q{i}": [("party", "Q7")] for i in range(10, 45)}
        facts = _facts(Q1=[("party", "Q7")], Q2=[("party", "Q7")], **edges)
        graph = association.relations(facts)

        items = list(
            association.build(
                _links("cablegate:1", ("Q1", "Meier", 0.99), ("Q2", "Weber", 0.99)),
                graph,
                PEOPLE,
                max_degree=20,
            )
        )

        assert items == []

    def test_the_most_specific_shared_neighbour_wins(self):
        """Two affiliations in common: the rarer one is the more informative answer."""
        common = {f"Q{i}": [("party", "Q7")] for i in range(10, 25)}
        facts = _facts(
            Q1=[("party", "Q7"), ("employer", "Q9")],
            Q2=[("party", "Q7"), ("employer", "Q9")],
            **common,
        )
        graph = association.relations(facts)

        items = list(
            association.build(
                _links("cablegate:1", ("Q1", "Meier", 0.99), ("Q2", "Weber", 0.99)),
                graph,
                PEOPLE,
            )
        )

        assert items[0].shared == "Q9"

    def test_an_organisation_pair_is_not_an_association(self):
        """Two airlines sharing a trade body is a coincidence, not a shared affiliation."""
        facts = _facts(Q1=[("member_of", "Q7")], Q2=[("member_of", "Q7")])
        graph = association.relations(facts)

        items = list(
            association.build(
                _links("cablegate:1", ("Q1", "Swissair", 0.99), ("Q2", "Crossair", 0.99)),
                graph,
                people=set(),
            )
        )

        assert items == []

    def test_a_low_confidence_mention_may_not_anchor_a_question(self):
        """A bridge built on a mislink costs a pair; this costs a question with a false premise."""
        facts = _facts(Q1=[("party", "Q7")], Q2=[("party", "Q7")])
        graph = association.relations(facts)

        items = list(
            association.build(
                _links("cablegate:1", ("Q1", "Meier", 0.5), ("Q2", "Weber", 0.99)),
                graph,
                PEOPLE,
            )
        )

        assert items == []

    def test_the_pairs_per_document_are_capped(self):
        """A document naming forty people yields 780 pairs, almost all incidental."""
        facts = _facts(**{f"Q{i}": [("party", f"Q{100 + i}")] for i in range(1, 21)})
        graph = association.relations(facts)
        entities = tuple((f"Q{i}", f"name{i}", 0.99) for i in range(1, 21))

        items = list(
            association.build(
                _links("cablegate:1", *entities),
                graph,
                people={f"Q{i}" for i in range(1, 21)},
                max_per_document=6,
            )
        )

        assert len(items) <= 15  # six entities give fifteen pairs at most
