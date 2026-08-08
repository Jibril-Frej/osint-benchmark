"""Two people a private document puts together, and the public affiliation they share.

The previous project's largest question type: 180 of its 404. Ported rather than reinvented,
including every threshold, because each one is there for a reason that is not visible from
the outside.

**Why both documents are needed, by construction rather than by measurement.** The pair
comes only from the private document — the two are required *not* to be linked in the
public catalogue, so a solver holding only public sources has no reason to consider them
together. The shared affiliation comes only from the public graph, and is absent from the
private text. Neither side alone can answer, and that is a property of how the item is
built, not something an ablation discovers afterwards.

That is the difference between this and a question written from a shared mention. The
generic ``bridge`` type is necessary only when the model happens to write it that way, and
a human check found three quarters of them were not.

Four constraints, all ported:

* **People only.** Organisation pairs produce coincidental links — two airlines sharing a
  trade body, two parties sharing a district — rather than a shared affiliation.
* **Relational predicates only.** A taxonomic one connects any two people through "human"
  or "politician": a true edge and a worthless connection. ``position_held`` is excluded
  even though it is relational, because two leaders sharing an office are usually
  successors, which is public and obvious.
* **Discriminative neighbours only.** A node thousands of entities point at is a category.
  Above ``MAX_NEIGHBOUR_DEGREE`` shared neighbours, sharing it says nothing.
* **The pair must not already be linked publicly.** This is the public-only failure
  condition, and without it the question is answerable from the catalogue alone.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

# The linker's confidence floor for a mention that may anchor a question. Higher than the
# floor used for building the graph: a bridge that turns out wrong costs one pair, an
# association built on a mislink costs a question whose premise is false.
MIN_CONFIDENCE = 0.90

# Predicates that carry a real affiliation. Taxonomic ones (instance_of, occupation,
# citizenship) connect any two people through a class. headquarters and part_of link any
# two bodies sharing a city or a parent, which is coincidence rather than relationship.
RELATIONAL = frozenset({"party", "member_of", "founder", "employer"})

# A neighbour shared by more than this many entities is a category, not a connection.
MAX_NEIGHBOUR_DEGREE = 20

# Entities considered per document. A document naming forty people yields 780 pairs, nearly
# all of them incidental co-mention rather than association.
MAX_PER_DOCUMENT = 6


@dataclass(frozen=True)
class Association:
    """One private document, two people it names, and the affiliation they share.

    Attributes:
        doc_id: The private document putting the two together.
        a: One person's QID.
        b: The other person's QID.
        a_surface: How the private document wrote the first.
        b_surface: How it wrote the second.
        shared: The QID of the affiliation both hold publicly — the answer.
        predicate: Which relation carries it, for a reviewer to check.
        degree: How many entities point at the shared neighbour; lower is more specific.
    """

    doc_id: str
    a: str
    b: str
    a_surface: str
    b_surface: str
    shared: str
    predicate: str
    degree: int
    evidence: dict = field(default_factory=dict)


def relations(facts: Iterable[dict]) -> dict[str, dict[str, set[str]]]:
    """Return ``QID -> predicate -> the QIDs it points at``, for relational predicates only."""
    graph: dict[str, dict[str, set[str]]] = {}
    for record in facts:
        statements = record.get("statements") or {}
        edges = {
            predicate: {v for v in values if isinstance(v, str) and v.startswith("Q")}
            for predicate, values in statements.items()
            if predicate in RELATIONAL
        }
        edges = {predicate: values for predicate, values in edges.items() if values}
        if edges:
            graph[record["qid"]] = edges
    return graph


def degrees(graph: dict[str, dict[str, set[str]]]) -> dict[str, int]:
    """Return how many entities point at each neighbour, the signal for how generic it is."""
    counts: dict[str, int] = {}
    for edges in graph.values():
        for values in edges.values():
            for value in values:
                counts[value] = counts.get(value, 0) + 1
    return counts


def neighbours(graph: dict[str, dict[str, set[str]]], qid: str) -> set[str]:
    """Return everything an entity points at across every relational predicate."""
    return {value for values in graph.get(qid, {}).values() for value in values}


def build(
    links: Iterable[dict],
    graph: dict[str, dict[str, set[str]]],
    people: set[str],
    max_per_document: int = MAX_PER_DOCUMENT,
    max_degree: int = MAX_NEIGHBOUR_DEGREE,
) -> Iterator[Association]:
    """Yield one association per (document, person pair) that satisfies every condition.

    ``people`` is the set of QIDs that are humans; organisation pairs are excluded because
    their shared neighbours are coincidences.
    """
    counts = degrees(graph)
    for row in links:
        named: list[tuple[str, str]] = []
        seen: set[str] = set()
        for entity in row.get("entities", []):
            qid = entity["qid"]
            if entity.get("confidence", 1.0) < MIN_CONFIDENCE:
                continue
            if qid in seen or qid not in people or qid not in graph:
                continue
            seen.add(qid)
            named.append((qid, entity.get("surface_form", "")))
        named = named[:max_per_document]

        for index, (a, a_surface) in enumerate(named):
            for b, b_surface in named[index + 1 :]:
                first, second = neighbours(graph, a), neighbours(graph, b)
                # The public-only failure condition: if the catalogue already links them,
                # the question is answerable without the private document.
                if b in first or a in second:
                    continue
                shared = {c for c in first & second if c not in {a, b}}
                # A neighbour thousands of entities point at is a category, and sharing a
                # category is not a connection.
                shared = {c for c in shared if counts.get(c, 0) <= max_degree}
                if not shared:
                    continue
                # The most specific available: the fewest other entities point at it.
                best = min(shared, key=lambda c: (counts.get(c, 0), c))
                predicate = next(
                    (p for p, values in graph[a].items() if best in values),
                    "",
                )
                yield Association(
                    doc_id=row["doc_id"],
                    a=a,
                    b=b,
                    a_surface=a_surface,
                    b_surface=b_surface,
                    shared=best,
                    predicate=predicate,
                    degree=counts.get(best, 0),
                )
