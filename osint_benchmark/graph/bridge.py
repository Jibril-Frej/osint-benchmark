"""Invert per-document entity links into the entity-document graph.

Step 2 produces one row per document listing the entities it mentions. This inverts that:
one row per entity, listing the documents on each side of the trust boundary that mention
it. An entity with documents on both sides is a **bridge** — a candidate for a question
that needs one of each.

The inversion is the cheap part. What matters is the counting: a bridge naming one cable
and one article is a different proposition from one naming four hundred cables and the
article on the United States, and the pairing step needs to tell them apart.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field


@dataclass
class Bridge:
    """One entity, and the documents mentioning it on each side.

    Attributes:
        qid: The Wikidata entity.
        private: Document ids from the confidential corpora.
        public: Document ids from the public sources.
        surface_forms: The spellings the entity was recognised under, with counts. A
            bridge resting on one rare spelling is weaker than one resting on many.
    """

    qid: str
    private: set[str] = field(default_factory=set)
    public: set[str] = field(default_factory=set)
    surface_forms: dict[str, int] = field(default_factory=dict)

    @property
    def bridges(self) -> bool:
        """True when the entity is named on both sides of the boundary."""
        return bool(self.private and self.public)

    def to_json(self) -> dict:
        """Return the on-disk form, with ids sorted so the output is reproducible."""
        return {
            "doc_id": self.qid,
            "qid": self.qid,
            "private": sorted(self.private),
            "public": sorted(self.public),
            "private_count": len(self.private),
            "public_count": len(self.public),
            "surface_forms": dict(sorted(self.surface_forms.items())),
        }


def build(links: Iterable[dict]) -> dict[str, Bridge]:
    """Return ``QID -> Bridge`` from per-document link rows.

    Each row is ``{doc_id, side, entities}`` where ``side`` is ``private`` or ``public``
    and each entity is ``{qid, surface_form}``. ``surface_form`` is optional: the tabular
    sources reconcile by code and have no spelling to record.

    Raises:
        ValueError: If a row's side is neither private nor public. Silently dropping it
            would quietly shrink one leg of the graph.
    """
    bridges: dict[str, Bridge] = defaultdict(lambda: Bridge(qid=""))
    for row in links:
        side = row["side"]
        if side not in {"private", "public"}:
            raise ValueError(f"{row['doc_id']}: side must be private or public, got {side!r}")
        for entity in row["entities"]:
            qid = entity["qid"]
            bridge = bridges[qid]
            bridge.qid = qid
            getattr(bridge, side).add(row["doc_id"])
            form = entity.get("surface_form")
            if form:
                bridge.surface_forms[form] = bridge.surface_forms.get(form, 0) + 1
    return dict(bridges)


def bridging(bridges: dict[str, Bridge], max_private: int | None = None) -> Iterator[dict]:
    """Yield the entities named on both sides, as records, in QID order.

    ``max_private`` drops entities mentioned in more private documents than that. An entity
    appearing in hundreds of cables is a country or an institution, not a subject: it
    bridges everything and therefore distinguishes nothing. Leaving it in floods the
    pairing step with pairs that share only a continent.
    """
    for qid in sorted(bridges):
        bridge = bridges[qid]
        if not bridge.bridges:
            continue
        if max_private is not None and len(bridge.private) > max_private:
            continue
        yield bridge.to_json()
