"""The knowledge-graph payload, built from one ontology engine.

Shared by `GET /api/knowledge-graph` and the room route `GET /api/rooms/{id}/knowledge`
(#1357). The first picks an engine by looking an agent up among the chat surface's agents;
the second is handed the seat's own. Choosing the engine is the caller's decision, and it
is the one thing the two routes must not share -- the chat lookup cannot see a room seat,
and falling back to the shared engine described nothing the seat learned.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from uclone_x.memory.models import MemoryFact

__all__ = ["knowledge_graph", "remembered_statements", "saved_fact_statements"]


def knowledge_graph(
    engine: Any, *, session_id: str | None = None, agent_id: str | None = None
) -> dict[str, Any]:
    """Entity-relation triples (subject, predicate, object, provenance, tier), with nodes and edges."""
    triples: list[dict[str, Any]] = []

    # 1. From Relations (source_entity, predicate, target_entity)
    for r in getattr(engine, "_relations", []):
        evidence = getattr(r, "evidence", None)
        src_session = evidence.source_session if evidence else None
        originating = (
            list(evidence.originating_sessions)
            if evidence and hasattr(evidence, "originating_sessions")
            else []
        )
        r_agent = getattr(engine, "_agent_id", "default")

        if session_id:
            if src_session != session_id and session_id not in originating:
                continue

        if agent_id and agent_id != "all":
            if r_agent != agent_id and (
                not evidence or getattr(evidence, "model_id", None) != agent_id
            ):
                continue

        tier_val = r.tier.value if hasattr(r.tier, "value") else str(r.tier)
        triples.append(
            {
                "subject": r.source_entity,
                "predicate": r.predicate,
                "object": r.target_entity,
                "tier": tier_val,
                "provenance": {
                    "source_session": src_session,
                    "originating_sessions": originating,
                    "agent_id": r_agent,
                    "confidence": getattr(r, "confidence", 1.0),
                    "content_hash": getattr(r, "content_hash", ""),
                    "first_seen": evidence.first_seen if evidence else None,
                    "last_seen": evidence.last_seen if evidence else None,
                    "origin": (
                        "derived"
                        if tier_val in ("induced_enforcing", "induced_candidate")
                        else "axiomatic"
                    ),
                },
            }
        )

    # 2. From Concepts with parent_type (name, is_a, parent_type)
    for c in getattr(engine, "_concepts", {}).values():
        if not getattr(c, "parent_type", None):
            continue
        evidence = getattr(c, "evidence", None)
        src_session = evidence.source_session if evidence else None
        originating = (
            list(evidence.originating_sessions)
            if evidence and hasattr(evidence, "originating_sessions")
            else []
        )
        c_agent = getattr(engine, "_agent_id", "default")

        if session_id:
            if src_session != session_id and session_id not in originating:
                continue

        if agent_id and agent_id != "all":
            if c_agent != agent_id and (
                not evidence or getattr(evidence, "model_id", None) != agent_id
            ):
                continue

        tier_val = c.tier.value if hasattr(c.tier, "value") else str(c.tier)
        triples.append(
            {
                "subject": c.name,
                "predicate": "is_a",
                "object": c.parent_type,
                "tier": tier_val,
                "provenance": {
                    "source_session": src_session,
                    "originating_sessions": originating,
                    "agent_id": c_agent,
                    "confidence": getattr(c, "confidence", 1.0),
                    "content_hash": getattr(c, "content_hash", ""),
                    "first_seen": evidence.first_seen if evidence else None,
                    "last_seen": evidence.last_seen if evidence else None,
                    "origin": (
                        "derived"
                        if tier_val in ("induced_enforcing", "induced_candidate")
                        else "axiomatic"
                    ),
                },
            }
        )

    # 3. From Axioms with predicate and object_value
    for a in getattr(engine, "_axioms", {}).values():
        if not getattr(a, "predicate", None) or not getattr(a, "object_value", None):
            continue
        tier_val = a.tier.value if hasattr(a.tier, "value") else str(a.tier)
        triples.append(
            {
                "subject": a.subject_entity,
                "predicate": a.predicate,
                "object": a.object_value,
                "tier": tier_val,
                "provenance": {
                    "source_session": None,
                    "originating_sessions": [],
                    "agent_id": getattr(engine, "_agent_id", "default"),
                    "confidence": getattr(a, "confidence", 1.0),
                    "content_hash": getattr(a, "content_hash", ""),
                    "description": getattr(a, "description", ""),
                    "rule_expression": getattr(a, "rule_expression", ""),
                    "origin": "axiomatic",
                },
            }
        )

    nodes_map: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    for idx, tr in enumerate(triples):
        s = tr["subject"]
        p = tr["predicate"]
        o = tr["object"]
        t = tr["tier"]
        prov = tr["provenance"]

        if s not in nodes_map:
            nodes_map[s] = {
                "id": s,
                "name": s,
                "tier": t,
                "type": "entity",
                "provenance": prov,
            }
        if o not in nodes_map:
            nodes_map[o] = {
                "id": o,
                "name": o,
                "tier": t,
                "type": "concept" if p == "is_a" else "entity",
                "provenance": prov,
            }

        edges.append(
            {
                "id": f"edge_{idx + 1}",
                "source": s,
                "target": o,
                "predicate": p,
                "tier": t,
                "provenance": prov,
            }
        )

    nodes = list(nodes_map.values())
    return {
        "triples": triples,
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "total_triples": len(triples),
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "session_id": session_id,
            "agent_id": agent_id,
        },
    }


def _statement(subject: str, relation: str, obj: str) -> str:
    """One plain sentence: "X is a Y", or subject, relation and object with `_` spoken as spaces."""
    if relation == "is_a":
        return f"{subject} is a {obj}"
    return f"{subject} {relation.replace('_', ' ').strip()} {obj}"


def remembered_statements(triples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The triples as plain statements, for a reader who is not a programmer (#1357).

    One sentence per triple, from the triple's own words: an `is_a` reads "X is a Y" and
    anything else reads subject, relation and object with underscores spoken as spaces.
    The words a surface wraps around the sentence ("Remembers that...") are the head's to
    write; this supplies the statement and where it came from, and no tier or predicate
    vocabulary.
    """
    remembered: list[dict[str, Any]] = []
    for triple in triples:
        subject = str(triple["subject"])
        relation = str(triple["predicate"])
        obj = str(triple["object"])
        statement = _statement(subject, relation, obj)
        provenance = cast("dict[str, Any]", triple.get("provenance") or {})
        remembered.append(
            {
                "statement": statement,
                "subject": subject,
                "relation": relation,
                "object": obj,
                # The conversation it was learned in, when the engine recorded one. `None`
                # is "not recorded", not "nowhere".
                "source_session_id": provenance.get("source_session"),
                "confidence": provenance.get("confidence"),
                "learned": provenance.get("origin") == "derived",
            }
        )
    return remembered


def saved_fact_statements(facts: Sequence[MemoryFact], session_id: str) -> list[dict[str, Any]]:
    """A clone's saved memory facts as plain statements, in the shape of `remembered_statements` (#1401).

    `saved_here` is whether the fact was saved in the conversation `session_id` names; a
    clone's memory is its own across every conversation, so most of what it holds may have
    been saved elsewhere.
    """
    return [
        {
            "statement": _statement(fact.subject, fact.predicate, fact.object_value),
            "subject": fact.subject,
            "relation": fact.predicate,
            "object": fact.object_value,
            "source_session_id": fact.source_session_id,
            "saved_here": fact.source_session_id == session_id,
            "confidence": fact.confidence,
        }
        for fact in facts
    ]
