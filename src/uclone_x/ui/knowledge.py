"""The knowledge-graph payload, built from one ontology engine.

Shared by `GET /api/knowledge-graph` and the room route `GET /api/rooms/{id}/knowledge`
(#1357). Both are handed the named clone's one rules engine (`ontology_for`); there is no
shared engine to fall back on any more (#1869), since a fallback described nothing the
clone learned.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from uclone_x.memory.models import MemoryFact
from uclone_x.ontology.models import OntologyAxiom
from uclone_x.ontology.reasoner import worked_out

__all__ = ["knowledge_graph", "known_facts", "worked_out_list"]


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


def known_facts(
    facts: Sequence[MemoryFact], room_id: str, session_id: str | None
) -> list[dict[str, Any]]:
    """A clone's facts as plain statements, one clone-wide list (clone-knowledge-graph §3.8).

    `learned_here` is whether the fact was learned in the conversation `room_id` names.
    A fact saved before facts recorded their conversation (#1716) has no `source_room_id`;
    it counts as learned here when its session is this seat's session in this
    conversation, which is how such a fact was marked before.

    The field names are the store's own (`predicate`, `object_value`), not the design's
    earlier `relation` / `value`.
    """
    listed: list[dict[str, Any]] = []
    for fact in facts:
        if fact.source_room_id is not None:
            here = fact.source_room_id == room_id
        else:
            here = session_id is not None and fact.source_session_id == session_id
        listed.append(
            {
                "fact_id": fact.fact_id,
                "statement": _statement(fact.subject, fact.predicate, fact.object_value),
                "subject": fact.subject,
                "predicate": fact.predicate,
                "object_value": fact.object_value,
                "origin": fact.origin,
                "learned_here": here,
                "source_turn_id": fact.source_turn_id,
                "confidence": fact.confidence,
                "created_at": fact.created_at,
            }
        )
    return listed


def worked_out_list(
    statements: Mapping[str, tuple[str, str, str]], axioms: Iterable[OntologyAxiom]
) -> list[dict[str, Any]]:
    """What a clone's rules work out from its facts, each with the facts it rests on (step 6).

    `statements` is `fact_id -> (subject, predicate, object)` of the facts that hold now
    (`CrossSessionMemory.statements_now`, the knowledge store's `facts_at` with the ids
    kept). Computed on every read and never saved (clone-knowledge-graph §3.1), so a fact
    corrected or forgotten takes what followed from it with it. `because` holds the
    `fact_id`s of the statements the answer rests on.
    """
    return [
        {
            "statement": _statement(w.subject, w.predicate, w.object),
            "because": list(w.because),
        }
        for w in worked_out(statements, axioms)
    ]
