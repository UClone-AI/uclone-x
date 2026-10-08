"""Checking a scene's facts against what the story says was true then (#1557).

The model reads a scene and **proposes** facts from it (`vane status alive`), each with the
quote it read it from. Nothing here trusts that reading: a fact without a quote, or with a
quote the scene does not contain, is rejected before anything is checked. What is left is
checked **deterministically** -- by the ontology reasoner, not by the model -- against the
codex as it stands once the scene has ended in story time (`uclone_x.story.timeline`), under
the story's rules (`story.yaml` `axioms`).

How facts are made from the codex: the codex is read as the story's uGraph store
(`uclone_x.story.codex_store`), and the facts are the approved edges that hold once the
scene has ended. So:

- every state value of every entry is a fact `(entry id, key, value)`; a list gives one fact
  per element, and a value that is a set of named fields is reported as not checked;
- a relation (`relations: {harin: parent}`) is the fact `(entry id, relation:harin,
  parent)` (`relation_predicate`): one word per pair, so a rule that the property is
  functional says a scene may not give the pair another;
- the key `status` becomes a type: `status: dead` is `type Dead`, so a rule like
  "Alive and Dead are disjoint" applies to it;
- names are matched to entries by id, name or alias, ignoring case, so `Lord Vane` and
  `vane` are one subject. A name no entry has is checked as it is, and listed.

A rule's classes and properties are read the way the facts are (#1584): classes are one
class whatever their case, separators and Unicode normalisation form (`_Classes`), and a
property is spelled as `_predicate` does, so `disjointWith alive half_dead` is the same
rule as `disjointWith Alive HalfDead`. A class is shown as the rules spell it (#1601). A rule that cannot be applied is reported as
the story wrote it, not by the name the reasoner gave it.

A contradiction comes back with where each fact came from: a codex entry's starting state,
a progression at a scene, or a submitted fact with its quote. A fact the reasoner derived
(`yerin type Dead`, from `yerin type Translucent` and `Translucent subClassOf Dead`) lists
the asserted facts it was derived from, with theirs, so a contradiction that rests on a
scene fact only through a rule still involves the scene. Every rule that was not
applied is listed (P6): a check never says a scene is consistent with a rule it did not
run.

The ontology modules are the adapter side of the reasoner; this module builds one
in-memory engine per check and keeps nothing.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from uclone_x.ontology.engine import OntologyEngine
from uclone_x.ontology.justification import (
    DISJOINT_WITH,
    DOMAIN,
    FUNCTIONAL_PROPERTY,
    INVERSE_FUNCTIONAL_PROPERTY,
    INVERSE_OF,
    MAX_CARDINALITY,
    MIN_CARDINALITY,
    RANGE,
    SUBCLASS_OF,
    SUBPROPERTY_OF,
    SYMMETRIC_PROPERTY,
    TRANSITIVE_PROPERTY,
    Fact,
)
from uclone_x.ontology.models import OntologyAxiom, OntologyRelation
from uclone_x.ontology.reasoner import OntologyReasoner
from uclone_x.story.codex_store import CodexYamlStore
from uclone_x.story.context import CodexIndex
from uclone_x.story.names import CodexNames
from uclone_x.story.quotes import MIN_QUOTE_CHARACTERS, quote_found, quote_too_short
from uclone_x.story.quotes import folded as _folded
from uclone_x.story.schemas import Outline, StoryAxiom
from uclone_x.story.timeline import assumptions

__all__ = ["RELATION_PREFIX", "SubmittedFact", "audit_scene", "relation_predicate"]

#: How a relation reads to the reasoner: `relation:<the other entry's id>`, its word the
#: object, so one pair has one property (`relation_predicate`).
RELATION_PREFIX = "relation:"


def relation_predicate(other_id: str) -> str:
    """The property a relation to entry `other_id` is checked as, by the codex and a scene."""
    return f"{RELATION_PREFIX}{other_id}"


_WORD_JOIN = re.compile(r"[\s_-]+")
_KIND_PREFIX = re.compile(r"^(?:rdf|rdfs|owl)\s*:\s*", re.IGNORECASE)

#: The rule kinds the reasoner reads, keyed by their spelling with case, prefix, spaces,
#: `_` and `-` removed, so `DisjointWith`, `disjoint_with` and `owl:disjointWith` are one
#: kind. The property kinds are also read without the word `property`, as the reasoner
#: reads them.
_KINDS = {
    kind.casefold(): kind
    for kind in (
        DISJOINT_WITH,
        DOMAIN,
        FUNCTIONAL_PROPERTY,
        INVERSE_FUNCTIONAL_PROPERTY,
        INVERSE_OF,
        MAX_CARDINALITY,
        MIN_CARDINALITY,
        RANGE,
        SUBCLASS_OF,
        SUBPROPERTY_OF,
        SYMMETRIC_PROPERTY,
        TRANSITIVE_PROPERTY,
    )
} | {
    kind.casefold().removesuffix("property"): kind
    for kind in (
        FUNCTIONAL_PROPERTY,
        INVERSE_FUNCTIONAL_PROPERTY,
        SYMMETRIC_PROPERTY,
        TRANSITIVE_PROPERTY,
    )
}


def _kind(raw: str) -> str | None:
    """The reasoner's name for a rule kind as written, or `None` if it reads no such kind."""
    return _KINDS.get(_WORD_JOIN.sub("", _KIND_PREFIX.sub("", raw.strip())).casefold())


@dataclass(frozen=True)
class SubmittedFact:
    """One fact the model read from a scene, and the quote it read it from."""

    subject: str
    predicate: str
    object: str
    quote: str | None


def _nfc(raw: str) -> str:
    """`raw` in NFC, so a composed and a decomposed `é` are one letter (#1601)."""
    return unicodedata.normalize("NFC", raw.strip())


def _predicate(raw: str) -> str:
    return _WORD_JOIN.sub("_", _nfc(raw)).casefold()


class _Classes:
    """Class names, compared ignoring case, spaces, `_`, `-` and Unicode form (#1584, #1601).

    `half dead`, `HalfDead`, `HALF_DEAD` and `halfdead` are one class, and so are `dead`,
    `DEAD` and `Dead`. A class is spelled as it was first seen in one check, each word's
    first letter raised: `half dead` is `HalfDead`, and `HalfDead` stays `HalfDead`. In a
    name of several words, a word in capitals is written as a word, so `HALF-dead` and
    `HALF_DEAD` are `HalfDead` rather than `HALFDead` (#1613); a one-word name keeps its
    capitals (`DEAD`, `NPCAlive`). The rules are read first, so a class a rule names is shown as the rule spells it
    (`NPCAlive`, not `NpcAlive` from a codex `npc alive`); then the scene's facts, then
    the codex.
    """

    def __init__(self) -> None:
        self._spelling: dict[str, str] = {}

    def __call__(self, raw: str) -> str:
        words = [part for part in _WORD_JOIN.split(_nfc(raw)) if part]
        if len(words) > 1:
            words = [word.capitalize() if word.isupper() else word for word in words]
        spelled = "".join(word[:1].upper() + word[1:] for word in words)
        return self._spelling.setdefault(spelled.casefold(), spelled)


def _axiom_terms(axiom: StoryAxiom, class_name: _Classes) -> tuple[str, str]:
    """A rule's subject and object, each spelled as the facts spell a class or property.

    A kind the reasoner does not know is left as written: it is reported, not applied.
    """
    kind = _kind(axiom.kind)
    subject, obj = axiom.subject, axiom.object
    if kind in (DISJOINT_WITH, SUBCLASS_OF):
        return class_name(subject), class_name(obj)
    if kind in (DOMAIN, RANGE):
        return _predicate(subject), class_name(obj)
    if kind in (MIN_CARDINALITY, MAX_CARDINALITY):
        prop, colon, count = obj.rpartition(":")
        if colon and prop.strip():
            obj = f"{_predicate(prop)}:{count.strip()}"
        return class_name(subject), obj
    if kind in (SUBPROPERTY_OF, INVERSE_OF):
        return _predicate(subject), _predicate(obj)
    if kind is not None:
        return _predicate(subject), obj
    return subject, obj


def _written(axiom: StoryAxiom) -> str:
    """The rule as `story.yaml` has it."""
    return " ".join(part for part in (axiom.kind, axiom.subject, axiom.object) if part)


#: The reasoner's field names, in the words `story.yaml` uses.
_FIELD_WORDS = (("subject_entity", "subject"), ("object_value", "object"))


def _plain_reason(reason: str) -> str:
    for field_name, word in _FIELD_WORDS:
        reason = reason.replace(field_name, word)
    return reason


@dataclass
class _Triples:
    """The facts to check, each with every place it came from."""

    names: CodexNames
    class_name: _Classes
    facts: dict[str, Fact]
    origins: dict[str, list[dict[str, Any]]]
    unresolved: set[str]

    def add(self, subject: str, predicate: str, value: str, origin: dict[str, Any]) -> None:
        subject_id = self.names.resolve(subject)
        if subject_id is None:
            self.unresolved.add(subject)
            subject_id = _folded(subject)
        pred = _predicate(predicate)
        if pred == "status":
            pred, obj = "type", self.class_name(value)
        elif pred == "type":
            obj = self.class_name(value)
        else:
            obj = self.names.resolve(value) or _folded(value)
        fact = Fact(subject=subject_id, predicate=pred, object=obj)
        self.facts.setdefault(fact.id, fact)
        self.origins.setdefault(fact.id, []).append(origin)


def _sources(contradiction: dict[str, Any]) -> list[dict[str, Any]]:
    """Where every fact a contradiction rests on came from, including the facts a derived
    one was derived from."""
    found: list[dict[str, Any]] = []
    for part in cast("list[dict[str, Any]]", contradiction["facts"]):
        found += part.get("sources", ())
        for leaf in cast("list[dict[str, Any]]", part.get("derived_from", ())):
            found += leaf["sources"]
    return found


def audit_scene(
    *,
    story_id: str,
    outline: Outline,
    scene_id: str,
    scene_text: str,
    codex: CodexIndex,
    facts: Sequence[SubmittedFact],
    axioms: Sequence[StoryAxiom],
    axioms_are_defaults: bool,
) -> dict[str, Any]:
    """Check `facts`, read from `scene_id`, against the codex once that scene has ended.

    Raises:
        ValueError: the outline has no scene `scene_id` (the caller checks first).
    """
    store = CodexYamlStore.read(story_id, outline, codex)
    placements = store.placements
    if scene_id not in placements:
        raise ValueError(f"the outline has no scene '{scene_id}'")

    class_name = _Classes()
    # The rules are read before the facts, so their spelling of a class is the one shown
    # (#1601); reading them again below returns the same names.
    for axiom in axioms:
        _axiom_terms(axiom, class_name)
    triples = _Triples(CodexNames(codex), class_name, {}, {}, set())
    rejected: list[dict[str, Any]] = []
    accepted = 0
    for number, fact in enumerate(facts, start=1):
        if fact.quote is None or not fact.quote.strip():
            reason = f"Fact {number} has no quote from the scene, so it was not checked."
        elif quote_too_short(fact.quote):
            reason = (
                f"Fact {number}'s quote is too short to show anything: a quote needs at "
                f"least {MIN_QUOTE_CHARACTERS} letters, so it was not checked."
            )
        elif not quote_found(fact.quote, scene_text):
            reason = f"Fact {number}'s quote is not in scene '{scene_id}', so it was not checked."
        elif not (fact.subject.strip() and fact.predicate.strip() and fact.object.strip()):
            reason = (
                f"Fact {number} is missing its subject, predicate or object, so it was not checked."
            )
        else:
            triples.add(
                fact.subject,
                fact.predicate,
                fact.object,
                {"from": "scene", "scene_id": scene_id, "fact": number, "quote": fact.quote},
            )
            accepted += 1
            continue
        rejected.append({"fact": number, "reason": reason})

    not_checked: list[dict[str, str]] = []
    for row in store.rows_at(store.point(scene_id, ended=True)):
        origin: dict[str, Any] = dict(row.record["origin"])
        edge = row.edge
        if not row.record["single"]:
            not_checked.append(
                {"file": origin["file"], "field": origin["field"], "reason": "not a single value"}
            )
        elif edge.object_id is not None:
            triples.add(edge.subject_id, relation_predicate(edge.object_id), edge.predicate, origin)
        else:
            triples.add(edge.subject_id, edge.predicate, str(edge.value), origin)

    engine = OntologyEngine(
        agent_id=f"story:{story_id}",
        namespace_iri=f"https://uclone-x.ai/story/{story_id}",
    )
    written: dict[str, str] = {}
    for index, axiom in enumerate(axioms):
        name = f"story_axiom_{index}"
        written[name] = _written(axiom)
        subject, obj = _axiom_terms(axiom, class_name)
        engine.register_axiom(
            OntologyAxiom(
                name=name,
                subject_entity=subject,
                predicate=axiom.kind,
                object_value=obj,
                description=axiom.note or "",
            )
        )
    for fact in triples.facts.values():
        engine.register_relation(
            OntologyRelation(
                source_entity=fact.subject, predicate=fact.predicate, target_entity=fact.object
            )
        )
    loaded = [
        Fact(subject=r.source_entity, predicate=r.predicate, object=r.target_entity)
        for r in engine.list_relations()
    ]
    reasoner = OntologyReasoner.from_engine(engine, loaded)
    closure = reasoner.materialize()
    found = reasoner.check_consistency()

    contradictions: list[dict[str, Any]] = []
    for inconsistency in found:
        involved: list[dict[str, Any]] = []
        for fact_id in inconsistency.facts:
            if fact_id in triples.origins:
                fact = triples.facts[fact_id]
                involved.append(
                    {"fact": " ".join(fact.triple), "sources": triples.origins[fact_id]}
                )
            else:
                rule = closure.facts.get(fact_id)
                if rule is not None:
                    part: dict[str, Any] = {"rule": " ".join(rule.triple)}
                    leaves = sorted(
                        {
                            premise
                            for step in closure.explain(fact_id)
                            for premise in step.premises
                            if premise in triples.origins
                        }
                    )
                    if leaves:
                        part["derived_from"] = [
                            {
                                "fact": " ".join(triples.facts[leaf].triple),
                                "sources": triples.origins[leaf],
                            }
                            for leaf in leaves
                        ]
                    involved.append(part)
        contradictions.append(
            {
                "kind": inconsistency.kind,
                "explanation": inconsistency.explanation,
                "facts": involved,
            }
        )
    # A contradiction the codex has on its own is not this scene's; it is still shown.
    for contradiction in contradictions:
        contradiction["involves_this_scene"] = any(
            source.get("from") == "scene" for source in _sources(contradiction)
        )

    unsupported = closure.unsupported_axioms
    applied_rules = len(axioms) - len(unsupported)
    summary = (
        f"{accepted} submitted fact(s) were checked against {applied_rules} rule(s): "
        f"{len(contradictions)} contradiction(s) found"
    )
    if rejected:
        summary += f"; {len(rejected)} fact(s) were rejected and not checked"
    if unsupported:
        summary += f"; {len(unsupported)} rule(s) could not be applied"
    summary += "."

    result: dict[str, Any] = {
        "scene_id": scene_id,
        "summary": summary,
        "checked_against": (
            f"The codex as it stands once scene '{scene_id}' has ended in story time, "
            "including the changes placed at that scene."
        ),
        "contradictions": contradictions,
        "rejected_facts": rejected,
        "rules": [
            {"kind": a.kind, "subject": a.subject, **({"object": a.object} if a.object else {})}
            for a in axioms
        ],
        "rules_source": (
            "the defaults (story.yaml has no axioms)" if axioms_are_defaults else "story.yaml"
        ),
    }
    if unsupported:
        result["rules_not_applied"] = [
            {"rule": written.get(u.source, u.source), "reason": _plain_reason(u.reason)}
            for u in unsupported
        ]
    if triples.unresolved:
        result["names_without_an_entry"] = sorted(triples.unresolved)
    if not_checked:
        result["codex_values_not_checked"] = not_checked
    if codex.unreadable:
        result["unreadable_files"] = [
            {"file": u.file, "reason": u.reason} for u in codex.unreadable
        ]
    placed = [a for a in assumptions(placements) if a["scene_id"] == scene_id]
    if placed:
        result["story_time_assumptions"] = placed
    return result
