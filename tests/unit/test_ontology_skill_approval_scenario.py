"""Regression scenario: incident #22, replayed against the ontology v2 entailment engine.

THE INCIDENT (GitHub #22 — "A synthesized skill can self-declare status:active and
bypass the audit verdict entirely")

``SkillRegistry.register()`` admitted a skill the moment its manifest declared
``status: active``, and it did this *before* it ever looked at the audit verdict::

    if skill.manifest.status is SkillStatus.ACTIVE:        # <-- early return
        self._skills[skill.manifest.name] = skill
        return

    if report.is_safe and report.recommendation is AuditVerdict.APPROVE:
        ...

``status`` is a field the manifest declares about *itself*. A synthesized
``SKILL.md`` is written by the very code being judged, so nothing stopped it from
writing ``status: active`` about itself. Confirmed by execution at the time: a
package registered successfully against a ``REJECT`` verdict with
``is_safe=False`` and ``risk_score=1.0``.

The shape of the bug is general, not specific to the ``status`` field: **an
artifact was allowed to assert a value about itself on the path that decides
whether the artifact is admitted.** Renaming the field, or patching the one call
site (as ``src/uclone_x/skills/auditor.py`` now does — see
``tests/unit/test_skills.py::test_skill_registry_self_declared_active_status_cannot_bypass_verdict``
for that code-level regression), fixes *this* occurrence but proves nothing about
whether the *next* field with the same shape is safe.

WHAT THE ONTOLOGY FORM BUYS, AND WHAT IT DOES NOT (docs/ontology-architecture-v2.md,
issue #138)

Model approval as a class membership *entailed* from two independent facts —
never as a field an artifact can assert about itself::

    Approved ≡ Skill ⊓ ∃hasVerdict.(AuditVerdict ⊓ {APPROVE})
                     ⊓ ∃verdictSubject.(sha256 = skill.content_sha256)
    Approved ⊓ Rejected ⊑ ⊥
    status  —  DERIVED, never asserted by the artifact

``Approved(skill)`` follows only from a *verdict fact* — sourced from the auditor,
not the skill — whose subject hash matches the skill's own content hash. A
self-asserted ``status`` triple can be ingested as an ordinary asserted fact
(skills legitimately do carry a ``status`` field) and it changes nothing: the
derivation rule for ``Approved`` never reads that predicate, so its value —
``active``, ``pending``, or anything else — cannot move the entailment. Tests 1
and 4 below exercise the two ways the historical exploit replays in ontology
terms: self-assertion, and verdict reuse across different content.

**State the guarantee at its true strength.** What holds is *"this vocabulary,
written this way, has no path from a self-asserted field to Approved"*. What does
**not** hold is *"the ontology form makes the bug unrepresentable"* — an earlier
revision of this docstring said exactly that, and it was false. The five tests
below all hold the rule set fixed and vary the data. Nothing here constrains what
a *rule* may do, and that is where the remaining exposure lives: review of PR #159
found, and this Builder independently reproduced, a route that goes around all
five by promoting attacker data into the schema (route D6, recorded in §9 of the
design). Incident #22 is blocked by how the ``Approved`` axiom is written, not by
anything the reasoner enforces about axioms in general. The design document has
said so since it was written; this file previously did not, and this file is the
more-read artifact.

API CONTRACT (from the reasoner Builder's module, landing alongside this file):

    from uclone_x.ontology.justification import Fact, ProofStep, Inconsistency, Closure
    from uclone_x.ontology.reasoner import materialize, check_consistency

    Fact(id=..., subject=..., predicate=..., object=..., tier=..., derived=False)
    materialize(facts: Iterable[Fact], axioms: Iterable[OntologyAxiom]) -> Closure
    closure.is_entailed(subject, predicate, object) -> bool
    closure.explain(fact_id) -> tuple[ProofStep, ...]   # empty for an asserted fact
    check_consistency(closure) -> tuple[Inconsistency, ...]

Now that ``uclone_x.ontology.{justification,reasoner,rules}`` have landed, the shapes
this file leans on beyond the four calls above are confirmed as:

* ``Closure.facts`` is a ``Mapping[str, Fact]`` keyed by fact id, not a bare iterable
  — a caller locating a *derived* fact to hand to ``explain`` iterates ``.values()``.
* ``ProofStep.premises`` is ``tuple[str, ...]`` of fact ids, as assumed.
* ``Inconsistency`` names the conflicting facts on ``.facts`` (not ``.fact_ids``).
* ``closure.explain(fact_id)`` raises ``KeyError`` for a fact id absent from the
  closure entirely — an empty tuple is reserved for a fact that *is* present but
  asserted (P6: a miss must not read the same as "nothing to explain").
* Predicates are canonicalised on ingestion (``rdf:type`` folds to ``type``,
  ``rdfs:subClassOf`` to ``subClassOf``, …) — this file keeps the namespaced
  spellings below because they read clearly, and they resolve to the same facts as
  the canonical forms.

WHAT THIS FILE SURFACED, AND WHERE EACH FINDING LANDED

*The join.* None of the reasoner's schema-driven rules combines more than one
premise fact with one schema axiom, so none can express a **conjunctive body with
a join**. The ``Approved`` definition needs exactly that join — the verdict's
subject hash must equal the skill's own content hash. The restricted Horn form
(conjunctive body, single head, safe, no negation or disjunction, equality via a
shared variable) landed for this reason, and tests 2 and 3, once expected to fail,
now pass against it. ``_approved_definition_axiom`` still carries
``predicate="equivalentClass"``, which is *not* a supported axiom kind: its entire
semantics is the ``rule_expression``, and the structured triple beside it is inert.

*What the Horn form did not buy.* It was derived from what this replay needed to
**express**, not from what the reasoner needs to **guarantee**. Its restrictions
buy safety and termination — the easy guarantee. The guarantee the design actually
needs is *a rule may not grant authority from a premise the subject controls*, and
the restrictions say nothing about that. Two consequences, both open findings on
PR #159:

* **D6 — schema lifting.** A rule head over a schema predicate promotes ordinary
  attacker-controlled data into the TBox. Reproduced: one benign taxonomy rule
  plus one triple reaches ``Approved`` with **zero verdict facts in existence**.
  This is not covered by the five tests below, which vary data against a fixed
  rule set; D6 varies neither and goes around both.
* **The general mechanism does not cover the specific gap.** ``sameAs``
  substitution — the twelfth rule the design conceded was missing — needs a
  predicate *variable* (``?p(?x,?z)``); the grammar requires a literal predicate
  name and refuses it. So identity/crosswalk (E9) has no path in the
  implementation, and "write it as a Horn rule" is not the escape valve it was
  offered as.

*D6b — a disjointness axiom is inert unless something derives both sides.* Test 5
originally asserted ``Approved`` and ``Rejected`` directly, which no real pipeline
produces: nothing derived ``Rejected``, so ``disjointWith`` never fired on a
realistic input. ``_rejected_definition_axiom`` below is the mirror rule, and
tests 6 and 7 exercise the contradiction as it would actually arise — both sides
*derived*, from verdict facts. Unlike the ``sameAs`` gap this one was expressible
in the Horn form, and the fix was one axiom.
"""

from __future__ import annotations

import hashlib

from uclone_x.ontology.justification import Closure, Fact, ProofStep
from uclone_x.ontology.models import OntologyAxiom, OntologyTier
from uclone_x.ontology.reasoner import check_consistency, materialize

# ---------------------------------------------------------------------------
# Fixed vocabulary for the scenario. These are IRIs/predicates in name only —
# plain strings are all the contract requires.
# ---------------------------------------------------------------------------

SKILL_ID = "skill:evil-echo"

CONTENT_HASH_AUDITED = hashlib.sha256(b"synthesized-skill:evil-echo:v1").hexdigest()
CONTENT_HASH_AFTER_EDIT = hashlib.sha256(b"synthesized-skill:evil-echo:v2-edited").hexdigest()

RDF_TYPE = "rdf:type"
PRED_CONTENT_SHA256 = "contentSha256"
PRED_STATUS = "status"
PRED_HAS_VERDICT = "hasVerdict"
PRED_VERDICT_SUBJECT = "verdictSubject"

CLASS_APPROVED = "Approved"
CLASS_REJECTED = "Rejected"
VERDICT_APPROVE = "AuditVerdict.APPROVE"
VERDICT_REJECT = "AuditVerdict.REJECT"


def _approved_definition_axiom() -> OntologyAxiom:
    """The load-bearing rule: Approved is derived from (verdict, content-hash match).

    Mirrors ``Approved ≡ Skill ⊓ ∃hasVerdict.(AuditVerdict ⊓ {APPROVE}) ⊓
    ∃verdictSubject.(sha256 = skill.content_sha256)`` from the design. Note what is
    absent: no clause reads ``status``, or any other field a skill could assert
    about itself.
    """
    return OntologyAxiom(
        name="ApprovedRequiresMatchingApproveVerdict",
        subject_entity=CLASS_APPROVED,
        predicate="equivalentClass",
        object_value="Skill",
        rule_expression=(
            "Skill(?s) ^ hasVerdict(?s,?v) ^ rdf:type(?v,'AuditVerdict.APPROVE') "
            "^ verdictSubject(?v,?h) ^ contentSha256(?s,?h) -> rdf:type(?s,'Approved')"
        ),
        tier=OntologyTier.ASSERTED,
    )


def _rejected_definition_axiom() -> OntologyAxiom:
    """The mirror of the ``Approved`` rule, and the reason test 5 was not enough.

    ``Approved ⊓ Rejected ⊑ ⊥`` is inert unless *something derives both sides*. Before
    this axiom existed, nothing in the vocabulary ever produced ``Rejected``, so the
    disjointness check could only fire on a hand-asserted pair — which no real
    pipeline emits. A genuine REJECT verdict on the genuine content hash, plus a
    forged APPROVE verdict object on the same hash, therefore yielded
    ``Approved=True``, ``Rejected=False`` and **zero** inconsistencies: the
    contradiction was real and the reasoner was silent about it.

    Note the symmetry with ``_approved_definition_axiom``: same body shape, same
    hash join, only the verdict type differs. Neither rule reads any predicate the
    skill asserts about itself.
    """
    return OntologyAxiom(
        name="RejectedRequiresMatchingRejectVerdict",
        subject_entity=CLASS_REJECTED,
        predicate="equivalentClass",
        object_value="Skill",
        rule_expression=(
            "Skill(?s) ^ hasVerdict(?s,?v) ^ rdf:type(?v,'AuditVerdict.REJECT') "
            "^ verdictSubject(?v,?h) ^ contentSha256(?s,?h) -> rdf:type(?s,'Rejected')"
        ),
        tier=OntologyTier.ASSERTED,
    )


def _approved_rejected_disjoint_axiom() -> OntologyAxiom:
    """``Approved ⊓ Rejected ⊑ ⊥`` — the two verdict classes cannot both hold.

    ``disjointWith`` is one of the reasoner's twelve supported schema kinds, so this
    axiom is fully interpreted through its structured fields alone; no
    ``rule_expression`` is needed (or wanted — see the module docstring on why the
    other axiom below cannot get the same treatment).
    """
    return OntologyAxiom(
        name="ApprovedRejectedDisjoint",
        subject_entity=CLASS_APPROVED,
        predicate="disjointWith",
        object_value=CLASS_REJECTED,
        tier=OntologyTier.ASSERTED,
    )


def _skill_type_fact(skill_id: str) -> Fact:
    """Assert that the subject is a ``Skill``.

    The ``Approved`` rule body opens with ``Skill(?s)``, so without this the body
    cannot match at all. Requiring skill-hood is deliberate rather than incidental:
    the rule should not approve an arbitrary subject that merely happens to carry a
    verdict and a matching hash.
    """
    return Fact(
        subject=skill_id,
        predicate=RDF_TYPE,
        object="Skill",
        tier=OntologyTier.ASSERTED,
    )


def _skill_content_fact(skill_id: str, content_hash: str) -> Fact:
    return Fact(
        subject=skill_id,
        predicate=PRED_CONTENT_SHA256,
        object=content_hash,
        tier=OntologyTier.ASSERTED,
        derived=False,
    )


def _self_asserted_status_fact(skill_id: str, status: str) -> Fact:
    """The attack fact: the skill's *own* manifest declaring its own status.

    This is precisely what ``SkillManifest.status`` was in the original incident —
    a field the artifact under judgment writes about itself.
    """
    return Fact(
        subject=skill_id,
        predicate=PRED_STATUS,
        object=status,
        tier=OntologyTier.ASSERTED,
        derived=False,
    )


def _verdict_facts(verdict_id: str, verdict_type: str, subject_hash: str) -> tuple[Fact, Fact]:
    """Facts describing an audit verdict: its type, and the content hash it covers.

    These are sourced from the auditor, never from the skill being judged — the
    asymmetry that makes the derivation rule trustworthy.
    """
    return (
        Fact(
            subject=verdict_id,
            predicate=RDF_TYPE,
            object=verdict_type,
            tier=OntologyTier.ASSERTED,
            derived=False,
        ),
        Fact(
            subject=verdict_id,
            predicate=PRED_VERDICT_SUBJECT,
            object=subject_hash,
            tier=OntologyTier.ASSERTED,
            derived=False,
        ),
    )


def _has_verdict_fact(skill_id: str, verdict_id: str) -> Fact:
    return Fact(
        subject=skill_id,
        predicate=PRED_HAS_VERDICT,
        object=verdict_id,
        tier=OntologyTier.ASSERTED,
        derived=False,
    )


def _approved_fact_id(closure: Closure, skill_id: str) -> str | None:
    """Locate the id of the (derived) ``rdf:type Approved`` fact for a skill.

    Uses ``Closure.get`` rather than scanning ``closure.facts``: predicates are
    canonicalised on the way in, so ``rdf:type`` is stored as ``type`` and a raw
    string comparison against the spelling used here would never match.
    """
    fact = closure.get(skill_id, RDF_TYPE, CLASS_APPROVED)
    return fact.id if fact is not None else None


def _reachable_premise_ids(closure: Closure, fact_id: str) -> set[str]:
    """Walk ``explain`` recursively and collect every premise fact id touched.

    ``ProofStep.premises`` is ``tuple[str, ...]`` of fact ids.
    """
    seen: set[str] = set()
    stack: list[str] = [fact_id]
    while stack:
        current = stack.pop()
        steps: tuple[ProofStep, ...] = closure.explain(current)
        for step in steps:
            for premise_id in step.premises:
                if premise_id not in seen:
                    seen.add(premise_id)
                    stack.append(premise_id)
    return seen


# ---------------------------------------------------------------------------
# 1. The historical attack fails: self-asserted status cannot buy approval.
# ---------------------------------------------------------------------------


def test_self_asserted_active_status_does_not_entail_approval_against_reject_verdict() -> None:
    """Replay of #22: a synthesized skill asserts status:active about itself while
    the only verdict on its content is REJECT. Approved(skill) must not be entailed
    — and asserting the self-declared status must not change that either way.
    """
    verdict_id = "verdict:v-reject"
    type_fact = _skill_type_fact(SKILL_ID)
    content_fact = _skill_content_fact(SKILL_ID, CONTENT_HASH_AUDITED)
    verdict_type_fact, verdict_subject_fact = _verdict_facts(
        verdict_id, VERDICT_REJECT, CONTENT_HASH_AUDITED
    )
    has_verdict_fact = _has_verdict_fact(SKILL_ID, verdict_id)
    axioms = (_approved_definition_axiom(), _approved_rejected_disjoint_axiom())

    # Without the self-asserted attack fact at all.
    baseline_closure = materialize(
        (type_fact, content_fact, verdict_type_fact, verdict_subject_fact, has_verdict_fact),
        axioms,
    )
    assert not baseline_closure.is_entailed(SKILL_ID, RDF_TYPE, CLASS_APPROVED)

    # With the synthesized manifest self-asserting status:active — the historical
    # exploit's exact move. The derivation rule for Approved never reads `status`,
    # so admitting this fact into the closure must not change the outcome.
    attack_status_fact = _self_asserted_status_fact(SKILL_ID, "active")
    attack_closure = materialize(
        (
            type_fact,
            content_fact,
            verdict_type_fact,
            verdict_subject_fact,
            has_verdict_fact,
            attack_status_fact,
        ),
        axioms,
    )
    assert not attack_closure.is_entailed(SKILL_ID, RDF_TYPE, CLASS_APPROVED)

    # The self-asserted fact is present in the closure (it was legitimately
    # ingested as an ordinary asserted fact) — it simply has no path to Approved.
    assert attack_status_fact.id in attack_closure.facts


# ---------------------------------------------------------------------------
# 2. A genuine approval is entailed.
# ---------------------------------------------------------------------------


def test_genuine_approve_verdict_on_matching_content_entails_approval() -> None:
    """Same skill; an APPROVE verdict whose subject is the skill's own content
    hash. Approved(skill) is entailed.
    """
    verdict_id = "verdict:v-approve"
    type_fact = _skill_type_fact(SKILL_ID)
    content_fact = _skill_content_fact(SKILL_ID, CONTENT_HASH_AUDITED)
    verdict_type_fact, verdict_subject_fact = _verdict_facts(
        verdict_id, VERDICT_APPROVE, CONTENT_HASH_AUDITED
    )
    has_verdict_fact = _has_verdict_fact(SKILL_ID, verdict_id)
    axioms = (_approved_definition_axiom(), _approved_rejected_disjoint_axiom())

    closure = materialize(
        (type_fact, content_fact, verdict_type_fact, verdict_subject_fact, has_verdict_fact),
        axioms,
    )

    assert closure.is_entailed(SKILL_ID, RDF_TYPE, CLASS_APPROVED)


# ---------------------------------------------------------------------------
# 3. explain() names the reason — a derivation, not a narration (E1).
# ---------------------------------------------------------------------------


def test_explain_reaches_the_verdict_fact() -> None:
    """For the genuine-approval scenario, explain() must return a chain that
    bottoms out at the verdict facts, not an opaque or empty answer.
    """
    verdict_id = "verdict:v-approve"
    type_fact = _skill_type_fact(SKILL_ID)
    content_fact = _skill_content_fact(SKILL_ID, CONTENT_HASH_AUDITED)
    verdict_type_fact, verdict_subject_fact = _verdict_facts(
        verdict_id, VERDICT_APPROVE, CONTENT_HASH_AUDITED
    )
    has_verdict_fact = _has_verdict_fact(SKILL_ID, verdict_id)
    axioms = (_approved_definition_axiom(), _approved_rejected_disjoint_axiom())

    closure = materialize(
        (type_fact, content_fact, verdict_type_fact, verdict_subject_fact, has_verdict_fact),
        axioms,
    )

    approved_fact_id = _approved_fact_id(closure, SKILL_ID)
    assert approved_fact_id is not None, "expected a derived rdf:type Approved fact in the closure"

    steps = closure.explain(approved_fact_id)
    assert len(steps) > 0, "a derived fact must carry at least one proof step"

    reached = _reachable_premise_ids(closure, approved_fact_id)
    assert has_verdict_fact.id in reached
    assert verdict_type_fact.id in reached
    assert verdict_subject_fact.id in reached
    assert content_fact.id in reached

    # An asserted fact carries no derivation of its own — the chain bottoms out.
    assert closure.explain(content_fact.id) == ()
    assert closure.explain(verdict_type_fact.id) == ()


# ---------------------------------------------------------------------------
# 4. Content binding holds: an audit cannot be inherited by different code.
# ---------------------------------------------------------------------------


def test_approve_verdict_on_different_content_does_not_entail_approval() -> None:
    """An APPROVE verdict whose subject is a *different* content hash must not
    entail approval for this skill's current content — e.g. the skill was edited
    after the audit, or a stale approval is being replayed against new code.
    """
    verdict_id = "verdict:v-approve-stale"
    type_fact = _skill_type_fact(SKILL_ID)
    content_fact = _skill_content_fact(SKILL_ID, CONTENT_HASH_AFTER_EDIT)
    # The verdict approves CONTENT_HASH_AUDITED, not the skill's current content.
    verdict_type_fact, verdict_subject_fact = _verdict_facts(
        verdict_id, VERDICT_APPROVE, CONTENT_HASH_AUDITED
    )
    has_verdict_fact = _has_verdict_fact(SKILL_ID, verdict_id)
    axioms = (_approved_definition_axiom(), _approved_rejected_disjoint_axiom())

    closure = materialize(
        (type_fact, content_fact, verdict_type_fact, verdict_subject_fact, has_verdict_fact),
        axioms,
    )

    assert not closure.is_entailed(SKILL_ID, RDF_TYPE, CLASS_APPROVED)


# ---------------------------------------------------------------------------
# 5. Disjointness fires: Approved and Rejected cannot both hold for one skill.
# ---------------------------------------------------------------------------


def test_approved_and_rejected_are_disjoint() -> None:
    """Asserting both Approved and Rejected for the same skill is a contradiction
    the reasoner must surface, naming both facts.

    ``Inconsistency.kind`` is ``str`` and ``Inconsistency.facts`` is
    ``tuple[str, ...]`` naming the conflicting facts.
    """
    approved_fact = Fact(
        subject=SKILL_ID,
        predicate=RDF_TYPE,
        object=CLASS_APPROVED,
        tier=OntologyTier.ASSERTED,
        derived=False,
    )
    rejected_fact = Fact(
        subject=SKILL_ID,
        predicate=RDF_TYPE,
        object=CLASS_REJECTED,
        tier=OntologyTier.ASSERTED,
        derived=False,
    )
    axioms = (_approved_rejected_disjoint_axiom(),)

    closure = materialize((approved_fact, rejected_fact), axioms)
    inconsistencies = check_consistency(closure)

    assert len(inconsistencies) >= 1
    disjoint_inconsistencies = [inc for inc in inconsistencies if inc.kind == "disjoint"]
    assert len(disjoint_inconsistencies) >= 1
    named_fact_ids = {fact_id for inc in disjoint_inconsistencies for fact_id in inc.facts}
    assert approved_fact.id in named_fact_ids
    assert rejected_fact.id in named_fact_ids


# ---------------------------------------------------------------------------
# 6. The mirror rule derives Rejected from a genuine REJECT verdict.
# ---------------------------------------------------------------------------


def test_genuine_reject_verdict_on_matching_content_entails_rejection() -> None:
    """A REJECT verdict on the skill's own content hash entails ``Rejected``.

    The precondition for test 7, and a gap in its own right: until
    ``_rejected_definition_axiom`` existed, ``Rejected`` was a class no rule could
    ever produce, which made the disjointness axiom decorative.
    """
    verdict_id = "verdict:audit-2026-09-02"
    verdict_type_fact, verdict_subject_fact = _verdict_facts(
        verdict_id, VERDICT_REJECT, CONTENT_HASH_AUDITED
    )
    facts = (
        _skill_type_fact(SKILL_ID),
        _skill_content_fact(SKILL_ID, CONTENT_HASH_AUDITED),
        _has_verdict_fact(SKILL_ID, verdict_id),
        verdict_type_fact,
        verdict_subject_fact,
    )
    axioms = (_approved_definition_axiom(), _rejected_definition_axiom())

    closure = materialize(facts, axioms)

    assert closure.is_entailed(SKILL_ID, RDF_TYPE, CLASS_REJECTED)
    assert not closure.is_entailed(SKILL_ID, RDF_TYPE, CLASS_APPROVED)

    # Derived, not asserted: nothing in `facts` mentions the class Rejected.
    rejected = closure.get(SKILL_ID, RDF_TYPE, CLASS_REJECTED)
    assert rejected is not None
    assert rejected.derived is True
    assert closure.explain(rejected.id), "a derived fact must carry a proof (P6)"

    # A consistent world: one verdict, one class, no contradiction to report.
    assert check_consistency(closure) == ()


# ---------------------------------------------------------------------------
# 7. D6b: the contradiction as a real pipeline produces it — both sides derived.
# ---------------------------------------------------------------------------


def test_forged_approve_verdict_alongside_genuine_reject_is_caught_as_contradiction() -> None:
    """An injected APPROVE verdict object does not silently win — it contradicts.

    This is the case test 5 was standing in for, and could not actually reach. An
    attacker who can inject a verdict *object* (rather than a self-asserted field
    on the skill) satisfies the ``Approved`` body exactly: the forged verdict names
    the genuine content hash, so the hash join holds. The ``Approved`` rule alone
    therefore concludes approval, and before the mirror rule existed the closure
    reported **zero** inconsistencies while a genuine REJECT sat in the same graph.

    With both rules present the two derivations collide and ``disjointWith`` fires,
    naming both *derived* facts. Note what this does and does not buy: the forged
    verdict is not rejected — it is *contradicted*. The reasoner's job here is to
    make the conflict impossible to miss, not to adjudicate it.
    """
    genuine_id = "verdict:audit-2026-09-02"
    forged_id = "verdict:forged-by-skill"
    genuine_type, genuine_subject = _verdict_facts(genuine_id, VERDICT_REJECT, CONTENT_HASH_AUDITED)
    forged_type, forged_subject = _verdict_facts(forged_id, VERDICT_APPROVE, CONTENT_HASH_AUDITED)
    facts = (
        _skill_type_fact(SKILL_ID),
        _skill_content_fact(SKILL_ID, CONTENT_HASH_AUDITED),
        _has_verdict_fact(SKILL_ID, genuine_id),
        _has_verdict_fact(SKILL_ID, forged_id),
        genuine_type,
        genuine_subject,
        forged_type,
        forged_subject,
    )
    axioms = (
        _approved_definition_axiom(),
        _rejected_definition_axiom(),
        _approved_rejected_disjoint_axiom(),
    )

    closure = materialize(facts, axioms)

    approved = closure.get(SKILL_ID, RDF_TYPE, CLASS_APPROVED)
    rejected = closure.get(SKILL_ID, RDF_TYPE, CLASS_REJECTED)
    assert approved is not None and approved.derived is True
    assert rejected is not None and rejected.derived is True

    inconsistencies = check_consistency(closure)
    disjoint = [inc for inc in inconsistencies if inc.kind == "disjoint"]
    assert len(disjoint) >= 1, (
        "a forged APPROVE beside a genuine REJECT must be reported, not silently approved"
    )
    named = {fact_id for inc in disjoint for fact_id in inc.facts}
    assert approved.id in named
    assert rejected.id in named

    # The proof names the forged verdict, so an operator can see which object to
    # revoke. This is the audit surface the whole design rests on.
    premises = _reachable_premise_ids(closure, approved.id)
    assert forged_type.id in premises
