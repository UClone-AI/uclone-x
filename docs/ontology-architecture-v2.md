# Ontology v2 — Entailment, Justification and Consistency

> [!IMPORTANT]
> **Implementation status (2026-09-02).** This document is a **design**, in the sense
> [`a2a-protocol-spec.md`](a2a-protocol-spec.md#0-how-to-read-this-document--specified-conformant-implemented)
> fixes the word: it states what this repository intends to build. Of everything
> specified below, **only Phase 2 of [§7](#7-phasing-each-phase-delivers-one-working-experience)
> is being implemented** under issue
> [#138](https://github.com/UClone-AI/uclone-x/issues/138) — the thirteen-capability
> justification-recording entailment engine and `explain`. Every other phase, every
> other store, and every CLI command named here is **Planned** and has no code.
> [§8](#8-implementation-status) is the authoritative per-item table; where §8 and
> prose disagree, §8 wins.
>
> This document **extends** [`agent-ontology-architecture.md`](agent-ontology-architecture.md)
> and [`ontology-alignment-spec.md`](ontology-alignment-spec.md). It does not replace
> either, and it preserves their three-tier model
> (`asserted` / `induced-enforcing` / `induced-candidate`) unchanged. It makes exactly
> one deliberate change to those documents' commitments — what the tiers *govern* —
> stated in [§5.4](#54-what-changes-about-what-the-tiers-govern) rather than
> introduced silently.

---

## 0. How to read this document

### 0.1 Status vocabulary

Borrowed verbatim from [`a2a-protocol-spec.md` §0](a2a-protocol-spec.md#0-how-to-read-this-document--specified-conformant-implemented),
so that the two documents cannot drift into using the same words for different things:

| Term | Meaning here |
| :--- | :--- |
| **Specified** | Written down in this repository. Costs nothing and proves nothing. |
| **Implemented** | Code exists in `src/uclone_x/`, is exercised by tests, and passes `./ucx test check`. |
| **Verified by execution** | A claim in this document that was checked by running the shipped code and recording the output. [§1.2](#12-three-demonstrations-verified-by-execution) is the only place this label is used, and it names what was run. |
| **Unverified** | A claim this document could not confirm against a primary source. Collected in [§9.4](#94-unverified-claims). An unverified claim is left labelled rather than silently asserted. |
| **Derived requirement** | A structural commitment that [§4](#4-derivation-experiences--capabilities--structure) traces back to a specific numbered experience. Anything in [§4.4](#44-five-stores)–[§4.5](#45-four-operations) that carries no such trace is a design error, not a feature. |

### 0.2 What this document is arguing

The argument has one shape, and it is a methodological claim as much as a technical
one: **the architecture below is derived from ten user experiences, not from
description-logic theory.** [§3](#3-ten-end-to-end-experiences) is therefore the
requirements section, and everything after it is consequence. Three of the structural
commitments — storing *justifications* alongside derived facts, *retaining
counterexamples* for refuted hypotheses, and typing every predicate by whether the
artifact supplies it about *itself* — exist only because specific experiences were
taken seriously. A design that started from "what does a reasoner do?" would have
shipped none of them. That is the point;
[§4.6](#46-what-the-derivation-shows) states it as a falsifiable claim rather than a
slogan, and [§4.7](#47-a-capability-the-experiences-did-not-surface-predicate-provenance)
is its strongest instance.

### 0.3 Relationship to the two sibling documents

| Document | Owns | This document's relation |
| :--- | :--- | :--- |
| [`agent-ontology-architecture.md`](agent-ontology-architecture.md) | The two pillars (self-construction, human steering), the Tier 1 / Tier 2 performance split | Extended. The Tier 1 fast validator becomes one of four operations (`validate`); Tier 2 background work gains `entail` and `diff_closure`. |
| [`ontology-alignment-spec.md`](ontology-alignment-spec.md) | Three tiers, evidence records, crosswalks, arbitration, the `content_hash` determinism split | Preserved. Its §2 matching procedure, §3 arbitration, §4.3 contradiction handling and §5 determinism rules are adopted unchanged and are **not** reopened here. [§5.3](#53-a-falsifiable-promotion-criterion) proposes one amendment to its §4.2 promotion criterion, marked as such. |
| [`docs/principles/details/p7-evolving-ontology.md`](principles/details/p7-evolving-ontology.md) | The law | **Not touched.** [§1.4](#14-the-consequence-p7s-stated-purpose-is-unachievable-by-the-current-design) argues the current implementation cannot deliver P7's stated purpose. That is a claim about the implementation, not a proposal to amend the principle. |

---

## 1. The defect, stated precisely

### 1.1 What is actually there

`src/uclone_x/ontology/` is over 1,400 lines of working, tested code (`engine.py`,
`models.py`, `protocols.py` at commit `f590f65`). It is not a stub, and
the "type stubs only" banners at the top of both sibling documents are now stale
([§9.3](#93-contradictions-found-in-the-existing-specifications-and-code)). What it is,
precisely, is a **record-schema validator with tier metadata**:

* **`OntologyConcept` is a struct definition.** `name`, `parent_type`, `attributes:
  Mapping[str, str]`, `required_fields: tuple[str, ...]`
  (`src/uclone_x/ontology/models.py`, `class OntologyConcept`). Nothing about it says
  what a concept *means*; it says what fields a dictionary is allowed to carry.
* **`OntologyRelation` is a labelled edge type.** `source_entity`, `predicate`,
  `target_entity`, `is_directed`. There is no rule anywhere that consumes a relation
  to conclude anything.
* **`OntologyAxiom.rule_expression` is a `str`** (`models.py`, `rule_expression: str =
  ""`). The repository's own test suite writes `rule_expression="balance >= 0"`
  (`tests/unit/test_ontology_session_induction.py`). Nothing parses it. Its only
  functional consumer in the entire codebase is
  `SessionKnowledgeExtractor.inject_rules_to_prompt` in
  `src/uclone_x/ontology/engine.py`, which appends it to an LLM system prompt as
  free text. **An axiom's rule is enforced, today, by asking the model nicely.**
* **`parent_type` is the giveaway.** It is a `str | None` that is written into LinkML
  `is_a` on export, compared for equality during induction conflict detection, and
  scanned during `forget()` to list dependents. It is never used to *derive* anything.
  We have the notation of subsumption with none of its consequence: `parent_type`
  is a string that looks like `subClassOf` and behaves like a comment.

### 1.2 Three demonstrations (verified by execution)

Run against the code at commit `f590f65`, on the shipped `OntologyEngine`, with no
mocks. These are not hypotheticals about a future refactor; they are the current
behaviour.

**(a) Subsumption has no consequence.** A subclass does not inherit its parent's
required fields, because `validate_entity` iterates `concept.required_fields` and
`concept.attributes` for the *named* concept only and never walks `parent_type`:

```python
e.teach_concept("Task", attributes={"id": "str"}, required_fields=("id",))
e.teach_concept(
    "ReviewTask", parent_type="Task", attributes={"reviewer": "str"}, required_fields=("reviewer",)
)

e.validate_entity("ReviewTask", {"reviewer": "r1"})
# -> is_valid=True, errors=()
```

A `ReviewTask` with no `id` is valid. Every `ReviewTask` is a `Task`; no `Task` may
lack an `id`; this `ReviewTask` lacks an `id`; the validator says yes. The premises
are all present in the store and the conclusion is never drawn.

**(b) The axiom rule is inert.**

```python
e.teach_concept("Account", attributes={"balance": "int"})
e.teach_axiom("nonneg", "Account", rule_expression="balance >= 0")

e.validate_entity("Account", {"balance": -100})
# -> is_valid=True, errors=()
```

`validate_entity` checks axioms only through the `predicate == object_value` equality
path. An axiom whose content lives in `rule_expression` contributes nothing to
validation. It is stored, hashed into `content_hash`, exported, and pasted into a
prompt.

**(c) Natural-language teaching has a silent fallback.** `teach_directive` matches
three regexes and, failing all three, manufactures a concept named from the first 40
characters of the input. Fed the exact example
[`agent-ontology-architecture.md` §3.2](agent-ontology-architecture.md#32-human-guided-steering-developer-ingestion)
advertises for `./ucx ontology teach`:

```python
e.teach_directive("all internal gRPC calls must use mTLS encryption")
# -> OntologyConcept(name='all_internal_gRPC_calls_must_use_mTLS_en')
```

The user taught a security constraint and received a concept named after a sentence
fragment. Nothing failed, nothing warned. This is the substitution shape
[P6](principles/details/p6-fail-fast-observability.md) forbids unconditionally,
sitting in the primary human-steering entry point of the subsystem P7 governs. It is
also, concretely, why [E2](#e2--teach-a-constraint-in-natural-language-and-see-what-it-would-have-blocked)
is a real experience and not a nicety.

### 1.3 The dividing test

There is one question that separates an ontology from a schema:

> **Can it tell you something you did not put in?**

Stated that loosely, a schema validator scrapes past — it does tell you something you
did not type, namely that a payload is malformed. So state it in the form that
actually divides:

> **Can it produce a new assertion — a fact about the domain, of the same kind as the
> facts a human asserts — that no human wrote and no tool observed?**

A schema validator cannot, by construction. Its entire output is a verdict about an
input. An ontology can: given `subClassOf(ReviewTask, Task)` and `type(t, ReviewTask)`
it produces `type(t, Task)`, which nobody wrote down, and which is now available to
every subsequent query, constraint and refusal. Everything the current design can say
is something someone typed.

### 1.4 The consequence: P7's stated purpose is unachievable by the current design

[P7](principles/details/p7-evolving-ontology.md) requires that "Decisions and A2A
collaboration must be validated against the agent's active ontology **to prevent
semantic drift**." That is the stated purpose, and it is the one thing the current
design cannot deliver — not because it is incomplete, but because of what it is.

Schema validation catches **malformed messages**. Semantic drift is **two well-formed
messages meaning different things**. When agent A sends `{"status": "completed"}`
meaning *the artifact was produced and reviewed*, and agent B sends
`{"status": "completed"}` meaning *my turn ended without an exception*, a schema
passes both — necessarily, because both satisfy the same field constraints. Every
schema always will. Drift lives in the gap between the field and its meaning, and a
validator that only sees fields cannot see the gap.

What closes it is not a stricter schema. It is a definition with consequences: if
`Completed` is a class whose membership is *derived* — from a produced artifact, a
recorded review, a terminal state that is not `CANCELED` — then B's message does not
mean `Completed` unless those facts are present, and asking *why* the system believes
`Completed(t)` returns either a derivation or nothing. That is the capability P7 needs
and this document specifies. This is a claim about the implementation, not a
challenge to the principle text: P7 as written is achievable; the record-schema
validator is not a way to achieve it.

---

## 2. What an ontology adds over a schema

Three things, in the order they become load-bearing.

### 2.1 Identity — what makes two descriptions the same thing

A schema tells you a payload is well-formed. It cannot tell you that the
`Task` in agent A's store and the `Task` in agent B's store are the *same task*.
UClone-X hits this immediately: [`ontology-alignment-spec.md` §1.2](ontology-alignment-spec.md)
has two agents exchanging `OntologyFragment`s on a shared `contextId`, and A2A gives
each side its own `Task.id` namespace. Two agents both reporting on `Task#7` are, as
far as the current code is concerned, reporting on two unrelated records.

The ontology answer is a *key*, declared and reasoned over. Take
`SkillManifest.content_sha256` (`src/uclone_x/skills/`): declare `content_sha256` an
**inverse-functional** property — no two distinct skills share one — and the reasoner
concludes `sameAs(skill:pr-summarizer@agentA, skill:summarize-pr@agentB)` from the
digests alone, without a human writing a crosswalk, and records that it did so and
why. Declare `approval_of` **functional** — a skill version has at most one approval
record — and two conflicting approvals for one digest become a detectable
contradiction instead of two rows. Neither conclusion is expressible in a field
constraint, because both are statements about *which descriptions co-refer*, and a
field constraint only sees one description at a time.

**Correction.** This section states what identity-by-key should deliver; it is not a
description of what shipped. As built, inverse-functional (and functional) violations
are reported as **contradictions**, not resolved into `sameAs` — the reasoner never
concludes the `sameAs` fact this paragraph describes, and no rule in the shipped code
substitutes through one if it existed. See [§6.3](#63-the-thirteen-capabilities) row 12
and [§6.3.1](#631-the-thirteenth-capability-cannot-supply-the-twelfth).

### 2.2 Entailment — what follows

A schema is closed under nothing. An ontology is closed under its rules: a fact
asserted once is available everywhere its consequences reach.

The UClone-X case is [§1.2(a)](#12-three-demonstrations-verified-by-execution) exactly.
`teach_concept("ReviewTask", parent_type="Task")` is a human writing down a
subsumption. `subClassOf(ReviewTask, Task) ∧ type(t, ReviewTask) ⊨ type(t, Task)` is
the consequence, and the consequence is what makes the human's statement worth making.
Without it, every invariant on `Task` must be re-typed on `ReviewTask`, on
`AuditTask`, on every future child — and the moment someone forgets, the hierarchy is
decorative and the drift is silent. With it, one asserted edge propagates every
`Task` constraint to every descendant, and `explain` can say which edge did it.

The same rule is what makes an A2A skill contract enforceable across a hierarchy: a
skill declaring it accepts `Task` accepts a `ReviewTask` because the reasoner says so,
not because someone remembered to widen the type.

### 2.3 Consistency — what cannot hold together

A schema validates values independently. It has no vocabulary for *combinations that
cannot both be true*, so the errors it cannot catch are exactly the interesting ones.

`TaskStatus.COMPLETED` and `TaskStatus.CANCELED` (`src/uclone_x/a2a/models.py`) are
each individually valid; both are terminal states in
[A2A §5.1](a2a-protocol-spec.md#51-task-lifecycle--the-real-state-names). A task
history that records both, or a remote agent's result asserting `COMPLETED` over a
`contextId` whose prior artifact asserted `CANCELED`, is not a malformed message — it
is a contradiction. `disjointWith(Completed, Canceled)` is one asserted axiom, and it
converts an undetectable disagreement into a typed error naming both facts and the
axiom that made them incompatible.

The same shape covers the interesting case of incident
[#22](https://github.com/UClone-AI/uclone-x/issues/22): `Approved` and `Rejected` are
disjoint, and `approval_of` is functional, so a skill manifest that asserts its own
approval while an audit report asserts rejection produces an inconsistency at
*write* time rather than a silent admission at registration time
([E8](#e8--a-past-incident-made-representationally-impossible)).

---

## 3. Ten end-to-end experiences

These are the requirements. Everything in [§4](#4-derivation-experiences--capabilities--structure)
onward is derived from them, and anything in this architecture that no experience
below demands should be cut. They are numbered as in issue
[#138](https://github.com/UClone-AI/uclone-x/issues/138).

Each is written as: **Trigger** (what happened) · **Does** (what the user does) ·
**Sees** (what comes back) · **Requires** (the capability, in one line).

### E1 — "Why did you do that?"

* **Trigger.** The agent refused to register a synthesized skill, and the developer
  does not believe the refusal.
* **Does.** `./ucx ontology explain 'Approved(skill:pr-summarizer@a1b2c3)'`
* **Sees.** Either a derivation tree — the rule that fired, its premises, each premise
  expanded in turn, down to leaves that are all human-asserted axioms or recorded
  observations — or the sentence *"not entailed; no rule derives it from the current
  assertions."* Not a paragraph of LLM narration about what the agent was probably
  thinking. A chain the developer can walk and disagree with at a specific step.
* **Requires.** Forward entailment whose every derived fact stores the rule id and the
  premise set that produced it.

### E2 — Teach a constraint in natural language, and see what it would have blocked

* **Trigger.** A post-mortem concluded "deployments should never go out without a
  human-signed approval," and the developer wants that written into the agent, now.
* **Does.** `./ucx ontology teach "every Deployment must reference an Approval whose decided_by is a human"`
* **Sees.** First, the parse read back in the ontology's own terms — a named shape
  over `Deployment`, its target class, its cardinality — so the developer can confirm
  the system understood, or get an explicit parse failure. **Never** a concept named
  `every_Deployment_must_reference_an_Appr`
  ([§1.2(c)](#12-three-demonstrations-verified-by-execution)). Then the retroactive
  report: *"Applied to the 412 assertions on record, this constraint would have
  blocked 3 — here they are, with dates and sessions."* One of them is the
  post-mortem's incident, which is how the developer knows the rule they wrote is the
  rule they meant.
* **Requires.** Constraint validation over a *stored* assertion history, not only over
  the next incoming payload.

### E3 — Reject an A2A message with the specific contradiction

* **Trigger.** A remote agent returns a task result asserting `status: completed` for a
  `contextId` whose earlier artifact asserted `canceled`.
* **Does.** Nothing. The developer is watching a task run.
* **Sees.** The task transitions to `TASK_STATE_REJECTED` with a typed
  `OntologyInconsistencyError` that names *both* conflicting facts, the axiom
  (`disjointWith(Completed, Canceled)`), where each fact came from (which message,
  which agent, which timestamp), and — if either fact was derived rather than asserted
  — its derivation. Not "validation failed on field `status`."
* **Requires.** Typed consistency checking that reports the conflicting fact set, not
  a boolean.

### E4 — "What did the agent learn this week?"

* **Trigger.** Friday. The developer wants to know what their agent has been inducing
  while unattended.
* **Does.** `./ucx ontology learned --since 7d`
* **Sees.** Three lists, and the third is the reason the command is worth running.
  *Promoted* (2): now enforcing, with the evidence and the human who approved.
  *Pending* (9): candidates awaiting review, with observation counts and sessions.
  ***Rejected* (14)**: hypotheses the agent formed and then killed — and, for each, the
  **counterexample that killed it**, quoted, with its session and turn. *"Hypothesised:
  every `ToolResult` with `exit_code=0` implies `Succeeded`. Refuted 2026-08-29 by
  session `s_4471` turn 12: `exit_code=0` with `stderr` non-empty and a downstream
  failure."* This is the most informative screen an operator ever sees, and almost
  every system throws its contents away.
* **Requires.** Hypotheses tested against the closure, with refutations retained rather
  than discarded, over a queryable time window.

### E5 — Impact analysis before changing a term

* **Trigger.** The developer wants to reparent `ReviewTask` from `Task` to a new
  `AuditTask`, and has no idea what depends on it.
* **Does.** `./ucx ontology impact ReviewTask --set parent=AuditTask`
* **Sees.** A three-part diff computed without committing anything: derived facts that
  would **disappear** (147, with the top-level ones named), derived facts that would
  **appear** (38), and — the part that matters — validations that currently pass and
  would begin to **fail** (2, each with the assertion and the constraint). Each entry
  is expandable into the same derivation tree E1 returns, so "why would this
  disappear?" has an answer.
* **Requires.** Closure diffing between a current and a hypothetical TBox, with
  per-fact justifications to explain each delta.

### E6 — Code questions answered by symbol traversal, with a freshness state

* **Trigger.** The agent is about to change `OntologyEngine.promote` and needs to know
  what calls it — accurately, not by grep.
* **Does.** `./ucx code callers uclone_x.ontology.engine.OntologyEngine.promote`
* **Sees.** The call sites from the SCIP symbol graph, *transitively* where asked
  (`--depth 3`), each with a path and range — and, in-band on the result, the index
  state: `freshness: STALE — index built 4h ago, 12 files changed since`. Never an
  empty list that reads as "nothing calls this."
* **Requires.** Symbol-graph ingestion whose relations are traversed by the same
  transitive-closure machinery, carrying an explicit freshness state.

### E7 — A browsable domain map, coloured by tier

* **Trigger.** A new developer joins and asks what the agent actually knows.
* **Does.** `./ucx ontology map --serve`
* **Sees.** A navigable graph of the agent's concepts and relations, coloured by tier —
  `asserted` solid, `induced-enforcing` outlined, `induced-candidate` dashed — with
  asserted edges visually distinct from derived ones, so it is obvious at a glance
  which parts of the map a human wrote and which the machine concluded. Clicking any
  derived edge shows its derivation.
* **Requires.** A canonical, tier-labelled TBox that can be serialized and rendered,
  with derived edges distinguishable from asserted ones.

### E8 — A past incident made representationally impossible

* **Trigger.** Incident [#22](https://github.com/UClone-AI/uclone-x/issues/22): a
  synthesized skill whose manifest declared its own `status: active` registered
  successfully against a `REJECT` audit verdict with `risk_score=1.0`.
* **Does.** The developer replays the incident: synthesizes a skill whose manifest
  asserts `status: active`, with a rejecting audit report, and attempts registration.
* **Sees.** Registration refused, with the reason: `Approved(skill:X)` is **not
  entailed**. `Approved` is a *derived* class — membership follows from an approving
  audit verdict bound to *this* content digest, and from nothing else — so the
  manifest's self-assertion is not a premise of any rule and derives nothing.
  `./ucx ontology explain 'Approved(skill:X)'` returns the empty derivation, naming
  the missing premise.
* **Requires.** Derived-class membership plus justification: the ability to say
  "not entailed, and here is the premise you are missing."

> **Scope of the claim, stated exactly.** What phase 2 + 3 deliver is that *this*
> incident is blocked, because of how the `Approved` axiom is written: it joins only on
> auditor-sourced facts (`hasVerdict`, the verdict type, `verdictSubject`) plus the
> skill's content digest as a key. It does **not** follow that no axiom expressible in
> this system could recreate the bug. Nothing stops a future axiom author from writing
> a rule that joins on `status` — a predicate the artifact supplies about itself —
> instead of on `hasVerdict`. The honest claim is therefore: **the vocabulary does not
> force self-assertion in, and the one axiom that matters here does not use it.** That
> is a review discipline for axiom authors, not a guarantee. [§4.7](#47-a-capability-the-experiences-did-not-surface-predicate-provenance)
> closes the class rather than the instance.

### E9 — Crosswalk review when connecting an external agent

* **Trigger.** The developer connects their agent to a partner team's agent for the
  first time.
* **Does.** `./ucx ontology align --peer https://partner.example/.well-known/agent-card.json`
* **Sees.** A side-by-side review table: terms matched exactly by IRI (14, no action),
  terms already crosswalked (3), and terms with only a *proposed* mapping (6) — each
  showing both definitions, the proposed SKOS relation, the confidence, and what the
  score was computed from. Nothing is applied. The developer accepts, rejects or
  defers each one, and every acceptance mints a `CrosswalkAssertion` with
  `declared_by: "human"`, exactly as
  [`ontology-alignment-spec.md` §2.2–§2.4](ontology-alignment-spec.md) already
  requires. Any pair whose definitions cannot both hold is flagged as a conflict for
  arbitration rather than shown as a near-match.
* **Requires.** A crosswalk store over SKOS mapping relations, identity resolution,
  and consistency checking across the two term sets.

### E10 — "What did the agent believe at time T?"

* **Trigger.** A task failed on 2026-08-30 and the developer is reconstructing why. The
  ontology has changed twice since.
* **Does.** `./ucx ontology as-of 2026-08-30T12:00:00Z --query 'Approved(skill:pr-summarizer)'`
* **Sees.** The answer the agent *would have given then* — `Approved` held, derived
  from an audit record that has since been retracted — together with the derivation as
  it stood at that moment, and a note that the answer today is different and why. Not
  today's closure with old timestamps pasted on it.
* **Requires.** Bi-temporal assertions (valid time and transaction time), so the
  closure can be recomputed as of any past instant.

---

## 4. Derivation: experiences → capabilities → structure

### 4.1 The capabilities the ten experiences name

Reading only the **Requires** lines of [§3](#3-ten-end-to-end-experiences), with no
reference to what reasoners conventionally provide:

| Id | Capability |
| :--- | :--- |
| **K1** | Canonical, tier-labelled TBox — concepts, relations and axioms as data, serializable and renderable |
| **K2** | Forward entailment — derive facts that follow from asserted axioms and assertions |
| **K3** | Justification recording — every derived fact stores the rule and the premise set that produced it |
| **K4** | Typed consistency detection — report the conflicting fact *set* and the axiom violated, not a boolean |
| **K5** | Constraint validation — shape/cardinality checks over instance data |
| **K6** | Identity resolution — conclude that two descriptions co-refer, from declared keys |
| **K7** | Closure diff — what changes between two TBoxes or two instants |
| **K8** | Hypothesis testing with counterexample retention — candidates tested, refutations kept |
| **K9** | Bi-temporal assertions — valid time and transaction time on every assertion |
| **K10** | Crosswalk mapping — declared equivalences to terms outside this agent's namespace |
| **K11** | Symbol-graph ingestion with an explicit freshness state |

### 4.2 The capability matrix

Which experiences demand which capability. `●` = the experience is impossible without
it; `○` = the experience is materially degraded without it.

| | K1 TBox | K2 Entail | K3 Justify | K4 Consist | K5 Constrain | K6 Identity | K7 Diff | K8 Hypoth | K9 Bi-temp | K10 Crosswalk | K11 Symbols |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **E1** why | ● | ● | ● | | | | | | | | |
| **E2** teach + retro | ● | ○ | ○ | | ● | | ● | | ● | | |
| **E3** reject A2A | ● | ● | ● | ● | ● | | | | | ○ | |
| **E4** learned | ○ | ○ | | | | | | ● | ● | | |
| **E5** impact | ● | ● | ● | ○ | | | ● | | | | |
| **E6** code | ○ | ● | | | | | | | | | ● |
| **E7** map | ● | ○ | ○ | | | | | | | ○ | |
| **E8** incident #22 | ● | ● | ● | ● | ○ | ○ | | | | | |
| **E9** crosswalk | ● | ○ | | ● | | ● | | | | ● | |
| **E10** as-of T | ● | ● | ● | | | | ○ | | ● | | |
| **Demanded by (●)** | **8** | **6** | **5** | **3** | **2** | **1** | **2** | **1** | **3** | **1** | **1** |

Two readings of this table matter.

**K3 (justification) is demanded outright by five experiences and degrades two more.**
It is not an observability nicety layered on a reasoner; it is the second most
load-bearing column in the table. E1 *is* justification — the experience has no
content without it. E5 needs it to explain each delta, E8 to name the missing premise,
E3 to attribute each conflicting fact, E10 to reproduce a past derivation.

**K8 (counterexample retention) is demanded by exactly one experience.** That is not an
argument against it; it is the argument for reading requirements this way.
[§4.3](#43-the-two-requirements-a-theory-first-design-would-have-missed) says why.

### 4.3 The two requirements a theory-first design would have missed

This is the document's central methodological claim, and it is checkable: take the two
requirements below, ask what a design derived from description-logic theory would
produce instead, and compare.

**Justification storage (K3), demanded by E1, E5 and E8.** A theory-first design asks
"what is the entailment closure?" and answers "the set of facts derivable from the
axioms." That answer is a *set of facts*. Nothing in the definition of entailment
mentions why any member is a member — the closure is identical whether or not you
remember how you got each element, and every textbook presentation drops the
derivation the moment the fixpoint is reached. So a theory-first implementation stores
facts, and `explain` becomes a feature request. It is then very expensive: recovering
a derivation after the fact means re-running the closure with tracing bolted on, and
the trace you recover is *a* derivation, not *the* one that fired, because the fixpoint
has no memory of order. Deriving from E1 instead produces the opposite default — a
derived fact is not a fact, it is a `(fact, rule_id, premises[])` triple, and the
closure is a store of triples from the first line of code. Off-the-shelf reasoners
demonstrate the point: `owlrl` computes the right closure and cannot answer E1
([§6.1](#61-rejected-owlrl)).

**Counterexample retention (K8), demanded by E4.** A theory-first design treats
induction as hypothesis generation feeding a promotion gate: a candidate either meets
the criteria and is promoted or does not and is dropped. A refuted hypothesis is, in
that frame, *noise that has been correctly filtered* — there is no theoretical reason
to keep it, and the current
[`ontology-alignment-spec.md` §4.3](ontology-alignment-spec.md) table indeed says a
contradicted observation against an asserted term is "discarded for promotion purposes
and logged." Deriving from E4 inverts the value judgment: the refuted hypotheses and
the observations that killed them are the single most informative thing an operator can
read, because they show what the agent *tried to conclude* about the domain and what
reality did about it. Retention is nearly free — the refutation is a reference to an
observation that is already stored, so the marginal cost is one row and a foreign key.
A design that never asked "what does a human want to read on Friday afternoon" spends
nothing to save nothing and loses the best screen in the product.

Neither requirement can be recovered cheaply later. Justification must be in the
closure's data model from the start; counterexamples must be retained from the moment
induction runs, because a refutation not stored at refusal time is gone.

### 4.4 Five stores

The capabilities in [§4.1](#41-the-capabilities-the-ten-experiences-name) partition
into five stores. Each row names the experiences that force it to exist; a store with
no forcing experience would be a design error.

| Store | Holds | Capabilities | Forced by |
| :--- | :--- | :--- | :--- |
| **S1 — Asserted TBox** | Concepts, relations, axioms, tier, provenance, `content_hash`, and each predicate's `source_class` ([§4.7](#47-a-capability-the-experiences-did-not-surface-predicate-provenance)). The **only** input to the closure ([§5](#5-the-induction-safety-rule)). | K1, K12 | E7, and every other experience transitively |
| **S2 — Assertions (bi-temporal)** | Ground facts about instances. Each carries **valid time** (`valid_from`, `valid_to` — when the fact was true of the world) and **transaction time** (`recorded_at`, `invalidated_at` — when the agent believed it). | K5, K9 | E2 (retroactive), E4 (`--since`), E10 (as-of) |
| **S3 — Closure, with justifications** | Derived facts as `(fact, rule_id, premises[], derived_at)`. Never a bare fact. `explain` is a traversal of this store; `diff_closure` is a set operation over two of them. | K2, K3, K7 | E1, E5, E8, E10 |
| **S4 — Hypotheses, with retained counterexamples** | `induced-candidate` axioms, their `EvidenceRecord`, and — for each refuted candidate — the **counterexample**: the assertion, session and turn that refuted it, retained permanently. | K8 | E4 |
| **S5 — Crosswalks** | `CrosswalkAssertion` and `ArbitrationRecord` per [`ontology-alignment-spec.md` §2.2/§3.2](ontology-alignment-spec.md), plus `sameAs` conclusions from K6 with their justifications in S3. | K6, K10 | E9, E3 |

The symbol graph (K11) is **not** a sixth store. It is an ingestion path *into* S1 and
S2: SCIP symbols become instances, SCIP relationships (`references`, `implementation`,
`type_definition`) become relations, and E6's transitive traversal is `R-TRANS-PROP`
([§6.3](#63-the-thirteen-capabilities)) applied to them. `IndexFreshness`
(`src/uclone_x/code_intel/models.py`) rides on the query result, exactly as
[`code-intelligence-lsp-scip.md` §5](code-intelligence-lsp-scip.md#5-dependency-tiers-and-degradation-behaviour)
already requires. This is a derived conclusion worth stating: E6 does not need a
reasoning subsystem of its own.

Note what falls out about **S3's lifetime**: because S1 is the only input and S1 is
versioned by `content_hash`, S3 is a pure function of `(S1@version, S2@instant)`. It is
a cache, fully recomputable, and never the system of record. That is what makes E10
possible at all — the closure "as of T" is not stored, it is recomputed from the
bi-temporal S2 and the S1 version in force at T.

### 4.5 Four operations

| Operation | Signature (shape, not final) | Reads | Writes | Serves |
| :--- | :--- | :--- | :--- | :--- |
| `validate` | `(entity_name, data, pinned_content_hash) -> ValidationResult` | S1, S3 | — | E2, E3 |
| `entail` | `(assertions, as_of=None) -> Closure` | S1, S2 | S3 | E1, E5, E6, E8, E10 |
| `check_consistency` | `(closure) -> tuple[Inconsistency, ...]` | S1, S3 | — | E3, E8, E9 |
| `diff_closure` | `(closure_a, closure_b) -> ClosureDiff` | S3 ×2 | — | E2, E5, E10 |

Three notes, each a decision rather than a description:

* **`explain` is not a fifth operation.** It is a read over S3 — follow a fact's
  `premises[]` recursively until every leaf is an asserted axiom or a recorded
  assertion. Making it an operation would imply it is a computation; it is a
  traversal, and it is only a traversal because K3 is in S3's data model. If `explain`
  ever needs to *compute* anything, K3 has been implemented wrong.
* **`validate` reads S3, not only S1.** This is the visible consequence of
  [§1.2(a)](#12-three-demonstrations-verified-by-execution): a `ReviewTask` must be
  checked against every constraint on every class it is *entailed* to belong to. The
  current `validate_entity` reads S1 only, which is precisely why the demonstration
  fails.
* **`check_consistency` returns a tuple, not a bool.** E3 needs the conflicting fact
  set and the axiom; a boolean cannot carry either, and a raised exception cannot carry
  more than one inconsistency. Each `Inconsistency` names the axiom, the conflicting
  facts, and each fact's justification if derived.

### 4.6 What the derivation shows

The claim, stated so it can be checked rather than admired: **two of the five stores
carry structure that no theory of entailment asks for, and both were produced by
reading a user experience.** S3 stores `(fact, rule, premises)` where the theory needs
only `fact`, because E1 asks *why*. S4 retains refutations where the theory needs only
the surviving candidates, because E4 asks *what did you get wrong*. Delete E1 and E4
from [§3](#3-ten-end-to-end-experiences) and a competent theory-first design reproduces
the rest of this architecture almost exactly — the same closure, the same tiers, the
same four operations — and ships a system that cannot explain itself and forgets its
own mistakes.

That is the argument for deriving from experiences: it is not that theory produces
*wrong* structure, it is that theory produces *exactly* the structure its own success
criterion needs, and "the user understood the answer" is not one of description
logic's success criteria.

[§4.7](#47-a-capability-the-experiences-did-not-surface-predicate-provenance) is the
third and strongest instance of the same method, and it arrives by a different route:
not from reading an experience, but from taking one seriously enough to ask what its
*general form* would require.

### 4.7 A capability the experiences did not surface: predicate provenance

**K12 — predicate provenance.** Every predicate in S1 is declared with a **source
class**, and a derivation rule that concludes an `authority`-class fact from a
`self_asserted` premise is **rejected when the axiom is registered**, not when
inference runs.

| Source class | Meaning | Examples in this repository |
| :--- | :--- | :--- |
| `self_asserted` | The artifact supplies this value about itself | `SkillManifest.status`, `AgentEvent.source`, an artifact's requested `isolation_level` |
| `authority` | The runtime, an auditor, or a computation produced it | An audit verdict, a bus-stamped `sequence`, a content digest computed over bytes |

No experience in [§3](#3-ten-end-to-end-experiences) asks for this. E8 asks for *one
incident* to become impossible, and [§3's scope note](#e8--a-past-incident-made-representationally-impossible)
records what that actually buys: this incident is blocked because of how the `Approved`
axiom happens to be written, and a future axiom keyed on `status` rather than
`hasVerdict` would recreate the bug inside a system that had lost nothing structurally.
Taking E8's requirement seriously — *make the incident impossible*, not *make this
instance fail* — is what produces K12. The difference between the two readings is the
difference between a review discipline and a static guarantee, and only the second is
worth calling a capability.

**Why it belongs in the design and not in a follow-up.** This repository has hit the
identical shape three times, in one day, in three unrelated subsystems:

| Finding | The self-asserted value on the admission path |
| :--- | :--- |
| [#22](https://github.com/UClone-AI/uclone-x/issues/22) | `SkillManifest.status` — a synthesized skill declared itself `active` and registered against a `REJECT` verdict |
| `2026-09-02-039` | `AgentEvent.source` — any in-process component can publish `source="user"`, so a human-approval step delivered as an event is forgeable by the code it gates |
| `2026-09-02-034` | The artifact's requested isolation level — an artifact naming its own ceiling |

One sentence covers all three: **a value an artifact supplies about itself cannot gate
that artifact.** Three code fixes closed three instances. A typed predicate provenance
closes the *shape* — and it does so at axiom-registration time, so the failure is a
rejected axiom with a named premise rather than a wrong answer discovered later.

This is the strongest argument in this document for building an ontology at all,
stronger than any single experience in [§3](#3-ten-end-to-end-experiences): it is the
one place where the ontology prevents a class of bug that code review demonstrably did
not catch three times running.

**Where it lives.** K12 is a property of **S1**, not a sixth store: two extra fields on
a predicate declaration (`source_class`, and the authority that may assert it) plus one
check in the axiom-registration path. It constrains rule *authoring*; it does not change
the reasoner's capabilities in [§6.3](#63-the-thirteen-capabilities), which are
unaffected by where their premises came from. The check is a static test over an
axiom's premise/conclusion predicates, so it costs nothing at inference time.

**What it does not reach.** K12 constrains which *premises* may license an
`authority`-typed *conclusion*. It says nothing about a rule's *head* landing on a
schema predicate instead — `declaresCategory(?c,?parent) -> subClassOf(?c,?parent)`
has no `authority`-typed conclusion for K12 to object to; the escalation it enables
happens one round later, through the ordinary type-propagation rule, over a predicate
K12 does not govern. [§6.3.1](#631-the-thirteenth-capability-cannot-supply-the-twelfth)
(route D6) is the reproduction.

**Two things it does not do**, stated so the claim is not read wider than it is:

* It does not make `self_asserted` facts unusable. They remain assertable, storable,
  queryable and reportable. What is forbidden is a *derivation* that turns one into an
  `authority`-class conclusion.
* It does not classify predicates for you. Someone must declare each predicate's source
  class, and a predicate mislabelled `authority` when it is really self-asserted
  reopens the hole. The mechanism converts a per-axiom review into a per-predicate one
  — far fewer decisions, made once, in a place a reviewer can enumerate — but it does
  not eliminate judgment.

---

## 5. The induction safety rule

### 5.1 The rule

> **Only `asserted` axioms enter the closure.**
>
> An `induced-candidate` or `induced-enforcing` axiom is a **testable candidate**: it
> may be evaluated *against* the closure, and it derives nothing *into* it. Its
> consequences are computed in a scratch closure, reported, and discarded. It becomes
> a premise of a real derivation only by being promoted to `asserted` by a human
> ([§5.3](#53-a-falsifiable-promotion-criterion)).

Correspondingly, every fact in S3 has a derivation whose leaves are all human-asserted
axioms and recorded observations. `explain` cannot bottom out in a guess. That property
is what makes E1 worth reading, and it holds by construction rather than by discipline.

### 5.2 Why: one wrong induction, a thousand poisoned facts

An induced axiom that *validates* is bounded: it can wrongly reject some actions, which
is bad and visible. An induced axiom that *entails* is not bounded, because entailment
multiplies.

Take the concrete case. The synthesizer observes three tool executions where
`exit_code == 0` coincided with a successful task, and induces
`∀x. exitCode(x, 0) → Succeeded(x)`. Suppose it enters the closure. The assertion store
holds ten thousand `ToolResult` observations, of which — say — a thousand have
`exit_code == 0` and were not, in fact, successes. R2 immediately derives
`Succeeded(x)` for all thousand. R4/R5 propagate types from `Succeeded`'s domain and
range. Any constraint keyed on `Succeeded` now passes for all thousand. Any
`disjointWith(Succeeded, Failed)` axiom now reports a thousand inconsistencies, which
buries the real ones. And every one of those derived facts is *justified* — `explain`
will produce a clean, honest derivation tree ending in an axiom the machine made up
from three observations. **Justification does not protect against this; it makes the
poison look rigorous.** The blast radius of one bad axiom is the size of the assertion
store, not the size of the evidence.

Three observations bought a thousand facts. The exchange rate is the problem, and no
confidence score fixes it: a threshold tuned to admit a good axiom at 0.9 admits a bad
one at 0.9. What fixes it is refusing the exchange — an induced axiom's consequences
are computed and *shown*, never *stored*.

This is `2026-09-02-003`'s
"hallucination laundering" one level up. -003 was about an induced axiom being reported
as validated. This is about an induced axiom generating validated-looking *facts*, and
it is strictly worse, because those facts then feed further rules.

It also has a pleasant consequence for determinism: since S1 is the only closure input
and `content_hash` is computed over the asserted set only
([`ontology-alignment-spec.md` §5.1](ontology-alignment-spec.md), already implemented
in `OntologyEngine.compute_content_hash`), the closure is reproducible from a commit.
`./ucx test check` validating the committed asserted snapshot
(`2026-09-02-013`) and the
induction safety rule are the same mechanism seen from two directions.

### 5.3 A falsifiable promotion criterion

The current criterion, in `OntologyEngine.promote` and
[`ontology-alignment-spec.md` §4.2](ontology-alignment-spec.md), is
`observation_count >= 5` across distinct sessions, no open contradiction, no conflict
with an asserted term, plus explicit human action.

**Observation count measures repetition, not explanatory power.** Five sessions in which
nothing interesting happened will produce five confirmations of a vacuous axiom — every
`Task` this week had a `created_at`, therefore `Task` requires `created_at` — and an
axiom that is true of everything constrains nothing. Repetition is evidence that the
agent keeps seeing the same thing, which is largely a fact about the agent's workload.

The replacement is a conjunction, and each conjunct is falsifiable by running it:

> An induced axiom `h` is promotable to `asserted` when, computed in a scratch closure
> over the current S1 and S2:
>
> 1. **Consistent.** `check_consistency(entail(S1 ∪ {h}, S2))` returns empty. Adding `h`
>    introduces no inconsistency that `S1` alone does not already have.
> 2. **Explanatory.** `h` **explains at least N previously unexplained assertions** —
>    where an assertion `a ∈ S2` is *explained* if `a ∈ entail(S1 ∪ {h})` and
>    `a ∉ entail(S1)`. `h` must account for facts the agent already recorded and could
>    not previously derive.
> 3. **Not in conflict with an asserted term** — unchanged from
>    [`ontology-alignment-spec.md` §4.2](ontology-alignment-spec.md) clause 3. Asserted
>    wins unconditionally.
> 4. **Explicit human action** — unchanged from clause 4. Auto-promotion off by default.

Criterion 2 is the substantive change and it is a different kind of measurement: it
asks what the hypothesis *does*, against data already on record, rather than how often
it was seen. An axiom that explains nothing previously unexplained is, by construction,
adding no derivations — so promoting it can only add risk. An axiom that explains
twenty recorded assertions is doing work, and the twenty are enumerable, reviewable,
and shown to the human in the E4 review screen.

Two honesty notes, because this criterion is not free:

* **`N` is a tuning value**, in the sense
  [`nfr-performance-budgets.md`](nfr-performance-budgets.md) uses the word — revisable
  by configuration, not a principle amendment. This document does not pick a number,
  and picking one without data would be the same mistake as `5`.
* **Do not delete the evidence floor.** Issue #138 proposes explanatory power
  *replacing* observation count. Replacing it loses something real: a hypothesis can
  explain twenty assertions that all came from one session, which is one situation
  observed twenty times, not twenty independent confirmations. The recommendation here
  is **conjunction, not replacement** — keep distinct-session evidence as a low floor
  and add explanatory power as the substantive gate. [§9.1](#91-where-the-argument-in-138-is-weak)
  records this as a deliberate divergence from the issue.

### 5.4 What changes about what the tiers govern

The three tiers are preserved exactly. What changes is the *scope of their authority*,
and this is the one place this document alters a commitment in
[`ontology-alignment-spec.md`](ontology-alignment-spec.md). Stating it explicitly:

| Tier | Matching (align spec §2/§4.1) | Validation (Tier 1) | **Entailment (new)** |
| :--- | :--- | :--- | :--- |
| `asserted` | Yes | Yes | **Yes — the only closure input** |
| `induced-enforcing` | Yes | Yes | **No** |
| `induced-candidate` | No | No (advisory) | **No** |

Before this document, the tiers governed **matching and enforcement**: an
`induced-enforcing` term could satisfy a cross-agent match and could reject an action.
Both of those remain true and are not weakened. This document adds a third column,
and in that column `induced-enforcing` sits with `induced-candidate` rather than with
`asserted`. Entailment is a strictly narrower gate than enforcement, for the reason in
[§5.2](#52-why-one-wrong-induction-a-thousand-poisoned-facts): rejecting an action
wrongly costs one action, deriving a fact wrongly costs everything downstream of it.

An `induced-enforcing` axiom therefore keeps the authority it has today and gains none.
Nothing in [`ontology-alignment-spec.md` §4.1](ontology-alignment-spec.md)'s table is
contradicted; a column is added to it.

---

## 6. Stack, with the trade-offs stated

### 6.1 Rejected: `owlrl`

`owlrl` expands an rdflib graph in place under OWL 2 RL / RDFS semantics. It computes
the correct closure. It **records no justifications** — the expanded graph contains the
derived triples and nothing about which rule produced which triple from which premises
([§9.4](#94-unverified-claims): stated from the library's documented API surface, not
verified against its source in this repository). Therefore it cannot serve
[E1](#e1--why-did-you-do-that), which is the watershed experience, nor
[E5](#e5--impact-analysis-before-changing-a-term) or
[E8](#e8--a-past-incident-made-representationally-impossible), which need the same
data. Recovering justifications afterwards means re-running expansion with tracing
around it — which is writing the rules anyway, plus a wrapper.

The trade genuinely given up: `owlrl` implements the full OWL 2 RL rule set (roughly
sixty rules) and is battle-tested. We implement thirteen and inherit no testing. That is
a real loss on rules we do not write; [§6.3](#63-the-thirteen-capabilities) argues the ten
experiences need none of them, and that argument is falsifiable — if an eleventh
experience needs `someValuesFrom` propagation, this decision should be revisited rather
than defended.

### 6.2 Rejected: full OWL DL

A complete OWL DL reasoner (HermiT, Pellet, ELK) buys sound and complete reasoning over
a far richer logic. Two costs:

1. **Worst-case exponential.** OWL DL satisfiability is N2ExpTime-complete. Real
   ontologies usually behave, but "usually" is not a property you want on the path of
   an agent turn, and there is no way to bound it by configuration.
2. **It effectively requires a JVM.** The mature complete reasoners are Java. Running
   one means shipping a JVM dependency and an out-of-process call.

On cost 2, be precise about which principle is at stake, because #138 overstates it.
[P3](principles/details/p3-single-machine-acceleration.md) forbids **out-of-process
message brokers on the single-machine dispatch path**; it does not forbid subprocesses
in general, and [`code-intelligence-lsp-scip.md` §4.3](code-intelligence-lsp-scip.md)
already accepts one-shot subprocess SCIP indexers and LSP server subprocesses as
legitimate. A JVM reasoner invoked as a subprocess is not a P3 violation by the letter
of P3. What it is: an unbounded-latency, external-toolchain dependency on the
**inline turn-validation path** — Tier 1 in
[`agent-ontology-architecture.md` §4](agent-ontology-architecture.md#4-tiered-performance-design),
whose whole design premise is an in-memory check — plus an install burden that
`pip install uclone-x[ontology]` cannot satisfy, the same burden
[`code-intelligence-lsp-scip.md` §4.3](code-intelligence-lsp-scip.md) documents for
SCIP binaries and accepts only because SCIP is *optional* and *background*. Entailment
is neither.

So the honest statement is: **full OWL DL is rejected on latency bounds and install
burden on a path that must be in-process and fast, not because P3's text forbids a
subprocess.** [§9.1](#91-where-the-argument-in-138-is-weak) records this correction.

### 6.3 The thirteen capabilities

Pure-Python, in-process, hand-written, each recording `(rule_id, premises)` on every
fact it derives. Nine fixed derivation rules, three fixed contradiction detectors, and
a thirteenth capability — a restricted Horn-rule admission mechanism — that lets a
caller add a rule the fixed nine cannot express, without a new Python function. This
table was previously wrong about a third of its own entries; see the note below the
table for exactly which, and [§9.1](#91-where-the-argument-in-138-is-weak) for why an
"authoritative" table being wrong is a distinct, worse failure from having no table.

| # | Rule id | Premises | Conclusion | Serves |
| :-- | :--- | :--- | :--- | :--- |
| 1 | `R-SUBCLASS-TRANS` | `subClassOf(A,B)`, `subClassOf(B,C)` | `subClassOf(A,C)` | E1, E5, E7 |
| 2 | `R-TYPE-PROP` | `type(x,A)`, `subClassOf(A,B)` | `type(x,B)` | E1, E3, E5, E8 |
| 3 | `R-SUBPROP-TRANS` | `subPropertyOf(p,q)`, `subPropertyOf(q,r)` | `subPropertyOf(p,r)` | E1, E8 |
| 4 | `R-SUBPROP-VAL` | `subPropertyOf(p,q)`, `p(x,y)` | `q(x,y)` | E1, E8 |
| 5 | `R-DOMAIN` | `domain(p,A)`, `p(x,y)` | `type(x,A)` | E3, E8 |
| 6 | `R-RANGE` | `range(p,B)`, `p(x,y)` | `type(y,B)` | E3, E8 |
| 7 | `R-INVERSE` | `inverseOf(p,q)`, `p(x,y)` | `q(y,x)` | E5, E6 |
| 8 | `R-TRANS-PROP` | `transitiveProperty(p)`, `p(x,y)`, `p(y,z)` | `p(x,z)` | E5, **E6** (symbol traversal) |
| 9 | `R-SYM` | `symmetricProperty(p)`, `p(x,y)` | `p(y,x)` | E1, E5 |
| 10 | `C-DISJOINT` | `disjointWith(A,B)`, `type(x,A)`, `type(x,B)` | **Inconsistency** naming both type facts and the axiom | E3, E8 |
| 11 | `C-CARD` | `maxCardinality`/`minCardinality(p,A,n)`, the `p`-fillers for `x:A` that violate it | **Inconsistency** naming the fillers and the axiom | E3, E8, E9 |
| 12 | `C-FUNC` / `C-IFP` | `functionalProperty(p)`, `p(x,y)`, `p(x,z)`, `y≠z` — or `inverseFunctionalProperty(p)`, `p(x,z)`, `p(y,z)`, `x≠y` | **Inconsistency** naming the conflicting value facts and the axiom (no `sameAs` is derived) | E8 |
| 13 | Horn admission (`hornRule`) | any asserted `hornRule(name, "<body> -> <head>")` fact whose text parses under the restricted grammar in `rules.py` | fires the rule's head for every match of its body against the closure, citing the rule fact itself as an additional premise | E8 (`Approved`, `Rejected`), and any future join the fixed nine cannot express |

Notes on the choices, corrected against the shipped `src/uclone_x/ontology/rules.py`:

* **Row 9 (`R-SYM`) had no row before this revision**, despite `rule_property_symmetry`
  being implemented: it derives `p(y,x)` from an asserted `symmetricProperty(p)` and
  `p(x,y)`, the same shape as `R-INVERSE` self-paired.
* **Rows 8–9 of the previous table (`R-FUNC`, `R-IFP`) claimed to derive `sameAs`. The
  shipped code detects a contradiction instead, and derives nothing.**
  `detect_functional` in `rules.py` reports a `KIND_FUNCTIONAL` or
  `KIND_INVERSE_FUNCTIONAL` inconsistency when a functional property holds two distinct
  values for one subject, or an inverse-functional property holds for two distinct
  subjects — and stops there. No `sameAs` fact is ever produced. Row 12 above replaces
  the previous rows 8 and 9.
* **`R-SAMEAS` — row 10 of the previous table — does not exist in the shipped code, in
  any form.** No rule reads a `sameAs(x,y)` fact and substitutes `y` for `x` in another
  predicate. Combined with the point above, `sameAs` is now neither derived nor consumed
  anywhere in this implementation: **E9 (crosswalk / identity resolution,
  [§3](#e9--crosswalk-review-when-connecting-an-external-agent)) has no entailment path
  today.** [§6.3.1](#631-the-thirteenth-capability-cannot-supply-the-twelfth) records why
  row 13's Horn admission — this design's own stated answer to "if a rule is missing,
  write it as a Horn rule" — cannot fill this particular gap.
* **Rows 10–12 detect; they do not derive.** They are counted among the thirteen because
  they are rules with premises that fire during the same fixpoint pass. Their output
  goes to `check_consistency` — **implemented**, not planned; [§8](#8-implementation-status)
  previously said otherwise — not into S3.
* **Row 8 (`R-TRANS-PROP`) is what makes E6 free.** Transitive-property closure over
  SCIP `references` edges *is* call-graph traversal. E6 needs no reasoning machinery of
  its own.
* **Row 13 (Horn admission) is what makes E8's `Approved` and `Rejected` derivations
  possible**, and is examined in detail — including a serious gap that was found in it,
  reproduced, and has since been closed — in
  [§6.3.1](#631-the-thirteenth-capability-cannot-supply-the-twelfth).
* **Not implemented, deliberately:** `someValuesFrom` / `allValuesFrom` propagation,
  equivalentClass beyond the two-way `subClassOf` encoding, nominals, property chains,
  negation. No experience in [§3](#3-ten-end-to-end-experiences) needs those. `sameAs`
  substitution *is* needed, by E9, and is missing for a different reason: see
  [§6.3.1](#631-the-thirteenth-capability-cannot-supply-the-twelfth).
* **Termination.** All thirteen capabilities — the nine fixed rules, the three
  detectors, and every rule admitted through row 13 — are Horn-shaped over a finite
  Herbrand base fixed by S1's vocabulary and S2's individuals, so naive semi-naive
  iteration to fixpoint terminates. That guarantee is real and it is the cheap one;
  [§6.3.1](#631-the-thirteenth-capability-cannot-supply-the-twelfth) explains why it is
  not, by itself, the guarantee this design needs. No numeric latency claim is made
  here; per `2026-09-02-009` that
  belongs in [`nfr-performance-budgets.md`](nfr-performance-budgets.md) with a
  benchmark, and no benchmark exists.

Writing thirteen capabilities with justification recording is *less* work than
wrapping `owlrl` and retrofitting `explain` onto it, stays pure-Python and in-process,
and puts K3 in the data model from the first line rather than bolting it on — which
[§4.3](#43-the-two-requirements-a-theory-first-design-would-have-missed) argues does not
work.

### 6.3.1 The thirteenth capability cannot supply the twelfth

Row 13's Horn admission was introduced as a general escape valve: if a needed rule is
missing from rows 1–12, express it as a safe Horn rule rather than adding a fourteenth
Python function. Three findings — all reproduced by execution against the shipped code,
not argued from its grammar description — show that valve to be both narrower, and (in
its first shipped form) more dangerous, than that framing suggests.

**It cannot express `R-SAMEAS`.** Substituting through an identity fact into an
arbitrary other predicate needs a *predicate variable* — `?p(?x,?z)`, where which rule
fires depends on what `?p` is bound to. The grammar in `rules.py` refuses this:
`HornAtom.predicate` is always a literal string (`_ATOM_PATTERN` requires
`name(...)`, where `name` matches `[A-Za-z_][A-Za-z0-9_.:-]*` and is never a
`?`-prefixed term), so `?p(?x,?z) -> ...` fails to parse. This was confirmed against
the parser directly, not inferred from the grammar's description. The consequence is
specific: a rule that is missing because the fixed nine cannot express a *conjunctive
join* (what row 13 was built for) is not automatically expressible through row 13
either, when what is actually missing is a *predicate variable* — a different shape
restriction, for a different reason, that the join-shaped gap gave no warning of.

**D6 — schema lifting reached `Approved` around every one of the other restrictions at
once, and the fix needed a restriction the design had not stated.** Found in review of
this PR and reproduced against the reasoner as it stood at the time:

```
lift  : declaresCategory(?c,?parent) -> subClassOf(?c,?parent)     # a benign taxonomy rule
data  : declaresCategory(Skill, Approved)                          # one attacker-controlled triple
result: hasVerdict facts 0 · is_entailed(Approved) True · tier asserted · inconsistencies 0
```

`declaresCategory` is exactly the kind of predicate a skill legitimately owns — a
self-declared category is not, by itself, an authority claim — so [§4.7](#47-a-capability-the-experiences-did-not-surface-predicate-provenance)'s
K12 (predicate provenance) neither blocks this nor should: K12 constrains which
*premises* may license an `authority`-typed *conclusion*, and this escalation happens
in the rule's *head*, over `subClassOf`, a predicate K12 does not govern. Nor, at the
time this was found, was `lift` caught by row 13's safety restriction: it is safe (its
one head variable, `?parent`, is bound in the body), single-headed, and uses no
negation. **The restriction was sound and the exclusion it enforced was the wrong one:
it excluded what makes termination easy, and admitted what makes authority forgeable.**
Safety and acyclicity are what a Datalog textbook promises and what
[§6.3](#63-the-thirteen-capabilities)'s termination note proves; they were the easy
guarantee. The guarantee this design actually needs — *a rule may not grant authority
from a premise the subject controls* — does not follow from safety or from acyclicity,
and the mechanics of D6 show why the old restriction did nothing toward it: the rule
fires once, over asserted data, and derives `subClassOf(Skill, Approved)` as a *derived*
fact whose tier is nonetheless `ASSERTED` (its only premises — the data triple and the
`hornRule` fact itself — are both asserted-tier, and a derivation's tier is the weakest
of its premises' tiers). A tier gate that admits a schema-shaped fact whenever
`fact.tier is OntologyTier.ASSERTED`, without also requiring `fact.derived is False`,
cannot tell that fact apart from one a human wrote by hand, and licenses `R-TYPE-PROP`
from it exactly the same way on the next round. This restriction was derived from what
the #22 replay in `tests/unit/test_ontology_skill_approval_scenario.py` needed to
**express** — its five tests hold the rule set fixed and vary only the data — not from
what the reasoner needs to **guarantee**; D6 varies neither the rule set (`lift` is one
ordinary rule) nor, in the sense those five tests mean it, the data (`declaresCategory`
is one ordinary triple) — it goes *around* both by promoting attacker-controlled data
into the TBox through a channel none of the five tests, or the restriction as first
written, had reason to consider.

**Status: closed in this PR, in `rules.py`, by two independent changes**, made after
this finding and verified against the current code: `parse_horn_rule` now refuses any
rule whose head predicate is a member of `SCHEMA_PREDICATES` (not merely `hornRule`,
which is all the original restriction excluded) — the exact restriction the paragraph
above says was missing — and `AxiomIndex` separately refuses to admit any schema-shaped
fact with `fact.derived is True`, regardless of its tier, so that even a Horn rule
admitted before the parser change existed cannot license further derivation from a
fact it produced. Two layers rather than one: the parser-time refusal stops the rule
from being admitted at all, and the runtime refusal is what stops a *stored* rule,
written before the parser change, from acting on schema-shaped output it produces. This
document is corrected to state the restriction that should exist, and does, rather than
the one that previously stopped at `hornRule` alone. The general lesson survives the
fix: a safety restriction sized to what a test suite needed to express is not the same
restriction as the one the threat model needs, and the two can diverge silently until
someone goes looking for the gap between them.

**D6b — a disjointness axiom is inert unless something derives both of the classes it
names.** `disjointWith(Approved, Rejected)` only fires `C-DISJOINT` when both
`type(x, Approved)` and `type(x, Rejected)` are present in the closure. Row 13's
`ApprovedRequiresMatchingApproveVerdict` derives the first from an `APPROVE` verdict;
nothing derived the second, in the version of the scenario test that first shipped.
Reproduced: a genuine `REJECT` verdict on the skill's genuine content hash, plus any
second verdict object typed `AuditVerdict.APPROVE` on that same hash — exactly what an
attacker who can inject a verdict *object*, rather than a self-asserted field, would
try — yielded `Approved=True`, `Rejected=False`, and **zero** inconsistencies: nothing
in the closure had ever produced a `Rejected` fact for `disjointWith` to conflict with.
The test asserting both classes directly, which is what let it pass, is not a shape any
real registration pipeline produces. Unlike `R-SAMEAS`, this gap **is** expressible in
the existing Horn grammar — the mirror of `ApprovedRequiresMatchingApproveVerdict`,
reading `AuditVerdict.REJECT` instead of `AuditVerdict.APPROVE`, parses under the same
grammar with no new capability — and it has since been added
(`RejectedRequiresMatchingRejectVerdict` in the scenario test), closing this instance.
The design lesson is the reusable part: **a `disjointWith` pair is only as strong as the
weaker of the two derivation paths that feed it**, and a design that derives one side of
a disjointness axiom while leaving the other to be hand-asserted, or to arise from
nothing, has built a detector with one eye shut. Any future disjoint pair introduced
into this vocabulary needs this checked explicitly; nothing in the reasoner enforces it
structurally.

**Where the overclaim actually lived, and a structural point worth stating on its own.**
None of the three findings above corrects an overclaim *in this document*.
[§8](#8-implementation-status) and [§9.1](#91-where-the-argument-in-138-is-weak) point 3
already stated the honest form before any of this was found: incident #22 "is blocked
by how the `Approved` axiom is written, not by anything the reasoner enforces." The
overclaim lived in `tests/unit/test_ontology_skill_approval_scenario.py`'s module
docstring, whose header read *"WHY THE ONTOLOGY FORM MAKES IT UNREPRESENTABLE"* — a
stronger claim than this design ever made, and one D6 and D6b together show to be false
as originally stated (the ontology form, as it was then interpreted, made *that one
instance* of the exploit unrepresentable, not the shape of the exploit in general; D6
is precisely a different-shaped instance of the same exploit, reachable through a
route the header did not anticipate). The header has since been corrected in that file
to *"WHAT THE ONTOLOGY FORM BUYS, AND WHAT IT DOES NOT."* The point survives the fix and
is worth recording on its own: a design document can be, and here was, more honest than
the regression suite claiming to enforce it, and the test file is the more-read
artifact — an engineer extending this system is far more likely to open a test's module
docstring for the contract it claims than to open a design document's open-questions
section for its caveats. A codebase's candor is not uniform across its own artifacts,
and the least candid one tends to be the one read under the least suspicion.

### 6.4 Taken off the shelf

| Piece | Role | Status in this repo | Trade |
| :--- | :--- | :--- | :--- |
| **LinkML** (`linkml-runtime>=1.6.0`) | Canonical TBox source and serialization for S1 | **Already a declared dependency** (`pyproject.toml`, `ontology` extra); `OntologyEngine.export_linkml_yaml` already emits LinkML | Adds no dependency. LinkML has no reasoning; it is a schema language, which is exactly the role S1 needs and exactly why it cannot be the whole answer. |
| **pySHACL** | K5 constraint validation — shapes, cardinality, value ranges | **Not declared.** Adding it is a new dependency in the `ontology` extra | Pure-Python, no JVM. Cost: a second constraint vocabulary alongside LinkML's, and a rule about which owns what (proposal: LinkML owns structure, SHACL owns instance constraints; not yet decided). |
| **rdflib** (`>=7.0.0`) | Triple substrate, IRI handling, serialization | **Already declared** (`ontology` extra) | S3 is **not** an rdflib graph — a triple store cannot hold `(fact, rule, premises)` without reification, and reified justifications are painful to query. rdflib carries S1/S2 interchange; S3 is our own structure. |
| **PROV-O** | Vocabulary for provenance on derived facts (`prov:wasDerivedFrom`, `prov:wasGeneratedBy`) | Vocabulary only | A **vocabulary citation, not a conformance claim** — the same wording [`ontology-alignment-spec.md` §1.1](ontology-alignment-spec.md) uses for SKOS. Using established predicate names costs nothing and makes S3 exportable. |
| **SKOS** mapping relations | Crosswalk predicates in S5 | **Already decided** by [`ontology-alignment-spec.md` §2.2](ontology-alignment-spec.md) | Adopted verbatim, including the rule that `broadMatch`/`narrowMatch`/`relatedMatch` may never satisfy a required field. Not reopened here. |
| **Graphiti** (Zep) bi-temporal edge model | **Reference, not dependency** — the valid-time/transaction-time pair on every edge, and edge invalidation rather than deletion | Not declared, not proposed | We take the data model for S2 and write it ourselves. We do not take the library: it is backed by an external graph database process, which puts it on the wrong side of P3's in-process requirement for the single-machine path ([§9.4](#94-unverified-claims): backend requirement unverified against current upstream docs). |
| **SCIP** | K11 symbol graph feeding S1/S2 | Specified in [`code-intelligence-lsp-scip.md` §4.3](code-intelligence-lsp-scip.md); indexer binaries **Planned**, external toolchain | Adopted with that document's decisions intact, including its unresolved NetworkX-vs-Kùzu question and its `IndexFreshness` requirement. This document does not reopen either. |

### 6.5 Not taken: Mem0, Cognee, Letta

Each of these is a capable agent-memory system, and each is rejected for the same
structural reason rather than on quality: **they impose a memory *policy*, and we need
a storage layer, because we already have a policy.**

UClone-X's policy is written and normative: the three tiers, the evidence rules and
contradiction handling of
[`ontology-alignment-spec.md` §4](ontology-alignment-spec.md), and the induction safety
rule of [§5](#5-the-induction-safety-rule) above. It says what may be believed, what may
enforce, what may derive, what is retained on refutation, and who promotes. That is
precisely the layer these systems supply for themselves — what to extract, when to
consolidate, what to forget, how to score salience, when a memory supersedes another.

Adopting one leaves two options, both bad. Run its policy alongside ours and there are
two authorities over the same facts, with no rule for which wins — the multi-authority
shape [P6](principles/details/p6-fail-fast-observability.md) forbids for values and
[`ontology-alignment-spec.md` §3.1](ontology-alignment-spec.md) had to write an explicit
precedence ladder to avoid. Or disable its policy, and what remains is a store we would
wrap anyway, with its extraction and scoring machinery carried as dead weight.

There is a second, independent objection for the service-shaped ones: Letta is an agent
*server*, and the hosted paths of these systems assume an external service or vector
store. That is the wrong shape for P3's single-machine, in-process path — the same
objection [§6.2](#62-rejected-full-owl-dl) makes to a JVM reasoner, and it applies
regardless of the policy argument ([§9.4](#94-unverified-claims): current deployment
shapes unverified against upstream docs).

None of this says these systems are wrong. It says the part of them that is most
valuable — the policy — is the part we cannot accept, and the part we need — durable
justified storage — is the part they treat as an implementation detail.

---

## 7. Phasing: each phase delivers one working experience

No phase is "infrastructure." Each ends with a command a developer can run.

| Phase | Delivers | Experience | Depends on | Status |
| :--- | :--- | :--- | :--- | :--- |
| **1** | LinkML as canonical TBox source; `ontology/<agent>.yaml` round-trips; tier-labelled render | **E7** — browsable domain map | — | Planned |
| **2** | **The thirteen capabilities, S3 with justifications, `entail`, `explain`, `check_consistency` with typed inconsistencies** | **E1 — the watershed** | 1 (usable standalone on the current S1) | **In progress, issue #138** — `check_consistency` has landed with the reasoner, ahead of where this row's phase-3 sibling below still expects it |
| **3** | pySHACL constraints; `validate` reading S3; **predicate `source_class` and the axiom-registration check** ([§4.7](#47-a-capability-the-experiences-did-not-surface-predicate-provenance)) | **E3**, **E8** | 2 | Planned |
| **4** | `diff_closure`; impact analysis; constraint applied to the current assertion set | **E5**; **E2 (partial)** | 2, 3 | Planned |
| **5** | Hypothesis testing in a scratch closure; S4 with retained counterexamples; explanatory promotion criterion | **E4** | 2, 4 | Planned |
| **6** | Bi-temporal S2; `as_of`; closure recomputation at a past instant | **E10**; **E2 (completed)** | 2, 4 | Planned |
| **7** | S5 crosswalk store and review flow; SCIP ingestion into S1/S2 with `IndexFreshness` | **E9**, **E6** | 2, 3, 6 | Planned |

**Phase 2 is the watershed and the phrase is meant literally.** Before it, the
subsystem is a type-checker with tier metadata, which is what
[§1](#1-the-defect-stated-precisely) demonstrates it is today. The moment `explain`
returns a derivation, the system can tell you something you did not put in, and answer
for it. Everything in phases 3–7 is additive to that; nothing in phases 3–7 substitutes
for it.

**One ordering honesty note.** [E2](#e2--teach-a-constraint-in-natural-language-and-see-what-it-would-have-blocked)
is split across phases 4 and 6, because its two halves have different dependencies. The
*prospective* half — parse the constraint, apply it to the assertions currently on
record, report what it blocks — needs only S2 as a flat store, and lands in phase 4.
The *retroactive* half as E2 actually describes it — what this constraint would have
blocked over the recorded history — needs transaction time, and cannot land before
phase 6. Phase 4 therefore delivers E2 in a reduced form, and the reduction must be
visible in the CLI output (*"evaluated against 118 current assertions; history before
2026-09-14 not available"*) rather than presented as the full report. Issue #138's
phasing places E2 wholly at `diff_closure`; that is not achievable, and
[§9.1](#91-where-the-argument-in-138-is-weak) records it.

---

## 8. Implementation status

Authoritative. Where this table and the prose above disagree, this table is correct.

| Item | Status |
| :--- | :--- |
| S1 asserted TBox (as `OntologyEngine` concepts/relations/axioms, in-memory + YAML) | **Implemented** — `src/uclone_x/ontology/engine.py`, without entailment |
| Three tiers, `EvidenceRecord`, promote/demote/forget, `content_hash` over asserted only | **Implemented** — `engine.py`, `models.py` |
| `validate` (as `validate_entity`, S1-only, no inheritance, `rule_expression` inert) | **Implemented, and defective** — [§1.2](#12-three-demonstrations-verified-by-execution) |
| The thirteen capabilities, S3 with justifications, `entail`, `explain` | **In progress** — issue #138, phase 2. Target modules `src/uclone_x/ontology/{justification,rules,reasoner}.py`, being written concurrently with this document |
| S2 bi-temporal assertion store | **Planned** — phase 6, no code |
| S4 hypothesis store with retained counterexamples | **Planned** — phase 5, no code. `EvidenceRecord.contradicting_observations` exists but holds free-text reasons, not counterexample references |
| S5 crosswalk store, `CrosswalkAssertion`, `ArbitrationRecord` | **Planned** — phase 7, no code. Specified in [`ontology-alignment-spec.md`](ontology-alignment-spec.md) |
| `check_consistency` (the three fixed detectors: `C-DISJOINT`, `C-CARD`, `C-FUNC`/`C-IFP`) | **Implemented** — `src/uclone_x/ontology/reasoner.py`, `rules.py`. This row previously said "Planned — phase 3, no code," which was false the moment this PR's `reasoner.check_consistency` landed; corrected here per [§9.1](#91-where-the-argument-in-138-is-weak) |
| pySHACL constraints (instance-level shape/value-range validation beyond the three fixed detectors) | **Planned** — phase 3, no code, dependency not declared |
| **K12 predicate provenance** — `source_class` on every predicate, and rejection at axiom-registration time of any rule concluding an `authority` fact from a `self_asserted` premise ([§4.7](#47-a-capability-the-experiences-did-not-surface-predicate-provenance)) | **Designed, not built.** Specified here; **not** in #138's vertical slice, which is the reasoner and `explain` only. Targeted at phase 3. Until it exists, incident [#22](https://github.com/UClone-AI/uclone-x/issues/22) is blocked by how the `Approved` axiom is written, not by anything the reasoner enforces |
| `diff_closure` | **Planned** — phase 4, no code |
| SCIP ingestion (K11) | **Planned** — phase 7; upstream state per [`code-intelligence-lsp-scip.md` §4.3](code-intelligence-lsp-scip.md), indexer binaries not installable via `pip` |
| `./ucx ontology explain / impact / learned / map / as-of / align` | **Planned** — no such commands exist, in the sense [`cli-specification.md`](cli-specification.md) uses the word |
| The explanatory promotion criterion ([§5.3](#53-a-falsifiable-promotion-criterion)) | **Specified only** — `OntologyEngine.promote` still applies the observation-count rule, and applies it to concepts only ([§9.3](#93-contradictions-found-in-the-existing-specifications-and-code)) |
| Any latency figure for `entail` | **None asserted.** Per `2026-09-02-009`, numeric budgets live in [`nfr-performance-budgets.md`](nfr-performance-budgets.md) and none has been measured |

---

## 9. Open questions, corrections and unverified claims

Recorded rather than smoothed over, per this repository's convention that a design
document states what it could not establish.

### 9.1 Where the argument in #138 is weak

Four points where this document deliberately diverges from the issue it implements.

1. **The P3 argument against a JVM reasoner is overstated.** #138 says a JVM dependency
   violates P3. P3's text forbids out-of-process *brokers on the dispatch path*, and
   [`code-intelligence-lsp-scip.md` §4.3](code-intelligence-lsp-scip.md) already accepts
   subprocess toolchains elsewhere in this repository. The defensible objection is
   unbounded latency on an in-process inline-validation path plus a non-`pip` install
   burden — which is sufficient, and is what [§6.2](#62-rejected-full-owl-dl) argues.
   Invoking P3 here would set a precedent that also condemns SCIP.
2. **"Replacing" the observation-count criterion loses independent evidence.**
   Explanatory power and repetition measure different things, and neither subsumes the
   other: twenty explained assertions from one session is one situation seen twenty
   times. [§5.3](#53-a-falsifiable-promotion-criterion) specifies conjunction — a low
   distinct-session floor *and* explanatory power — rather than replacement.
3. **"Representationally impossible" is two claims, and #138 only earns the weaker
   one.** Modelling `Approved` as a derived class blocks *this instance* — the
   `Approved` axiom joins on auditor-sourced facts and a content digest, and the
   manifest's `status` is not among its premises. It does not block the *class*:
   nothing in the vocabulary stops a later axiom from joining on `status` instead. The
   claim the design can honestly make without [§4.7](#47-a-capability-the-experiences-did-not-surface-predicate-provenance)
   is *"the vocabulary does not force self-assertion in, and the axiom that matters does
   not use it"* — a review discipline. K12 is what converts it into a static guarantee,
   and it is **designed, not built** ([§8](#8-implementation-status)). Separately, and
   independently of K12: `SkillRegistry.register()` must actually consult the ontology
   instead of reading `manifest.status`, or the ontology holds a correct belief that
   nothing acts on. Phase 3 must include that call site or E8 is not delivered. A third,
   independent gap in the same neighbourhood was found and closed during this PR: K12
   constrains premises, not a rule's *head*, and a rule head can promote attacker data
   straight into the schema without K12 ever having an opinion — [§6.3.1](#631-the-thirteenth-capability-cannot-supply-the-twelfth)
   (route D6) has the reproduction and the fix.
4. **The dividing test needed sharpening.** "Can it tell you something you did not put
   in?" admits a cheap yes — a validator tells you a payload is malformed, which you did
   not type. [§1.3](#13-the-dividing-test) restates it as *can it produce a new
   assertion of the same kind as the facts a human asserts*, which divides cleanly.

Additionally — stated plainly rather than smoothed into a footnote, because a table
[§8](#8-implementation-status) calls *"authoritative"* being simply wrong is a worse
failure than having no table — #138's rule list named **eleven** rules while calling
them twelve, and that was already an undercount by one. Measured against the code that
shipped in this PR, [§6.3](#63-the-thirteen-capabilities)'s own table (before this
revision) was wrong about **four** of its twelve entries, and the errors ran in both
directions: `R-FUNC` and `R-IFP` were written as rules that derive `sameAs`, when the
shipped code makes them detectors that derive nothing; `R-SAMEAS` was given a row, a
premise pattern and a served-experience list, when no such rule exists anywhere in
`rules.py`; and `R-SYM` (`symmetricProperty`), which does exist and does fire, had no
row at all. That is not "off by one, in the same direction as before" — it is four
independent errors, some capabilities claimed that are absent and one present that was
never claimed — and [§8](#8-implementation-status)'s status table separately called an
already-implemented `check_consistency` "Planned" in the same revision. A reader who
checks unsupported prose against nothing at least knows to go read the code; a reader
who checks it against a table marked authoritative, and wrong, stops one step earlier,
having confirmed a mistake instead of a fact. Both tables are corrected in this
revision: [§6.3](#63-the-thirteen-capabilities)'s rule table now has thirteen rows that
match `rules.py`, and [§8](#8-implementation-status)'s `check_consistency` row now says
**Implemented**.

### 9.2 Genuinely open, not decided here

* **The value of `N`** in the explanatory promotion criterion. No data exists.
* **Which vocabulary owns which constraint** when LinkML and SHACL overlap
  (`required` vs `sh:minCount`). A rule is needed before phase 3, or the same constraint
  will be expressible twice with different enforcement.
* **`R-SAMEAS` does not exist, and the escape valve cannot express it.** Earlier
  drafts of this document treated the open question as *scoping* a substitution filter
  — keeping it from rewriting provenance, session ids or content digests. That question
  is now moot in the near term: no rule performs the substitution at all, row 13's Horn
  admission cannot express one (it needs a predicate variable the grammar refuses — see
  [§6.3.1](#631-the-thirteenth-capability-cannot-supply-the-twelfth)), and E9 has no
  entailment path as a result. The open question is no longer "what should the filter
  exclude" but "what mechanism — a grammar extension, a small fixed fourteenth rule, or
  something else — could add identity substitution at all without reopening the same
  safety question [§6.3.1](#631-the-thirteenth-capability-cannot-supply-the-twelfth)
  found row 13 wanting on." Not designed here.
* **Closure recomputation cost at phase 6.** Recomputing "as of T" from scratch is
  correct and possibly slow; incremental maintenance is a known hard problem and is not
  designed here.
* **Cross-agent trust for S5.** Inherited from
  [`ontology-alignment-spec.md` §8](ontology-alignment-spec.md), which defers it to
  `2026-09-02-016`. Unchanged.

### 9.3 Contradictions found in the existing specifications and code

Reported, **not fixed here** — those files are owned elsewhere.

1. **Both sibling documents' status banners are stale.**
   [`agent-ontology-architecture.md`](agent-ontology-architecture.md) says "Nothing in
   this document is implemented. `src/uclone_x/ontology/` holds type stubs only," and
   [`ontology-alignment-spec.md`](ontology-alignment-spec.md) says "`models.py` and
   `protocols.py` are type stubs … none of them carry a tier, a confidence score, an
   IRI, evidence metadata." Both are now false: `models.py` carries `tier`, `iri`,
   `confidence` and `EvidenceRecord`, and `engine.py` implements teach, induce, promote,
   demote, forget, validate and LinkML export. The banners should be updated to the
   [`a2a-protocol-spec.md`](a2a-protocol-spec.md) form, which lists what exists.
2. **`OntologyEngine.promote` does not apply its own evidence gate to axioms.** The
   `min_observations` and open-contradiction checks are inside the concept branch only;
   the axiom branch promotes unconditionally. Verified by execution: an
   `induced-candidate` axiom with `observation_count=1` promotes to
   `induced-enforcing` on the first call. This is
   `2026-09-02-003`'s
   exact complaint, surviving on the element type where it matters most. Out of scope
   for #138; should be filed.
3. **`teach_directive` has a silent fallback** ([§1.2(c)](#12-three-demonstrations-verified-by-execution)):
   an unparseable directive becomes a concept named after a truncated sentence, with no
   error and no warning. This is a
   [P6](principles/details/p6-fail-fast-observability.md) substitution in the human
   steering path P7 depends on. Out of scope for #138; should be filed.
4. **`OntologyAxiom.rule_expression` is hashed into `content_hash` but never
   evaluated.** Two axioms that differ only in `rule_expression` have different content
   hashes and identical behaviour, so `content_hash` is currently sensitive to a field
   that changes nothing. Phase 3 either gives the field semantics or removes it.

### 9.4 Unverified claims

Claims this document could not confirm against a primary source in this environment.
Each is labelled at its use site.

| Claim | Where | Why unverified |
| :--- | :--- | :--- |
| `owlrl` records no justifications for derived triples | [§6.1](#61-rejected-owlrl) | Stated from its documented API shape; the library is not a dependency of this repository and its source was not read here. If it does expose per-triple derivations, [§6.1](#61-rejected-owlrl) must be revisited. |
| No mature pure-Python complete OWL DL reasoner exists; the complete ones are JVM-based | [§6.2](#62-rejected-full-owl-dl) | Believed accurate as of the current knowledge cutoff; not checked against a package index here. |
| Graphiti requires an external graph-database process | [§6.4](#64-taken-off-the-shelf) | Not checked against current upstream documentation. The *reference* use of its bi-temporal edge model does not depend on this claim; only the "not taken as a dependency" reasoning does. |
| Mem0 / Cognee / Letta deployment shapes assume an external service or vector store | [§6.5](#65-not-taken-mem0-cognee-letta) | Not checked against current upstream documentation. The policy argument in [§6.5](#65-not-taken-mem0-cognee-letta) stands independently of it. |

---

## 10. Related

* [#138](https://github.com/UClone-AI/uclone-x/issues/138) — the issue this document answers; [§9.1](#91-where-the-argument-in-138-is-weak) records where it diverges
* [#22](https://github.com/UClone-AI/uclone-x/issues/22) — the incident [E8](#e8--a-past-incident-made-representationally-impossible) replays
* [`agent-ontology-architecture.md`](agent-ontology-architecture.md) — the two pillars and the Tier 1 / Tier 2 split this document extends
* [`ontology-alignment-spec.md`](ontology-alignment-spec.md) — three tiers, evidence, crosswalks, arbitration, determinism; preserved, with one amendment proposed to its §4.2
* [`a2a-protocol-spec.md`](a2a-protocol-spec.md) — status vocabulary ([§0](#0-how-to-read-this-document)), task lifecycle ([§2.3](#23-consistency--what-cannot-hold-together)), extension mechanism carrying `OntologyFragment`
* [`code-intelligence-lsp-scip.md`](code-intelligence-lsp-scip.md) — SCIP, `IndexFreshness`, and the subprocess-toolchain precedent [§6.2](#62-rejected-full-owl-dl) relies on
* [`nfr-performance-budgets.md`](nfr-performance-budgets.md) — where any latency figure for `entail` must go, once measured
* `2026-09-02-003` — the verifiability gap [§5](#5-the-induction-safety-rule) answers at the entailment level
* `2026-09-02-039`, `2026-09-02-034` — the second and third instances of the self-asserted-gate shape that [§4.7](#47-a-capability-the-experiences-did-not-surface-predicate-provenance) closes
* `2026-09-02-012` — cross-agent alignment, resolved by the alignment spec
* `2026-09-02-013` — the determinism split the induction safety rule reinforces
* [`docs/principles/details/p7-evolving-ontology.md`](principles/details/p7-evolving-ontology.md), [P3](principles/details/p3-single-machine-acceleration.md), [P6](principles/details/p6-fail-fast-observability.md) — the principles this document operates under; none is amended by it
