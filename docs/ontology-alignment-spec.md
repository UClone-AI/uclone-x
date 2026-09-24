# Cross-Agent Ontology Alignment Specification

> [!IMPORTANT]
> **This document specifies an intent. It does not describe working software.**
>
> Nothing under `src/uclone_x/` implements any part of this specification.
> `src/uclone_x/ontology/models.py` and `src/uclone_x/ontology/protocols.py` are
> type stubs for a single-agent ontology only (`EntitySchema`, `RelationSchema`,
> `OntologyValidationResult`, and three `Protocol` classes) — none of them carry a
> tier, a confidence score, an IRI, evidence metadata, or anything cross-agent. The
> repository-root `ontology/` directory referenced by
> [`agent-ontology-architecture.md`](agent-ontology-architecture.md) §3.2 as the home
> of human-authored schemas is **empty**. `./ucx ontology validate`,
> `./ucx ontology drift`, `./ucx ontology review`, `./ucx ontology forget` and
> `./ucx ontology arbitrate` — every command named below — are **Planned**, in the
> sense [`cli-specification.md`](cli-specification.md) uses the word: no such module
> or command exists yet. Every normative statement below is a design commitment for
> an implementation that has not started, not a report of current behaviour.
>
> This document resolves `2026-09-02-012`
> (the alignment mechanism P2 and P7 require had no specification) and answers the
> promotion/contradiction/retraction gap raised by
> `2026-09-02-003`.
> It does not resolve -003 itself — -003 is `approval_required: true` because it
> edits `docs/principles/details/p7-evolving-ontology.md`, which is out of scope
> here — but the tiering and evidence rules this document specifies are exactly the
> mechanism -003 asks for, and -003's resolution should point at this file rather
> than re-deriving it.

---

## 0. Why this document exists, and what it must not do

[P7](principles/details/p7-evolving-ontology.md) requires "Decisions and A2A
collaboration must be validated against the agent's active ontology to prevent
semantic drift." [P2](principles/details/p2-google-a2a-interoperability.md) requires
cross-agent communication to logically conform to A2A. Neither principle, nor
[`agent-ontology-architecture.md`](agent-ontology-architecture.md), says what happens
when two agents validate the *same* payload against two *different* active
ontologies — which, under P7, is the normal case, not the exceptional one: P7 makes
each agent's ontology self-constructed from its own experience, so two agents that
have never shared a training signal will independently induce different terms,
different IRIs, and different axioms for the same real-world concept.

The pre-refactor version of this document tried to solve that with "semantic
embedding similarities" deciding equivalence at runtime. Issue `2026-09-02-012`
is explicit that this was itself a defect, not a feature worth restoring: a
probabilistic mapping that silently substitutes one agent's concept for another's is
exactly the silent, unattributable substitution [P6](principles/details/p6-fail-fast-observability.md)
forbids. This specification is written so that similarity scoring can only ever
**propose**, never **decide** — §2 makes that a hard boundary, not a guideline.

---

## 1. What is exchanged

### 1.1 Unit of exchange: the ontology fragment, not the ontology

Agents never exchange their full ontology. They exchange an **`OntologyFragment`**: the
subset of terms relevant to one task — concretely, the terms referenced by the
declared input/output contract of the [`AgentSkill`](a2a-protocol-spec.md#32-required-structure)
being invoked (see [§10.6](a2a-protocol-spec.md#106-typed-skill-contracts) of the A2A
spec: `AgentSkill` carries no schema fields, so a typed contract is itself a
declared extension — the fragment below is the payload that extension needs, and the
two should share one wire format rather than inventing a second).

```text
OntologyFragment
├── ontology_iri        str   — namespace IRI this fragment's terms are minted under
├── ontology_version     int   — monotonic per-agent version, bumped on every
│                                promotion/retraction that changes the asserted or
│                                induced-enforcing set (§5.2)
├── content_hash          str   — sha256 over the canonical serialization of `terms`
│                                at this version (§5.1); identifies the fragment
│                                independent of transport
├── terms[]              TermDescriptor
└── crosswalks[]          CrosswalkAssertion   — equivalences this agent already
                                                  holds for terms outside its own
                                                  namespace (§2.2)

TermDescriptor
├── iri              str            — globally unique: "{ontology_iri}#{local_name}"
├── kind             "entity" | "relation" | "axiom"
├── tier             "asserted" | "induced-enforcing" | "induced-candidate"  (§4.1)
├── labels[]         str            — lexical surface forms, for §2.3 similarity only
├── parent_iri       str | null     — corresponds to EntitySchema.parent_type
├── attributes       map[str, str]  — corresponds to EntitySchema.attributes
├── required_fields[] str
└── evidence         EvidenceRecord | null   — null iff tier == "asserted" (§4.2)

CrosswalkAssertion
├── local_iri        str
├── remote_iri       str
├── remote_ontology_iri str
├── relation         "exactMatch" | "closeMatch" | "broadMatch" | "narrowMatch" | "relatedMatch"
│                     (SKOS mapping-relation vocabulary, borrowed for its precision —
│                      this is a vocabulary citation, not an SKOS conformance claim)
├── declared_by      "human" | "arbitration"   — never "induced" (§2.2)
└── decided_at       str (ISO 8601 UTC)

EvidenceRecord
├── observation_count        int
├── first_seen / last_seen   str (ISO 8601 UTC)
├── originating_sessions[]   str
├── confidence                float in [0, 1]
├── induced_by                {provider: str, model: str}
└── contradicting_observations[]  str   — session/turn refs, empty until §4.3 fires
```

`TermDescriptor` and `EvidenceRecord` are a superset of today's `EntitySchema` /
`RelationSchema` (`src/uclone_x/ontology/models.py`) — the existing fields map
straight across (`name` → `local_name` component of `iri`, `attributes`,
`required_fields`, `is_directed` for relations). Nothing here proposes replacing
those types; it proposes what they are missing to be exchanged and matched across
agents at all: an IRI, a tier, and evidence.

### 1.2 How it rides an A2A message

[§10.6 of `a2a-protocol-spec.md`](a2a-protocol-spec.md#106-typed-skill-contracts)
already establishes the pattern this must follow: `AgentSkill` has no schema field,
so anything typed travels as a **declared, optional A2A extension** (§4.6 of the A2A
spec), never as an invented Agent Card field.

* **Extension URI**: `https://github.com/UClone-AI/uclone-x/extensions/ontology-alignment/v1`,
  listed in `AgentCard.capabilities.extensions` as an `AgentExtension` with
  `required: false`. An agent that does not recognise the extension MUST still be
  able to call the skill using the standard natural-language `description` /
  `examples` / media-type fields — per the A2A rule that a `required: false`
  extension cannot be the only way to use a capability. This is what keeps ontology
  alignment additive to A2A conformance rather than a fork of it.
* **Activation**: the calling agent sends the extension's URI in the
  `A2A-Extensions` service parameter on the request that starts a new `contextId`
  (A2A §3.2.6 / §3.6.1, `docs/a2a-protocol-spec.md` §7.3).
* **Payload placement**: the `OntologyFragment` JSON is carried as one `Part` inside
  `Message.parts[]`, with `mediaType: "application/vnd.uclone-x.ontology-fragment+json"`.
  `Message.extensions[]` (A2A §5.2 core objects) MUST include the extension URI above
  on any message carrying this part, so a receiver can locate it without parsing
  every part's `mediaType` speculatively.
* **When it is sent**: attached to the first `Message` of a new `contextId` between
  two agents (or the first message after either side's `ontology_version` has
  changed for terms relevant to that `contextId` — see §4.4). It is not resent on
  every message in a task; both sides pin the negotiated fragment for the life of
  that task (§5.3).
* **Reply channel is the existing state machine, not a new one**: a receiver that
  cannot map a required term does not invent a new `TaskState`. It transitions the
  `Task` to the existing `TASK_STATE_INPUT_REQUIRED` (A2A §5.1) and attaches a
  `TaskStatus.message` whose `Part` (same media type) names the unmapped
  `TermDescriptor.iri`, any candidate crosswalk(s) from §2.3 with their confidence,
  and what would resolve it (§2.4, §3). A receiver that cannot map at all, with no
  interactive party able to answer, fails the task (`TASK_STATE_FAILED`) with a
  typed error rather than guessing — see §2.4.

This mechanism is deliberately symmetric with, and reuses the payload envelope of,
the typed-skill-contract extension in §10.6: a fragment describing a skill's
declared input/output entities *is* that skill's typed contract, expressed in terms
of IRIs instead of a bespoke JSON Schema. Whoever defines the §10.6 payload should
consider pointing it at this type rather than inventing a second one — noted for the
`a2a-protocol-spec.md` owner, not decided here.

---

## 2. How terms are matched

Given a required term IRI on one side and the counterpart `OntologyFragment` from
the other, matching is a **strict, ordered procedure** — later steps run only if
every earlier step fails to produce a match. No step may be skipped, and no step
below the line in §2.4 may stand in for one above it.

### 2.1 Tier 1 — exact IRI match

The required IRI appears verbatim in the counterpart's `terms[]` (including via a
shared upstream namespace both sides import, e.g. a common seed ontology loaded by
[`./ucx ontology teach`](agent-ontology-architecture.md#32-human-guided-steering-developer-ingestion)
at both agents). Confidence is definitionally `1.0`. No further step runs.

### 2.2 Tier 2 — declared equivalence (crosswalk)

A `CrosswalkAssertion` already exists between the two IRIs, with `declared_by`
either `"human"` (a developer ran the equivalent of `./ucx ontology teach --crosswalk`,
Planned) or `"arbitration"` (§3 ran to completion and recorded a decision). A
crosswalk with `declared_by == "induced"` **cannot exist** — §2.3 output is never
written back as a `CrosswalkAssertion` on its own; see §2.4.

The `relation` field controls whether the crosswalk can satisfy a *required* input:

| Relation | May satisfy a required field automatically? |
| :--- | :--- |
| `exactMatch` | Yes |
| `closeMatch` | Yes |
| `broadMatch` / `narrowMatch` / `relatedMatch` | **No** — informational only. A lossy or directional mapping satisfying a required field would be exactly the kind of mismatched-meaning acceptance `2026-09-02-012` warns about. These relations may satisfy an *optional* field, always flagged in the response's `metadata` as satisfied via a lossy crosswalk. |

### 2.3 Tier 3 — structural / lexical similarity (proposal only)

Runs only when Tiers 1–2 produce nothing. Computes a `confidence ∈ [0, 1]` from any
combination of: `labels[]` lexical distance, `attributes` key/type overlap, and
`parent_iri` chain overlap. The method is intentionally unspecified — embedding
similarity, edit distance, structural unification are all permissible
*implementations* — because the contract that matters is not the algorithm, it is
what the algorithm is allowed to produce: **a proposed `CrosswalkAssertion` with
`declared_by` left unset**, never an accepted mapping. A Tier 3 result:

* is surfaced to whichever party can act on it (§2.4), never applied silently;
* is never written into either agent's `crosswalks[]` as if declared;
* does not, on its own, change any term's tier (a good similarity score is not
  evidence for §4's promotion criteria — that evidence comes only from successful
  task executions using the mapping *after* a human or arbitration record accepts
  it).

### 2.4 Below the confidence threshold — explicit failure or explicit clarification, never a guess

A configurable minimum, `ONTOLOGY_ALIGNMENT_MIN_CONFIDENCE` (default **0.85**, a
tuning value in the sense [`nfr-performance-budgets.md`](nfr-performance-budgets.md)
uses the term — revisable by configuration, not a principle amendment), gates
whether a Tier 3 proposal is even shown as a candidate. This procedure is total —
every required term reaches exactly one outcome:

1. **Tier 1 or 2 succeeds** → accepted, `provenance`-equivalent tier recorded on the
   match (see §5.3 for what the task's artifacts must carry).
2. **Tier 3 produces a candidate ≥ threshold, and the caller is interactive** (a human
   or a human-attended session is reachable within the task's timeout) → `Task`
   transitions to `TASK_STATE_INPUT_REQUIRED`, presenting the candidate for a
   human decision. A `"yes"` mints a `CrosswalkAssertion` with `declared_by: "human"`
   before the task resumes; the task never resumes on the candidate alone.
3. **Tier 3 produces a candidate ≥ threshold, and no interactive party is reachable**
   → the task **fails** (`TASK_STATE_FAILED`, typed error `OntologyTermUnmappedError`
   carrying the candidate and its confidence in the error detail) rather than
   silently accepting a high-scoring guess. A high score is still a guess.
4. **Tier 3 produces nothing ≥ threshold, or nothing at all** → the task fails the
   same way, with an empty candidate list.

Outcomes 3 and 4 differ only in whether a human could plausibly resolve it faster by
being shown a near-miss; both are the *same* class of outcome under P6's
classification procedure (`docs/principles/details/p6-fail-fast-observability.md`):
no real, attributable mapping was produced, so nothing is returned as if a mapping
had been produced. **There is no fifth outcome where the system proceeds on the best
available guess.** This is the direct answer to why the pre-refactor "embedding
similarity" design was a defect: it collapsed outcomes 2–4 into automatic
acceptance.

Optional (non-required) terms follow the same table, except outcome 3/4 does not
fail the task — the field is simply omitted, and its omission is recorded in-band
(the response's `metadata`, not only in a trace span, per P6's "provenance travels
with the value" rule) so the receiving agent's own reasoning can see that a field it
might expect is missing *because of an alignment gap*, not because the sender chose
to omit it.

---

## 3. Who arbitrates a conflict

A **conflict** is two agents each holding a definition for what should be the same
concept that cannot both be true — e.g. incompatible `required_fields`, incompatible
`parent_iri` chains, or an axiom on one side that the other side's declared crosswalk
would violate. Conflicts are detected at Tier 2 lookup time (§2.2) or at
arbitration review time (below), never inferred from a Tier 3 score.

### 3.1 Precedence before arbitration is needed

Precedence is checked first because most apparent conflicts are not conflicts
between equals:

1. `asserted` beats `induced-enforcing` beats `induced-candidate`, **on each side
   independently, before the two sides are compared.** An agent's own induced term
   never gets to argue with its own asserted term — see §4.3.
2. If exactly one side's *conflicting* term is `asserted` and the other side's is
   induced (either tier): the induced side loses outright. Its agent must demote
   that term per §4.3 (contradiction handling) for the scope of this alignment; no
   human is needed to adjudicate an asserted-vs-induced conflict, because P7 already
   ranks human authorship over self-construction.
3. Only when **both sides' conflicting terms are `asserted`** — two humans (or one
   human wearing two hats, in a same-organization multi-agent deployment) each
   deliberately defined the term differently — does this require arbitration.

### 3.2 The arbitration record

Arbitration is a human-steering intervention, concretely: a reviewer inspects both
`TermDescriptor`s side by side (mirroring the review discipline
[`principle-amendment-policy.md`](governance/principle-amendment-policy.md) §2 Tier B
uses for non-normative text: exact current state, exact proposed state, one
decision) and records exactly one outcome as an `ArbitrationRecord`:

```text
ArbitrationRecord
├── term_a_iri / term_b_iri
├── outcome        "equivalent" | "distinct" | "deferred"
├── relation       CrosswalkAssertion.relation   — present iff outcome == "equivalent"
├── decided_by[]   str            — steward identity per participating organization;
│                                   a cross-organization conflict requires one
│                                   steward per organization, not one unilaterally
├── decided_at     str (ISO 8601 UTC)
└── rationale      str            — required, free text
```

* **`equivalent`** → mints a `CrosswalkAssertion` with `declared_by: "arbitration"`
  on both sides, per §2.2.
* **`distinct`** → records permanent non-equivalence. This is not a no-op: it
  suppresses §2.3 from re-proposing this exact pair, so the same unresolved
  ambiguity does not re-surface as a fresh clarification request on every task.
* **`deferred`** → no mapping now; the record is kept so a future arbitration
  attempt has the prior rationale available. Tasks needing both terms continue to
  fail per §2.4 until a later `equivalent` or `distinct` supersedes the deferral.

Planned interface: `./ucx ontology arbitrate <term-a-iri> <term-b-iri>` — not
implemented. Until it exists, an `ArbitrationRecord` is a hand-authored artifact, the
same way the retired design-review register's entries were hand-authored
(`builder-issue-resolution-protocol.md` §2).

### 3.3 Mid-task ontology change during arbitration

If either side's `ontology_version` advances while an arbitration is pending
(§4 can retract or demote terms independently of this negotiation), the pending
`ArbitrationRecord` is re-validated against the new `TermDescriptor` before being
applied — an arbitration decided against a term that no longer exists in that shape
is not silently carried forward.

---

## 4. Promotion, contradiction and retraction

This section is the direct answer to `2026-09-02-003`:
without it, an induced term is indistinguishable from ground truth the moment it
exists. It adopts -003's suggested tiers exactly.

### 4.1 Tiers

| Tier | Authored by | Used to reject an action? | Participates in §2 Tier 1/2 matching? |
| :--- | :--- | :--- | :--- |
| `asserted` | Human, via schema or `./ucx ontology teach` | Yes, always | Yes |
| `induced-enforcing` | Promoted from `induced-candidate` (below) | Yes | Yes |
| `induced-candidate` | Autonomous induction, freshly extracted | **Never** | No — visible only as §2.3 similarity input, never as a Tier 1/2 hit |

An `induced-candidate` term is recorded and surfaced for review; it must never gate
a Tier-1 turn validation (`OntologyValidatorProtocol.validate_entity`) and must
never satisfy a cross-agent match on its own. This is the direct fix for issue -003's
"hallucination laundering" complaint: a single observation can create a
`induced-candidate`, but nothing downstream treats a candidate as true.

### 4.2 Promotion: `induced-candidate` → `induced-enforcing`

All of the following must hold — this is a conjunction, not a scorecard:

1. **Evidence threshold**: `observation_count >= ONTOLOGY_PROMOTION_MIN_OBSERVATIONS`
   (default **5**, tuning value, same status as §2.4's threshold), drawn from
   distinct `originating_sessions` — five repeats within one session count as one
   observation for this purpose, because repetition inside a single context is not
   independent confirmation.
2. **No open contradiction**: `contradicting_observations` is empty at promotion
   time (§4.3 defines when an entry is added and removed).
3. **No conflict with an `asserted` term**: an induced term that contradicts an
   asserted one is not promoted — ever, regardless of evidence count. Asserted wins
   unconditionally per §3.1.
4. **Explicit human action**, one of:
   * a human runs the equivalent of `./ucx ontology review` (Planned) and approves
     the specific term, or
   * a per-agent auto-promotion policy is turned on by a human, as an explicit,
     declared configuration entry naming the agent and the threshold — never a
     silent default. This mirrors P6's "declared in advance" requirement for any
     recovery path: an unreviewed auto-promotion is the ontology equivalent of an
     undeclared fallback.

Default configuration ships with auto-promotion **off**. P7 says "human-guided";
the default must guide, not merely permit guidance.

### 4.3 Contradiction handling — at write time, not at enforcement time

A contradiction is a new observation whose data violates an existing term's stated
`required_fields`, attribute type, or an axiom derived from it. Detection happens
when the observation is written to the induction store, not later when something
tries to use the term to reject an action — issue -003 named this exact gap.

| Contradicted term's tier | Effect |
| :--- | :--- |
| `asserted` | The new observation is **discarded** for promotion purposes and logged as a rejected candidate. An asserted term is never overridden by contradicting evidence — a human must change it deliberately. |
| `induced-enforcing` | **Immediate demotion** to `induced-candidate`, the instant the contradiction is detected. The term stops rejecting actions from that point on, and a review flag is raised. A bad induced axiom must stop being authoritative the moment it is known to be shaky — leaving it enforcing while "under review" is exactly the false-rejection failure mode issue -003 describes. |
| `induced-candidate` | Both the existing candidate and the new observation are recorded; `contradicting_observations` grows on the older one; neither's `observation_count` advances toward promotion while unresolved. |

Demotion is logged with a `retraction_reason` (§4.4) even though the term is not
removed — a demotion is not silent just because the term still exists.

### 4.4 Retraction and dependents

Retraction removes a term entirely: a human runs the equivalent of
`./ucx ontology forget <iri>` (Planned), or a term repeatedly re-contradicted after
demotion (a config'd rolling-window count, default **2** re-demotions without an
intervening promotion) is auto-retracted — always into an audit log, never deleted
without trace.

**Dependents** of a retracted term `T` are: any `TermDescriptor` whose `parent_iri`
is `T`; any relation term naming `T` as source or target; any `CrosswalkAssertion`
naming `T`'s IRI on either side; any `induced-enforcing` axiom whose derivation
recorded `T` as a premise.

| Dependent's tier | Effect of retracting `T` |
| :--- | :--- |
| `asserted` | **Retraction of `T` is blocked.** The error names every blocking `asserted` dependent. A human must retract or amend those first, or keep `T`. Cascading a retraction into human-authored structure without being asked is the same silent-mutation failure mode P6 forbids elsewhere. |
| `induced-enforcing` / `induced-candidate` | Cascade-demoted to `induced-candidate` (or cascade-retracted, if the dependent cannot be well-formed without `T` — e.g. a relation whose `target_entity` no longer exists), recorded with `retraction_reason` pointing at `T`. |
| Any `CrosswalkAssertion` naming `T` | **Invalidated immediately.** Any agent that had accepted this crosswalk must re-run §2 alignment before relying on it again. |

**Mid-task retraction** (issue -012's "what happens mid-task when either side's
ontology changes"): a task already in flight keeps using the `OntologyFragment` it
was handed at task start (§1.2, §5.3) — retraction takes effect *between* tasks on a
`contextId`, not inside one. The **next** message on that `contextId` MUST re-attach
a fresh `OntologyFragment` if `ontology_version` has advanced since the last one, and
the receiver MUST treat a stale-versioned fragment presented after it already
observed a newer version (via a prior message on the same `contextId`, or via its own
retraction) as invalid and re-request alignment (`TASK_STATE_INPUT_REQUIRED`). A task
is never silently continued under a mapping that has already been invalidated.

---

## 5. Determinism

`2026-09-02-013` observes that
a self-evolving ontology makes `./ucx test check` non-reproducible if the gate
validates live, mutating state. The same instability would make cross-agent
alignment non-reproducible too: two attempts to align the same two agents could
disagree if either side's induction ran in between. The fix is one mechanism serving
both:

### 5.1 Content hash, computed once, over one tier only

`content_hash` (§1.1) is a sha256 over the canonical serialization (keys sorted,
whitespace normalized, no floating-point confidence fields — confidence lives only
on `induced-candidate` evidence, which is excluded, see below) of exactly the
**`asserted`** term set at a given `ontology_version`. `induced-candidate` and
`induced-enforcing` terms are **excluded from the hash** — they mutate in the
background by design (P7), and a hash that included them would defeat the purpose
of hashing.

### 5.2 Two tracks, matching issue -013's suggested split exactly

| Track | Validates | Storage | Blocking? |
| :--- | :--- | :--- | :--- |
| `./ucx test check` (existing gate, Planned ontology step) | `asserted` tier only, at the `content_hash` pinned in the commit's `ontology/` tree | Repository-root `ontology/<agent_name>.yaml`, version-controlled | **Yes** — deterministic, bisectable, identical on any machine checking out the same commit |
| `./ucx ontology drift` (Planned) | Delta between the committed `content_hash` and the live `induced-candidate` / `induced-enforcing` state | A runtime state directory outside the validated tree (e.g. `.uclone/ontology-state/<agent_id>/`, gitignored) | **No** — advisory, surfaced for human review, never fails a commit |

Cross-agent alignment (§2) is permitted to draw on the live `induced-enforcing` tier
in addition to `asserted` — that is the entire point of P7's self-construction being
usable at all — but every match an agent makes must record **which tier** satisfied
it (§5.3). A validation run that only exercises `asserted`-tier matching is
reproducible by construction (§5.1); one that also drew on `induced-enforcing` is
labelled as such and is out of `./ucx test check`'s blocking path, exactly as
issue -013 prescribes for the single-agent case.

### 5.3 What a task's artifacts must record

Every `Task` whose alignment drew on §2 MUST attach, in an `Artifact.metadata` (or
equivalent in-band field — never telemetry-only, per the P6 precedent this document
follows throughout), for each matched term:

```text
{
  "term_iri": "...",
  "matched_tier": "asserted" | "induced-enforcing",
  "match_kind": "exact-iri" | "crosswalk",
  "crosswalk_relation": "..."        // present iff match_kind == "crosswalk"
  "sender_ontology_version": <int>,
  "receiver_ontology_version": <int>
}
```

This is what makes a divergence between two alignment attempts on the same commit
detectable after the fact: if `sender_ontology_version` or `receiver_ontology_version`
differs between two runs that were supposed to be identical, the run is provably not
reproducible, and the artifact says why — rather than the two runs silently
disagreeing with no recorded cause, which is the failure mode issue -013 opened
against.

---

## 6. Where P7 and P2 cannot both fully hold

This is reported as a genuine tension, not resolved by anything in this document,
because resolving it would mean weakening one of the two principles — a Tier A
change this Builder is not authorized to make.

**P7 guarantees autonomy of construction: each agent's ontology is its own, induced
from its own experience, in its own namespace, un-coordinated with any other agent's
induction process.** **P2 guarantees interoperability: cross-agent communication
must logically conform to A2A.** A2A conformance is a *protocol*-level guarantee — it
says the message envelope, the task lifecycle, and the transport are mutually
intelligible. It says nothing about whether the *meaning* of the payload inside that
envelope is shared, and per P7 there is no reason it would be: two independently
self-constructing agents have no mechanism that would cause them to converge on the
same IRI, the same `labels[]`, or the same axioms for the same real-world concept,
because nothing in P7 coordinates the induction process across agents.

The consequence, stated plainly: **full automatic semantic interoperability between
two ontologies that have only ever self-constructed, with no prior declared
crosswalk and no shared seed ontology, is not achievable without either (a)
weakening P7's autonomy** — e.g. by imposing a shared upper ontology or a shared IRI
minting authority that agents must induce *into*, which is no longer purely
self-constructed — **or (b) weakening P2's guarantee** to "protocol-level
interoperability, with semantic interoperability conditional on an alignment record
existing for the terms in play." This specification adopts (b): §2's fail-fast /
clarify behavior at the confidence threshold is the honest expression of that
scope-limit — every message is A2A-conformant and transportable regardless of
ontology alignment, but a message whose *meaning* has no declared crosswalk is
refused or escalated, never guessed into an appearance of interoperability. Two
agents that have truly never aligned anything will, correctly, be unable to
collaborate on novel concepts until a human (§3) or a prior crosswalk (§2.2) bridges
the gap — that outcome is not a bug this document can specify away; it is what P7's
autonomy costs P2's promise, made visible instead of papered over.

Whoever eventually reconciles the *principle* text — not this document — should
decide, with the project owner per
[`principle-amendment-policy.md`](governance/principle-amendment-policy.md) §2
Tier A, whether P2's "interoperability" should be qualified this way explicitly, or
whether P7 should instead require every agent to seed from a shared minimal upper
ontology (narrowing self-construction to *extension* of a common core rather than
construction from nothing). Both are legitimate designs; this document does not pick
one, because picking one changes what a principle requires.

---

## 7. Requested changes to `agent-ontology-architecture.md`

Reported per this task's instructions — **not applied here**; that file is owned by
the Builder Manager.


1. Add a "§5 Cross-Agent Semantic Alignment" section pointing at this document as the
   normative source, replacing the pre-refactor §4 that
   `2026-09-02-012`
   found deleted with nothing put in its place.
2. §3.1 ("Autonomous Self-Construction") currently describes a single undifferentiated
   pipeline — `Synthesizer -->|Auto-Extend Classes & Rules| SchemaStore` in the §2
   diagram implies immediate enforcement. It should show the three-tier model of this
   document's §4.1: induction lands in an `induced-candidate` staging store distinct
   from the enforcing `SchemaStore`, with an explicit, human-gated promotion edge
   between them — otherwise the diagram keeps contradicting
   `2026-09-02-003`.
3. §4 ("Tiered Performance Design") should note that Tier 1's fast validator is only
   deterministic against a pinned `ontology_version` / `content_hash` (this
   document's §5.1) — otherwise the "0.01ms" figure and the ontology it is validating
   are both moving targets at once, which is a second latency-figure problem in the
   spirit of `2026-09-02-009`.
4. Add an implementation-status banner matching the one at the top of this document
   and the one already in [`a2a-protocol-spec.md`](a2a-protocol-spec.md) — currently
   `agent-ontology-architecture.md` carries no such disclaimer at all, despite
   describing a `Synthesizer`, a `FastValidator`, and a `GraphStore` none of which
   exist in `src/uclone_x/ontology/`.

---

## 8. Related

* `2026-09-02-012` — the finding this document resolves
* `2026-09-02-003` — the tiering and evidence rules this document supplies
* `2026-09-02-004` / [`a2a-protocol-spec.md`](a2a-protocol-spec.md) §10.6 — the extension mechanism this document's exchange rides on
* `2026-09-02-013` — the determinism split this document's §5 implements
* `2026-09-02-016` — cross-organization crosswalk exchange has trust and authentication implications this document does not cover; owed to the security threat model, not here
* [`agent-ontology-architecture.md`](agent-ontology-architecture.md) — this document's primary input; §7 above lists requested changes, not applied by this Builder
* [`docs/principles/details/p7-evolving-ontology.md`](principles/details/p7-evolving-ontology.md), [`docs/principles/details/p2-google-a2a-interoperability.md`](principles/details/p2-google-a2a-interoperability.md) — the two principles this document reconciles operationally, and the genuine tension between them reported in §6
