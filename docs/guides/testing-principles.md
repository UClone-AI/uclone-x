# UClone-X Testing Principles

**Governance Tier**: Tier 3 Guide (`docs/guides/`, Docs Fast-Track per `documentation-governance-policy.md`)
**Governing Norms**: [`docs/principles/core-principles.md`](../principles/core-principles.md) — this document applies P1, P6 and P8 to testing. It establishes no new law.

Industry consensus is assumed, not restated: ISTQB's principles and test levels, FIRST, the
test pyramid, behaviour over implementation, hermeticity, trunk-based development. Standard
terms are used with their standard meanings, `ad hoc test` and `exploratory testing`
included.

What follows is only what **this** repository's structure forces, and could not be derived
from general practice. Every entry states the question it answers and the structural fact it
comes from. An entry that answers no question a test author faces is deleted.

Cases — issue numbers, dates, measurements, paths, commands — belong in
`quality-testing-and-evaluation-guide.md`.

---

## What these principles optimize for

Goals are not decision rules — they do not settle a concrete case, and nothing below is
derived from them by argument. They say what the principles are tuned for, so that an
amendment can be judged by whether it serves them.

**Consistency.** The same situation produces the same choice, whoever is writing and whenever
they write it. This is why a principle must answer a question with one answer: a rule that
merely advises leaves each author to re-decide, and the suite ends up holding several
different opinions about the same thing.

**Cost and time.** The gate runs on every commit and is the only verification that exists.
Every test is paid for on every run and again on every refactor, so the cheapest method that
proves a claim is the right one, and a level that outgrows its budget gets bypassed rather
than tolerated. *Served by R16, R17, R19, R28.*

**Only the tests that are needed.** More tests are not more safety. Two tests that fail
together on the same defect carry one signal at twice the cost, and a test whose subject has
moved carries none at full cost. Adding a test and deleting a test are the same kind of
decision. *Served by R11, R14, R18, R28.*

**What the user experiences.** This product is worth what a user gets from it, so a suite can
be entirely green while the thing itself is broken for the person using it. Tests assert what
is observable where the user stands, treat the disappearance of a promised behaviour as a
failure, and cover each surface people actually use. *Served by R6, R16, R20, R21.*

The first three pull against the fourth, and that tension is deliberate: the cheapest suite
that proves nothing about the user's experience has optimized the wrong thing.

---

## R1 — Substitute only as deep as your claim allows

*Answers: which substitute do I use for the LLM — or must it be real?*
*From: the LLM sits behind a provider-neutral interface, so a substitute can be inserted at four different depths, and each answers a different question.*

A collaborator is replaced **only when the test's claim is not about that collaborator**.
What the test asserts decides what may be substituted.

| The claim is about… | Substitute at | Why |
| :--- | :--- | :--- |
| our code's behaviour given some model output | the **connector** (scripted mock) | the model is not the subject; scripting it is what makes the claim decidable |
| our handling of malformed, degraded or hostile output | the **connector**, producing exactly that output | the failure mode must be producible on demand |
| our parsing of a response shape we already know | the **transport**, with a hand-written payload | the claim is about our parsing, and the shape is an input to it |
| whether the provider actually sends that shape | the **transport**, with a **recorded** exchange | a payload we wrote ourselves cannot testify to what the provider sends |
| the model's own behaviour — native tool calls, convergence, retention | **nothing**: a real model, or a recording of one | a double asked about itself answers about itself |
| cost, latency or quota behaviour | **nothing** | a double has no cost |

The default is a double, because most claims are about our code. The exception is narrow and
explicit. A result produced entirely against doubles says nothing about anything outside our
code, however green it is.

The same question decides every other nondeterministic collaborator — the clock, the random
source, identifier generation, iteration order. Each is either the subject of the claim, or
it is pinned; it is never left free while something else is being asserted. Pinning uses the
one mechanism the suite provides for that collaborator, per R12.

## R2 — Choose the level by what runs, and the file by what the claim addresses

*Answers: where does this test go?*
*From: the unit level has absorbed tests that start processes and tests that read documents, while the integration level stands empty.*

One module and nothing else is a unit test. Several modules in one process is an integration
test. A real process observed from outside is a system test. Directory placement follows
this answer; it never produces it.

The system level is not "the most layers combined" — it is **what can be observed from
outside the process**. A claim about several layers meeting, which is only visible from
inside, is an integration claim no matter how many layers it spans. Combination depth never
promotes a test to the system level.

One module has one test file. Two files addressing one module means one of them is really
addressed to something else and has not said so.

Within a level, placement is decided by **what the claim is addressed to**, and every test
falls into exactly one of four cases:

| The claim is addressed to… | The file | Named after |
| :--- | :--- | :--- |
| one module | mirrors that module | the module |
| several modules reached through one entry point | mirrors the entry point | the entry point, not the collaborators behind it |
| a norm the whole system must satisfy, with no entry point | one file per norm | the norm |
| a repository document or policy | outside the levels entirely (see R3) | the policy it guards |

A test with no single entry point and no norm is not a placement problem — it is a test
that has not decided what it claims. Split it until each part has an address.

This is what makes "where does this test already live?" answerable without searching.

## R3 — Guard a document by pinning its structure

*Answers: how do I encode a governance rule as a check?*
*From: this repository encodes governance in executable checks, and those checks currently assert exact sentences of the documents they guard.*

Such a check is a fitness function. It lives in its own location outside the test levels —
not inside any level's directory — and outside coverage, because it exercises no product
code. It asserts that a section, link or declaration **exists**, never that a particular
sentence is present.

Pinning prose freezes the document — improving the wording breaks the gate, so the wording
never improves. A guard that prevents what it protects from getting better is inverted.

## R4 — Commit a baseline that can fail the run

*Answers: may this eval exist, and is this an eval at all?*
*From: model quality is measured by a separate scoring layer whose baselines are recorded as human-readable reports.*

If an outcome is pass/fail without a threshold, it is a test and belongs in a tier. If it is
a rate or a score, it is an eval — and then it requires a baseline a run can be compared
against **mechanically**, plus a stated consequence when the comparison fails. A baseline
only a human reads is a report, not a gate.

With R1: an eval scored against a double measures the harness, not the subject.

A performance budget is a score with a threshold, so this applies unchanged: a budget with
no test that measures it is a target, not a budget, and must not be cited as an observed
property of the system.

## R5 — Make the suite prove it actually ran

*Answers: what does an empty scope, an unplayed recording or an unreached assertion mean?*
*From: P6 forbids silent substitution; the same failure mode appears in the suite as silent non-execution.*

They are failures. Reporting the unverified as verified is worse than failing, because a
failure is visible and this is green.

## R6 — Assert the change the turn left behind

*Answers: what may I assert about an agent turn?*
*From: every subject here produces natural language, and natural language is stochastic.*

Assert the change left behind — in a file, in session state, in the ontology graph, in the
event stream, on the screen. Anything a user is claimed to see is confirmed where the user
sees it.

This rules prose out; it does not choose the effect for you. Which change counts as evidence
for a given claim stays with the author (see the closing section).

## R7 — Wait on the event the system publishes

*Answers: how do I wait for an asynchronous result?*
*From: P1 forbids polling and busy-wait in the runtime; a suite that polls asserts a behaviour the runtime is not allowed to have.*

Wait by subscribing to the event the system publishes, or on an observable condition. Never
on elapsed time.

## R8 — Place a check where its result survives to the next stage

*Answers: which stage does this check run at?*
*From: verification runs only on a developer's machine, and branches reach the mainline as a squashed result that no verified commit ever had.*

A result proven on a branch may not survive combination with others. A check whose result
expires before the artifact merges belongs later in the sequence.

## R9 — Keep `main` runnable and experiment beside it

*Answers: where do I do exploratory work?*
*From: there is no staging environment; `main` is what gets demonstrated.*

Any commit of `main` runs as it is. Because it is the reference to compare against, it is
not where experiments happen: exploratory work runs in an isolated workspace, and that
workspace and `main` must be able to run **at the same time**.

## R10 — Take every coordinate from the environment

*Answers: which port, path or directory may I use?*
*From: many worktrees run on one machine at once.*

A fixed port, a fixed path, a shared directory: each makes concurrent execution impossible.
The working directory is a fixed coordinate too — anything loaded from the repository is
located relative to the repository, never to wherever the runner was invoked.

This binds local execution generally — dev servers included — not tests alone.

## R11 — Promote an ad hoc test only when no existing test observes the failure

*Answers: does this one-off check become permanent?*
*From: one-off checks have no home here, so they survive by being committed as unit tests.*

Promotion to a regression test is justified when **no existing test observes this failure**.
That is settled by looking: if the defect can be reintroduced and the suite stays green, the
test is promoted. If some test already turns red, it is not.

What is not promoted is deleted, not left in place.

## R12 — Extend the component that exists, and assert a norm through its one helper

*Answers: may I write my own fixture or my own way of asserting a rule?*
*From: the server-boot fixture exists in several copies, and the repository's own compliance rule is asserted by hand in dozens of places.*

Starting a server, finding a free port, waiting for readiness: implemented once, in the
setup module of the level that uses it. A fixture defined inside a test file is thereby
private to that file.

Where a rule is normative for the system, one shared helper asserts it, so that changing the
rule changes one place. Many hand-written variants of one assertion are many slightly
different definitions of it.

## R13 — Give a tier its selector before you give it tests

*Answers: is this new grouping a tier?*
*From: some tiers here spend tokens, and the ladder that gates merges is expressed as scope selectors.*

Running only that set, and excluding that set from another run, must both be mechanically
possible. A set distinguished only by folder or by prose is not a tier: it mixes into other
runs, and expensive tests ride along with cheap ones.

## R14 — Adopt a component everywhere it applies, or delete it

*Answers: what do I do with a helper nobody calls or a scope nothing populates?*
*From: helpers and scopes exist here for tiers that were never seeded.*

Each is a claim that the suite does something it does not do. Adopt it everywhere it
applies, or remove it.

## R15 — Watch a regression test fail before committing it

*Answers: is this test evidence, or just green?*
*From: verification runs only on the author's machine — there is no second party that will independently observe whether the test can fail.*

A test written to pin a defect is verified by reproducing the defect and watching that
specific test fail. Until then it is an assertion about nothing, and it will stay green
through the defect's return.

**Purge `__pycache__` and set `PYTHONDONTWRITEBYTECODE=1` before every mutation — both, not
either.** Otherwise the run you are watching may not be the mutation you applied. CPython
invalidates a cached `.pyc` on `(source mtime in whole seconds, source size)`; a
size-preserving mutation — a flipped comparison, a swapped string constant, an equal-length
regex — reverted and re-applied inside one wall-clock second matches both components, so the
previous mutation's bytecode runs and reports its failure set with no error and a plausible
result. **A correct `cmp -s` / `git diff --quiet` presence check does not detect this**: the
source is genuinely mutated; only the executed bytecode is stale, so this principle followed
perfectly still yields a wrong answer. Neither half of the remedy is redundant — the variable
suppresses *writing*, never *reading*, and a purge is a point-in-time act the next run undoes.
`./ucx` exports the variable; a bare `pytest` harness inherits neither and must set both.
Mechanism, measurements and executable demonstration:
Swarm Guide §6.9 *Stale bytecode*
and the stale-bytecode governance test (#482).

## R16 — Verify a UI claim at the shallowest level that can answer it

*Answers: component test, system test, or core test — and how many viewports?*
*From: P8 makes the GUI an observational testbench while the truth lives in the headless core, and the frontend has no test level of its own, so every UI claim currently escalates to a real browser.*

Logic and state belong to the core and are verified there. Rendering and visibility are
verified in a browser. Layout behaviour that changes with width is the only claim that
justifies repeating a test across viewports; repeating everything across viewports buys
nothing and costs the most expensive level.

## R17 — Keep each level inside its stated time budget

*Answers: may I add this test here?*
*From: the fast gate runs on every commit and is the only verification that exists; when it becomes slow it gets bypassed, and bypassing it is a single flag away.*

Each level carries a stated time budget. A test that pushes a level past its budget is
moved to a cheaper level, replaced by a cheaper claim, or admitted by explicitly raising
the budget — never by silently absorbing it.

## R18 — Retarget a test when its subject moves, or remove it

*Answers: what happens to existing tests when the model, schema or interface changes?*
*From: fixed probe batteries and scripted model outputs age against a moving target — a swapped model or a changed schema can leave a suite passing while measuring nothing.*

A test whose subject has moved is either updated to the new subject or removed. A suite
retained because deleting tests feels like losing coverage measures the past, and its green
result is the most expensive kind of false evidence.


## R19 — Measure coverage over the change, counting product code only

*Answers: what does the coverage requirement actually require?*
*From: the whole-repository figure sits far above the floor, so an entire new module can land untested while the gate stays green.*

The floor applies to the change as well as to the whole. A total that clears the floor says
nothing about the lines just added, and is not permitted to stand in for them.

What counts is product code only. Checks that exercise no product code — fitness functions,
evaluation harnesses — belong in neither the numerator nor the denominator; including them
raises the number without raising the verification.

## R20 — Write an absence test for every user-visible requirement

*Answers: what must exist after a UI change?*
*From: user-visible requirements have been deleted here by changes that each passed the gate, because every test asserted what was present and none asserted that it must be.*

For anything the product promises a user will see, some test fails when it stops being
there. Assertions that only describe what is currently rendered cannot detect removal; the
valuable assertion is the one that treats absence as failure.

This decides what a UI change owes: not "some E2E test", but a test that would have caught
the deletion.

## R21 — Test each external surface from that surface

*Answers: where is the boundary of an end-to-end test?*
*From: this product is used through three separate external surfaces — a CLI, an HTTP/UI surface, and the agent protocol — and only one of them is currently observed from outside.*

Each external surface is its own entry point. A claim proven at one surface is not evidence
about another, however much code they share underneath.

So "end-to-end" is not one suite. It is one suite per surface, and a surface with no
system-level test has no end-to-end coverage regardless of how well the code beneath it is
tested.

## R22 — Quarantine an intermittent test and find its cause

*Answers: what do I do with a test that fails sometimes?*
*From: verification is local and unattended by any second party, so a re-run that passes ends the investigation and the signal is lost.*

A test that fails intermittently has already reported something: either the subject is
nondeterministic, or the test left a coordinate free (R10) or waited on time (R7). Re-running
until green discards that report.

Until the cause is found, the test is removed from the gate explicitly, so that its absence
is visible. It is never left in place to fail occasionally.


## R23 — Read a degraded result as a failure unless provenance declares it

*Answers: the double returned an empty, default or fallback value — is that a pass?*
*From: P6 forbids a substituted result from standing in for a real one, and a test that accepts one re-introduces at verification time exactly what the runtime is forbidden to do.*

When a substitute returns nothing, a default, or a degraded result, the test asserts that the
result carries in-band `provenance` saying so. A run that produced a substituted value and no
declaration is asserted to fail.

The same reading applies to a model that emits a tool call as text while leaving the
structured call empty: assert that the turn fails, never that it succeeded.

## R24 — Build test data through a shared builder, stating only the fields the claim rests on

*Answers: do I write this model out in full, or build it?*
*From: the subjects here are strictly typed models with many required fields, so writing one out inline states many things the test does not mean, and one model change reaches every site that did.*

A builder supplies valid defaults; the test overrides exactly the fields its claim depends on.
Every value a test states is thereby part of its claim, and a reader can tell what matters by
what is written.

Where no builder exists for a subject a test needs, adding one is part of writing the test
(R12).

## R25 — Verify a contract you do not own against an artifact you did not write

*Answers: how do I test conformance to a format, a protocol, or a provider's payload?*
*From: session state and the ontology graph outlive the process that wrote them; the agent protocol is an external specification pinned at a version; provider payloads are authored by the provider. In all three the other side owns the contract.*

Writing and reading with the same code proves the two sides agree with each other, and stays
green when both move together. Conformance therefore needs a reference **this repository did
not produce**:

| Contract | Reference the test reads |
| :--- | :--- |
| a format persisted to disk | a sample committed in the older shape, read by today's code |
| an external protocol | an example carried by the pinned specification |
| a provider's response payload | a recorded exchange (see R1) |

Adding a field means adding a sample without it. Moving to a new specification version means
replacing the examples, not adjusting our own expectations to match our own output.

A test that exercises our types against our own interface definitions verifies internal
consistency. That is worth having, and it is not conformance — so it does not carry the
word.

## R26 — Verify a control by exercising what it must refuse

*Answers: how do I test an isolation level, a permission, or a boundary?*
*From: the sandbox exposes controls that some levels genuinely cannot enforce, and the threat model treats a control that is accepted and quietly ignored as the failure P6 names.*

The test drives the forbidden operation and asserts it is refused. Showing that permitted
operations succeed says nothing about the boundary.

Where a level cannot enforce a control, the test asserts that the level **says so** — that the
control is rejected or absent at that level, rather than accepted and discarded.


## R27 — Read attribution from the result, not from the trace

*Answers: the run recorded a failover, a retry or a degraded path — where do I assert it?*
*From: P6 accepts a declared failover only when it is attributable in band, and states that a telemetry span alone is not sufficient.*

Assert on the `provenance` carried by the result envelope. A span may be asserted when
telemetry itself is the subject of the claim; it is never the evidence for a claim about what
the system returned.

A test that reads a span to learn what happened would pass on a build that emits perfect
traces and returns substituted results — the exact failure P6 exists to prevent.


## R28 — Prove each claim once, by the cheapest means that proves it

*Answers: several tests could establish this — which do I write, and do I write one at all?*
*From: the gate runs on every commit and is the only verification that exists, and the tiers here are defined by what they cost, so a second proof of the same claim is not spare safety — it is a bill paid on every run and every refactor.*

If a test already proves the claim, do not add another. Extend the existing one if it is
close, and otherwise leave it alone: two tests that go red together on the same defect carry
one signal and two maintenance costs.

When more than one method would prove the claim, take the cheapest one that actually proves
it. Cost here runs in a fixed order — a real model is dearer than a real process, a real
process dearer than modules combined in one process, and that dearer than a module alone.
"Cheapest that proves it" is the whole rule: a cheaper method that proves something adjacent
proves nothing, and R1 decides which methods are even eligible.

This never licenses deleting a claim that nothing else makes. An absence test (R20), a
refusal test (R26) and a conformance reference (R25) each assert something no ordinary test
asserts, so they are not duplicates of the tests they sit beside.

---

## What these principles do not decide

A principle earns its place by producing a single answer for a concrete case. The following
are genuine decisions that remain with the test author on every test, and no rule above
settles them:

| Decision | What the principles do constrain |
| :--- | :--- |
| What is left untested, and why | nothing — state it where the change is reviewed |
| Which failure modes are worth reproducing | R1 says a chosen failure mode must be producible on demand |
| Which effect counts as evidence for this claim | R6 rules out prose; it does not pick the effect |
| Which entry point a multi-module claim is addressed to | R2 fixes the naming once the entry point is identified |
| What value a baseline is set to | R4 requires a baseline that can fail; not its value |
| Whether exceeding a level's budget is worth it | R17 requires the decision to be explicit, not silent |
| Which metric expresses a component's quality at all | R4 requires that whatever is chosen can fail |
| Whether a surface is worth an end-to-end suite | R21 fixes what such a suite must observe, not whether to build one |

Two candidates were considered and rejected, so they are not re-proposed:

* **What a failure message must say.** The runner already reports observed and expected
  values, so a rule here would restate the tooling rather than decide anything.
* **What a reviewer checks.** The principles above are already the checklist; a separate
  review rule would decide nothing that R1–R28 does not.
* **How to test an evolving ontology, and how to test hot-reloaded skills.** Both were
  examined and both are already decided: an accumulated graph is built explicitly by the test
  like any other subject (R24), a graph whose shape has legitimately moved is retargeted
  (R18), and a skill loaded from disk at runtime takes its location from the environment
  (R10) with the synthesising model substituted per R1.

A rule that only frames one of these belongs here, in this table — not above it as a
principle.

---

## How this document grows

It is not finished, and it is not meant to be completed in one sitting. It is completed by
writing tests. The procedure is part of the document, so that a gap becomes a change rather
than a private workaround:

1. **When a principle answers, follow it** — including when the answer is inconvenient. An
   inconvenient answer is the only kind that carries information.
2. **When a principle answers and the answer is unreasonable, change the principle — in the
   same change as the test.** Never take a local exception. An exception decides nothing on
   the next case and leaves the next author facing the same question with less information,
   whereas an amendment carries the case that forced it.

   The amendment states the concrete case, the answer the old wording produced, and the new
   wording. It lands the way any change to this document lands; no separate approval gates
   it. Amending is expected — a principle that has never been rewritten has probably never
   been tested against a hard case.

3. **When no principle answers, decide and record the decision with the test.** One
   occurrence is a case, not a rule, and it stays with the test.
4. **When the same unanswered question appears a second time, it becomes a candidate.** Two
   occurrences are a pattern; the second author should not have to re-derive the first
   author's reasoning.
5. **A candidate is admitted only if it produces a single answer for a concrete case.**
   Otherwise it belongs in *What these principles do not decide*, which is a real place in
   this document and not a rejection.

A principle admitted this way arrives with the case that forced it, which is what keeps this
document from filling up with rules nobody needed.

