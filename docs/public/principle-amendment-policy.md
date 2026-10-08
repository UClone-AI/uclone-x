# Principle Amendment Policy

`docs/principles/` encodes the inviolable laws of UClone-X. This document is the
**only** place that says who may change them and how. It exists because an approval
mandate with no recorded procedure is unenforceable: a prohibition that names no
route, no approver and no record is prose, and prose does not stop an edit.

This is the public statement of the procedure. The amendment log — every edit made to
`docs/principles/`, with its issue, route and author — is kept in the development
repository, where the identifiers it references resolve.

---

## 1. Enforcement model: versioned principles, not decorative prose

The repository does not claim its principles are unchangeable. It claims they are
**versioned, and expensive to change**:

* **Versioned principles over decorative prose.** "Immutable / never change" is
  replaced by a stated amendment procedure and a version history.
* **Stability through visible cost.** Immutability here means governance stability
  backed by auditable costs — a filed issue, a linked task, a named commit, a
  proposal record, and an append-only log entry — rather than an unbacked
  prohibition.
* **A complete audit trail.** Every edit to `docs/principles/`, including those made
  before this policy existed, is registered in the amendment log.

The distinction matters because the first version of this rule was the prose kind. It
said principles must never change without approval, named no procedure, and was
therefore satisfied by anyone who simply edited one. A rule nobody can violate
*procedurally* is not a rule.

---

## 2. The 2026-09 waiver, and its expiry

A time-boxed waiver granted on **2026-09-02** allowed direct amendment of
`docs/principles/` for one week, without a prior approval request per amendment, so
that approval-blocked findings could be worked while the specification was still
stabilising. It **lapsed on 2026-09-09**, and §3 has been in effect since. A waiver is
not extended by silence: ending it required no action, and renewing it would have
required the owner to say so in this document.

Two things the waiver never covered, which are worth stating because they are the
cases people assume a waiver reaches:

* **Deleting a principle, or hollowing one out.** Dropping a requirement, or
  weakening it until nothing is required, is a normative change wearing the shape of a
  cleanup. That was Tier A during the waiver and is Tier A now.
* **Adding a principle**, and **changing this policy**. The document that says who may
  amend the principles was never amendable by the people it binds.

Even under the waiver, traceability still bound every edit: a filed issue first, a
task carrying it, the identifier in the commit subject, and a row in the amendment
log — the log itself is kept in the development repository (§5).

---

## 3. The two routes, effective 2026-09-09

Every amendment travels one of two routes, and the classification is not the
author's convenience.

### Tier A — a normative change

A change to **what a principle requires**: adding or removing a requirement,
weakening or strengthening one, changing a mandated default, or adding a new
principle.

* Requires **explicit written approval from the project owner, before the edit is
  made**.
* The approval must be **attributable to the owner's own account** — a review, an
  issue comment, a pull-request approval. A contributor cannot record an approval on
  the owner's behalf, and a claim of approval with no attributable evidence is not an
  approval. This repository has filed findings against itself for exactly that shape
  of record.
* Requires a proposal document stating rationale, alternatives considered, and
  sign-off.

### Tier B — a non-normative repair

A change that leaves the requirement intact and repairs how it is stated:

* resolving a contradiction between two principle documents,
* correcting a factual or reference error,
* supplying a definition the text already relies on,
* moving a tuning constant out of the law into a configuration or budget document
  (the *requirement* stays; the *number* moves).

Route: the contributor writes a complete, ready-to-merge proposal containing the
exact current text, the exact proposed text, the issue it resolves, and why the
change is non-normative. **The contributor does not touch `docs/principles/`.** The
owner's decision is then a single merge rather than a design conversation — which is
the point: it converts "blocked until a human designs it" into "blocked until a human
approves it".

**If you cannot tell whether a change is Tier A or Tier B, it is Tier A.**

---

## 4. Versioning and proposal format

The core principles reference ([`../principles/core-principles.md`](../principles/core-principles.md))
and the detail specifications under `../principles/details/`
are versioned documents.

* **Major / minor revision** — a Tier A normative change, addition, or scope change.
* **Patch revision** — a Tier B clarification, reference update, or typo correction.

A version bump links to the issue that prompted it, the change that carried it, the
proposal record, and a row in the amendment log.

A proposal contains:

1. **Header** — target principle, classification (Tier A or Tier B), author, approver.
2. **Problem statement** — the ambiguity, contradiction, or gap that necessitates the change.
3. **Exact text diff** — current principle text against proposed text.
4. **Architectural and falsifiability impact** — how the new text stays falsifiable, and which system boundaries move.
5. **Alternatives considered** — other formulations, and why they were rejected.

Point 4 is not ceremony. A principle that cannot be contradicted by an observation
cannot be verified either, and several of this repository's own findings are about
principle text that read as law while asserting nothing checkable.

---

## 5. Amendment log

Every edit to `docs/principles/` gets a row: the date, the principle, what changed and
why, the issue that prompted it, the route it travelled, and the author. The log is
append-only.

It is kept in the development repository rather than here, because each row references
identifiers — issues, tasks, commits — that do not resolve outside it. A row is
evidence only where the things it cites can be read, and a log of dangling references
would look like an audit trail without being one.

---

## 6. Related

* [`../principles/core-principles.md`](../principles/core-principles.md) — the principles this policy governs
* [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md) — how patches reach this repository
