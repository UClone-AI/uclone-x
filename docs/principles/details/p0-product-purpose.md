# Principle 0: Product Purpose — Non-Expert Operability & Extension Points

## Detailed Specification & Rationale

* **Core Law**: UClone-X ships as an installable product for people who do not read its code. The Core is headless and portable to any head — CLI, desktop, cloud, mobile — and the default repository ships a desktop head together with a first-run path that requires no prior knowledge of the system. **Ease comes from complete defaults, not from a reduced feature set; extensibility comes from declared extension points the Core actually consumes, not from Core edits.**

* **Strict Rule**:
  - **Complete Default Composition**: The default configuration is a finished, opinionated product, not a scaffold to be assembled. A first-time user reaches a working agent without editing a configuration file, resolving a dependency, or reading architecture documentation. Installation either succeeds or fails naming its remediation — **a partial success is a failure** (P6 applied to installation).
  - **Head-Agnostic Core**: The Core exposes one contract that every head consumes. No head-specific logic in the Core; no Core logic in a head. Removing any head leaves the Core running and its sessions resumable.
  - **Shipped Desktop Head**: The default repository contains a desktop head a non-technical user can run. It is a product surface rather than a testbench, and it holds no primary state (P8).
  - **Declared, Consumed Extension Points**: Customisation happens at extension points the Core **declares, documents, and actually consumes**, and requires modest expertise — not Core source edits and not a fork. **An extension point the Core does not consume is not an extension point**: declaring one without wiring it violates this rule rather than deferring it. **The set is open.** This law constrains an extension point's *properties*, never its membership: new extension points are added as the product needs them, and doing so requires no amendment here. The set in force at any time is recorded in the implementation references below, not in this text — a law that fixes the list would make extending the harness a governance action, which is the opposite of what this principle exists to protect.
  - **Falsifiability**: Verified by a first-run measurement — a clean machine, no prior setup, timed to the first successful agent turn. Unmeasured, this principle is a claim and not a law. `2026-09-02-022` is the same failure in miniature: "Fast Resolution" was written as law without a way to fail it.

* **Precedence — what P0 being first does and does not mean**:
  - **Every principle binds by default.** Conformance to all of P0–P9 is the normal case and the expectation. P0 is not a budget to be spent against the others.
  - **Precedence applies only to a discovered contradiction** — a case where no design satisfies both principles. It is a resolution rule, not a trade-off licence. "This would be easier for a beginner" is not a contradiction with P6; it is a design constraint to be solved within P6.
  - **On a genuine contradiction, P0 is evaluated first and decides.**
  - **A discovered contradiction is itself a finding.** File it in `docs/issues/` naming both principles, so the conflict is repaired in the text rather than silently resolved in P0's favour on every encounter. A precedence rule invoked repeatedly for the same pair is a defect report that nobody wrote.

* **Why**: P1–P9 govern the runtime's internals, and every one of them is satisfiable by a system that only its own authors can install and run. Without a stated purpose at the head of the list, "who this is for" is decided implicitly, one commit at a time, by whoever is optimising the gate that week. Placing it first makes the audience a design input rather than an afterthought — while the precedence rule above keeps it from becoming a warrant to disable the safety laws that are *how* a non-expert's trust is earned.

* **Implementation Reference**: [`docs/cli-specification.md`](../../cli-specification.md), [`docs/ui-dashboard-architecture.md`](../../ui-dashboard-architecture.md), [`docs/skill-system-architecture.md`](../../skill-system-architecture.md)
