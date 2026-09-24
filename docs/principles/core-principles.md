# UClone-X Core Principles (Compact Reference)

> [!IMPORTANT]
> **SOLE NORMATIVE SOURCE**: This document (`docs/principles/core-principles.md`) along with its companion detail specifications under `details/` is the **sole normative text** for the ten core principles (P0–P9) of UClone-X. All other documents (`AGENTS.md`, `docs/PRD.md`, `README.md`, `CLAUDE.md`, `GEMINI.md`, etc.) are non-normative references or navigation and MUST link directly here without duplicating or paraphrasing principle rules.

> [!CAUTION]
> **IMMUTABILITY MANDATE**: No principle here may be altered or bypassed without **explicit, direct written approval from the project owner** per [`docs/governance/principle-amendment-policy.md`](../governance/principle-amendment-policy.md).
> 
> **BUILDER vs. ucx agent SEPARATION**: `Builder` (AI coding swarms building this codebase in Worktrees) must never confuse their own prompts/identities with `ucx agent` (`uclone_x.agent.BaseAgent`) executing inside the framework runtime. Keep framework abstractions pure and generic.


---

### 🔒 The 10 Inviolable Laws of UClone-X

**[P0: Product Purpose — Non-Expert Operability & Extension Points](details/p0-product-purpose.md)**: UClone-X ships as an installable product for people who do not read its code. Headless, head-portable Core (CLI / desktop / cloud / mobile); the default repository ships a desktop head and a first-run path needing no prior knowledge of the system. Ease comes from **complete defaults**, extensibility from extension points the Core **declares, documents and actually consumes** — currently skills (P9), MCP and tools, and lifecycle hooks, though **the set is open and adding to it needs no amendment**. An extension point the Core does not consume is not an extension point.

> **Precedence.** Every principle binds by default; conformance to all of P0-P9 is the expectation, and P0 is not a budget to be spent against the others. Precedence applies **only to a discovered contradiction** - a case where no design satisfies both principles. There, P0 is evaluated first and decides. A contradiction found this way is itself a finding: file it in `docs/issues/` naming both principles, so the conflict is repaired in the text rather than silently resolved on every encounter.

The nine laws below are ordered by number, not by precedence; they bind equally.


1. **[P1: Reactive Event-Driven Loop](details/p1-reactive-event-driven.md)**: Agents are non-blocking state machines. Never use busy-wait/polling sleep loops.
2. **[P2: A2A Dual-Transport](details/p2-google-a2a-interoperability.md)**: Logical conformance to the A2A (Agent2Agent) Specification — an Agentic AI Foundation project — at the version pinned by [`docs/a2a-protocol-spec.md`](../a2a-protocol-spec.md); zero-copy in-memory fastpath locally, a standard A2A binding remotely.
3. **[P3: Single-Machine Acceleration & Pluggable Sandboxing](details/p3-single-machine-acceleration.md)**: Zero-broker dispatch — in-process only, never an out-of-process broker on the single-machine path; numeric budgets live in [`docs/nfr-performance-budgets.md`](../nfr-performance-budgets.md), not in this law. Sandbox `isolation_level` defaults to `workspace`; `container`/`wasm` by user configuration; `none` only as an explicit opt-in.
4. **[P4: Dynamic Specialization & Bounded Execution](details/p4-dynamic-specialization.md)**: Solve tasks with no wasted agent steps, under explicit step budgets (`max_turns`) and token ceilings — *no wasted steps*, never *the fewest steps regardless of answer quality*; delegate to ephemeral sub-agents with clean context isolation. Background knowledge and skill enrichment (P7, P9) operates strictly off the critical response path.
5. **[P5: LLM Provider Agnosticism, Token Management & Auto-Compaction](details/p5-llm-token-management.md)**: Core agent, tool and orchestration code depends only on a provider-neutral interface — no provider SDK import or provider-native type outside `uclone_x.llm.connectors.*`. Centralized per-provider quota/budget control; automatic context pruning on thresholds.
6. **[P6: Fail-Fast & Zero Silent Fallbacks](details/p6-fail-fast-observability.md)**: Substituted mock, empty or default results are forbidden unconditionally. A declared retry or provider failover is permitted only when it is attributable in-band, via `provenance` on the result envelope — telemetry alone is not sufficient.
7. **[P7: Evolving Ontology Grounding](details/p7-evolving-ontology.md)**: Self-constructed through experience + human-guided; decisions semantically grounded.
8. **[P8: Strict Typing, Language Boundaries & Core-First Headless Autonomy](details/p8-strict-typing-boundaries.md)**: Python 3.11+ Core (`pyright --strict`, Pydantic v2); TypeScript React 19 GUI. Verification is **local** — `./ucx test check` on the developer's machine is the verification; the published repository's CI repeats it for contributions and does not replace it. All business logic, session state and conversation persistence reside strictly in the headless Core; heads hold no primary state ([P0](details/p0-product-purpose.md)).
9. **[P9: Self-Evolving & Pluggable Skills](details/p9-dynamic-skills.md)**: Autonomous skill synthesis (`SKILL.md`) + developer injection with hot-reloading.
