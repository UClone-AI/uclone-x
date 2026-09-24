# NFR: Performance Budgets

This document is the single source for UClone-X's numeric performance targets.
It exists because `2026-09-02-009`
found the same figure (`< 0.05ms`) attached to three different quantities —
turn latency, dispatch latency, and ontology validation latency — across
`core-principles.md`, `local-collaboration-engine.md`,
`agent-ontology-architecture.md`, and the PRD, with the turn-latency framing
being physically unreachable (a turn includes an LLM round-trip of 10²–10³ ms).

## Status of the figures in this document

**These budgets are revisable without a principle amendment.** They are
hardware- and workload-dependent tuning values, not law. What is immutable
(per [P3](principles/details/p3-single-machine-acceleration.md)) is the
*requirement* that single-machine dispatch never traverses an out-of-process
broker — not any number attached to how fast that dispatch is. Changing a
number in this table is a normal documentation/config edit; changing the
zero-broker requirement itself is a principle amendment under
[`docs/governance/principle-amendment-policy.md`](governance/principle-amendment-policy.md).

**A figure with no stated measurement point is not a budget.** Every row
below must name what is measured, where measurement starts and ends, and how
it is checked — a bare number ("< 0.05ms") with no measurement point, as
appeared in the documents above, does not belong in this table and should be
treated as a defect if reintroduced.

## Two axes the issue conflates — kept separate here

| Axis | What it bounds | Dominated by |
| :--- | :--- | :--- |
| **Dispatch overhead** | Time for the in-process event bus to move one event from publisher to subscriber wake-up. | Python/asyncio scheduling, object copy, queue operations. No network, no model call. |
| **Agent turn latency** | Time for a full turn: event ingest → agent processing (including any LLM call) → final emitted event. | The LLM provider round-trip (10²–10³ ms), which the framework does not control. |

A single number cannot bound both. The budgets below are named and scoped
separately so a benchmark against one is never read as a claim about the
other.

## Budget table

| Budget | Measured | Measurement point | Target | Verification |
| :--- | :--- | :--- | :--- | :--- |
| `dispatch_latency` | In-process `EventBus` publish-to-delivery time: one `publish()`/`publish_nowait()` call to the matching subscriber's `deliver()` returning, single process, no sandbox, no network. | Wall-clock delta bracketing `EventBus.publish()` → `EventSubscription.deliver()` for one event, on a single subscriber, warm event loop. | **Sub-millisecond (target: < 1 ms p99)** on standard Apple Silicon / Linux x86-64. **Unmeasured — target only.** No benchmark exists in this repository as of this writing; `src/uclone_x/engine/event_bus.py` and `tests/unit/test_event_bus.py` contain correctness tests (ordering, backpressure, pub/sub fan-out) but no timing assertions. | To be verified by a benchmark under `tests/` (or a `benchmarks/` suite) that asserts a p50/p99 threshold on CI hardware, with the hardware and Python version recorded alongside the result. Until such a benchmark exists, this row must not be cited as an observed figure. |
| `turn_latency_framework_overhead` | The framework's *own* added latency within one turn — event ingest, sub-agent dispatch/routing, ontology validation, tool-call marshalling, response emission — **excluding** time spent waiting on an LLM provider or an external tool/network call. | Wall-clock delta for one turn with the LLM call point stubbed/mocked out (i.e., provider latency subtracted or replaced with a fixed no-op), so only UClone-X's own code is timed. | **Unmeasured — target only.** No number is asserted here pending a benchmark; see Open Items below. | Same as above: a benchmark that mocks the LLM call and times everything else, with the mock's own overhead documented so it isn't miscounted as framework cost. |
| `turn_latency_end_to_end` | Full turn including the LLM provider round-trip. | Event ingest to final emitted event, model time included. | **Not a UClone-X performance budget.** This number is provider- and prompt-dependent (typically 10²–10³ ms or more) and is explicitly out of scope for a framework-overhead target. Track it as an observability metric (see `docs/telemetry-opentelemetry.md`) for operational visibility, never as a pass/fail gate on the framework itself. | Observed via OpenTelemetry span duration in production/dev traces; not a CI-gated assertion. |
| `ontology_validation_latency` | A single ontology invariant check (the "Tier-1 validation" referenced in `docs/agent-ontology-architecture.md`). | Wall-clock delta around one invariant-check call. | **Unmeasured — target only.** The `0.01ms` figure previously attributed to "Principle 3 (Microsecond Single-Machine Execution)" in `agent-ontology-architecture.md` is corrected here: P3 has no such name, and this quantity is validation latency, not dispatch or turn latency. No benchmark exists yet. | To be verified by a benchmark colocated with the ontology validation code once it exists. |

| `tool_result_inline_floor` (**T0**) | The result size at or below which a tool result stays inline verbatim, in this turn and in every later turn. Below it, referencing costs more than it saves: the reference's own id, preview, size and retrieval instructions can exceed the content, and following it spends a step from the P4 `max_turns` budget. | Estimated tokens of the formatted tool result, via `ContextCompactor.estimate_tokens`, measured at the `POST_TOOL_USE` boundary before the result reaches the surface. | **Unset — no figure is asserted.** The threshold is named here so the design can reference it before it is chosen; see the context-assembly design note. Choosing it requires measuring the reference's own token cost against the retrieval round trip. | To be set by measurement: render the reference notice for a range of result sizes, compare its token count plus one retrieval turn against the inline content, and take the crossover. Record the notice template used, since the answer moves with it. |
| `tool_result_verbatim_ceiling` (**T1**) | The largest result admitted verbatim into the turn that requested it. Recency admits a result; it does not exempt it from the turn's context budget. | Same measurement point and unit as **T0**. | **Unset — no figure is asserted.** Necessarily a function of the model's context window and the budget split across context layers, so it is a ratio to resolve per deployment rather than one constant. | To be set once the per-layer budget split exists. A verbatim result that pushes the assembled context past its ceiling is the failure this bounds. |
| `tool_result_structural_preview_floor` (**T2**) | The size above which a head-and-tail preview stops being informative and the preview becomes **structural** instead — a symbol outline, a key schema, a column list — selected by the result's declared kind. | Same measurement point and unit as **T0**. | **Unset — no figure is asserted.** Unlike T0 and T1 this is a legibility threshold rather than a capacity one: on a 200,000-line file the first and last five lines carry no information, and the crossover is a property of the content kind, so it may resolve to a per-kind value rather than one number. | To be set by evaluation rather than by benchmark: measure whether a model given a structural outline answers questions about the content more accurately than one given head and tail, at several sizes. The existing `evals/` machinery is the place for it. |

## What is actually true today, grounded in code

`src/uclone_x/engine/event_bus.py` implements `EventBus` as a pure in-process
mechanism:

* Ingress is `asyncio.PriorityQueue` (in-memory); delivery to subscribers is
  also via per-subscriber `asyncio.PriorityQueue` instances (`EventSubscription`).
* There is no import of, or dependency on, any external broker client
  (no Redis/RabbitMQ/AMQP/Kafka client in this module). `publish()` /
  `publish_nowait()` enqueue directly; the background `_dispatch_loop()` pulls
  from the ingress queue and calls `deliver()` on matching subscriptions and
  registered callbacks in-process.
* This satisfies the **structural** P3 requirement (no out-of-process broker
  on the single-machine path) by inspection of the code, independent of any
  timing number.

What is **not** yet true today: there is no benchmark in `tests/unit/test_event_bus.py`
(or anywhere else in the repo at this writing) that measures publish-to-delivery
wall-clock time. `test_event_bus.py` covers priority ordering, multi-subscriber
fan-out, backpressure policies, and clean shutdown — all correctness tests, zero
timing assertions. Any dispatch-latency number quoted before such a benchmark
exists is a target, not an observation, and must be labeled as such.

## Open items

* No dispatch-latency benchmark exists yet. Until one is added, `dispatch_latency`
  and `turn_latency_framework_overhead` remain **Unmeasured — target only** and
  must not be cited as measured results in any document, PRD table, or
  marketing/comparison claim (see `2026-09-02-017`
  for the related unfalsifiable-benchmark finding).
* The three tool-result thresholds **T0**, **T1** and **T2** are declared with no figures.
  This is deliberate — the design that consumes them
  (the context-assembly design note)
  needs to name them before they can be chosen, and a placeholder number would be cited as a
  decision. They must not be treated as set, and no code should hardcode a value for them
  without adding the measurement described in their rows. T0 and T1 are measurements; T2 is
  an evaluation.
* `docs/local-collaboration-engine.md`, `docs/PRD.md`, and
  `docs/agent-ontology-architecture.md` still contain the pre-existing
  `< 0.05ms` / `0.01ms` figures this document was scoped to replace in
  principle text; those files are outside this task's edit scope and are
  flagged for follow-up rather than edited here.

## Related

* [P3: Single-Machine Acceleration & Pluggable Sandboxing](principles/details/p3-single-machine-acceleration.md)
* `2026-09-02-009` — Impossible and mutually inconsistent latency targets
* `2026-09-02-017` — Unfalsifiable benchmarks
* [`docs/governance/principle-amendment-policy.md`](governance/principle-amendment-policy.md)
