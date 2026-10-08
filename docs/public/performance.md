# Performance budgets

The principles fix no latency number. P3 requires that dispatch on one machine never
passes through an out-of-process broker; how fast that dispatch is belongs here, and a
figure on this page can change without a principle amendment.

A budget names what is measured, where the measurement starts and ends, and how it is
checked. A bare number is not a budget.

| Budget | What it bounds | Target | State |
| :--- | :--- | :--- | :--- |
| Dispatch overhead | One event from `EventBus.publish()` to the subscriber's `deliver()` returning: one process, warm event loop, no sandbox, no network | Under 1 ms at p99 | **Unmeasured.** No benchmark exists yet, so this is a target, not an observation |
| Framework overhead per turn | UClone-X's own time in one turn — ingest, routing, tool-call marshalling, emission — with the model call stubbed out | None set | Unmeasured |
| Ontology validation | One invariant check | None set | Unmeasured |
| Turn latency end to end | A whole turn including the model's round trip | Not a budget | Depends on the provider and the prompt; observe it through tracing, never gate on it |

What holds today without a timing number: the event bus in
[`engine/event_bus.py`](../../src/uclone_x/engine/event_bus.py) is in-process asyncio
queues with no broker client, which is P3's structural requirement.
