# Principle 1: Reactive Event-Driven Loop

## Detailed Specification & Rationale

* **Core Law**: Agents and subsystems must operate as reactive event-driven state machines.
* **Strict Rule**: No component may use blocking sleep/busy-wait loops (e.g., `while True: time.sleep(1)`). When an agent awaits external tools, sub-agent completion, or user input, it **must** yield control and register a reactive listener on the event bus.
* **Why**: Polling wastes CPU cycles, degrades multi-agent concurrency, and breaks scalability on single machines. Reactive coroutines yield instantly and resume via event notifications.
* **Implementation Reference**: `docs/event-driven-agent-core.md`
