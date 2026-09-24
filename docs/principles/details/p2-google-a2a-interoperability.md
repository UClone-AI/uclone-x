# Principle 2: A2A Dual-Transport

## Detailed Specification & Rationale

* **Core Law**: All external Agent-to-Agent (A2A) interfaces must adhere to the open **A2A (Agent2Agent) Specification** — an Agentic AI Foundation / Linux Foundation project — implemented via an optimized Dual-Transport model.
* **Strict Rule**: 
  - Cross-agent communication must logically conform to the A2A specification at the version and in the binding pinned by [`docs/a2a-protocol-spec.md`](../../a2a-protocol-spec.md). Concrete discovery paths, endpoints, method names and object shapes are defined there and MUST NOT be restated in this principle.
  - **In-Memory Fastpath**: For co-located agents on the same machine, transport uses zero-copy direct memory dispatch without network or serialization overhead.
  - **Remote Wire Protocol**: For distributed or remote agents, transport uses a standard A2A protocol binding as pinned by the specification document. WebSocket is not an A2A binding.
* **Why**: Prevents framework vendor lock-in while keeping co-located agent interactions off the network path entirely. Numeric budgets live in [`docs/nfr-performance-budgets.md`](../../nfr-performance-budgets.md), not in this law.
* **Implementation Reference**: [`docs/a2a-protocol-spec.md`](../../a2a-protocol-spec.md)
