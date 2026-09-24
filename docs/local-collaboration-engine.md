# Local Collaboration Engine Specification

## 1. Motivation: Single-Machine High-Throughput Collaboration

Distributed message brokers (Kafka, RabbitMQ, Redis) introduce network hops, serialization overhead, and operational complexity. While necessary for multi-node deployments, multi-agent development and edge executions thrive on **co-located, single-machine execution**.

Inspired by Google Antigravity, the **UClone-X Local Collaboration Engine** is optimized for ultra-fast, in-memory concurrency, zero-overhead message passing, and reactive wakeups on a single host.

---

## 2. Core Concurrency Architecture

```mermaid
flowchart TD
    subgraph HostEngine["Single-Machine In-Memory Engine"]
        Bus["Zero-Copy Event Bus (Asyncio / In-Process Channels)"]
        Scheduler["Reactive Work Scheduler (Worker Pool)"]
        StateRegistry["Active Agents & Context Registry"]
        TimerService["High-Precision Timer & Cron Engine"]
    end

    subgraph Agents["Co-located Agents (In-Process Coroutines / Workers)"]
        LeadAgent["Agent: Lead Architect"]
        Worker1["Agent: Backend Developer"]
        Worker2["Agent: QA Reviewer"]
    end

    LeadAgent -->|Publish Task Event (Zero Copy)| Bus
    Bus -->|Instant Wakeup Notify| Scheduler
    Scheduler -->|Dispatch Event| Worker1
    Scheduler -->|Dispatch Event| Worker2
    Worker1 -->|Stream Tool Calls / Output| Bus
    Worker2 -->|Stream Validation Result| Bus
    TimerService -->|Timeout / Reminder Tick| Bus
```

---

## 3. Key Technical Pillars

### 3.1 Zero-Copy In-Process Event Dispatch
- Events passed between co-located agents bypass JSON serialization and network sockets.
- Employs lock-free ring buffers or `asyncio.Queue` / channel structures, achieving sub-millisecond turn dispatching latency.

### 3.2 Reactive Task Scheduler (No Polling)
- Agents register interest in specific topics or conversation turns.
- When an agent emits a background task, the agent immediately yields.
- The scheduler wakes up the listening coroutine via native notification primitives (`asyncio.Event` / signal) as soon as data is ready.

### 3.3 Selective & Pluggable Sandboxing
- **`isolation_level="none"` (Direct Host)**: Zero virtualization overhead, bare-metal process execution for ultra-fast local iteration.
- **`isolation_level="workspace"` (Worktree/Dir Sandbox)**: Restricts edits to isolated git branches / temporary worktrees, preventing collisions.
- **`isolation_level="container" | "wasm"` (Strict Sandbox)**: Runs destructive or untrusted code in ephemeral isolated micro-containers.


---

## 4. Performance Benchmarks (Target Specifications)

| Metric | Traditional Broker (Redis/HTTP) | UClone-X Local Collaboration Engine | Improvement |
|---|---|---|---|
| **Inter-Agent Event Latency** | 2.5 ms – 8.0 ms | **< 0.05 ms (In-Memory)** | ~50x – 100x Faster |
| **Throughput (Events/sec)** | 10,000 / sec | **> 500,000 / sec** | 50x Higher |
| **Memory Footprint per Idle Agent** | ~45 MB (Process overhead) | **< 500 KB (Coroutine/State)** | 90% Less RAM |
| **External Dependencies** | Redis / RabbitMQ / DB required | **Zero (Embedded Pure Runtime)** | Self-contained |
