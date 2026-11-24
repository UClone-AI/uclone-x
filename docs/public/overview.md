# Architecture overview

UClone-X is one Python package, `src/uclone_x/`, and one React
dashboard, `frontend/`. The Python package is the runtime: it holds
every agent, conversation and setting. The dashboard, the `ucx` command line and the
protocol servers are *heads* over it, and hold no state of their own.

This page describes the code as it is. Where a subsystem is only partly built, it says so.

## Layers

The package is split by one question — *does this module touch the outside world?* — and
imports only point downward.

| Layer | What it holds | Examples |
| :--- | :--- | :--- |
| **Shell** | The process, configuration and every inbound surface | `ui/` (the dashboard's FastAPI backend), `cli/` (`ucx`), `shells/` (`reconciler.py` for `ucx up`, `gateway_server.py` for the unified Model Gateway, stateless cluster worker, A2A and ACP servers) |
| **Adapter** | Implementations of a kernel protocol against something real: a provider, a file, a subprocess, or cluster stores | `llm/connectors/` (including `gateway_client.py`), `adapters/cluster/` (Kafka, Valkey, PostgreSQL, Centrifugo), `tools/builtin/`, `memory/store.py`, `telemetry/` exporters, `code_intel/` |
| **Kernel** | Pure Python: contracts, models and decisions | `agent/`, `room/`, `engine/`, `core/`, `llm/`, `tools/`, `sandbox/`, `memory/`, `skills/`, `ontology/`, `a2a/`, `acp/` |

No kernel or adapter module imports a shell module. The dashboard reaches the runtime only
over HTTP, SSE and WebSocket.

### Two consumption tracks over one Core

* **Declarative stateless cluster stack (`ucx up`)** — `shells/reconciler.py` converges
  Docker Compose infrastructure (`kafka`, `postgres`, `valkey-1..6`, `centrifugo`) together
  with supervised local daemons (`model_gateway`, stateless `agent_workers`, and
  `web_ui_server`). Workers and the UI server hold no sticky in-process session state and
  send every LLM request through the standalone **Model Gateway** (`GatewayModelClient`),
  which isolates provider API keys and optional remote-GPU SSH tunnels in one process.
* **Zero-broker Python Agent Kernel SDK (`pip install uclone-x`) & headless CLI** —
  importing `uclone_x` (`compose_agent`) or running `ucx run` / `ucx acp serve` executes
  `BaseAgent` turns directly in memory over `engine/event_bus.py` with zero external
  brokers or Docker containers required.

## One agent

An agent is a `BaseAgent` ([`agent/base.py`](../../src/uclone_x/agent/base.py)), a
non-blocking state machine:

```text
IDLE → INGESTING → REASONING ⇄ CALLING_TOOL → EMITTING_RESPONSE → IDLE
                       └──→ AWAITING_INPUT (a question back to the user)
```

plus the terminal states `ERROR` and `TERMINATED`. An agent waits on events, never on a
polling loop: it wakes when a message arrives, a tool returns or it is interrupted. Each
turn runs under a step budget (`max_turns`) and a token ceiling.

Every agent — a terminal session, a dashboard conversation, a seat in a group chat, a peer
called over A2A — is built by one composition root,
[`agent/composition.py`](../../src/uclone_x/agent/composition.py). A *clone* is an agent
built from a persona file plus its avatar, memory and tool scope; the six built-in clones
are YAML files in `src/uclone_x/personas/`.

## Many agents

There are two ways agents work together, and they are different things:

| Path | Where | Shape |
| :--- | :--- | :--- |
| Room | `room/` | Several clones in one shared conversation. An orchestrator decides who speaks next; each clone keeps its own session |
| Sub-agent / A2A | `tools/builtin/subagent.py`, `tools/builtin/a2a.py`, `a2a/` | One agent hands a task to another as a tool call and gets the result back |

The multi-agent layer depends on the single-agent layer, never the reverse.

## Subsystems

| Package | Does | State |
| :--- | :--- | :--- |
| `engine/` | In-process event bus with priorities and backpressure; no external broker | Implemented |
| `adapters/cluster/` | Stateless cluster adapters (Kafka KRaft, PostgreSQL, Valkey cluster, Centrifugo SSE edge) behind `cluster_protocols.py` | Implemented |
| `llm/` | One provider-neutral interface, token budgets and context compaction. Connectors for Gemini, Anthropic, OpenAI, Ollama, vLLM and the standalone Model Gateway (`gateway_client.py`); provider SDKs are imported only inside `llm/connectors/` | Implemented |
| `tools/` | Tool registry, MCP client, built-in tools (files, shell, web, images, plan, sub-agents, skills) | Implemented |
| `memory/` | Per-clone memory across conversations, with ranked recall (embeddings when an embedder is configured, lexical overlap otherwise) | Implemented |
| `sandbox/` | Where a tool may read, write and run. Host and workspace isolation work; container and WASM levels exist as configuration models only | Partial |
| `skills/` | `SKILL.md` packages: loading, audit, approval, and proposals synthesized from use | Implemented, evolving |
| `ontology/` | Per-clone knowledge with asserted and induced tiers, a Datalog reasoner and justifications; `ucx ontology` manages it | Implemented, evolving. Nothing learns into it automatically yet |
| `code_intel/` | Tree-sitter parsing, an in-memory symbol graph and SCIP export | Implemented; LSP client not yet |
| `telemetry/` | Tracing and metrics with OTLP export | Implemented |
| `a2a/`, `acp/` | Agent-to-Agent and Agent Client Protocol surfaces | See [protocols.md](protocols.md) |

## Principles

Every part of the runtime is built against ten principles, P0–P9. They are stated in one
place only, [`docs/principles/core-principles.md`](../principles/core-principles.md), and
are not restated here; how they change is in
[principle-amendment-policy.md](principle-amendment-policy.md).

## Where to go next

* [getting-started.md](getting-started.md) — set up a checkout and run the tests
* [cli.md](cli.md) — the `ucx` commands
* [protocols.md](protocols.md) — A2A and ACP
* [security.md](security.md) — what the runtime protects and what it does not
