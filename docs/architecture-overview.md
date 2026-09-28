# UClone-X Architecture Overview

> [!IMPORTANT]
> **This document is the target architecture; §0 says how much of it is built.** Most
> subsystems below now have code behind them, but several are partial, and §1's diagram
> still draws the target topology rather than the running one. For the module layout as it
> exists on `main` — packages, layers and the real dependency edges — read
> [module-structure.md](module-structure.md). Treat any statement in §1–§4 as design
> intent unless §0 or that document confirms it.

---

## 0. Implementation Status: Specification vs. Shipped Code

The architecture below is derived from the nine immutable principles in
[`docs/principles/core-principles.md`](principles/core-principles.md) and the functional
requirements in [`docs/PRD.md`](PRD.md). The principles are normative; most of the code
is not yet written. This section exists because this document was previously mistaken
for a description of a working system.

### What exists today

As of 2026-09-26 (`a42a62d1`) every package listed in §3.2 has code. The as-built layout,
the kernel / adapter / shell layer of each package and the known departures from the
layering rule are in [module-structure.md](module-structure.md); this section keeps only
the per-subsystem status.

### Per-subsystem status

| Subsystem | Governing principle / FR | Specification | Code status |
| :--- | :--- | :--- | :--- |
| Developer CLI (`./ucx`) | P8, FR-8 | [cli-specification.md](cli-specification.md) | **Implemented (partial)** — `src/uclone_x/cli/`: `run`, `room`, `loop`, `ui`, `llm`, `key`, `skill`, `ontology`, `a2a`, `acp`, `eval`, `report`, `dev` and the test gate |
| Builder issue/task protocol | — | builder-issue-resolution-protocol.md | **Implemented (partial)** — `src/uclone_x/cli/commands/dev.py`; issues themselves are tracked on GitHub |
| Reactive event bus | P1, P3, FR-1 | [event-driven-agent-core.md](event-driven-agent-core.md), [local-collaboration-engine.md](local-collaboration-engine.md) | **Implemented (partial)** — `uclone_x.engine.event_bus` provides `AgentEvent`, priority-queue `EventBus`, subscriptions and backpressure policies |
| Reactive scheduler, timer service & context registry | P1, P3, FR-1 | [local-collaboration-engine.md](local-collaboration-engine.md) | **Implemented (partial)** — recurring jobs in `uclone_x.agent.loop` (scheduler, runner with watchdog); no separate timer service or context registry package |
| Agent state machine | P1, P4, FR-1 | [event-driven-agent-core.md](event-driven-agent-core.md) | **Implemented (partial)** — `uclone_x.agent.base.BaseAgent`, built by `agent.composition.compose_agent` |
| Multi-agent rooms | P4 | design/multi-agent-conversational-orchestration.md | **Implemented (partial)** — `uclone_x.room`: orchestrator, speaker selectors, room service and store |
| A2A dual-transport adapter | P2, FR-2 | [a2a-protocol-spec.md](a2a-protocol-spec.md) | **Implemented (partial)** — `uclone_x.a2a` (in-memory fastpath, HTTP transport, discovery) and `uclone_x.shells.a2a_server` |
| Dynamic persona & sub-agents | P4, FR-3 | [dynamic-persona-interface.md](dynamic-persona-interface.md) | **Implemented (partial)** — persona registry and store in `uclone_x.agent`, bundled personas in `src/uclone_x/personas/`, `tools/builtin/subagent.py` |
| Ontology Engine | P7, FR-4 | [agent-ontology-architecture.md](agent-ontology-architecture.md), design/clone-knowledge-graph.md | **Implemented (partial)**, re-read 2026-09-25: tiers, reasoner and `ucx ontology` CLI exist; no step learns from conversations (#1638) |
| Skill subsystem & registry | P9, FR-5 | [skill-system-architecture.md](skill-system-architecture.md) | **Implemented (partial)** — `uclone_x.skills` (models, auditor, synthesizer) and `tools/builtin/skill_loader.py`; the web app and CLI heads load the approved skills in the runtime store `ucx-agent-skills/` at startup; the open-source release ships this store empty: its skill packages are kept with the development repository, so a fresh checkout or an installed package has none until `ucx skill` writes one |
| LLM-agnostic layer | P5, FR-6 | [llm-agnostic-interface.md](llm-agnostic-interface.md) | **Implemented (partial)** — `uclone_x.llm`: router, budget, compactor, catalog; connectors for Anthropic, Gemini, OpenAI, Ollama and vLLM |
| Sandbox Runner | P3, FR-7 | [sandbox-execution-architecture.md](sandbox-execution-architecture.md) | **Implemented (partial)** — `none` and `workspace` levels run (`sandbox/workspace_runner.py`); `container` and `wasm` exist as policy models only |
| Developer UI dashboard | P8, FR-9 | [ui-dashboard-architecture.md](ui-dashboard-architecture.md) | **Implemented (partial)** — React head in `frontend/`, FastAPI backend in `uclone_x.ui`, bundle served from `ui_static/` |
| OpenTelemetry exporter | P6, P8, FR-10 | [telemetry-opentelemetry.md](telemetry-opentelemetry.md) | **Implemented (partial)** — `uclone_x.telemetry`: in-memory tracer, OTLP and Langfuse exporters |
| Code Intelligence (AST/LSP/SCIP) | FR-11 | [code-intelligence-lsp-scip.md](code-intelligence-lsp-scip.md) | **Implemented (partial)** — `uclone_x.code_intel`: Tree-sitter AST parser, symbol graph, SCIP indexer |

Milestone sequencing for these subsystems is owned by [`docs/PRD.md`](PRD.md) §5, not by
this document.

---

## 1. Executive Summary

**UClone-X** is an open-source, high-performance, event-driven agent core and multi-agent
collaboration framework. Originating from the architecture of UClone2, UClone-X abstracts
and generalizes the agent reasoning loop into a low-latency, developer-friendly and
LLM-agnostic foundation.

It is designed to power single autonomous agents, dynamic sub-agent swarms, and
cross-system agent-to-agent interactions over the Google A2A specification — with every
agent decision grounded in an evolving ontology (P7), every capability packaged as a
hot-reloadable skill (P9), every tool invocation routed through a selectable sandbox (P3),
every code edit validated against a real symbol graph (FR-11), and every span exported as
OpenTelemetry (P6/P8).

The diagram below is the **target** runtime topology. For what is built, and how the
packages actually depend on each other, see [module-structure.md](module-structure.md).

```mermaid
flowchart TD
    subgraph External["External World / Host Surfaces"]
        Human["Developer via ./ucx CLI"]
        WebUI["React 19 + Vite Developer Dashboard"]
        RemoteAgent["Remote A2A Agent Service"]
        ExtTools["MCP / External Tool Servers"]
        Repo["Target Codebase / Repository"]
        Backends["Observability Backends: Jaeger / Langfuse / Prometheus"]
    end

    subgraph Core["UClone-X Core Runtime"]
        subgraph EngineBox["Local Collaboration Engine — P1 / P3"]
            EventBus["Zero-Copy In-Memory Event Bus"]
            Scheduler["Reactive Task Scheduler"]
            StateRegistry["Active Agent & Context Registry"]
            TimerService["High-Precision Timer Service"]
        end

        subgraph AgentBox["Agent Instances — P4"]
            PrimaryAgent["Primary Orchestrator ucx agent"]
            SubAgents["Ephemeral Sub-Agents, max_depth 2"]
            AuditorAgent["Skill Auditor ucx agent"]
        end

        subgraph ProtocolBox["Protocol & Delegation — P2 / P4"]
            A2AAdapter["A2A Dual-Transport Adapter: in-memory fastpath plus HTTP/SSE"]
            PersonaMgr["Dynamic Persona & Sub-Agent Manager"]
        end

        subgraph OntologyBox["Ontology Engine — P7"]
            OntSynth["Ontology Synthesizer, self-learning worker"]
            OntStore["Active Ontology & Semantic Graph: LinkML / JSON-LD / RDF"]
            OntValidator["Fast In-Memory Pydantic Invariant Validator"]
        end

        subgraph SkillBox["Skill Subsystem — P9"]
            SkillSynth["Autonomous Skill Synthesizer"]
            SkillRegistry["Skill Registry: ucx-agent-skills/ with hot-reload"]
        end

        subgraph CodeIntelBox["Code Intelligence Subsystem — FR-11"]
            TreeSitter["Tree-sitter AST Parser"]
            LSPClient["LSP Client, JSON-RPC over stdio"]
            SCIPIndexer["SCIP Repository Indexer"]
            SymbolGraph["Code Symbol Graph & Live Diagnostics"]
        end

        subgraph ExecBox["Execution Layer — P3"]
            ToolRegistry["Tool Registry & MCP Interop"]
            SandboxRunner["Sandbox Runner: none / workspace / container-wasm"]
        end

        subgraph LLMBox["LLM-Agnostic Layer — P5"]
            Compactor["Context Compactor & Semantic Pruner"]
            Budgeter["Token Budgeter & Quota Controller"]
            Router["Model Router with Observable Failover"]
            Providers["Provider Connectors: Gemini / Claude / OpenAI / Local"]
        end

        subgraph TelemetryBox["Observability — P6 / P8"]
            Tracer["OpenTelemetry Tracer & Meter"]
            Exporter["OTLP Exporter, gRPC / HTTP"]
        end
    end

    %% Host surfaces into the runtime
    Human <--> EventBus
    WebUI <--> EventBus
    Human -->|"seed / steer ontology"| OntStore
    Human -->|"inject / approve skills"| SkillRegistry

    %% A2A boundary: only remote peers speak A2A
    RemoteAgent <-->|"A2A HTTP / SSE"| A2AAdapter
    A2AAdapter <--> EventBus
    A2AAdapter -->|"validate task envelope"| OntValidator

    %% Engine core
    EventBus <--> Scheduler
    Scheduler <--> StateRegistry
    TimerService -->|"timeout / reminder tick"| EventBus
    Scheduler -->|"dispatch event"| PrimaryAgent
    Scheduler -->|"dispatch event"| SubAgents
    Scheduler -->|"dispatch event"| AuditorAgent

    %% Delegation
    PrimaryAgent <--> PersonaMgr
    PersonaMgr -->|"spawn with isolated context"| SubAgents
    SubAgents <--> EventBus
    PrimaryAgent <--> EventBus

    %% Ontology grounding
    PrimaryAgent <-->|"pre/post invariant check"| OntValidator
    SubAgents <-->|"pre/post invariant check"| OntValidator
    OntValidator <--> OntStore
    PrimaryAgent -->|"harvest task outcomes"| OntSynth
    OntSynth -->|"auto-extend classes & rules"| OntStore

    %% Skills
    PrimaryAgent -->|"post-task synthesis"| SkillSynth
    SkillSynth -->|"SKILL.md plus scripts and tests"| AuditorAgent
    AuditorAgent -->|"policy gate: always / safe_only / never"| SkillRegistry
    AuditorAgent -->|"high-risk notification"| Human
    SkillRegistry -->|"hot-load skills"| PrimaryAgent
    SkillRegistry -->|"hot-load skills"| SubAgents

    %% Tools and sandboxing
    PrimaryAgent <--> ToolRegistry
    SubAgents <--> ToolRegistry
    SkillRegistry -->|"skill scripts"| SandboxRunner
    ToolRegistry --> SandboxRunner
    SandboxRunner <--> ExtTools
    SandboxRunner <--> Repo

    %% Code intelligence
    Repo -->|"incremental parse"| TreeSitter
    Repo <-->|"live def / ref / type query"| LSPClient
    Repo -->|"offline index build"| SCIPIndexer
    TreeSitter --> SymbolGraph
    LSPClient --> SymbolGraph
    SCIPIndexer --> SymbolGraph
    SymbolGraph -->|"ingest symbols as ontology entities"| OntStore
    PrimaryAgent <-->|"query symbols / validate edits"| SymbolGraph
    SubAgents <-->|"fix diagnostics on edit"| SymbolGraph

    %% LLM layer
    PrimaryAgent -->|"full context"| Compactor
    SubAgents -->|"full context"| Compactor
    Compactor <--> Budgeter
    Budgeter --> Router
    Router --> Providers

    %% Telemetry: every plane emits spans
    PrimaryAgent --> Tracer
    SubAgents --> Tracer
    A2AAdapter --> Tracer
    SandboxRunner --> Tracer
    Router -->|"failover event, never silent"| Tracer
    OntSynth --> Tracer
    AuditorAgent --> Tracer
    Tracer --> Exporter
    Exporter --> Backends
```

**Reading the diagram.** Three relationships are load-bearing and easy to get wrong:

1. **A2A is agent-to-agent only.** The `A2A HTTP / SSE` label belongs on the edge to the
   remote peer. The provider connectors in the LLM layer are *not* A2A peers; they are
   the model-provider adapter layer (P5).
2. **Code Intelligence feeds the Ontology Engine.** The symbol graph is not a parallel
   store — per FR-11.3 its output is ingested into the agent's ontology, so P7's semantic
   grounding covers code entities too.
3. **Nothing reaches the host except through the Sandbox Runner.** Tool calls and skill
   scripts both pass the sandbox boundary, whose level is selected per P3
   (`none` / `workspace` / `container` / `wasm`).

---

## 2. Core Architectural Pillars

Each pillar is traceable to exactly one immutable principle. The principle documents in
`docs/principles/` are the normative source; this table is a summary and
must never contradict them.

| Pillar | Principle | Description |
| :--- | :--- | :--- |
| **Reactive Event-Driven Loop** | P1 | Agents are non-blocking state machines processing discrete events. No busy-wait or `while sleep` polling; awaiting components yield and register a reactive listener on the bus. |
| **Google A2A Dual-Transport** | P2 | External agent interfaces conform logically to the Google A2A specification (discovery via `/.well-known/agent.json`, capability negotiation, task envelopes), with a zero-copy in-memory fastpath for co-located agents and HTTP/SSE/WebSocket for remote peers. |
| **Single-Machine Acceleration & Pluggable Sandboxing** | P3 | Zero-broker in-process dispatch by default; external brokers are an optional distributed extension. Tool execution runs through a selectable sandbox level. |
| **Dynamic Specialization & Context Isolation** | P4 | Complex tasks are delegated to ephemeral sub-agents with bounded recursion (`max_depth = 2`) and isolated context windows, so parent context is never polluted. |
| **LLM-Layer Token Management & Auto-Compaction** | P5 | Quota tracking, provider switching and semantic context compaction live entirely in the LLM layer, decoupled from agent business logic. |
| **Fail-Fast & Observable Failover** | P6 | No silent suppression, no mock or empty fallback data. Failures propagate to the decision plane; any permitted failover is logged as an OpenTelemetry event. |
| **Evolving Ontology Grounding** | P7 | Every agent operates over a structured ontology that it self-constructs from experience and that developers can steer; decisions and A2A collaboration are validated against it. |
| **Strict Typing & Language Boundaries** | P8 | Python 3.11+ core under `pyright --strict` with Pydantic v2 models; TypeScript React 19 + Vite GUI. Untyped `dict`/`Any` contracts are forbidden across agent interfaces. |
| **Self-Evolving & Pluggable Skills** | P9 | Successful workflows are autonomously packaged as `ucx-agent-skills/<name>/SKILL.md` plus scripts and tests, audited by a Skill Auditor ucx agent under a configurable auto-approval policy, and hot-reloadable by developers. |

---

## 3. High-Level Component Structure

### 3.1 Repository layout

```text
uclone-x/
├── AGENTS.md                                   # Canonical assistant guidelines (Builder vs. ucx agent)
├── CLAUDE.md / GEMINI.md                       # Assistant entrypoints delegating to AGENTS.md
├── README.md                                   # Open source entrypoint & quickstart
├── LICENSE                                     # Apache License 2.0
├── pyproject.toml                              # Hatchling build, src-layout, ruff/pyright/pytest config
├── ucx                                         # Developer CLI shim
├── docs/                                       # Specifications, guides, design records — see docs/README.md
├── src/uclone_x/                               # Python 3.11+ runtime package (src-layout)
│   ├── core/  engine/  errors.py               # Foundation: contracts, event bus, error root
│   ├── llm/  tools/  sandbox/  memory/         # Subsystems (kernel + their adapters)
│   ├── ontology/  skills/  a2a/  acp/          #
│   ├── telemetry/  log/  evaluation/  i18n/    #
│   ├── code_intel/  adapters/                  # Adapter-only packages
│   ├── agent/                                  # One agent: BaseAgent, composition root, sessions
│   ├── room/                                   # Many agents: orchestrator, selectors, room store
│   ├── story/  artifacts/                      # Writing domain: stories and the files clones wrote
│   ├── cli/  ui/  shells/                      # Heads: ucx CLI, web backend, A2A / ACP servers
│   ├── personas/                               # Bundled persona definitions
│   └── ui_static/                              # Built dashboard bundle served by the web head
├── frontend/                                   # TypeScript React 19 + Vite web head (P8)
├── ontology/                                   # Runtime ontology store (LinkML/JSON-LD)
├── ucx-agent-skills/                           # Runtime skill store (SKILL.md packages)
└── tests/                                      # pytest suites — unit / integration / scenarios / fitness
```

> Each package's layer and its real dependency edges are in
> [module-structure.md](module-structure.md); the index of every document under `docs/` is
> [docs/README.md](README.md).

### 3.2 Package-to-principle mapping

The Python runtime lives under `src/uclone_x/` (src-layout, built by Hatchling per
`pyproject.toml`). Note that the §1 diagram groups nodes by *logical plane*, not by
package: the Dynamic Persona & Sub-Agent Manager is drawn beside the A2A adapter because
both mediate delegation, but it belongs to `uclone_x.agent` — there is no separate
`persona/` package.

| Package | Principle / FR | Responsibility |
| :--- | :--- | :--- |
| `uclone_x.cli` | P8, FR-8 | Typer CLI, quality gate, Builder tooling |
| `uclone_x.engine` | P1, P3, FR-1 | Zero-copy event bus, reactive scheduler, timer service, context registry |
| `uclone_x.agent` | P1, P4, FR-1/FR-3 | `BaseAgent` state machine, persona specification, sub-agent spawning |
| `uclone_x.a2a` | P2, FR-2 | A2A discovery, task envelopes, in-memory fastpath and remote wire transport |
| `uclone_x.llm` | P5, FR-6 | Provider connectors, token budgeter, compactor, observable-failover router |
| `uclone_x.ontology` | P7, FR-4 | Ontology synthesizer, semantic graph, fast invariant validator |
| `uclone_x.skills` | P9, FR-5 | Skill synthesizer, Skill Auditor agent, registry and hot-reload |
| `uclone_x.sandbox` | P3, FR-7 | Sandbox mode selection and `none` / `workspace` / `container` / `wasm` adapters |
| `uclone_x.code_intel` | FR-11 | Tree-sitter AST, LSP client, SCIP indexer, code symbol graph |
| `uclone_x.telemetry` | P6, P8, FR-10 | OTel tracer and meter, span conventions, OTLP exporter |
| `uclone_x.ui_static` | FR-9 | Packaged dashboard assets served by the web head |
| `uclone_x.core` | P8 | Cross-subsystem contracts: `Host`, capabilities, session store protocol, workspace, secrets, provenance |
| `uclone_x.room` | P4 | Multi-agent rooms: shared transcript, orchestrator, speaker selectors, room store |
| `uclone_x.tools` | P3, P4 | `BaseTool`, tool registry and scoping, MCP manager, built-in tools |
| `uclone_x.memory` | P7 | Cross-session memory, retrieval and vector store |
| `uclone_x.story`, `uclone_x.artifacts` | P0 | Writing domain: open stories, codex proposals, the files clones wrote |
| `uclone_x.ui`, `uclone_x.shells` | P8, FR-9 | Web backend routes; console entry point, A2A and ACP servers |
| `uclone_x.acp` | P2 | Agent Client Protocol conformance registry |
| `uclone_x.log`, `uclone_x.evaluation`, `uclone_x.i18n` | P6, P8 | Session log storage, evaluation backend seam, user-facing language |
| `uclone_x.adapters` | P2 | Integrations with foreign ecosystems (uclone2) |

---

## 4. Key Differences from the UClone2 Monolith

1. **Decoupled from web/chat infrastructure.** UClone-X strips out platform-specific
   database tables, web servers and UI dependencies, making it a pure framework usable
   from a CLI, a backend service, or an embedded runtime.
2. **Standardized A2A communication (P2).** Custom internal Pub/Sub topics are replaced by
   Google A2A envelopes, discovery descriptors and capability negotiation — with a
   zero-copy in-memory fastpath so standards compliance costs nothing locally.
3. **Dynamic agent evolution with hard context isolation (P4).** Sub-agents are spawned
   programmatically with bounded recursion depth, strict tool permissions and isolated
   context windows, rather than sharing one growing monolithic context.
4. **Zero-broker local acceleration (P3).** In-process event dispatch removes the Redis or
   RabbitMQ dependency for single-node setups; brokers become an optional distributed
   extension instead of a prerequisite.
5. **Knowledge is a first-class subsystem, not prompt text (P7).** UClone2 encoded domain
   knowledge in hand-written prompts. UClone-X gives every agent an ontology it
   self-constructs from task outcomes and that developers steer explicitly, with
   invariant validation on every decision. This pillar postdates the original UClone2
   comparison entirely.
6. **Capabilities are synthesized, audited and hot-reloaded, not hardcoded (P9).** Instead
   of tools being compiled into the codebase, successful workflows are packaged as
   `SKILL.md` skill bundles, security-audited by a dedicated Skill Auditor ucx agent under
   a configurable auto-approval policy, and loaded without a restart. This pillar also
   postdates the original comparison.
7. **Selectable execution isolation (P3).** UClone2 executed tools directly in the host
   process. UClone-X routes every tool and skill invocation through a Sandbox Runner with
   three explicit levels, so isolation becomes a per-workload decision rather than an
   architectural constant.
8. **Code understanding is semantic, not textual (FR-11).** Grep-and-regex code handling is
   replaced by a Tree-sitter AST, a live LSP client for definitions, references and
   diagnostics, and a SCIP repository index feeding the ontology's symbol graph.
9. **Observability and failure semantics are architectural (P6, P8).** Every reasoning
   span, A2A event and provider failover emits OpenTelemetry natively; silent fallbacks
   and mock data are forbidden rather than discouraged.
10. **Strict typing across an explicit language boundary (P8).** The core is Python 3.11+
    under `pyright --strict` with Pydantic v2 contracts; the GUI is TypeScript React 19.
    Untyped `dict`/`Any` payloads may not cross agent interfaces.

---

## 5. Maintenance Rule

> [!IMPORTANT]
> **Any change that adds, removes or renames a subsystem MUST update this document in the
> same commit.** A "subsystem" is anything that would appear as a node or subgraph in the
> §1 diagram, a package row in §3.2, or a directory in §3.1.
>
> Concretely, the same commit must update:
> 1. the **§1 Mermaid diagram** — the new node plus its real edges, not just the box;
> 2. the **§3.1 repository layout**, the **§3.2 package-to-principle mapping** and
>    [module-structure.md](module-structure.md);
> 3. the **§0 status table** — moving the subsystem's row to `Implemented` (or
>    `Implemented (partial)`) as soon as any of its code lands, so the specification/code
>    distinction in this document never goes stale again;
> 4. **§2** and **§4** if the change alters a pillar or the UClone2 comparison.
>
> If the change appears to require altering a principle, stop and request written approval
> from **the project owner** per [`AGENTS.md`](../AGENTS.md); do not reconcile the conflict by
> editing this document.
>
> Mermaid authoring note: node labels must be single-line and use `<br/>` for line breaks —
> raw newlines inside quoted labels do not render portably on GitHub.
