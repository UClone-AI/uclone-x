# Documentation Index

Every document under `docs/`, grouped by what it is for. Status words follow each
document's own banner: a **specification** says what should be built, **as built**
describes `main`, and a **record** is a dated account that is not kept current.

## Start here

| Document | What it is |
| :--- | :--- |
| [PRD.md](PRD.md) | Product requirements (FR / NFR) and roadmap |
| [principles/core-principles.md](principles/core-principles.md) | P0–P9, the normative principles. Immutable without the owner's approval; one detail document per principle under `principles/details/` |
| [architecture-overview.md](architecture-overview.md) | Target architecture, with a per-subsystem implementation status |
| [module-structure.md](module-structure.md) | **As built**: packages, their layers, real dependency edges, known departures |

## Contracts and cross-cutting specifications

| Document | Covers |
| :--- | :--- |
| core-shell-architecture.md | Kernel / adapter / shell layering rule (lab repository only), enforced by `tests/fitness/test_core_shell_boundaries.py` |
| [a2a-protocol-spec.md](a2a-protocol-spec.md) | Google A2A contracts and dual transport |
| [acp-protocol-spec.md](acp-protocol-spec.md) | Agent Client Protocol surface and conformance |
| [cli-specification.md](cli-specification.md) | `./ucx` command reference and quality gate |
| [ontology-alignment-spec.md](ontology-alignment-spec.md) | Ontology alignment contract |
| [nfr-performance-budgets.md](nfr-performance-budgets.md) | Performance budgets |
| [security-threat-model.md](security-threat-model.md) | Threat model |

## Subsystem architecture

| Document | Subsystem (package) |
| :--- | :--- |
| [event-driven-agent-core.md](event-driven-agent-core.md) | Agent state machine (`agent/`) |
| [local-collaboration-engine.md](local-collaboration-engine.md) | Event bus and scheduling (`engine/`, `agent/loop/`) |
| [dynamic-persona-interface.md](dynamic-persona-interface.md) | Personas and sub-agents (`agent/`) |
| [llm-agnostic-interface.md](llm-agnostic-interface.md) | Providers, budgeting, compaction (`llm/`) |
| [agent-ontology-architecture.md](agent-ontology-architecture.md), [ontology-architecture-v2.md](ontology-architecture-v2.md) | Ontology (`ontology/`) |
| [semantic-retrieval.md](semantic-retrieval.md) | Retrieval (`memory/`) |
| [skill-system-architecture.md](skill-system-architecture.md) | Skills (`skills/`) |
| [sandbox-execution-architecture.md](sandbox-execution-architecture.md) | Sandbox levels (`sandbox/`) |
| [code-intelligence-lsp-scip.md](code-intelligence-lsp-scip.md) | Code intelligence (`code_intel/`) |
| [telemetry-opentelemetry.md](telemetry-opentelemetry.md) | Tracing and export (`telemetry/`) |
| [ui-dashboard-architecture.md](ui-dashboard-architecture.md) | Web head (`frontend/`, `ui/`) |

## Guides

| Document | For |
| :--- | :--- |
| [local-development-guide.md](local-development-guide.md) | Setting up and running locally |
| [guides/agent-runtime-terminology.md](guides/agent-runtime-terminology.md) | Runtime vocabulary |
| [guides/logging-and-observability-policy.md](guides/logging-and-observability-policy.md) | Logging policy |
| [guides/testing-principles.md](guides/testing-principles.md) | How tests are written and judged |

## Engineering record (lab repository only)

These are not part of the published product documentation.

| Where | Holds |
| :--- | :--- |
| Design records | Design proposals per feature; read before implementing against one, update when the code departs |
| Dated notes | Investigations, benchmarks and design reviews, each true as of its date |
| Evaluation record | Every evaluation figure taken, with its commit |
| Plans | Multi-phase plans for internal work |
| Quality and evaluation guide | The quality gate and evaluation workflow |
| Governance | Decision ledger, amendment log, documentation and issue policies |
| Principle proposals | Proposed principle amendments awaiting the owner |
| Swarm guides, Builder manager guide, Builder issue protocol, Meta-agent guide | How Builder agents work in this repository |

The three Builder manuals sit at the top level for a historical reason: evaluation datasets
and hundreds of issue and document references cite their paths, so moving them would
break ground truth. New Builder-facing documents go under the swarm guides directory.

## Where a new document goes

| If it is… | Put it in |
| :--- | :--- |
| A principle change | A principle proposal; never edit the principles directly |
| A contract other code or peers depend on | The top level, beside the specifications above |
| A design for one feature | The design records directory |
| A how-to that stays current | `guides/` |
| A dated finding, benchmark or review | The dated notes directory, file name prefixed with the date |
| A measured evaluation figure | The evaluation record |

A new file under `docs/` must also be classified as published or unpublished in the export
manifest; nothing classifies it automatically.
