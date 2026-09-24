<div align="center">

# ⚡ UClone-X

**Event-driven AI agent core and multi-agent collaboration framework**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![LLM Agnostic](https://img.shields.io/badge/LLM-Gemini_|_Claude_|_OpenAI_|_Local-orange.svg)](docs/llm-agnostic-interface.md)

</div>

---

## Install

```bash
uv tool install --managed-python --python 3.12 "uclone-x[cli,http]"
ucx --help
```

Or `uvx --managed-python --python 3.12 --from "uclone-x[cli,http]" ucx --help` to try
it without installing anything permanently. Keep both flags: without `--python 3.12` uv
uses the first `python3` on PATH, which on macOS is 3.9, and the install fails; without
`--managed-python` it still runs that `python3` to check its version, and on a Mac
without the Xcode Command Line Tools that opens an install dialog. With both, uv
downloads its own Python 3.12 and touches nothing else.

The two extras are not optional in practice: `cli` is typer and rich, which the
`ucx` shell is written in, and `http` is FastAPI and uvicorn, which the
dashboard and the A2A gateway serve through. The base install is the runtime
without a shell — usable as a library, and what you want if you are importing
`uclone_x` rather than running `ucx`. Install without them and `ucx` tells you
which one is missing rather than failing obscurely.

Provider SDKs and the heavier stacks are the genuinely optional extras — install
what you use:

```bash
pip install "uclone-x[llm]"         # Gemini, Claude, OpenAI SDKs
pip install "uclone-x[ontology]"    # LinkML, rdflib, networkx
pip install "uclone-x[code_intel]"  # tree-sitter AST parsing
pip install "uclone-x[all]"
```

Python 3.11 or newer, and every extra resolves on every version of it, 3.13
included. `code_intel` used to be capped at 3.12, because the grammar bundle it
declared — `tree-sitter-languages` — publishes no wheel above cp312, and since
`all` includes `code_intel` that ceiling was `uclone-x[all]`'s too. It now
declares `tree-sitter-language-pack`, which ships cp313 wheels, so neither is
capped. Without the extra the AST parser falls back to Python's own `ast` and
says so.

## First run

Configure at least one provider, then start a chat session:

```bash
cp .env.example .env    # then fill in one API key, or point at a local Ollama
ucx run
```

`ucx run` is an interactive terminal REPL with streaming responses, tool calls
and slash commands. A local model needs no API key at all:

```bash
OLLAMA_FAST_BASE_URL=http://localhost:11434/v1 OLLAMA_FAST_MODEL=qwen3:8b ucx run
```

The dashboard — an agent and event-bus explorer — is a single command:

```bash
ucx ui
```

It serves a prebuilt interface, so it needs no Node toolchain.

## What UClone-X is

An agent runtime built around a non-blocking reactive event loop rather than a
blocking request/response cycle. Agents yield when idle and wake on tool
returns, interrupts or messages from other agents. The same event bus carries
single-agent execution and multi-agent collaboration on one machine, without an
external broker.

* **Event-driven agent core** — a six-stage reactive state machine over an
  in-memory priority event bus with backpressure policies and immutable event
  envelopes.
* **LLM-agnostic** — Gemini, Claude, OpenAI and local models (Ollama, vLLM)
  behind one interface, with schema translation and fallback routing.
* **Dynamic personas and sub-agents** — agents create, supervise and terminate
  child agents with their own prompts and strict tool boundaries.
* **Pluggable sandboxing** — host, workspace, container or WASM isolation
  selected per tool rather than globally.
* **Ontology grounding** — LinkML domain schemas and a semantic graph the agent
  reasons against instead of inventing structure per prompt.
* **OpenTelemetry native** — traces, metrics and OTLP export built in.

## Implementation status

UClone-X is pre-1.0 and under active development. Read this section before
depending on it.

| Area | State |
| :--- | :--- |
| Event bus, agent state machine, sessions | Implemented |
| Tool runtime, MCP client, built-in tools | Implemented |
| LLM connectors (Gemini, Claude, OpenAI, Ollama, vLLM) | Implemented |
| Sandbox isolation modes | Implemented (host and workspace); container and WASM partial |
| Ontology engine, skills, code intelligence | Implemented, evolving |
| Developer dashboard | Implemented |
| A2A protocol | **Specified, not implemented.** No conformance is claimed |

## Something went wrong?

`ucx` keeps a record of what failed, on your machine, and only if you allow it:

```bash
ucx report --enable     # start recording failures locally
ucx report              # show what has been recorded
ucx report --open       # open a pre-filled bug report in your browser
```

**Nothing is uploaded on its own.** There is no telemetry server and no account
to sign into. `ucx report` prints the exact text that would be sent; `--open`
fills in a GitHub issue form that you still have to submit yourself. If you have
the GitHub CLI, `ucx report --submit` files it through your own `gh` login.

What gets recorded: the error type, the code path it failed on, your UClone-X,
Python and OS versions, and the error message. Your prompts and the contents of
your files are never included, and paths are reduced — code paths to the file
name, home directories to `~`.

Masking is pattern-based, like every credential scrubber: it recognises the
shapes that occur in practice — `sk-…`, `ghp_…`, home directories on Linux,
macOS and Windows — and cannot promise to catch a token shape or a path layout
nobody has seen. That is why `ucx report` shows you the text before anything is
sent, and why the text is the last word on what leaves your machine.

Turn it off with `ucx report --disable` and delete what was recorded with
`ucx report --clear`.

## Documentation

* [Architecture overview](docs/architecture-overview.md)
* [Event-driven agent core](docs/event-driven-agent-core.md)
* [Core principles P0–P9](docs/principles/core-principles.md) — the normative
  rules every part of the runtime is built against
* [Product requirements](docs/PRD.md)
* [CLI specification](docs/cli-specification.md)
* [LLM-agnostic interface](docs/llm-agnostic-interface.md)
* [Dynamic persona and sub-agent interface](docs/dynamic-persona-interface.md)
* [Local collaboration engine](docs/local-collaboration-engine.md)
* [Sandbox execution](docs/sandbox-execution-architecture.md)
* [Ontology architecture](docs/agent-ontology-architecture.md)
* [Skill system](docs/skill-system-architecture.md)
* [Code intelligence (AST, LSP, SCIP)](docs/code-intelligence-lsp-scip.md)
* [Telemetry](docs/telemetry-opentelemetry.md)
* [A2A protocol specification](docs/a2a-protocol-spec.md)
* [Security threat model](docs/security-threat-model.md)

## Development

An installed build is for *using* the agent. To change it, work from a checkout:
the development commands — `setup`, `test`, `dev` — exist only there, because
each of them assumes the repository. `ucx --help` in an installed build says so
rather than leaving you to guess.

```bash
git clone https://github.com/UClone-AI/uclone-x.git
cd uclone-x
uv sync --all-extras
npm ci --prefix frontend   # the gate runs the frontend's vitest suite
./ucx test check      # ruff format + lint, pyright strict, pytest with branch coverage, vitest
```

`./ucx test check` is the whole gate: zero ruff findings, zero pyright errors in
strict mode, the test suite at or above 70% branch coverage, and a passing
frontend vitest suite. A change that does not pass it is not ready.

See [CONTRIBUTING.md](CONTRIBUTING.md) for how patches reach this repository,
and [docs/local-development-guide.md](docs/local-development-guide.md) for the
longer setup walkthrough.

## About this repository

This repository is the UClone-X runtime and its documentation. It is published
from a private development repository, in periodic snapshots rather than as a
mirror of that repository's history. The evaluation suites and multi-agent
build tooling used to develop it are not part of the published subset; the
runtime detects their absence and degrades cleanly, so `ucx eval` reports that
no evaluation backend is installed rather than failing to start.

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
