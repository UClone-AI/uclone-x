<div align="center">

# ⚡ UClone-X

**Event-driven AI agent core and multi-agent collaboration framework**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![LLM Agnostic](https://img.shields.io/badge/LLM-Gemini_|_Claude_|_OpenAI_|_Local-orange.svg)](docs/llm-agnostic-interface.md)

</div>

---

## Install

On macOS or Linux, paste this into a terminal:

```bash
curl -fsSL https://raw.githubusercontent.com/UClone-AI/uclone-x/main/install.sh | bash
```

That is the whole install. It needs nothing but `curl` — no Python, Homebrew or
admin rights of its own. It installs the core (~31 MB), then asks before each
larger download: a private Python 3.12 when your machine has nothing newer than
3.10, then a local AI model through Ollama (~1.4–5.2 GB, depending on memory; say
no to use an API key instead). Ollama's own installer may ask for your password.
At the end it asks **Start UClone-X now?** — press Enter, answer any remaining
setup question, and the dashboard opens in your browser at
<http://127.0.0.1:5180>. The agents work in the folder you ran it from.

To answer, UClone-X needs a model: the local one above, which needs no account,
or an API key for OpenAI, Anthropic or Gemini, pasted into the dashboard's
**Settings**. Nothing needs editing in a file.

Next time, start it with:

```bash
ucx start
```

If your shell says `command not found`, the installer printed one `export PATH=…`
line at the end; add it to `~/.zshrc` (or `~/.bashrc`) and open a new terminal.
Until then, `~/.local/bin/ucx start` works.

UClone-X itself lives in `~/.uclone-x` and the `ucx` link in `~/.local/bin`;
`rm -rf ~/.uclone-x ~/.local/bin/ucx` removes it. What the installer fetched
alongside it stays: uv (`~/.local/bin/uv`, `uvx`) and the Python it downloaded
(`uv python uninstall 3.12` removes that), and Ollama with its models.

### Other ways to install

If you already use [uv](https://docs.astral.sh/uv/):

```bash
uv tool install --managed-python --python 3.12 "uclone-x[cli,http]"
ucx start
```

Keep both flags: without `--python 3.12` uv uses the first `python3` on PATH,
which on macOS is 3.9, and the install fails; without `--managed-python` it still
runs that `python3` to check its version, and on a Mac without the Xcode Command
Line Tools that opens an install dialog.

With pip, into a Python 3.11+ environment of your own:
`pip install "uclone-x[cli,http]"`. The two extras are not optional in practice:
`cli` is the `ucx` shell and `http` serves the dashboard. The base install is the
runtime without a shell — what you want if you are importing `uclone_x` as a
library. Provider SDKs and the heavier stacks are the genuinely optional extras:

```bash
pip install "uclone-x[llm]"         # Gemini, Claude, OpenAI SDKs
pip install "uclone-x[ontology]"    # LinkML, rdflib, networkx
pip install "uclone-x[code_intel]"  # tree-sitter AST parsing
pip install "uclone-x[all]"
```

Every extra resolves on Python 3.11, 3.12 and 3.13.

### From a terminal

`ucx run` is an interactive terminal REPL with streaming responses, tool calls
and slash commands. It reads its model from the environment — an API key such as
`OPENAI_API_KEY`, or a local Ollama:

```bash
OLLAMA_FAST_BASE_URL=http://localhost:11434/v1 OLLAMA_FAST_MODEL=qwen3:8b ucx run
```

`ucx llm status` reports which models are reachable.

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
