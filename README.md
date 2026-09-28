<div align="center">

# ⚡ UClone-X

**AI clones that run on your own machine — chat with them one-to-one or in a group,
and let them remember, read your files, search and draw.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![LLM Agnostic](https://img.shields.io/badge/LLM-Gemini_|_Claude_|_OpenAI_|_Local-orange.svg)](https://github.com/UClone-AI/uclone-x/blob/main/docs/llm-agnostic-interface.md)

</div>

---

UClone-X is a local AI workspace. You install it, open the dashboard in your
browser, and talk to *clones* — agents with their own instructions, memory and
tools. A local model through Ollama works with no account and no API key; a
Gemini, Claude or OpenAI key works too.

Underneath is an event-driven agent runtime you can also use on its own: see
[For developers](#for-developers).

## What it looks like

Ask a clone a question. Here the default **clone** answers with a structured
summary on a local model (`ollama:qwen3:8b`):

![A clone answering "Find latest AI technology in september" with a structured list of AI trends](https://raw.githubusercontent.com/UClone-AI/uclone-x/main/docs/assets/screenshots/research-conversation.webp)

Or ask the **artist** clone for a picture. The image appears in the conversation,
with links to open it in the browser or in Docs:

![The artist clone showing a generated image for "beautiful girl with beautiful background"](https://raw.githubusercontent.com/UClone-AI/uclone-x/main/docs/assets/screenshots/image-generation.webp)

---

## Install

> On PyPI? The `pip install uclone-x` line PyPI prints at the top of the page
> installs the library only, without the `ucx` command. Use one of the installs
> below; with pip, that is `pip install "uclone-x[cli,http]"`.

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

### Models

`ucx install` downloads the local language model and the image checkpoint and
checks that each one actually works; it reports a partial setup as incomplete
rather than as success. You can also do this later from the dashboard.

As a rough guide to what fits, from the installer's own advice:

| Memory | Local model | Image generation |
| :--- | :--- | :--- |
| under 8 GB | use a cloud model (API key) | no |
| 8–16 GB | `qwen3:1.7b`, or a cloud model | no |
| 16–24 GB | `qwen3:8b` | 512 px |
| 24 GB and up | `qwen3:8b` | 768 px |

## Connecting a model

Pick **one** of these:

* **Local, no key.** Install [Ollama](https://ollama.com). `ucx start` finds it
  and offers a model that fits; models can also be added and removed from
  Settings, or with `ucx llm pull` and `ucx llm rm`.
* **In the dashboard.** Open Settings, choose Gemini, Anthropic, OpenAI, vLLM or
  Ollama, and enter the key or server address there. It is remembered for the
  next start, in `~/.uclone/sessions/settings.json` as plain text — treat that
  file like the key itself.
* **From the environment.** `ucx` reads provider settings from environment
  variables, which take precedence over what Settings saved:

  ```bash
  export ANTHROPIC_API_KEY=...        # or GEMINI_API_KEY, OPENAI_API_KEY
  ucx start
  ```

  `ucx` does **not** read a `.env` file by itself. The repository's
  `.env.example` lists every variable it understands; if you keep yours in a
  `.env`, load it into the shell first (`set -a; . ./.env; set +a`).

`ucx llm status` shows which providers it can reach.

## What you can do

* **Clones.** Six ship built in — clone (the default), writer, artist,
  guardian, pioneer and scout. Create your own, or edit one, from Settings; the
  connected model can draft its instructions for you. Each clone is a YAML file
  in your workspace.
* **Conversations.** One-to-one chats and group chats with several clones,
  listed most recent first, with pinning, rename and delete. Replies stream as
  they are written, including each tool call; messages render Markdown and math.
  Group conversations can also be driven from the terminal with `ucx room`.
* **Memory.** Every clone can record, look up and retract facts, and remembers
  them across conversations. The dock's Remembers tab lists what each clone
  saved.
* **Files and images.** Clones read files in your workspace and in folders you
  mark read-only, write only inside the workspace, and generate images in-process
  (or through ComfyUI when one is running). What each turn produced appears in
  the dock beside the conversation.
* **Tools from elsewhere.** Connect MCP servers from Settings — a remote one by
  URL, a local one by command, or by pasting a vendor's `mcpServers` snippet.
* **The terminal.** `ucx run` is the same agent as an interactive REPL, with
  streaming, tool calls and slash commands; `ucx run --prompt "…"` answers once.
* **Your editor.** `ucx acp serve` lets an editor that speaks the Agent Client
  Protocol talk to a clone.

`ucx --help` lists every command.

## Where your data lives

UClone-X has no server of its own and no account. Conversations and dashboard
settings are kept under `~/.uclone/sessions/` (`UCLONE_SESSION_DIR` moves them),
each clone's memory under `~/.uclone/agents/` (`UCLONE_AGENTS_DIR`), and clone
definitions are YAML files in the workspace folder.

What leaves the machine is what you connect it to: your messages go to the model
provider you chose, a web search goes to the search service, and an MCP server
you add receives what its tools are called with. With a local model and no
remote tools, nothing leaves at all.

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

Common first-run problems:

* **"No active LLM provider connected."** No model is reachable. Start Ollama
  (`ollama serve`), or set a key in Settings or the environment — see
  [Connecting a model](#connecting-a-model). `ucx llm status` says which
  providers answered.
* **A key in `.env` is ignored.** `ucx` reads the environment, not `.env`; load
  the file into your shell first.
* **`ucx: command not found` after the installer.** It is not on your PATH yet;
  add the `export PATH=…` line the installer printed to `~/.zshrc` (or
  `~/.bashrc`), or run it as `~/.local/bin/ucx` until then — see [Install](#install).

## For developers

### The runtime

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

### Implementation status

UClone-X is pre-1.0 and under active development. Read this section before
depending on it.

| Area | State |
| :--- | :--- |
| Event bus, agent state machine, sessions | Implemented |
| Tool runtime, MCP client, built-in tools | Implemented |
| LLM connectors (Gemini, Claude, OpenAI, Ollama, vLLM) | Implemented |
| Dashboard: conversations, clones, memory, settings | Implemented |
| Local image generation | Implemented |
| Sandbox isolation modes | Implemented (host and workspace); container and WASM partial |
| Ontology engine, skills, code intelligence | Implemented, evolving |
| A2A protocol | **Specified, not implemented.** No conformance is claimed |

### Documentation

The documents below are the design and specification of the runtime; they are
written for contributors rather than as a user manual.

* [Documentation index](https://github.com/UClone-AI/uclone-x/blob/main/docs/README.md)
* [Module structure](https://github.com/UClone-AI/uclone-x/blob/main/docs/module-structure.md) — the packages as built, their layers
  and dependency edges
* [Architecture overview](https://github.com/UClone-AI/uclone-x/blob/main/docs/architecture-overview.md)
* [Event-driven agent core](https://github.com/UClone-AI/uclone-x/blob/main/docs/event-driven-agent-core.md)
* [Core principles P0–P9](https://github.com/UClone-AI/uclone-x/blob/main/docs/principles/core-principles.md) — the normative
  rules every part of the runtime is built against
* [Product requirements](https://github.com/UClone-AI/uclone-x/blob/main/docs/PRD.md)
* [CLI specification](https://github.com/UClone-AI/uclone-x/blob/main/docs/cli-specification.md)
* [LLM-agnostic interface](https://github.com/UClone-AI/uclone-x/blob/main/docs/llm-agnostic-interface.md)
* [Dynamic persona and sub-agent interface](https://github.com/UClone-AI/uclone-x/blob/main/docs/dynamic-persona-interface.md)
* [Local collaboration engine](https://github.com/UClone-AI/uclone-x/blob/main/docs/local-collaboration-engine.md)
* [Sandbox execution](https://github.com/UClone-AI/uclone-x/blob/main/docs/sandbox-execution-architecture.md)
* [Ontology architecture](https://github.com/UClone-AI/uclone-x/blob/main/docs/agent-ontology-architecture.md)
* [Skill system](https://github.com/UClone-AI/uclone-x/blob/main/docs/skill-system-architecture.md)
* [Code intelligence (AST, LSP, SCIP)](https://github.com/UClone-AI/uclone-x/blob/main/docs/code-intelligence-lsp-scip.md)
* [Telemetry](https://github.com/UClone-AI/uclone-x/blob/main/docs/telemetry-opentelemetry.md)
* [A2A protocol specification](https://github.com/UClone-AI/uclone-x/blob/main/docs/a2a-protocol-spec.md)
* [Security threat model](https://github.com/UClone-AI/uclone-x/blob/main/docs/security-threat-model.md)

### Working on UClone-X

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

See [CONTRIBUTING.md](https://github.com/UClone-AI/uclone-x/blob/main/CONTRIBUTING.md) for how patches reach this repository,
and the [local development guide](https://github.com/UClone-AI/uclone-x/blob/main/docs/local-development-guide.md) for the
longer setup walkthrough.

## About this repository

This repository is the UClone-X runtime, its dashboard and its documentation. It
is published from a private development repository, in periodic snapshots rather
than as a mirror of that repository's history. The evaluation suites and
multi-agent build tooling used to develop it are not part of the published
subset; the runtime detects their absence and degrades cleanly, so `ucx eval`
reports that no evaluation backend is installed rather than failing to start.

## License

Apache 2.0 — see [LICENSE](https://github.com/UClone-AI/uclone-x/blob/main/LICENSE) and [NOTICE](https://github.com/UClone-AI/uclone-x/blob/main/NOTICE).
