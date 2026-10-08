# Getting started with a checkout

This page is for working *on* UClone-X. To install and use it, follow the
[README](../../README.md#install) instead: an installed build has no development
commands.

## Requirements

* Python 3.11 or newer, and [uv](https://docs.astral.sh/uv/)
* Node.js and npm, for the dashboard and its tests
* A model to talk to: [Ollama](https://ollama.com) with no account, or an API key for
  Gemini, Anthropic or OpenAI. The unit tests need neither.

## Set up

```bash
git clone https://github.com/UClone-AI/uclone-x.git
cd uclone-x
uv sync --all-extras
npm ci --prefix frontend
```

`./ucx` at the repository root runs the CLI from the checkout, so a change to
`src/uclone_x/` takes effect without reinstalling.

## Run it

```bash
./ucx ui --dev          # dashboard with hot reload: backend on 5180, Vite on 5173
./ucx run               # the same agent as a terminal REPL
./ucx run --prompt "…"  # answer once and exit
./ucx llm status        # which model providers are reachable
```

The agents read and write in the directory you start from; `--cwd` on `ucx ui` points
them somewhere else.

Provider settings come from the dashboard's Settings or from the environment
(`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `OLLAMA_*`). `ucx` does not read
a `.env` file by itself; [`.env.example`](../../.env.example) lists every variable it
understands, and `set -a; . ./.env; set +a` loads yours into the shell.

### A local model

`ucx start` and `ucx install` pick an Ollama model that fits the machine. Two settings are
worth knowing when you run Ollama yourself:

* **`OLLAMA_KEEP_ALIVE`** — UClone-X sends a keep-alive with every request, and the
  request's value overrides the daemon's. It sends this variable from *its own*
  environment, or `30m` when it is unset, so export it where UClone-X runs, not only for
  `ollama serve`.
* **`OLLAMA_CONTEXT_LENGTH`** — UClone-X asks for a 16384-token window unless the agent sets
  its own `context_limit` or this variable is set. Ollama's own default on a small GPU is
  4096, which one persona's instructions can fill.

## Test it

```bash
./ucx test check     # the whole gate: ruff, pyright strict, pytest ≥ 70% branch coverage, vitest
./ucx test changed   # the same checks over what your diff reaches; faster while iterating
./ucx test unit      # the unit suite alone
./ucx test live --provider ollama   # against a real local model; costs no tokens
```

The unit suite uses a scripted model and needs no network. `./ucx test check` is what a
change must pass; [CONTRIBUTING.md](../../CONTRIBUTING.md) says what else a pull request
needs and how it reaches this repository.

## Find your way around

| Path | Holds |
| :--- | :--- |
| `src/uclone_x/` | The runtime; [overview.md](overview.md) explains its layers |
| `src/uclone_x/personas/` | The built-in clones |
| `frontend/src/` | The dashboard (React, TypeScript) |
| `tests/unit/`, `tests/integration/`, `tests/e2e/` | Tests by tier; `tests/fixtures/` holds their data |
| `examples/` | Example personas |
| `docs/public/` | This documentation |
| `docs/principles/` | The principles P0–P9 |
