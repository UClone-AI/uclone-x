# Command line reference

UClone-X installs one command, `ucx`. This page covers the commands meant for people using
UClone-X. Run `ucx <command> --help` for the full option list of any command.

Commands are defined in `src/uclone_x/cli/main.py` and `src/uclone_x/cli/commands/`.

## Installation extras

`pip install uclone-x` gives the core package (Python 3.11 or newer). Features that need
more dependencies are optional extras: `cli`, `http`, `code-intel`, `telemetry`, `llm`,
`ontology`, `media`, or `all`. For example, `pip install "uclone-x[all]"`.

If a command needs an extra you have not installed, it prints `ucx: ` followed by the
name of the missing extra on standard error and exits with status 1.

## Getting started

| Command | What it does |
|---|---|
| `ucx version` | Print the installed version |
| `ucx start` | Check the machine, prepare a local model, and open the dashboard in a browser |
| `ucx install` | Install the local model (Ollama) and the image engine without prompts |
| `ucx ui` | Start the dashboard without the setup steps |
| `ucx ui stop` | Stop the dashboard on a port |
| `ucx status` | Show a running dashboard's health |

`ucx start` and `ucx ui` listen on `127.0.0.1:5180` by default (`--host`, `--port`).
`--cwd` sets the directory agents work in; it defaults to the current directory.
`ucx start --no-open` skips opening the browser, and `--skip-setup` skips the model check.

`ucx install` accepts `--yes`, which is required when there is no terminal to answer
prompts, and `--no-llm` or `--no-image` to skip a part. Its last lines always have this
form, one per requested part:

```
setup llm: ready model=<name> endpoint=<url>
setup llm: unavailable reason=<short>
setup image: ready engine=<engine> checkpoint=<path>
setup image: unavailable reason=<short>
```

It exits 0 only when every requested part is ready.

`ucx ui stop --port N` stops only the dashboard process recorded for that port under
`~/.uclone/ui/`. If there is no record, or the record does not match a running dashboard,
it refuses and exits 1 instead of stopping whatever holds the port.

`ucx status` reads `http://127.0.0.1:8000` unless given `--url`; `--json` prints JSON.

## Talking to a clone: `ucx run`

```bash
ucx run                              # interactive session with the default clone
ucx run writer --prompt "Summarise README.md"
ucx run writer --session-id <id>     # continue an earlier conversation
```

| Option | Meaning |
|---|---|
| `-p`, `--provider` | `ollama`, `vllm`, `openai`, `anthropic`, `gemini` or `mock`. Omitted, it is chosen from `LLM_PROVIDER`, a key in the environment, or a configured local endpoint. |
| `-m`, `--model` | Model name |
| `-s`, `--system` | Replace the default system prompt |
| `--prompt` | Run one turn and exit |
| `--session-id` | Conversation to resume. Omitted, a new conversation is started and its id printed. |
| `--reset` | Clear the conversation back to its system prompt first |
| `--compact` | Compact the conversation's context first |
| `-i`, `--isolation` | Isolation level; see [security.md](security.md). Only `workspace` and `none` have an effect. |
| `-w`, `--cwd` | Directory tools work in |

Agent names use lowercase letters, digits, `-` and `_`, start and end with a letter or
digit, and are at most 64 characters. Conversations are stored under `~/.uclone/sessions`
(`UCLONE_SESSION_DIR`); agent data under `~/.uclone/agents/<name>/` (`UCLONE_AGENTS_DIR`).

With `--prompt`, standard output carries only the reply, so it can be piped. Notices go to
standard error. On a terminal, control characters in the reply are shown as `^[` and
similar rather than sent to the terminal.

In an interactive session, `/help`, `/reset`, `/compact`, `/status`, `/history`, `/exit`
and `/quit` are available.

| Exit status | Meaning |
|---|---|
| 0 | The clone answered and the conversation was saved |
| 1 | The turn failed (the reason is on standard error), the model could not be set up, or the reply could not be written |
| 2 | Bad agent name, or a `--session-id` that is invalid or belongs to a conversation without this agent |
| 3 | The clone answered but the conversation could not be saved |
| 130 | Interrupted with Ctrl+C |

`ucx agent run` is the same command as `ucx run`.

## Recurring prompts: `ucx loop run`

`ucx loop run "check the build log"` runs a prompt repeatedly. `-i` sets the interval
(`30s`, `5m`, `1h`), `-n` the maximum number of runs, `-t` a per-run timeout (default 600
seconds), `--until REGEX` stops when a reply matches, and `--max-failures` (default 3)
stops after that many failures in a row. `-a`, `-p`, `-m`, `-w` and `--session-id` work as
in `ucx run`. `ucx agent loop` is the same command.

## Models and keys

| Command | What it does |
|---|---|
| `ucx key set PROVIDER [KEY]` | Save a key for `gemini`, `anthropic`, `openai` or `vllm`. Omit KEY to be prompted without echo. |
| `ucx key list` | Show saved keys, masked, and any environment variable that overrides them |
| `ucx key remove PROVIDER` | Remove a saved key |
| `ucx key setup` | Choose a provider and save its key interactively |
| `ucx llm status` | Check configured model endpoints and installed models |
| `ucx llm pull [fast\|indepth\|NAME]` | Download a model with Ollama (`fast` is `qwen3:1.7b`, `indepth` is `qwen3:8b`) |
| `ucx llm rm MODEL` | Remove an Ollama model |
| `ucx llm use MODEL` | Make MODEL the default (`--provider`, default `ollama`; `--base-url`) |
| `ucx media status` | Show which image engine would run and what each one is missing |
| `ucx media probe` | Ask a running ComfyUI server what it is |

A key in the environment, such as `GEMINI_API_KEY`, takes priority over a saved key.

## Rooms

A room is a conversation with several participants.

| Command | What it does |
|---|---|
| `ucx room create` | Create a room: `--human`, `--agent`, `--persona AGENT=TEXT`, `--responder`, `--max-turns` (default 3) |
| `ucx room say ROOM SENDER MESSAGE` | Post a message and run the agent turns it causes. `@name` addresses a participant. |
| `ucx room list` / `show ROOM [--tail N]` | List rooms, or show one room's participants and transcript |
| `ucx room add` / `remove` | Seat or unseat a participant |
| `ucx room responder` | Set or clear who answers a message addressed to no one |
| `ucx room retry` | Re-run a failed turn |

## Skills, ontology and failure reports

| Command | What it does |
|---|---|
| `ucx skill list` | List skills and their approval status |
| `ucx skill approve NAME` | Approve a skill. You must type `yes` in a terminal; without a terminal it is refused. |
| `ucx skill reject NAME` | Reject a skill |
| `ucx skill audit` | Audit a skill package (`--policy safe_only\|never\|always`) |
| `ucx skill synthesize` | Draft a skill from a conversation; it stays pending until approved |
| `ucx ontology list\|teach\|review\|forget\|validate` | Inspect and edit a clone's domain ontology |
| `ucx report` | Review locally recorded failures. `--enable`/`--disable` recording, `--save PATH`, `--clear`, or `--open`/`--submit` a GitHub issue (`--submit` uses the `gh` command). |

## Connecting to other programs

| Command | What it does |
|---|---|
| `ucx a2a serve` | Serve a clone over A2A HTTP |
| `ucx acp serve` (or `ucx acp-server`) | Serve a clone over ACP on stdio for an editor |
| `ucx link uclone2 CONNECT_URL` | Link a local clone to a uClone2 clone (`--clone`, `--server`) |
| `ucx link list` / `remove` | Show or remove links |
| `ucx link run` | Keep every enabled link online until Ctrl+C |

See [protocols.md](protocols.md) for the A2A and ACP servers.
