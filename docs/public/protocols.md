# Protocols: A2A and ACP

UClone-X can expose a clone over two protocols:

- **A2A** (Agent-to-Agent): an HTTP server that lets another program or agent send a
  clone a task and read its answer.
- **ACP** (Agent Client Protocol): a JSON-RPC server on standard input and output that
  lets an editor or another local client hold a conversation with a clone.

Both are partial implementations. This page lists what is there and what is not. The
code is the reference: `src/uclone_x/shells/a2a_server.py`,
`src/uclone_x/shells/acp/server.py` and `src/uclone_x/a2a/`.

Neither server authenticates its callers. Read [security.md](security.md) before you bind
either one to anything other than your own machine.

## A2A

### Starting the server

```bash
pip install "uclone-x[http]"     # the server needs the http extra
ucx a2a serve                    # listens on http://127.0.0.1:8080
```

| Option | Default | Meaning |
|---|---|---|
| `-h`, `--host` | `127.0.0.1` | Address to bind |
| `-p`, `--port` | `8080` | Port to bind |
| `--agent-id` | `default` | Clone to answer as. An id that names a persona answers as that persona; any other id is a generic agent. |
| `--agent-card` | none | Path to a JSON agent card. A missing or invalid file exits with status 1. |
| `-d`, `--dev` | off | Reload the server when source files change |

Without `--agent-card`, the server publishes a card named `agent-<id>` with the skills
`general_reasoning` and `task_execution` and its own URL as the `http` endpoint. The card
model (`src/uclone_x/a2a/models.py`) has these fields only: `name`, `description`,
`version`, `endpoints`, `skills` (a list of strings), `input_schema`, `output_schema`.

The clone's tools work in the directory the server was started from. If no LLM provider
is configured, the server falls back to the built-in mock model and still answers, so
check `ucx llm status` or `ucx key list` first.

### Endpoints

| Method and path | Purpose |
|---|---|
| `GET /.well-known/agent-card.json` | The agent card, with `Cache-Control: max-age=3600` and an `ETag` |
| `POST /a2a/v1/tasks` | Create a task. Returns 201 with the task record. |
| `GET /a2a/v1/tasks/{id}` | Read a task's current state |
| `GET /a2a/v1/tasks/{id}/events` | Server-Sent Events for the task. Events already sent are replayed first. |
| `POST /a2a/v1/tasks/{id}/cancel` | Cancel a task |

The task body accepts `task_id`, `session_id`, `contextId`, `input_data`, `prompt`,
`message`, `sender_agent_id`, `target_agent_id` and `metadata`.

Task states are lowercase strings: `submitted`, `working`, `input_required`,
`auth_required`, `completed`, `failed`, `canceled`, `rejected`. The event stream carries
`status_changed`, `state`, `token`, `completed`, `failed` and `canceled` events. A
`completed` event carries one artifact named `response` with the reply text.

The server also has routes at `/message:send`, `/message:stream`, `/tasks/send`,
`/tasks/stream` and `POST /tasks`. Under `ucx a2a serve` these answer
`404 No task handler registered`, because the command does not install the handler they
need. Use the `/a2a/v1/tasks` routes.

### Version header

A request may send `A2A-Version`. If it is absent, `1.0` is assumed. Any value other than
`1.0` is refused with status 400 and `VersionNotSupportedError`.

### Conversations

Each `contextId` is its own conversation: a one-seat room with a saved session, so a
later task with the same `contextId` continues where the last one stopped. Only one turn
runs at a time per conversation. The server keeps at most 8 conversations loaded and
unloads the least recently used idle one when it needs room; its saved history stays on
disk. Task records themselves are held in memory and are lost when the server stops.

### A2A: not supported

- The JSON-RPC and gRPC bindings of the A2A specification.
- Push notifications, the authenticated extended card, and signed agent cards.
- `securitySchemes` or any authentication. Anyone who can reach the port can submit tasks.
- TLS. Put a reverse proxy in front if you need it.
- Multi-tenant hosting.
- The specification's camelCase field names and `TASK_STATE_*` enum values; this server
  uses the lowercase names above.
- No conformance test suite has been run against the server.

UClone-X also contains an A2A HTTP client (`src/uclone_x/a2a/http_transport.py`), but no
`ucx` command uses it yet. Rooms and clones talk to each other in-process.

## ACP

### Starting the server

```bash
ucx acp serve --persona clone
```

`ucx acp-server` is the same command. The server reads JSON-RPC 2.0 messages from
standard input and writes responses and notifications to standard output. Human-facing
notices go to standard error. An editor starts it as a child process; you do not connect
to it over a network.

| Option | Default | Meaning |
|---|---|---|
| `--persona` | `clone` | Clone to answer as. An unknown persona exits with status 2. |
| `--agent-id` | the persona's name | Id the clone's memory is kept under |

The server reports protocol version `0.12.1` and server name `uclone-x`. It is written
without an ACP SDK.

### Methods

| Method | Notes |
|---|---|
| `initialize` | Returns capabilities: session load and cancel, modes `auto`, `ask`, `readonly`, streaming prompts, permission requests |
| `new_session` | Starts a session. Each session is a one-seat room. |
| `load_session` | Reopens a saved session |
| `set_session_mode` | Accepts `auto`, `ask` or `readonly` |
| `set_config_option` | Stores a configuration value |
| `prompt` | Runs one turn. Progress arrives as `session_update` notifications, including `text_delta` and `tool_result`. |
| `cancel` | Cancels the running turn; the result is `canceled` or `no_active_turn` |

Method names are the snake_case names above. Any other method gets error -32601. Only one
turn runs at a time per session.

When a tool call needs approval, the server sends a `request_permission` request with
`sessionId`, `tool`, `arguments`, `reason` and `requestId`. A result of `allowed: true` or
`action: "allow"` lets the call run. Any other answer, or no answer within 30 seconds,
blocks it.

### Error codes

| Code | Meaning |
|---|---|
| -32700, -32600, -32601, -32602, -32603 | Standard JSON-RPC errors |
| -32000 | Session not found |
| -32001 | Turn cancelled |
| -32002 | Unsupported MCP transport: an MCP server passed with the `acp` transport is refused |
| -32003 | Permission denied |

### ACP: not supported

- `list_sessions`, `fork_session`, `resume_session`, `close_session`, `authenticate` and
  extension methods.
- Any transport other than stdio.
- Session modes and configuration options are stored and echoed back, but they do not
  change what the clone is allowed to do. Choosing `readonly` does not stop writes.
- Like A2A, the server falls back to the mock model when no provider is configured.

## See also

- [cli.md](cli.md) for the rest of the command line.
- [security.md](security.md) for what these servers do and do not protect.
