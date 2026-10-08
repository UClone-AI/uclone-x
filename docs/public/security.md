# Security model

UClone-X runs LLM-driven agents on your own machine, with your user account's
permissions. This page describes where its boundaries are, what they protect, and what
they do not. It describes the code as shipped; where a level or feature exists only as a
name, it says so.

To report a vulnerability, follow [SECURITY.md](../../SECURITY.md). Please report
privately rather than in a public issue.

## Trust boundaries

| Boundary | Trusted side | Untrusted side |
|---|---|---|
| Your terminal | You, typing commands and approvals | Everything the agent reads or produces |
| LLM output | Nothing: tool calls from the model are checked before they run | The model and anything in its context |
| Local dashboard (`ucx ui`, `ucx start`) | Processes on your machine | Web pages open in your browser |
| A2A and ACP servers | Whoever can reach them | See below: they do not authenticate |
| Skills | Skills you approved, byte for byte | Skills that are pending or changed since approval |

There is no boundary between content an agent reads and the tools it can call. A web
page, file or message that contains instructions can steer the agent (prompt injection).
The protections below limit where tools can write; they do not stop the model from
following injected text.

## Isolation levels

The isolation policy is defined in `src/uclone_x/sandbox/models.py`.

| Level | Status | What it does |
|---|---|---|
| `workspace` | Implemented, default | Confines working directories and file writes to the workspace |
| `none` | Defined, raised to `workspace` for agents | Would remove confinement and environment scrubbing |
| `container` | Not implemented | Defined as a name only; no runner exists |
| `wasm` | Not implemented | Defined as a name only; no runner exists |

Every agent runs with an isolation floor of `workspace`, and a weaker request is raised
to the floor before a tool runs (`effective_isolation_level` in the same file). No shipped
command lowers the floor, so `ucx run --isolation none` still runs tool calls under
`workspace` rules.

`ucx run --help` also lists `container` and `wasm`, but `ucx run` treats every value other
than `none` as `workspace`. Asking for `container` does not give you a container.

### What `workspace` actually enforces

- **File-writing tools** resolve the target path, following symlinks, and refuse any path
  outside the workspace. They also refuse the story library and the persona directory.
- **Shell commands** (`bash_run`) must start in a directory inside the workspace, have a
  timeout, and are killed with their whole process group when it expires. Environment
  variables whose names look like credentials are removed first.
- **On macOS**, shell commands run under `sandbox-exec` with a profile that denies writes
  to the workspace's story folder. If `sandbox-exec` is not available, the command is
  refused rather than run without it.

### What `workspace` does not enforce

- A shell command can write anywhere your user account can write. Only its starting
  directory is checked.
- There is no read boundary at any level. Agents can read files outside the workspace.
- Network access is not restricted.
- There is no memory, CPU or process-count limit beyond the command timeout.

If you need these guarantees, run UClone-X inside a virtual machine or container that you
control.

## Secrets

- API keys saved with `ucx key set` are written in plain text to `settings.json` under
  `~/.uclone/sessions` (or `UCLONE_SESSION_DIR`). They are not encrypted, and the file
  gets your default file permissions.
  Environment variables such as `GEMINI_API_KEY` take priority over saved keys.
- `ucx key list` shows keys masked.
- Environment variables with credential-shaped names are not passed to shell commands
  under `workspace`, and cannot be added to a tool's allowed environment
  (`src/uclone_x/core/secrets.py`).
- Logs, saved sessions, context summaries, the local failure journal and telemetry
  attributes pass through a pattern-based redactor. It recognises common key prefixes
  (for example `sk-`, `sk-ant-`, `ghp_`, `github_pat_`, `AKIA`), bearer tokens, private key
  blocks and `name=value` assignments with credential-like names.

The redactor is a heuristic. A key without a known prefix, or one the model has split,
encoded or reworded, is not caught. Do not paste secrets into a conversation.

## Skills

- A new or synthesized skill stays pending until approved. `ucx skill approve` asks you to
  type `yes` at the controlling terminal. A program with no terminal, including an
  agent's shell command, cannot answer, and piping `yes` in does not work. A program that
  creates its own pseudo-terminal can answer.
- The approved content is pinned by digest. A skill whose files change after approval is
  not loaded until approved again.
- `ucx skill audit` is LLM-assisted and can miss things. It supports approval; it does not
  replace reading the skill.
- The approval record is an ordinary file in your home directory. Any program running as
  your user can change it.

## Local dashboard

`ucx ui` and `ucx start` bind `127.0.0.1` by default and have no login.

- While bound to a loopback address, requests whose `Host` header is not a loopback name
  are refused. This blocks DNS-rebinding attacks from web pages.
- Several routes that change state also refuse requests with a cross-origin `Origin`
  header.
- Any process on the machine can still call the dashboard's API.
- If you bind it to `0.0.0.0` or another address, the `Host` check is not installed and
  anyone who can reach the port can drive your agents. Do not do this on a shared network.

## A2A and ACP servers

`ucx a2a serve` has no authentication, no TLS and no `Host` check. Anyone who can reach the
port can run tasks with the clone's tools in the directory the server was started from.
Keep the default `127.0.0.1` binding.

`ucx acp serve` talks only over standard input and output, so only the process that
started it can reach it. Its session modes (`auto`, `ask`, `readonly`) are recorded but do
not restrict tools. Tool approvals are sent to the client; an unanswered request is
denied after 30 seconds. See [protocols.md](protocols.md).

## Out of scope

- The security of LLM providers and what they do with prompts you send them.
- Vulnerabilities in dependencies, the operating system or container runtimes.
- Multi-user or multi-tenant deployments. UClone-X assumes one person on one machine.

## Reporting

See [SECURITY.md](../../SECURITY.md) for the private reporting address and supported
versions. There is no guaranteed response time.
