# UCX CLI Tool Specification (`./ucx`)

## 1. Overview

`ucx` is the unified developer command-line interface for **UClone-X**. It wraps agent lifecycle management, in-memory local collaboration execution, automated quality gates, ontology validations, and UI launching into a clean, memorable CLI.

---

## 2. Command Reference

This table reflects the commands as implemented in `src/uclone_x/cli/main.py` and
under `src/uclone_x/cli/commands/` (`dev.py`, `llm.py`, `ontology.py`, `room.py`, `run.py`, `skill.py`).
**Stub** means the command exists and exits successfully but performs no real work yet.
**Planned** means no such command exists in the CLI at all.

| Command | Category | Status | Description |
| :--- | :--- | :--- | :--- |
| `./ucx version` | Utility | Implemented | Prints the installed UClone-X version. |
| `./ucx setup` | Environment | Implemented | Initializes environment and installs automated Git pre-commit quality gate hook. |
| `./ucx run [agent]` | Runtime | Implemented | Interactive terminal Chat REPL and single-shot execution with live BaseAgent. `--session-id` selects the session to run in and resumes the conversation that ID had; `--reset` purges it to its system prompt first; `--compact` compacts its context first (P5). Sessions persist through the Core `SessionStore` (#183), whose root can be redirected with `UCLONE_SESSION_DIR`. The agent named here is also a directory name: each agent owns `~/.uclone/agents/<name>/` (redirectable with `UCLONE_AGENTS_DIR`) holding its `id` and its cross-session `memory.json`. A name that cannot be one unambiguous directory is refused with exit 2 rather than repaired -- lowercase letters, digits, `-` and `_`, starting and ending with a letter or digit, at most 64 characters. Uppercase is refused rather than folded: a case-insensitive filesystem would give two spellings one directory and a case-sensitive one would give them two. |
| `./ucx ui` | Developer GUI | Implemented | Launches developer dashboard with automatic port collision detection and auto-fallback. |
| `./ucx ui stop [--port N]` | Developer GUI | Implemented | Stops the dashboard on port N (default 5180) by signalling only the process that launched it (#927). `ucx ui` and `ucx start` record each dashboard they launch in `~/.uclone/ui/dashboard-<port>.json` (redirectable with `UCLONE_UI_STATE_DIR`; file `0600`, directory created `0700` — an existing directory is never re-permissioned, and one owned by another user or writable by others is refused) with the launcher's PID and its start time and command line as `ps` reports them. `stop` sends SIGTERM to that PID only while `ps` still reports the same start time and command line, waits up to 5 s for it to exit (a `--dev` launcher takes its reload worker down), and only then sends SIGKILL. A SIGKILL is reported as such with exit 1, never as "Stopped" — the dashboard's graceful shutdown did not run — and if the port still accepts connections afterwards the output says a `--dev` worker may have been orphaned. If `ps` fails after SIGTERM was sent, the outcome is reported as unconfirmed (exit 1), no SIGKILL is sent and the record is kept. Nothing a process on the port says is consulted. **Behaviour change:** a dashboard with no record — including any dashboard started before this version, one started with another `UCLONE_UI_STATE_DIR`, or another product on the port — is refused with exit 1 and receives no signal; stop it with Ctrl-C in the terminal running it. A record whose PID has exited or been reused is removed and reported. A record that is malformed, lacks a process identity, is a symlink, is not owned by this user, or is writable by others (or sits in such a directory) is refused and kept. With no record and nothing accepting connections on 127.0.0.1 or ::1, exit 0, and the message names the record path and says a dashboard bound only to another address is not detected. Needs only the `cli` extra; uses `ps`, not `lsof`. |
| `./ucx test check` | Quality Gate | Implemented | Runs Ruff format check, Ruff lint, Pyright strict typing, and Pytest with branch coverage >= 70%. See §3. |
| `./ucx llm status\|pull` | LLM Management | Implemented | Status probe with endpoint check across the 2-Tier Ollama topology and a configured vLLM endpoint, pulling Ollama models. |
| `./ucx skill list\|synthesize\|quarantine\|promote` | Dynamic Skills | Implemented | List skills, extract/synthesize from trace, quarantine audit verification, and promotion. |
| `./ucx ontology list\|validate\|generate` | Ontology | Implemented | Inspect ontologies, validate against LinkML schemas, and code-generation. |
| `./ucx dev task create\|list\|claim\|resolve` | Builder Tracker | Implemented | Create, list, claim, and resolve tasks with automated quality gate enforcement on resolve. Tasks are GitHub Issues on Project #3; the board is the only store. `-g` / `--github` is accepted as a no-op synonym, because the board-backed behaviour it used to select is now the only behaviour. |
| `./ucx loop [run] <prompt>` | Recurring Loop | Implemented | Executes recurring agent automation loop with interval parsing (structured `5m` or natural language `5분마다`), concurrency lock (uclone2 PID lock pattern), watchdog timeout (`--timeout`), and exit conditions (`--until`, `--max-runs`). Also available in REPL as `/loop`. |
| `./ucx room create\|list\|show\|add\|remove\|responder\|say\|retry` | Multi-Agent Rooms | Implemented | Create a titled room, list rooms by title, render its transcript, change its roster, post a human utterance that drives the agent turns it causes, and re-run a turn that failed. `say` drives `RoomOrchestrator.post` and `retry` drives `RoomOrchestrator.retry`, which reuses the turn slot the failure already spent rather than taking a fresh one from the room's budget; every other command is a shell over `RoomService`. `show` renders what the record already held and no surface printed: a decided silence with its reasoning, a failed turn with the command that re-runs it, and the served model that answered beside the selector — and the model — that gave it the floor. Rooms persist under `~/.uclone/rooms`, redirectable with `UCLONE_ROOM_DIR`. A join and a leave are written into the transcript, so the conversation explains its own gaps. `create` takes `--persona AGENT=TEXT` and `--alias AGENT=NAME` — keyed, so a mismatch names the agent rather than landing on the wrong one — because a seated agent with no persona is one a selector cannot route to. `responder` names or clears the agent that answers an unaddressed message, which `create` could previously only give and `remove` could only take away. |
| `./ucx a2a serve` | A2A Gateway | Implemented | Launches standalone A2A HTTP/SSE gateway and federated node (`/.well-known/agent-card.json`, task CRUD & SSE). |
| `./ucx test live` | Quality Gate | **Planned — not implemented** | No live/integration-test command exists; nothing consumes real LLM tokens today. |
| `./ucx agent create <name>` | Scaffold | **Planned — not implemented** | No agent-scaffolding command exists in the CLI. |
| `./ucx telemetry view` | Observability | **Planned — not implemented** | No such command exists in the CLI. |

---

## 3. Automated Quality Gate (`./ucx test check`)

Per **Principle 8 (Strict Typing & Automated Verification)**, `./ucx test check`
runs four steps, in this order, and **fails fast**: it exits immediately with the
first step's non-zero exit code and does not run any later step.

```bash
# Executed steps in quality gate, in order — first non-zero exit stops the gate
1. Ruff Format Check:     ruff format --check src tests
2. Ruff Linter:           ruff check src tests
3. Pyright Strict Typing: pyright
4. Pytest Unit Suite:     pytest -v --cov=src/uclone_x --cov-branch --cov-fail-under=70
```

Step 3 runs plain `pyright` with no `--strict` (or `--warnings`) flag on the
command line. Strict mode instead comes from `typeCheckingMode = "strict"` in
`[tool.pyright]` in `pyproject.toml`, which also pins `include = ["src", "tests"]`
and `exclude = [..., "frontend"]`. Pinning the mode in configuration, rather than
passing a flag, means the gate cannot silently drift away from Principle 8 by a
command-line edit alone — but it also means anyone auditing the gate must read
`pyproject.toml`, not just this document, to confirm strictness.

Step 4 enforces **branch coverage >= 70%** via `pytest-cov`, failing fast if coverage drops below the threshold.

---

## 4. CLI Implementation Architecture

The `ucx` CLI is built in Python using `typer` and `rich`, providing interactive progress bars, formatted error trees, and instant colorized execution logs.

### `ucx run` exit status and output streams

`ucx run --prompt TEXT` is the way a script drives an agent, and a script can tell a failed
turn from an answer only by the exit status. So a failed turn exits non-zero, and its error
goes to stderr rather than to stdout, where the reply goes (#953). Setup refusals that predate
#953 still print on stdout; the table says which.

**stdout carries the reply and nothing else (#960).** With `--prompt`, stdout is the model's
reply as it wrote it — exactly, through a pipe; with its control characters neutralised on a
terminal (see below) — followed by one newline, written as plain text, not through
Rich. Escaping markup alone would not be enough: Rich also wraps at the console width (80
columns when stdout is a pipe) and expands tabs, so `print(arr[i])`, a long line or a tab
would still not reach a script as written. An empty reply writes nothing to stdout. Everything
meant for a person — the agent's name above the reply, `Resumed session …`, the `--reset` and
`--compact` notices, the spinner, `(No response content)` — goes to stderr, with Rich styling.

Nothing is taken away either (#970): trailing whitespace, CRLF line endings and control
characters (ESC, BEL, BS, CR) reach stdout as the model wrote them, and a reply of only
whitespace is a reply — written, with no placeholder. Only an empty reply writes nothing.

**A terminal is a renderer; a pipe is a byte channel, and `ucx run` treats them differently
(#1282).** The open decision #970 left here is now taken:

* **Through a pipe or a file** — `sys.stdout.isatty()` is false — nothing is touched. The
  bytes above are the guarantee, and a caller redirecting `ucx run` somewhere is entitled to
  exactly what the model wrote. Rewriting them there would be data corruption, not safety.
* **On a terminal** — `isatty()` is true — the reply's control characters are rendered as
  visible carets before it is written: C0 except `\n` and `\t`, plus DEL and C1, so ESC
  becomes `^[`, BEL `^G`, CR `^M`, DEL `^?`, and a C1 byte the 7-bit sequence it is
  equivalent to (`0x9B` → `^[[`). Written raw, a reply *instructs* the terminal: the measured
  case is OSC 52, which writes the user's clipboard, and the family also covers CSI cursor
  moves that overwrite what is on screen and CR that rewrites the current line. A reply is
  model output, and on a tool-using turn it can carry text a third party put there.
  Neutralising rather than deleting is deliberate — the reader can still see that something
  was there, where a silent deletion would hide a manipulation attempt about as well as
  executing it does.

A stdout that cannot answer `isatty()` — a closed stream, a descriptor that cannot be
queried, a substitute that does not implement it — counts as **not** a terminal, which keeps
the byte-fidelity guarantee and cannot turn a writable reply into an unhandled exception.

**The REPL neutralises unconditionally.** It prompts and reads, so its output is always for a
person, and it is already not byte-faithful (Rich wraps and styles it), so no caller can
depend on those bytes. Before #1282 that path was protected only by accident — `escape()` let
ESC through and Rich's highlighter merely broke the sequence up, which `highlight=False`
would have silently removed.

| Exit | Meaning | stdout | stderr |
| :--- | :--- | :--- | :--- |
| `0` | The turn answered and the session was saved. | The reply — verbatim through a pipe, control characters neutralised on a terminal (#1282). | The agent's name; resume/reset/compact notices. |
| `1` | **The turn failed**: `execute_turn` returned a result with `error` set — an unreachable provider, a budget ceiling, the step budget (even when partial content came back), a turn a `PRE_TURN` hook blocked — or raised. `FAILED_TURN_EXIT_CODE`. | No reply. | `✖ Execution failed: <error>`, or `✖ Turn blocked by hook: <reason>` when `TurnResult.stop_reason` is `blocked_by_hook` (#970: the wording of `content` decides nothing). |
| `3` | **The turn answered, but the session was not saved** (the store raised). `UNSAVED_SESSION_EXIT_CODE`. A later `--session-id` call does not resume this turn. A turn that failed *and* was not saved exits `1`, with both reports on stderr. | The reply — verbatim through a pipe, control characters neutralised on a terminal (#1282). | `✖ Session not saved: <ExceptionClass>: <message>` |
| `1` | **The turn answered and was saved, but the reply could not be written** to stdout: a closed pipe (`BrokenPipeError`), any other error writing to the descriptor (`OSError`, such as stdout opened for reading), or an encoding that cannot represent the reply (`UnicodeEncodeError`). `UNWRITTEN_REPLY_EXIT_CODE`. After any `OSError`, file descriptor 1 is pointed at the null device, so the reply still in stdout's buffer is not flushed into the same failure at interpreter shutdown, which printed `Exception ignored … OSError` and exited `120` (#996). The session is saved *before* the reply is written (#970), so a later `--session-id` call resumes the turn; if the save failed too, both are reported, nothing says the turn was saved, and the status is still `1`. Whether this case should keep sharing `1` with a failed turn is an open decision on #970. | No complete reply. A write that fails part-way — a non-blocking pipe that fills (`BlockingIOError`), a disk that fills once some bytes fit (`ENOSPC`) — can leave the start of the reply on stdout, with nothing marking where it stops; only the status and stderr say it was not delivered (#1005). Bytes the kernel has accepted cannot be taken back, and a pipe write is all-or-nothing only up to `PIPE_BUF`. | `✖ Reply not written to stdout: <ExceptionClass>: <message>`, then `The turn was saved; --session-id resumes it.` only when the save succeeded; after `✖ Session not saved: …` when it did not. |
| `1` | The LLM connector could not be built (`LLMError`), before any turn ran. | `LLM Error: …` | — |
| `1` | A command needs an optional extra that is not installed (`MissingDependencyError`). Not relabelled as a turn failure; the console-script launcher reports it. | — | `ucx: Optional dependency …` |
| `2` | `--session-id` was refused, or collides with an existing session on this filesystem. | `Invalid --session-id` / `Unusable --session-id` | — |
| `130` | Interrupted (Ctrl+C) — the status Typer gives an interrupt in every other command. An interrupt that lands inside a session save is reported first (#970); saves are atomic, so the session is as it was at its last completed save. With `--prompt` the save comes before the reply, so no reply is written; a failed or hook-blocked turn is reported *before* the save, so an interrupt inside it still leaves `✖ Execution failed` / `✖ Turn blocked by hook` on stderr (#996). | — | `✖ Session not saved: interrupted before the save completed` (REPL: `✖ Session not saved on exit: …`) when a save was cut short; then `Interrupted by user.` |

What a failed turn leaves in the persisted session depends on where it failed. In no case is
an assistant message written for the failure, so `--session-id` never resumes an error as
something the agent replied:

| Failure | Persisted |
| :--- | :--- |
| A `PRE_TURN` hook blocked the turn | The session as it was before the prompt: the block happens before the prompt is appended, so the prompt is **not** saved. For a new session that is the system message alone. |
| The step budget ran out | The prompt, and the partial work (the assistant's tool calls and their results); no final reply. |
| Any other failure the engine reports (a provider error, a budget ceiling) | The prompt, which `execute_turn` appends right after the `PRE_TURN` hook allows the turn; no assistant message. |

**Which save and export failures change the exit status.** Saving the session does; exporting
telemetry does not:

| Where | On failure | Why |
| :--- | :--- | :--- |
| `--prompt`: saving the session after the turn — after a failed turn is reported, **before** the reply is written | Exit `3` (or `1` if the turn also failed or the reply could not be written), message on stderr. | The reply is real but lost to the next `--session-id` call; exit `0` hid that (#957). Saving first means a failure writing the reply cannot take the turn with it (#970). |
| REPL: saving the session after each turn | `⚠ Turn completed but the session was not saved` on stderr; the REPL continues and the status is unaffected. | The save on the way out retries it and decides the status. |
| REPL: saving the session on the way out | `✖ Session not saved on exit` on stderr; exit `3` when the REPL was left normally. An exception already leaving the REPL (an interrupt, a missing extra) keeps its own status. An interrupt *during* this save prints `✖ Session not saved on exit: interrupted before the save completed` and exits `130` (#970). Under `asyncio.run` a first Ctrl+C is delivered at the next `await`, after a save in progress finishes; only a second one lands inside it. | Whatever changed since the last successful save is gone. |
| Exporting telemetry spans (both modes) | Logged; the spans are counted as dropped (`cli_export_failed`, #187). Exit status unaffected. | Telemetry observes the turn and is not part of it: a collector outage must not turn an answered, saved turn into a failure. The loss is accounted for on the tracer rather than silent. |

**Text the CLI did not write is never read as Rich markup (#960).** In both modes every
interpolated value — the reply, the messages `/history` shows, agent names, session ids, model
names, tool names and descriptions, exception text — is escaped, and the consoles do not
substitute emoji codes (`:thumbs_up:` stays text). Before #960, `arr[i]` printed as `arr`, an
unbalanced `[/bold]` in a reply raised `MarkupError` after the turn had succeeded and before it
was saved, and `/history` never showed its `[user #1]` role labels.

In the **interactive REPL** a failed turn does not end the session: the failure is printed on
stderr with the same labels (`✖ Execution failed:` / `✖ Turn blocked by hook:`), is not printed
or stored as the agent's reply, and the next prompt runs normally. The REPL's replies, notices
and slash-command output stay on stdout. Leaving with `/exit` exits `0` (or `3`, above); an
interrupt that escapes the REPL loop exits `130`.

### REPL slash commands

| Command | Effect |
| :--- | :--- |
| `/help` | Command manual. |
| `/reset` | Purge the session to its system prompt and zero its turn counter, via `BaseAgent.reset_session`. |
| `/compact` | Compact the session context now, via `BaseAgent.compact_session` (P5), reporting which producer wrote the ledger. |
| `/status` | Agent state, session ID, message count, turns executed. |
| `/history` | The session's messages. |
| `/exit`, `/quit` | Leave, persisting the session first. |

`/reset`, `/status` and `/history` reach the agent only through its public API.
Until #183 they assigned and read `agent._history` behind three
`pyright: ignore[reportPrivateUsage]` pragmas — which was the falsifiable proof that no
Core session API existed — and `/reset` seeded `config.system_prompt or ""`, inserting an
**empty** `SYSTEM` message that `BaseAgent.__init__` never produces, while resetting
neither turn counter.

