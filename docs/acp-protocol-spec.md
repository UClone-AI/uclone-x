# ACP (Agent Client Protocol) Conformance Specification

> [!NOTE]
> **Implementation status (2026-09-12): Core ACP server shell adapter is implemented
> under `src/uclone_x/shells/acp/` (Issue #649, Decision #575).**
> Standard stdio JSON-RPC transport and core methods (`initialize`, `new_session`,
> `load_session`, `set_session_mode`, `set_config_option`, `prompt`, `cancel`,
> `request_permission`) are implemented and verified against unit tests.

---

## 0. How to read this document

The vocabulary is the one fixed by `docs/a2a-protocol-spec.md`, and is used identically here.
It is repeated rather than cross-referenced because substituting these words for one another is
what produced finding `2026-09-02-004`.

**Citation convention — deliberately *not* the A2A document's.** That document reserves a bare
`§N` for the external specification and hyperlinks its own sections. Here a bare `§N` is
**always a section of this document**, and references to ACP's own specification or to the SDK
name the source explicitly (`acp/schema.py`, `acp/interfaces.py`, a URL). An earlier version of
this paragraph said the convention was adopted "identically" and then used bare self-references
throughout, which meant the document violated the rule it cited in the sentence that cited it.
One convention consistently applied is worth more than the other one claimed.

| Term | Meaning |
| :--- | :--- |
| **Specified** | This repository has written down what it intends to do. A claim in a `docs/` file. Costs nothing and proves nothing. |
| **Conformant** | The written specification agrees with the published ACP specification at a named version. Checkable by reading two documents side by side. |
| **Implemented** | Code exists in `src/uclone_x/`, is exercised by tests, and passes `./ucx test check`. |
| **Verified conformant** | Implemented code has been run against a third-party ACP client. **Nothing in UClone-X has reached this state, and no such claim may be made until it does.** |

Two further labels are used in the tables:

* **Not implementable** — a method we cannot honour against today's architecture, with the
  reason named. This is distinct from *not implemented*, and the distinction is the point of
  §3: a method that is merely unbuilt is scheduled work, while a method that cannot be honoured
  is a design gap that must be visible before a client depends on it.
* **Out of scope** — deliberately not offered. A reason is required.

**A method is marked *Implemented* only once a test exercises it.** `docs/a2a-protocol-spec.md`
takes the same position — its §1.1 conformance rows carry `Implemented: No` against
requirements that are `Specified: Yes` — and this table says the same thing on day one.

Two corrections to an earlier version of this paragraph, both of which invented an authority:

* It quoted P2's A2A table as carrying "nothing is implemented yet and no conformance is
  claimed". **That sentence exists in no source in this repository.** It was a paraphrase
  presented as a quotation.
* **P2 has no table.** `docs/principles/details/p2-google-a2a-interoperability.md` is eleven
  lines and explicitly delegates the version pin and object shapes to
  `docs/a2a-protocol-spec.md`, forbidding their restatement. The conformance table lives in
  that document, not in the principle.

The position is right and did not need a fabricated citation to support it. Noted here rather
than silently corrected, because inventing a source is the failure this document exists to
prevent in the ACP surface.

---

## 1. Normative reference and version pinning

| | |
| :--- | :--- |
| Protocol | Agent Client Protocol (ACP), Zed Industries |
| Python SDK | **`agent-client-protocol==0.12.1`** |
| SDK source | `https://github.com/agentclientprotocol/python-sdk` |
| Surface read for this document | `acp/interfaces.py` (`Agent`, `Client` protocols), `acp/schema.py` |
| Date checked | 2026-09-07 |

The method surface in §2 and §4 is taken from the SDK's own `Agent` and `Client` Protocol
classes at 0.12.1, not from prose. That is deliberate: the SDK's protocol classes are what a
client will actually call, and a table derived from them cannot silently omit a method.

**Version drift.** Once ACP is adopted, `pyproject.toml` pins the version and a fitness test
asserts that the pin and the version named in this document are the same string. Until adoption
there is no dependency to compare against, so the test is specified in §7 and not yet written.
Naming a version in a document with nothing checking it is how a conformance claim goes stale
invisibly, so this is the one acceptance criterion that must not be deferred past the first
line of ACP code.

An earlier version of this paragraph called that "the shape
the principles-normative fitness test uses for the principles". It is not: that test
compares markdown against markdown and the `details/` directory, and never opens
`pyproject.toml`. **There is no existing document-to-dependency-pin assertion in this
repository to copy**, so the work is a new fitness test rather than a variation on one, and
saying otherwise understated it.

This is also a **deferral of a stated acceptance criterion**. #578 lists the fitness test
unconditionally; it cannot be written before the dependency exists. That is negotiated on #578
rather than absorbed here.

**ACP is pre-1.0 and its method surface is still moving.** deepagents pins
`agent-client-protocol>=0.10.1` and locks `0.10.1`; 0.12.1 adds `fork_session`,
`resume_session`, `close_session`, `list_sessions` and the elicitation methods. A conformance
table against a moving protocol is worth more than one against a frozen one, not less — but
its version line is load-bearing.

---

## 2. Agent-side conformance table

These are the methods **a client calls on us**. This is the surface an ACP server shell must
answer.

| ACP method | UClone-X counterpart | Specified | Implemented | Note |
| :--- | :--- | :--- | :--- | :--- |
| `initialize` | capability negotiation; `AgentCard` is the A2A analogue, not a substitute | Yes | **Yes** | Reports real capabilities (§3.3) |
| `new_session` | A per-session agent (`agent_config_for_persona` + `compose_agent`) + `BaseAgent.persist_session` | Yes | **Yes** | Rejects `AcpMcpServer` per §5. Each session gets its own agent, so no session's history reaches another's request (#1454) |
| `load_session` | `SessionStore.load` + `BaseAgent.hydrate_session` | Yes | **Yes** | The saved history is restored into the session's own agent (#1454) |
| `list_sessions` | `SessionStore.list_session_ids` | Yes | **No** | Ours returns ids only; ACP's response and its `cursor` pagination need more. §3.4 |
| `fork_session` | *(none)* | Yes | **No** | Needs a copy-with-new-id; `SessionStore` has no fork |
| `resume_session` | `SessionStore.load` | Yes | **No** | Distinct from `load_session` in ACP; the difference must be honoured, not collapsed |
| `close_session` | *(none)* — `delete` destroys | Yes | **No** | Close is not delete. Mapping it to `SessionStore.delete` would lose the session |
| `set_session_mode` | `ACPServer._handle_set_session_mode` | Yes | **Yes** | Configures session permission mode (`auto`, `ask`, `readonly`) |
| `set_config_option` | `ACPServer._handle_set_config_option` | Yes | **Yes** | Configures session options |
| `authenticate` | *(none)* | Yes | **No** | **Out of scope** for the local stdio head — see §3.5 |
| `prompt` | `BaseAgent` turn execution + `BaseAgent.persist_session` | Yes | **Yes** | The core mapping. Every turn is saved, with its event log; an id no `new_session` or `load_session` opened is refused (#1454). One turn at a time per session: a `prompt` while that session's turn is in flight is refused in words, not started beside it (#1494). A turn that ended in an error answers with a JSON-RPC error, not `completed` with empty output: a context-window refusal is sent in its own plain words; a usage limit, the step limit or a blocking hook is named plainly (only the usage limit is said to stop a retry; the step budget resets every turn); any other error is a plain failure. Every error not sent as is goes only to the log (#1509) |
| `cancel` | `ACPServer._handle_cancel` | Yes | **Yes** | Turn-scoped task cancellation (§3.1). The turn stays recorded as in flight until its task has actually ended, so its agent is not released while it is still running (#1494) |
| `ext_method` / `ext_notification` | — | Yes | **No** | ACP's extension escape hatch |
| `on_connect` | — | Yes | **No** | SDK plumbing, not a protocol decision |

---

## 3. The gaps, in order of how much they matter

### 3.1 `cancel` — turn-scoped cancellation already exists, on the other protocol surface

ACP's `cancel(session_id)` means *abort the prompt turn now in flight; the session remains
usable and the client may prompt again*. It is a notification, so there is no response the
agent can use to decline.

**This repository already implements that operation.** `A2AServer.cancel_task_endpoint`
(`src/uclone_x/a2a/server.py:401`) does:

```python
if record.async_task is not None and not record.async_task.done():
    record.async_task.cancel()
record.status = TaskStatus.CANCELED
```

`record.async_task` is `asyncio.create_task(self._execute_managed_task(record))`
(`server.py:345`), and `_execute_managed_task` awaits `self._agent.execute_turn(prompt)`
(`server.py:437`). Cancelling it aborts the turn and **leaves the agent untouched** — no
`stop()`, no `TERMINATED`, and the agent can take another prompt. That is ACP's semantics,
already built, forty lines away on the sibling protocol.

Two earlier statements about this are withdrawn, and both erred in the same direction:

* #575 said "nothing in `BaseAgent` cancels a turn in flight".
* This section previously said the *corrected* version was "the only cancellation we have is
  terminal, and ACP's is not", citing `BaseAgent.stop()` (`agent/base.py:1355`, which does
  cancel the loop task, close the subscription and transition to `AgentState.TERMINATED`).

`stop()` is indeed terminal and indeed the wrong mapping. But it is not the only cancellation,
and describing it as such told #576's implementer to build the mechanism from scratch or
declare the method unimplementable, when the model to follow is in `src/uclone_x/a2a/`.

**The real gap is addressing and cooperation, not absence:**

* **Addressing.** A2A cancels a *task id* it minted and holds in `self._tasks`. ACP cancels a
  *session id*. An ACP shell needs a session→in-flight-task map of its own; it cannot reuse
  `_tasks`, which is keyed differently and owned by the A2A server.
* **Cooperation.** `execute_turn` has no cancellation checkpoint. `asyncio.Task.cancel()`
  raises at the next `await`, so a long synchronous stretch — or a tool call that shields
  itself — delays the abort, and what a half-cancelled turn leaves behind (a partial log
  append, an open tool call) is unspecified. A2A does not answer this either; it sets
  `TaskStatus.CANCELED` and moves on.

**A trap worth naming, since §3.2 names its sibling.** `EventType.INTERRUPT` is the mapping an
implementer searching for "cancel" reaches for first. Its handler
(`src/uclone_x/agent/base.py:1482`) is:

```python
elif event.type == EventType.INTERRUPT:
    self.transition_to(AgentState.IDLE)
    return True
```

It cancels **nothing**. It marks the agent idle while the turn continues running. Routing ACP's
`cancel` there would report success for something that did not happen, which P6 forbids and
which is harder to notice than a missing method.

So #576's choice is between reusing A2A's mechanism under ACP's addressing, and declaring
`cancel` **not implementable** until `execute_turn` has a checkpoint. What is no longer on the
table is "we would have to invent this".

### 3.2 `close_session` must not be mapped to `delete`

`SessionStoreProtocol` (`src/uclone_x/core/session_store.py`) declares `load`, `save`, `delete`,
`list_session_ids`. The concrete `SessionStore` (`src/uclone_x/agent/session.py`) adds
`storage_dir`, `session_path`, `reap_orphaned_temp_files` and `cleanup_session_artifacts` — an
earlier version of this paragraph attributed the protocol's four methods to the class and
implied the enumeration was exhaustive.

Neither has a *close*. The tempting mapping — `close_session` → `delete` — **destroys the
session record**, when ACP means release the resources and keep the record. This is written down because it is the mapping a
reasonable implementer reaches for, and it is silently lossy.

`fork_session` has the same shape with the opposite risk: there is no fork, and building one on
`load` + `save` with a new id must decide what the append-only log
([#526](https://github.com/UClone-AI/uclone-x/issues/526)) does at a fork point. That interacts
with #565/#566 and is not a question a shell should answer alone.

### 3.3 `initialize` must report real capabilities

`initialize` is where an agent tells a client what it can do. Every **Not implementable** row
in this table has to be reachable from that response, or the client discovers the gap by
calling the method. A hardcoded capability block is the mechanism by which a conformance claim
becomes false without anyone editing a document — so the capability response must be derived
from the same source as this table, not written in parallel with it.

### 3.4 `list_sessions` is narrower on our side than it looks

`SessionStore.list_session_ids()` returns a tuple of ids. ACP's `ListSessionsRequest` takes a
`cwd` filter and an opaque `cursor` for pagination, and its response carries session
information rather than bare ids. Mapping ours onto it means either extending the store or
declaring the filter and pagination unsupported. Returning every id and ignoring `cursor` would
be a P6 violation of the same kind as §3.1.

### 3.5 `authenticate` is out of scope, with a reason

For a local stdio head the client is a process the user launched on their own machine; there is
no third party to authenticate. Declaring `authenticate` **out of scope** is correct there and
becomes wrong the moment a remote transport is offered (§6). It is recorded as out of scope
*for stdio*, which is a scoped claim rather than a permanent one.

---

## 4. Client-side surface — what we would call on the client

Listed because it is where the cost of adoption actually falls: these are capabilities the
*client* provides, and using them means our tools stop doing the work themselves.

| ACP client method | Bears on | Note |
| :--- | :--- | :--- |
| `session_update` | `AgentEvent` → ACP update translation | The main translation surface. **Blocked on [#566](https://github.com/UClone-AI/uclone-x/issues/566)**, which changes what the turn loop emits — #576 must not freeze a shape #566 will move |
| `request_permission` | [#474](https://github.com/UClone-AI/uclone-x/issues/474) approval seam | ACP makes permission a **first-class request/response**; our `TOOL_APPROVAL_RESPONSE` event with a 30s timeout (`agent/base.py:2095`) is the nearest existing thing. The timeout is a decision ACP does not have and would need stating |
| `read_text_file` / `write_text_file` | `tools/builtin/filesystem.py` | Routing file I/O through the client is how an editor shows unsaved buffers. Optional, and a real behaviour change |
| `create_terminal`, `terminal_output`, `release_terminal`, `wait_for_terminal_exit`, `kill_terminal` | `tools/builtin/shell.py` | Five methods. **This is where the isolation question bites**: a client-hosted terminal runs outside our sandbox model entirely, which is design-review finding `2026-09-02-042` in a new place (the Markdown design-review register was retired in #1037 and survives frozen as an eval fixture) |
| `create_elicitation` / `complete_elicitation` | *(none)* | Agent-initiated questions to the user. No counterpart |
| `ext_method` / `ext_notification` | — | ACP's extension escape hatch |
| `on_connect` | — | SDK plumbing, not a protocol decision. Listed because §1 promises this table cannot silently omit a method, and an earlier version omitted exactly this one while §2 listed it for the agent side |

None of these are required to answer `prompt`. They are the extension surface, and each one
adopted is a behaviour that leaves our control — so each needs its own decision rather than
being taken as a set.

---

## 5. MCP servers arrive over this protocol

`new_session`, `load_session`, `fork_session` and `resume_session` all carry
`mcp_servers: list[HttpMcpServer | SseMcpServer | AcpMcpServer | McpServerStdio]`.

Measured in [#577](https://github.com/UClone-AI/uclone-x/issues/577) —
the ACP/MCP descriptor-overlap note. Three results bear on conformance:

* **`AcpMcpServer` is not implementable against today's model.** It carries a `serverId` and no
  address, because the MCP server is reached back down the ACP connection to the client. There
  is no local process and no URL, and `MCPConnectionConfig` has no shape for it. Recorded here
  as **not implementable**, not as "MCP is supported".
* `McpServerHttp` is MCP streamable HTTP. `MCPTransport` has no member for it
  ([#579](https://github.com/UClone-AI/uclone-x/issues/579)), and a descriptor of that shape is
  silently inferred as SSE rather than refused.
* Client-supplied descriptors must not go through `MCPConfigFileLoader.parse_config_dict`. It
  expands `${VAR}` from the host environment, so a client could name any host environment
  variable and receive its value in the argv of a process the same client chose.

## 6. Transport scope

**Supported: local JSON-RPC over stdio. Deferred: everything else.**

The reason is ACP's, not ours. At 0.12.1 the SDK ships remote transports and labels them
itself: `acp/http/__init__.py` opens *"Streamable HTTP transport for ACP (experimental)"* and
`acp/ws/__init__.py` *"WebSocket transport for ACP (experimental)"*. Both sit behind an
optional `[http]` extra and raise `ImportError` without it.

This corrects the looser phrasing used when #578 was filed ("remote transport is a work in
progress"). Remote transport **exists** at 0.12.1; it is marked experimental by its own authors
and is not installed by default. That is a weaker statement and the accurate one.

Deferring remote transport is also what keeps §3.5 honest: `authenticate` is out of scope
*because* the client is a local process. Adding a remote transport reopens it, along with every
isolation question in §4.

---

## 7. Acceptance criteria for the first ACP code

Carried here so the implementation has them in one place:

- [ ] `pyproject.toml` pins `agent-client-protocol` to the version named in §1.
- [ ] A fitness test asserts that pin and §1's version string are equal, so document and
      dependency cannot drift apart silently.
- [x] Every row in §2 that becomes *Implemented* is exercised by a test in the same change.
- [x] `initialize`'s capability response is derived from the same source as §2's table (§3.3).
- [x] `cancel` is either turn-scoped or reported as not implementable (§3.1). It is not
      accepted-and-ignored, and not mapped to `stop()`.
- [x] `close_session` is not mapped to `SessionStore.delete` (§3.2).
- [x] `AcpMcpServer` is refused with a stated reason rather than silently dropped (§5).

## 8. What this document does not do

It amends no principle. It does not decide adoption — that was decided and approved in #575.

