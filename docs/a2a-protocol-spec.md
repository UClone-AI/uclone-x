# A2A (Agent2Agent) Protocol Conformance Specification

> [!NOTE]
> **Implementation status (2026-09-02)**: `src/uclone_x/a2a/` and `./ucx a2a serve` are implemented:
> - `models.py` (`AgentCard`, `TaskMessage`, `TaskResult`, `TaskStatus`, `AgentSkill` strictly conforming to A2A v1.0.1)
> - `protocols.py` (`A2ATransportProtocol`, `A2ADiscoveryProtocol`)
> - `in_memory.py` (`A2AInMemoryTransport` for zero-copy local fastpath)
> - `http_transport.py` (`A2AHttpTransport` for remote REST/SSE transport)
> - `server.py` (`A2AServer` exposing RFC 8615 `/.well-known/agent-card.json`, task dispatch, and `/a2a/v1/tasks*` REST/SSE endpoints)
> - `src/uclone_x/cli/commands/a2a.py` (`./ucx a2a serve` CLI subcommand hosting standalone A2A gateway and federated nodes)
> - `discovery.py` (`A2ADiscoveryService`)
> - `wire.py` (`task_result_to_wire`, `task_result_from_wire_json`)
> Transport and wire serialization preserve in-band Provenance and dynamically recompute degraded status per Principle 6.

---

## 0. How to read this document — *specified*, *conformant*, *implemented*

Conflating these three words is what produced finding 2026-09-02-004. They are used
here with fixed meanings and must not be substituted for one another anywhere in this
repository:

| Term | Meaning |
| :--- | :--- |
| **Specified** | This repository has written down what it intends to do. A claim in a `docs/` file. Costs nothing and proves nothing. |
| **Conformant** | The written specification agrees with the published A2A specification at a named version. Checkable by reading two documents side by side. A document can be *specified* and non-conformant — which is precisely the defect that was filed. |
| **Implemented** | Code exists in `src/uclone_x/`, is exercised by tests, and passes `./ucx test check`. |
| **Verified conformant** | Implemented code has been run against an independent A2A conformance suite or a third-party A2A client. **Nothing in UClone-X has reached this state, and no such claim may be made until it does.** |

Two further labels are used inline:

* **Extension** — a UClone-X-specific mechanism that goes beyond the standard. It is
  permitted by the standard's own extensibility rules (cited where used), but a
  third-party A2A client is not required to understand it and MUST still be able to
  interoperate without it.
* **Unverified** — a claim this document could not confirm against a primary source.
  Collected in [§12](#12-unverified-claims). An unverified claim is left labelled
  rather than silently asserted.

**Citation convention.** A bare `§N` refers to a section of the **A2A specification** at
the pinned version; sections of *this* document are always written as a hyperlink, e.g.
[§10](#10-uclone-x-extension-in-process-zero-copy-fastpath). The two numbering schemes
overlap, so the distinction matters: "§12" is A2A's Custom Binding Guidelines, while
[§12](#12-unverified-claims) is this document's list of unverified claims.

---

## 1. Conformance table

Checked against **A2A v1.0.1** (see [§2](#2-normative-reference-and-version-pinning) for
the exact sources) on **2026-09-02**.

Legend — **Specified**: written down in this document. **Conformant**: what is written
matches v1.0.1. **Implemented**: code exists in `src/uclone_x/`.

### 1.1 Mandatory requirements

| # | Requirement (A2A v1.0.1) | Specified | Conformant | Implemented |
| :--- | :--- | :--- | :--- | :--- |
| R1 | Server MUST publish an Agent Card (§8.1) | Yes | Yes | **No** |
| R2 | Agent Card discoverable at `https://{server_domain}/.well-known/agent-card.json` (§8.2) | Yes | Yes | **No** |
| R3 | Agent Card MUST declare `supportedInterfaces[]`, ordered by preference (§8.3.1) | Yes | Yes | **No** |
| R4 | Agent Card required fields: `name`, `description`, `supportedInterfaces`, `version`, `capabilities`, `defaultInputModes`, `defaultOutputModes`, `skills` (proto `AgentCard`) | Yes | Yes | **No** |
| R5 | `capabilities` is an `AgentCapabilities` object of feature flags, **not** a list of callable functions (§4.4.3) | Yes | Yes | **No** |
| R6 | Callable abilities are declared in `skills[]` as `AgentSkill` objects (§4.4.5) | Yes | Yes | **No** |
| R7 | Implement all eleven core operations of §3.1 | Yes | Yes | **No** |
| R8 | Task lifecycle uses the `TaskState` enum with ProtoJSON SCREAMING_SNAKE_CASE values (§4.1.3, §5.5) | Yes | Yes | **No** |
| R9 | JSON serialisations MUST use camelCase field names (§5.5) | Yes | Yes | **No** |
| R10 | At least one standard binding — JSON-RPC 2.0 (§9), gRPC (§10) or HTTP+JSON/REST (§11) | Yes | Yes | **No** |
| R11 | Streaming, where offered, uses the binding's mechanism — SSE for the JSON-RPC and REST bindings (§9.1, §11.1, §11.7) | Yes | Yes | **No** |
| R12 | Streaming events MUST be delivered in generation order (§3.5.2) | Yes | Yes | **No** |
| R13 | Clients MUST send the `A2A-Version` service parameter; servers MUST honour it or return `VersionNotSupportedError` (§3.6) | Yes | Yes | **No** |
| R14 | Undeclared optional capabilities MUST be refused with the mapped error (§3.3.4) | Yes | Yes | **No** |
| R15 | A2A error types MUST map to binding-native codes (§5.4) | Yes | Yes | **No** |
| R16 | Multiple supported bindings MUST be functionally equivalent (§5.1) | Yes | Yes | **No** |
| R17 | Credentials transmitted per the Agent Card's declared `securitySchemes`; no protocol-level handshake exists (§7) | Yes | Yes | **No** |

### 1.2 Optional requirements

| # | Requirement (A2A v1.0.1) | Specified | Conformant | Implemented |
| :--- | :--- | :--- | :--- | :--- |
| O1 | Streaming (`capabilities.streaming`) — `SendStreamingMessage`, `SubscribeToTask` | Yes | Yes | **No** |
| O2 | Push notifications (`capabilities.pushNotifications`) — the four config operations + webhook delivery (§3.5.3, §4.3) | Yes | Yes | **No** |
| O3 | Extended Agent Card (`capabilities.extendedAgentCard`) — `GetExtendedAgentCard` | **No** — deliberately out of scope for v0.2.0 | n/a | **No** |
| O4 | Agent Card JWS signatures (§8.4) | **No** — out of scope for v0.2.0 | n/a | **No** |
| O5 | Multi-tenant routing via `AgentInterface.tenant` (§8.3.2) | **No** — single-tenant by design (P3) | n/a | **No** |
| O6 | Protocol extensions (`capabilities.extensions`, §4.6) | Yes — used by the fastpath, [§10](#10-uclone-x-extension-in-process-zero-copy-fastpath) | Yes | **No** |
| O7 | Agent Card HTTP caching headers (§8.6) | Yes | Yes | **No** |

### 1.3 UClone-X extensions beyond the standard

| # | Extension | Standard basis | Specified | Implemented |
| :--- | :--- | :--- | :--- | :--- |
| X1 | In-process zero-copy fastpath as a **custom protocol binding** | §5.8 + §12 (Custom Binding Guidelines) permit custom bindings identified by a URI | Yes — [§10](#10-uclone-x-extension-in-process-zero-copy-fastpath) | **No** |
| X2 | Token metrics attached to task updates (cost is out of scope, #1392) | §4.6 protocol extension mechanism | Partial — extension URI reserved, payload not yet defined | **No** |

> **The single most important row of this table is the `Implemented` column: it is
> `No` throughout.** Any statement elsewhere in this repository that UClone-X "fully
> complies with" A2A is false today. See [§13](#13-corrections-owed-by-other-documents).

---

## 2. Normative reference and version pinning

### 2.1 Conformance target

UClone-X targets **A2A specification version 1.0.1**, wire protocol version **`1.0`**.

Patch numbers do not affect protocol compatibility and MUST NOT appear in requests,
responses or Agent Cards; only the `Major.Minor` pair is negotiated (§3.6). So the
`protocolVersion` value UClone-X emits is `"1.0"`, and the `A2A-Version` service
parameter value is `1.0` — never `1.0.1`.

### 2.2 Sources checked

Verified on **2026-09-02**:

| Source | URL | Role |
| :--- | :--- | :--- |
| A2A Protocol Specification v1.0.1 (prose) | <https://a2a-protocol.org/v1.0.1/specification/> · raw: `https://raw.githubusercontent.com/a2aproject/A2A/v1.0.1/docs/specification.md` | Normative prose requirements |
| `specification/a2a.proto` @ tag `v1.0.1` | `https://raw.githubusercontent.com/a2aproject/A2A/v1.0.1/specification/a2a.proto` | **Single authoritative normative definition** of every protocol data object (spec §1.4) |
| A2A release list | <https://github.com/a2aproject/A2A/releases> | v1.0.1 published 2026-05-28; v1.0.0 published 2026-03-12 |
| "A2A Protocol Ships v1.0" | <https://a2a-protocol.org/latest/blog/2026/03/12/a2a-protocol-ships-v10-production-ready-standard-for-agent-to-agent-communication/> | v1.0 stability announcement |
| "A New Chapter for A2A: Joining the Agentic AI Foundation" | <https://a2a-protocol.org/latest/blog/2026/08/27/a-new-chapter-for-a2a-joining-the-agentic-ai-foundation/> | Governance |
| "What's New in v1.0" | <https://a2a-protocol.org/latest/whats-new-v1/> | v0.3 → v1.0 breaking-change list |
| Generated JSON Schema bundle | <https://a2a-protocol.org/latest/spec/a2a.json> | **Explicitly non-normative** build artifact (spec §1.4) |

**Precedence.** Where the prose and the proto disagree, the proto wins: spec §1.4 states
that `a2a.proto` "is the single authoritative normative definition of all protocol data
objects and request/response messages", and that derived schemas MUST be regenerated
from it rather than hand-edited.

One such disagreement exists inside v1.0.1 itself and is called out here so that no
reader mistakes it for a UClone-X error: the sample Agent Card in v1.0.1 §8.5 writes the
security requirement as `"security": [{ "google": ["openid", ...] }]`, while the v1.0.1
proto defines `repeated SecurityRequirement security_requirements` → JSON
`securityRequirements`, whose entries are `{"schemes": {"<name>": {"list": [...]}}}`.
The A2A `main` branch has since corrected the sample to match the proto. **UClone-X
follows the proto** (`securityRequirements`).

### 2.3 Governance and attribution

A2A is **not** a Google standard, and this repository must stop calling it one.

* Announced by Google in April 2025 (<https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/>).
* Donated by Google to the **Linux Foundation** in June 2025 (<https://developers.googleblog.com/en/google-cloud-donates-a2a-to-linux-foundation/>).
* Accepted as a **Growth Stage project at the Agentic AI Foundation (AAIF)**, a
  Linux-Foundation-directed body, announced by the project on **2026-08-27**
  (<https://a2a-protocol.org/latest/blog/2026/08/27/a-new-chapter-for-a2a-joining-the-agentic-ai-foundation/>),
  alongside MCP, goose and AGENTS.md.

The correct attribution is therefore: **"A2A (Agent2Agent) Protocol, an Agentic AI
Foundation / Linux Foundation project, originally created by Google."**

> Finding 2026-09-02-004 dated the AAIF transfer to 2026-08-20. The project's own
> announcement is dated **2026-08-27**; press coverage appeared from 2026-08-17
> onward. The finding's substance stands; only the date needed correcting.

### 2.4 Re-verification duty

This document hard-codes concrete external details (a well-known path, method names, an
enum). That is acceptable *here* because this file is amendable. It is **not** acceptable
in `docs/principles/`, which is why P2 must reference this specification and its version
rather than restating any path or endpoint — see
[§13.3](#133-p2-docsprinciplesdetailsp2-google-a2a-interoperabilitymd).

Whoever raises the pinned version in [§2.1](#21-conformance-target) MUST re-check every
row of [§1](#1-conformance-table) and update the verification date in
[§2.2](#22-sources-checked) in the same change.

---

## 3. Discovery — the Agent Card

### 3.1 Location

```text
GET https://{server_domain}/.well-known/agent-card.json
```

Per v1.0.1 §8.2, clients may also obtain a card from a registry/catalogue or by direct
configuration. The well-known URI is the only path form the specification defines.

**The pre-0.3 path `/.well-known/agent.json` is wrong.** A v1.0 client never looks
there. Every occurrence of it in this repository is a defect
([§13](#13-corrections-owed-by-other-documents)).

There is no A2A schema URL to put in a `$schema` key. The only published JSON Schema
bundle (<https://a2a-protocol.org/latest/spec/a2a.json>) is generated from the proto and
labelled non-normative by §1.4; `https://schemas.google.com/a2a/v1/agent.json`, which
the previous version of this document asserted, does not exist. **UClone-X Agent Cards
MUST NOT emit a `$schema` field.**

Servers SHOULD send `Cache-Control: max-age=…` and an `ETag` derived from the card's
`version`; clients SHOULD revalidate with `If-None-Match` (§8.6).

### 3.2 Required structure

From the v1.0.1 proto (`AgentCard`), fields marked `REQUIRED`: `name`, `description`,
`supportedInterfaces`, `version`, `capabilities`, `defaultInputModes`,
`defaultOutputModes`, `skills`. Optional: `provider`, `documentationUrl`, `iconUrl`,
`securitySchemes`, `securityRequirements`, `signatures`.

`capabilities` is an **`AgentCapabilities` object of feature flags** — `streaming`,
`pushNotifications`, `extendedAgentCard` (booleans) and `extensions` (a list of
`AgentExtension`). It is **not** a list of things the agent can do. The list of things
the agent can do is `skills`, an array of `AgentSkill`:
`id`, `name`, `description`, `tags` (all required), plus optional `examples`,
`inputModes`, `outputModes`, `securityRequirements`.

`AgentSkill` has **no `inputSchema` / `outputSchema`**. A2A skills are described in
natural language and media types, not JSON Schema. UClone-X's need for typed
input/output contracts is therefore an extension concern
([§10.6](#106-typed-skill-contracts)), not something to be smuggled into the card's
standard fields.

Each `supportedInterfaces` entry is an `AgentInterface`: `url`, `protocolBinding`
(core values `JSONRPC`, `GRPC`, `HTTP+JSON`; an open string, so custom bindings use a
URI per §5.8), `protocolVersion` (`Major.Minor`), and optional `tenant`. The first
entry is the preferred one; clients pick the first entry whose binding they support
(§8.3).

### 3.3 UClone-X Agent Card — target shape

Conformant, and illustrative only: no code emits this yet.

```json
{
  "name": "UClone-X Security Auditor",
  "description": "Performs static security analysis and vulnerability scanning on source code.",
  "version": "0.2.0",
  "provider": {
    "organization": "UClone-AI",
    "url": "https://github.com/UClone-AI/uclone-x"
  },
  "documentationUrl": "https://github.com/UClone-AI/uclone-x/blob/main/docs/a2a-protocol-spec.md",
  "supportedInterfaces": [
    {
      "url": "https://agents.example.com/a2a/jsonrpc",
      "protocolBinding": "JSONRPC",
      "protocolVersion": "1.0"
    }
  ],
  "capabilities": {
    "streaming": true,
    "pushNotifications": false,
    "extendedAgentCard": false,
    "extensions": [
      {
        "uri": "https://github.com/UClone-AI/uclone-x/extensions/task-metrics/v1",
        "description": "Attaches token accounting to task status updates.",
        "required": false
      }
    ]
  },
  "defaultInputModes": ["application/json", "text/plain"],
  "defaultOutputModes": ["application/json", "text/plain"],
  "skills": [
    {
      "id": "audit-codebase",
      "name": "Codebase Security Audit",
      "description": "Scans a source tree for known CVEs, hardcoded secrets and insecure patterns, and returns findings with file, line and severity.",
      "tags": ["security", "static-analysis", "sast", "cve"],
      "examples": [
        "Audit /workspace/auth-service at high strictness.",
        "Find hardcoded secrets in the checkout service."
      ],
      "inputModes": ["application/json", "text/plain"],
      "outputModes": ["application/json"]
    }
  ],
  "securitySchemes": {
    "bearer": {
      "httpAuthSecurityScheme": {
        "scheme": "bearer",
        "description": "Bearer token issued by the deploying operator."
      }
    }
  },
  "securityRequirements": [{ "schemes": { "bearer": { "list": [] } } }]
}
```

The in-process fastpath is **not** advertised in this card. It is reachable only from
inside the same process, so publishing it to remote clients would be meaningless — see
[§10.4](#104-agent-card-declaration).

---

## 4. There is no handshake

**A2A v1.0.1 defines no handshake operation and no handshake phase.** The word
"handshake" appears exactly once in the specification, at §7.2, referring to the **TLS**
handshake. There is no session-establishment step and no session identifier in the
protocol.

The previous version of this document specified `POST /a2a/v1/handshake` returning a
"Session ID". That operation is invented; a real A2A server would answer 404. Every
mention of an A2A "handshake" or "capability negotiation handshake" in this repository
is a defect.

What actually happens before the first task:

1. **Agent Card retrieval** — fetch `/.well-known/agent-card.json` (§8.2).
2. **Binding selection** — pick the first `supportedInterfaces` entry whose
   `protocolBinding` the client supports, and use that entry's `url` (§8.3.2).
3. **Capability check** — read `capabilities` and do not attempt streaming, push
   notifications or the extended card unless the corresponding flag is `true` (§3.3.4).
4. **Credentials** — obtain credentials out of band for one of the declared
   `securitySchemes` and attach them to *every* request in the binding's native
   mechanism (§7.3). There is no login call, no token exchange, no session.
5. **Version declaration** — send `A2A-Version: 1.0` on every request (§3.6.1).

Grouping is a client-side concern, not a session: `contextId` logically groups related
tasks and messages (§3.4.1), and `taskId` identifies one unit of work (§3.4.2).

---

## 5. Data model

Object definitions are normatively the proto (§1.4). JSON uses **camelCase** field names
and **ProtoJSON enum strings** (§5.5).

### 5.1 Task lifecycle — the real state names

`TaskState`, v1.0.1 proto:

| Value | Kind | Meaning (spec wording) |
| :--- | :--- | :--- |
| `TASK_STATE_UNSPECIFIED` | — | Unknown or indeterminate. |
| `TASK_STATE_SUBMITTED` | in progress | Successfully submitted and acknowledged. |
| `TASK_STATE_WORKING` | in progress | Actively being processed by the agent. |
| `TASK_STATE_INPUT_REQUIRED` | **interrupted** | Agent requires additional user input to proceed. |
| `TASK_STATE_AUTH_REQUIRED` | **interrupted** | Authentication is required to proceed. |
| `TASK_STATE_COMPLETED` | **terminal** | Finished successfully. |
| `TASK_STATE_FAILED` | **terminal** | Finished with an error. |
| `TASK_STATE_CANCELED` | **terminal** | Canceled before completion. |
| `TASK_STATE_REJECTED` | **terminal** | Agent decided not to perform the task. |

Notes that matter for implementation:

* **"terminal" and "interrupted" are different.** Terminal =
  `COMPLETED | FAILED | CANCELED | REJECTED`; interrupted =
  `INPUT_REQUIRED | AUTH_REQUIRED`. A blocking `SendMessage` returns when either kind
  is reached (§3.2.2), but a stream closes only on a **terminal** state (§3.1.2).
* Messages sent to a task in a terminal state MUST be refused with
  `UnsupportedOperationError` (§3.1.1).
* The wire values are the SCREAMING_SNAKE_CASE strings above, not `"completed"`.
  v0.3 used kebab-case (`"input-required"`); v1.0 changed this
  (<https://a2a-protocol.org/latest/whats-new-v1/>). There is no `"progress"` state and
  no `progressPercentage` field anywhere in A2A.
* `TASK_STATE_AUTH_REQUIRED` is a request that the *client* fulfil an authorization
  need, not an authorization grant (§7.6). The agent MUST transition to it, MUST attach
  a `TaskStatus` message explaining what is required, and MUST arrange to receive the
  credential out of band unless an in-band mechanism was negotiated out of band or via
  an extension (§7.6.1). A2A `main` adds an explicit §7.6.4 stating that the transition
  alone authorizes nothing; that section is **not in v1.0.1**, but UClone-X adopts the
  rule regardless, as P6 requires it.

### 5.2 Core objects

* **`Task`** — `id` (required), `contextId`, `status` (required), `artifacts[]`,
  `history[]` (Messages), `metadata`.
* **`TaskStatus`** — `state` (required), `message`, `timestamp` (ISO 8601 UTC, §5.6.1).
* **`Message`** — `messageId` (required), `role` (required), `parts[]` (required),
  `contextId`, `taskId`, `metadata`, `extensions[]`, `referenceTaskIds[]`.
* **`Role`** — `ROLE_UNSPECIFIED`, `ROLE_USER`, `ROLE_AGENT`. Not `"user"`/`"agent"`.
* **`Part`** — a `oneof` over `text`, `raw` (bytes, base64 in JSON), `url`, `data`
  (arbitrary JSON), plus the common fields `mediaType`, `filename` and `metadata`,
  available for all part types. v1.0 removed the separate
  `TextPart`/`FilePart`/`DataPart` types and the `kind` discriminator; presence of the
  member field discriminates. `mimeType` was renamed `mediaType`.
* **`Artifact`** — `artifactId` (required), `parts[]` (required), `name`,
  `description`, `metadata`, `extensions[]`.
* **`StreamResponse`** — a `oneof` over `task`, `message`, `statusUpdate`,
  `artifactUpdate`. Used identically by streaming and by push-notification webhooks
  (§4.3.3).

### 5.3 Messages carry communication; Artifacts carry results

Per §3.7, Messages carry the conversation — task initiation, clarification requests,
status commentary, further input — and SHOULD NOT be used to deliver task outputs.
Results SHOULD be returned as `Artifact`s associated with the `Task`.

The previous version of this document put results in a `result` object on a
`task.completed` event. Conformant UClone-X agents emit results as `Artifact`s attached
to the `Task`, delivered via `TaskArtifactUpdateEvent` while streaming.

---

## 6. Operations

The eleven core operations of §3.1 are binding-independent. Method names below are the
canonical mapping from §5.3.

| Operation | JSON-RPC method | gRPC method | REST endpoint | UClone-X target |
| :--- | :--- | :--- | :--- | :--- |
| Send message | `SendMessage` | `SendMessage` | `POST /message:send` | Required |
| Send streaming message | `SendStreamingMessage` | `SendStreamingMessage` | `POST /message:stream` | Required |
| Get task | `GetTask` | `GetTask` | `GET /tasks/{id}` | Required |
| List tasks | `ListTasks` | `ListTasks` | `GET /tasks` | Required |
| Cancel task | `CancelTask` | `CancelTask` | `POST /tasks/{id}:cancel` | Required |
| Subscribe to task | `SubscribeToTask` | `SubscribeToTask` | `POST /tasks/{id}:subscribe` | Required |
| Create push notification config | `CreateTaskPushNotificationConfig` | idem | `POST /tasks/{id}/pushNotificationConfigs` | Deferred (O2) |
| Get push notification config | `GetTaskPushNotificationConfig` | idem | `GET /tasks/{id}/pushNotificationConfigs/{configId}` | Deferred (O2) |
| List push notification configs | `ListTaskPushNotificationConfigs` | idem | `GET /tasks/{id}/pushNotificationConfigs` | Deferred (O2) |
| Delete push notification config | `DeleteTaskPushNotificationConfig` | idem | `DELETE /tasks/{id}/pushNotificationConfigs/{configId}` | Deferred (O2) |
| Get extended Agent Card | `GetExtendedAgentCard` | idem | `GET /extendedAgentCard` | Not offered (O3) |

> Because O2 is deferred, a conformant UClone-X agent MUST publish
> `capabilities.pushNotifications: false` and MUST answer the four config operations
> with `PushNotificationNotSupportedError` (§3.3.4). Declaring a capability the code
> does not have is exactly the class of error this document exists to end, and is also a
> P6 (fail-fast, no silent fallback) violation.

There is **no** "create task" operation. Tasks are created by the agent as a side effect
of `SendMessage` / `SendStreamingMessage`; the client does not choose the task id. The
previous version's `POST /a2a/v1/tasks` with a client-supplied `task_id`, `capability`
name, `callback_url`, `signature` and `context.max_iterations` corresponds to nothing in
the specification.

### 6.1 Send Message semantics

Input is a `SendMessageRequest`: `message` (required), optional `configuration`
(`SendMessageConfiguration`), `metadata`, `tenant`. Output is **either** a `Task` **or**
a bare `Message` for simple interactions (§3.1.1).

`SendMessageConfiguration.returnImmediately` selects the execution mode (§3.2.2):

* unset / `false` (**the default**) — blocking: return only once the task reaches a
  terminal or interrupted state;
* `true` — non-blocking: return as soon as the task is created; the caller then polls
  `GetTask`, calls `SubscribeToTask`, or receives push notifications.

Other configuration fields: `acceptedOutputModes`, `historyLength`,
`taskPushNotificationConfig`.

### 6.2 Errors

A2A error types and their canonical mappings (§5.4):

| A2A error | JSON-RPC | gRPC | HTTP |
| :--- | :--- | :--- | :--- |
| `TaskNotFoundError` | `-32001` | `NOT_FOUND` | 404 |
| `TaskNotCancelableError` | `-32002` | `FAILED_PRECONDITION` | 400 |
| `PushNotificationNotSupportedError` | `-32003` | `FAILED_PRECONDITION` | 400 |
| `UnsupportedOperationError` | `-32004` | `FAILED_PRECONDITION` | 400 |
| `ContentTypeNotSupportedError` | `-32005` | `INVALID_ARGUMENT` | 400 |
| `InvalidAgentResponseError` | `-32006` | `INTERNAL` | 500 |
| `ExtendedAgentCardNotConfiguredError` | `-32007` | `FAILED_PRECONDITION` | 400 |
| `ExtensionSupportRequiredError` | `-32008` | `FAILED_PRECONDITION` | 400 |
| `VersionNotSupportedError` | `-32009` | `FAILED_PRECONDITION` | 400 |

Error payloads use `google.rpc.Status` with `google.rpc.ErrorInfo` in `details`; v1.0
dropped RFC 9457 Problem Details. Every custom binding MUST supply an equivalent
mapping that preserves these semantics (§12.4) — including the fastpath
([§10.5](#105-error-mapping)).

---

## 7. Transport bindings and streaming

Both JSON-RPC **and** SSE, plus gRPC — the question "JSON-RPC or SSE?" is a category
error: JSON-RPC is the request/response binding, SSE is how that binding streams.

| | JSON-RPC binding (§9) | gRPC binding (§10) | HTTP+JSON/REST binding (§11) |
| :--- | :--- | :--- | :--- |
| Transport | JSON-RPC 2.0 over HTTP(S) | gRPC over HTTP/2 with TLS | HTTP(S) + JSON |
| Content type | `application/json` | protobuf v3 | `application/a2a+json` SHOULD be used |
| Method naming | PascalCase, matching gRPC (`SendMessage`) | `A2AService` service, per `a2a.proto` | RESTful verbs + resource URLs |
| Streaming | **Server-Sent Events** (`text/event-stream`) | server-streaming RPCs | **Server-Sent Events** |
| Service params | HTTP request headers | gRPC metadata | HTTP request headers |

**WebSocket is not a standard A2A binding.** The previous version of this document
offered a `wss://…/a2a/v1/stream` endpoint and "SSE / WS" events as if both were
standard. A WebSocket transport is only reachable as a *custom binding* under §12, with
its own URI identifier — the specification even uses WebSocket as its worked example of
a custom binding (§5.8). UClone-X does not currently specify one.

### 7.1 UClone-X binding scope

* **v0.2.0 target: the JSON-RPC 2.0 binding with SSE streaming, only.** One binding is
  sufficient for conformance (§5.1 constrains agents that offer *several*), and it is
  the cheapest to verify.
* gRPC and HTTP+JSON/REST: not specified, not offered, and MUST NOT appear in
  `supportedInterfaces` until implemented.
* If a second binding is ever added, §5.1 requires it to be functionally equivalent —
  same operations, same semantics, same error mapping, same auth schemes.

### 7.2 Wire examples

Conformant JSON-RPC request/response (§9.4.1, §6.1):

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "SendMessage",
  "params": {
    "message": {
      "messageId": "msg-3f1c",
      "role": "ROLE_USER",
      "parts": [{ "text": "Audit /workspace/auth-service at high strictness." }]
    },
    "configuration": {
      "acceptedOutputModes": ["application/json"],
      "returnImmediately": true
    }
  }
}
```

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "task": {
      "id": "task-9a2e",
      "contextId": "ctx-main-flow",
      "status": { "state": "TASK_STATE_WORKING" }
    }
  }
}
```

Streaming (`SendStreamingMessage`, §9.4.2): HTTP 200 with
`Content-Type: text/event-stream`, each event a JSON-RPC envelope wrapping a
`StreamResponse`:

```text
data: {"jsonrpc":"2.0","id":1,"result":{"task":{"id":"task-9a2e","contextId":"ctx-main-flow","status":{"state":"TASK_STATE_WORKING"}}}}

data: {"jsonrpc":"2.0","id":1,"result":{"artifactUpdate":{"taskId":"task-9a2e","contextId":"ctx-main-flow","artifact":{"artifactId":"art-1","name":"Findings","parts":[{"data":{"file":"src/auth.ts","line":42,"severity":"CRITICAL","description":"Hardcoded JWT secret"}}]},"lastChunk":true}}}

data: {"jsonrpc":"2.0","id":1,"result":{"statusUpdate":{"taskId":"task-9a2e","contextId":"ctx-main-flow","status":{"state":"TASK_STATE_COMPLETED"}}}}
```

Stream shape rules (§3.1.2, §11.7): a message-only stream contains exactly one `Message`
then closes; a task stream begins with the `Task`, then zero or more `statusUpdate` /
`artifactUpdate` events, and closes when the task reaches a terminal state. Events MUST
NOT be reordered (§3.5.2). v1.0 removed the `final` boolean from
`TaskStatusUpdateEvent` and the `kind` discriminator from stream events. An agent MAY
serve multiple concurrent streams for one task; each MUST see the same events in the
same order, and closing one MUST NOT affect the others.

### 7.3 Versioning

`A2A-Version` is a **service parameter** (§3.2.6), carried as an HTTP header in the
HTTP-based bindings and as gRPC metadata in the gRPC binding; clients MAY instead pass
it as a request parameter (§3.6.1). Rules:

* Clients MUST send it on every request; UClone-X sends `1.0`.
* Servers MUST honour the requested `Major.Minor` or return `VersionNotSupportedError`.
* An **absent** value MUST be interpreted as `0.3`. UClone-X does not implement 0.3, so
  a UClone-X server MUST return `VersionNotSupportedError` for requests with no version
  rather than guessing — a direct application of P6.

`A2A-Extensions` is the companion service parameter carrying a comma-separated list of
extension URIs the client wishes to activate.

---

## 8. Security

Not a substitute for the security documentation owed under
`2026-09-02-016`; this section
records only what A2A itself requires.

* Transport security and identity: clients SHOULD validate the server's TLS certificate
  (§7.2). This is the only "handshake" in A2A.
* Credentials are obtained out of band and attached to every request per the declared
  `securitySchemes` (§7.3). Schemes available: API key, HTTP auth (Basic/Bearer),
  OAuth 2.0, OpenID Connect, mutual TLS. In the v1.0.1 proto the OAuth `implicit` and
  `password` flows are marked `deprecated` — not removed, contrary to the wording of
  the project's own "What's New in v1.0" page — and `device_code` (RFC 8628) plus
  `pkce_required` on the authorization-code flow were added. UClone-X MUST NOT offer
  the two deprecated flows.
* A2A carries no message-level signature field. The `"signature": "sha256_hmac_…"`
  field in the previous version of this document was invented. The only signing
  mechanism in A2A is JWS signing of the **Agent Card** (§8.4), which UClone-X does not
  yet offer (O4).
* Push-notification webhooks MUST authenticate to the client endpoint per
  `TaskPushNotificationConfig.authentication`, and clients MUST validate the source
  (§4.3.3, §13.2). Deferred with O2.
* In-task authorization via `TASK_STATE_AUTH_REQUIRED` (§7.6): credentials SHOULD reach
  the agent out of band over a secure channel; in-band exchange exposes credentials to
  every agent in a delegation chain and, if used, credentials SHOULD be bound to the
  originating agent (§7.6.3). The state conveys no authorization scope — see
  [§5.1](#51-task-lifecycle--the-real-state-names).

---

## 9. Conformance verification plan

No claim of conformance may be made on the strength of this document. Conformance
becomes checkable only through the following, in order:

1. **Schema round-trip tests.** UClone-X model types serialise to and parse from the
   generated v1.0.1 JSON Schema bundle. Catches camelCase, enum-string and
   `capabilities`-vs-`skills` regressions mechanically.
2. **Agent Card validation test.** The published card validates against the `AgentCard`
   schema, is served at `/.well-known/agent-card.json`, and declares only capabilities
   the build actually implements.
3. **Operation coverage tests.** One test per required row of §6, asserting the wire
   method name, the response `oneof` member, and the state transition.
4. **Negative tests.** Undeclared capability → mapped error; message to a terminal task
   → `UnsupportedOperationError`; missing `A2A-Version` → `VersionNotSupportedError`.
5. **Third-party interop test.** An off-the-shelf A2A client (e.g. an SDK from
   <https://a2a-protocol.org/latest/sdk/>) discovers a UClone-X agent and completes a
   streaming task. **Only after this step may any document in this repository use the
   word "conformant" without qualification.**
6. **Fastpath equivalence test.** The same task, executed over the fastpath and over
   the JSON-RPC binding, yields identical `Task`, `TaskState` sequence and `Artifact`
   content ([§10.7](#107-equivalence-obligation)).

Steps 1–4 and 6 belong in `./ucx test check`. Step 5 needs a network fixture and should
gate releases rather than every commit.

Tracked by `2026-09-02-004`; none of it
exists yet.

---

## 10. UClone-X extension: in-process zero-copy fastpath

**This section describes a UClone-X extension. It is not part of the A2A standard.**

P2 requires that co-located agents communicate by zero-copy in-memory dispatch, with no
network and no serialisation. A2A's data model is defined over serialisable objects, so
this is by construction a local optimisation and not standard A2A traffic.

The specification anticipates exactly this situation and provides the seam: §12 (Custom
Binding Guidelines) permits implementers to define custom bindings for "additional
transport mechanisms or communication patterns", provided they satisfy §5. The fastpath
is therefore specified **as a custom A2A protocol binding**, which makes its obligations
concrete and testable rather than a hand-wave.

### 10.1 What the fastpath is and is not

| | |
| :--- | :--- |
| **Is** | A UClone-X-internal binding of the A2A *operations* and *data model* onto direct async-queue dispatch between agents in one process. |
| **Is** | A performance optimisation that preserves the **logical** A2A contract: same operations, same object semantics, same lifecycle states, same errors. |
| **Is not** | Part of the A2A standard, in any version. |
| **Is not** | Reachable by any external client, and therefore not something a third party can be asked to interoperate with. |
| **Is not** | A licence to invent operations, states or fields. Anything the fastpath can do MUST be expressible over the JSON-RPC binding. |

The claim UClone-X is entitled to make is: *"logical A2A v1.0.1 conformance, with a
non-standard in-process binding for co-located agents."* The claim it is **not**
entitled to make is *"full A2A compliance"* — neither before nor after the fastpath
exists.

### 10.2 Binding identifier

Per §5.8 a custom binding SHOULD be identified by a URI, and a breaking change MUST take
a new URI:

```text
https://github.com/UClone-AI/uclone-x/bindings/inproc/v1
```

### 10.3 Obligations inherited from §12

Each is a requirement on the eventual implementation, not a description of it:

| §12 requirement | Fastpath obligation |
| :--- | :--- |
| 12.1.1 all core operations | Every required row of §6 dispatchable in-process. |
| 12.1.2 preserve data model | Pass the same `Task` / `Message` / `Artifact` objects by reference; no ad-hoc envelope. |
| 12.1.3 maintain semantics | Identical `TaskState` transitions and blocking/non-blocking behaviour. |
| 12.1.4 document completely | This section, kept in step with the code. |
| 12.2 data type mappings | Native Python objects; no serialisation. Where objects are shared by reference they MUST be immutable (frozen models) so zero-copy cannot become shared mutable state. |
| 12.3 service parameters | `A2A-Version` and `A2A-Extensions` carried in a call-scoped mapping, since there are no headers. |
| 12.4 error mapping | [§10.5](#105-error-mapping). |
| 12.5 streaming | Async generator yielding the same `StreamResponse` members, in generation order, closing on a terminal state. |
| 12.6 auth | In-process calls do not cross a trust boundary and carry no credentials; the fastpath MUST therefore be usable **only** between agents in the same process, and the loopback selection rule in [§10.4](#104-agent-card-declaration) MUST never be satisfied by a remote peer. |
| 12.7 Agent Card declaration | [§10.4](#104-agent-card-declaration). |
| 12.8 interop testing | [§10.7](#107-equivalence-obligation). |

### 10.4 Agent Card declaration

§12.7 requires custom bindings to be declared in the Agent Card with a URI and an
endpoint URL. A published card, however, is fetched by *remote* clients, for whom an
in-process binding is unusable — and §8.3.1 requires each declared interface to be
genuinely reachable at its URL.

**Rule:** the fastpath MUST NOT appear in the Agent Card served at
`/.well-known/agent-card.json`. Selection is a local decision: when the resolved target
agent is registered in the same process, the runtime uses the fastpath; otherwise it
selects a standard binding from the target's card exactly as §8.3.2 requires. This
choice is a local routing decision and MUST NOT change any observable task semantics.

If a future release needs the fastpath to be discoverable — e.g. by an in-process
extended card offered only to co-located agents — that is a change to this section, not
an implicit allowance.

### 10.5 Error mapping

The fastpath raises typed Python exceptions, one per A2A error type of §6.2, and the
JSON-RPC binding maps those same exceptions to the codes in that table. This gives a
single error taxonomy across both bindings, satisfying §12.4 and P6: a failure in the
fastpath surfaces as the same A2A error a remote caller would have received, never as a
silent fallback to the remote path.

### 10.6 Typed skill contracts

A2A `AgentSkill` has no `inputSchema` / `outputSchema` (see
[§3.2](#32-required-structure)). UClone-X's typed input/output contracts are therefore
declared through the §4.6 protocol-extension mechanism, under an extension URI, and
MUST be `required: false` so that a client ignorant of the extension can still call the
skill using the standard media-type-based description.

The extension payload is **not yet specified**. Until it is, no UClone-X Agent Card may
carry schema fields, and no document may claim schema negotiation exists.

### 10.7 Equivalence obligation

The fastpath is a valid optimisation only for as long as it is observationally
equivalent to the standard binding. Step 6 of [§9](#9-conformance-verification-plan) is
the test that keeps it honest, and it is a release gate: if the two paths ever diverge in
`Task`, state sequence, artifact content or error type, the fastpath is a
non-conformance, not a feature.

---

## 11. Known non-conformance in the current repository

`src/uclone_x/a2a/` contains type stubs only — no transport, no server, no client, no
discovery. Those stubs still do not match v1.0.1. **They are outside this document's
scope to change; they are recorded here so the gap is not lost.**

Re-checked on 2026-09-02 against the `a2a/models.py` and `a2a/protocols.py` that exist
today — this is a resync of the table below, not a re-audit from first principles. Each
row is marked **Resolved**, **Still stands**, or **Now different** against the current
code. The work that closes the remaining rows is tracked as GitHub issue #2 ("Align A2A
models and wire types with A2A v1.0.1"); the register finding behind it is
`2026-09-02-031`.

| Location | Defect (as filed) | Required | Status — 2026-09-02 resync |
| :--- | :--- | :--- | :--- |
| `a2a/__init__.py:1`, `a2a/models.py:1` | Docstrings say "Google A2A … v1.0.0" | AAIF governance ([§2.3](#23-governance-and-attribution)); target v1.0.1 / wire `1.0` | **Now different.** Both docstrings now read `v1.0.1`, not `v1.0.0` — the version half is fixed. Both still say "Google A2A" rather than the AAIF/Linux Foundation attribution, and neither distinguishes the spec version (`1.0.1`) from the wire version (`1.0`, [§2.1](#21-conformance-target)). The governance-wording defect still stands. |
| `a2a/models.py` `AgentCard.capabilities: tuple[str, ...]` | `capabilities` used as a list of ability names | `AgentCapabilities` object of boolean flags (R5) | **Now different.** There is no `capabilities` field at all any more — it was renamed to `skills` (next row) rather than split into a flags object. R5 (an `AgentCapabilities` boolean flag set) is still unimplemented; the field this row named no longer exists in the shape described. |
| `a2a/models.py` `AgentCard` | No `skills` field | `skills: tuple[AgentSkill, ...]`, required (R6) | **Now different.** `AgentCard.skills: tuple[str, ...]` exists now, so the missing-field defect is resolved. Its members are bare strings, not `AgentSkill` objects, so R6 is still not met. |
| `a2a/models.py` `AgentCard.input_schema` / `output_schema` | Not A2A fields | Remove; see [§10.6](#106-typed-skill-contracts) | **Still stands.** Both fields are unchanged in the current model. |
| `a2a/models.py` `AgentCard.endpoints: dict[str, str]` | Not an A2A field | `supported_interfaces: tuple[AgentInterface, ...]`, required (R3) | **Still stands.** `endpoints` (now typed `ImmutableStrMapping`, same shape) is unchanged; there is no `supported_interfaces` field. |
| `a2a/models.py` `AgentCard` | Missing required `default_input_modes`, `default_output_modes` | Add (R4) | **Still stands.** Neither field exists on `AgentCard`. |
| `a2a/models.py` `TaskResult.status: str = "completed"` | Free-form lowercase status | `TaskState` enum with `TASK_STATE_*` values (R8) | **Now different.** `status` is now `TaskStatus`, a closed `StrEnum` — no longer a free-form `str`, so the untyped half of R8 is met. It is still not R8-conformant: the enum is named `TaskStatus`, not `TaskState`; its five members (`completed`, `failed`, `rejected`, `input_required`, `canceled`) are lowercase, not `TASK_STATE_*` SCREAMING_SNAKE_CASE; and four of the spec's nine states (`UNSPECIFIED`, `SUBMITTED`, `WORKING`, `AUTH_REQUIRED`) are absent. |
| `a2a/models.py` `TaskMessage` / `TaskResult` | Invented envelopes with `session_id`, `input_data`, `output_data` | `Message` / `Task` / `Artifact` per [§5.2](#52-core-objects); results as Artifacts per [§5.3](#53-messages-carry-communication-artifacts-carry-results) | **Still stands.** `TaskMessage` still carries `session_id` and `input_data`; `TaskResult` still carries `output_data`. No `Message`, `Task` or `Artifact` model exists. |
| `a2a/models.py` `WireProtocolType.REST_SSE` | Implies the REST binding | `JSONRPC` per [§7.1](#71-uclone-x-binding-scope), plus the fastpath binding URI | **Still stands.** `WireProtocolType` is unchanged: `LOCAL_IN_MEMORY`, `REST_SSE`. |
| `a2a/protocols.py` `A2ATransportProtocol.send_task` | No such operation in A2A | `SendMessage` / `SendStreamingMessage` / `GetTask` / … per [§6](#6-operations) | **Still stands.** `async def send_task(target_endpoint, message) -> TaskResult` is unchanged. |
| `a2a/protocols.py` `stream_task` declared `async def ... -> AsyncIterator[str]` | An async generator function *is* a plain `def` returning an `AsyncIterator`. Declared `async def`, the caller had to `await` the call just to *obtain* the iterator, and then iterate it — a shape no async generator can satisfy, so no implementation could ever have conformed to this signature | Plain `def stream_task(...) -> AsyncIterator[...]` | **Resolved.** `stream_task` is now `def`, not `async def` (verified in `a2a/protocols.py`; the change lands in commit `0bde230`). Recorded here rather than deleted, with the reasoning kept, so the shape does not return; the method's own docstring carries the same rule and cites `2026-09-02-035`. |
| `a2a/protocols.py` `stream_task -> AsyncIterator[str]` payload | Untyped strings | `AsyncIterator[StreamResponse]` | **Still stands.** Only the `def`/`async def` shape above was fixed; the return type is still `AsyncIterator[str]`, not `AsyncIterator[StreamResponse]`. |
| all | No `A2A-Version` handling anywhere | R13 | **Still stands.** No transport, server or client code exists yet (this section's opening paragraph), so nothing implements the header. |

Field naming: these are Python models, so snake_case attributes are correct; the
camelCase requirement (R9) applies to the JSON serialisation, which is where the
aliasing must be configured.

**P6 in-band provenance.** `TaskResult` now declares `provenance: Provenance | None`
(`src/uclone_x/core/provenance.py`) as a required field with no default: `None` is
representable, so a non-conformant value can still be rejected by `require_provenance`,
but a provenance marker is never inherited silently. This meets P6's in-band-attribution
requirement at the type level for the one result-bearing envelope `a2a/models.py`
defines. `TaskMessage`, the dispatch (not result) envelope, carries no `provenance`
field — consistent with P6 applying to results rather than requests.

**What the gateway does with an unattributed result: it refuses (#157).** The rule is
one sentence — *a component may state its own attribution and may forward another's, but
may never invent one on another's behalf* — and it binds hardest here, because this is
the only boundary where the fabrication leaves the process. A peer receiving
`provenance` cannot audit where it came from; whatever the gateway writes is what the
peer believes.

`A2AServer._execute_managed_task` previously wrote
`turn_result.provenance or Provenance.primary(self._agent.agent_id)` (and
`result.provenance or Provenance.primary("handler")`), so a result that stated nothing
reached a peer as `path=primary`, `requested == served_by`, `degraded=False` — a clean
primary asserted by the component that was merely *carrying* the value. #136 is what
made this reachable: once `execute_turn` stopped substituting its own synthetic value
and began propagating `None` verbatim, an unattributed turn became this path's normal
input.

The task now **fails**: `status=failed`, `provenance` serialised as `null`, and `error`
naming the agent or handler that produced nothing to attribute. That matches what the
two synchronous paths in the same file already did — `send_task_endpoint` answers `500
MissingProvenanceError`, and the SSE generator raises it — so the async path is no
longer the one place with the opposite policy. A remote peer therefore sees an explicit
absence with a stated reason, which is what P6's "absence is a violation, not a default"
requires, and never a fabricated attestation it cannot distinguish from a real one.

Forwarding is unaffected: a stated `Provenance` passes through as the producer's own
object, not a copy the gateway rebuilds.

*Why no validator can enforce this.* The fabricated value is well-formed by
construction — it is exactly what an honest primary result looks like — so both
`Provenance` validators accept it, verified. The difference is not in the value but in
who called the constructor, which only the source shows. Enforcement is therefore a
static sweep over `src/`
(`test_no_component_manufactures_provenance_for_an_absent_one`), which fails the gate on
a construction reached from a `None`-check of another object's provenance.

**What the sweep covers, stated exactly, because a syntactic rule cannot be complete.**
It resolves `Provenance` through the module's own import bindings (so an alias is not an
escape hatch) and tracks locals bound from a `.provenance` read (so extracting a
variable is not one either — the likeliest rewrite of all, since the failure message
names the `or` form). Within that, it catches `x.provenance or Provenance...`, the
conditional-expression form, `if x.provenance is None:`, and `if not x.provenance:`.

It does **not** catch construction moved behind a helper function, an absence tested via
an intermediate boolean, or `getattr`-style reflective access. Those are listed and
pinned as known blind spots in `_KNOWN_UNCOVERED_SHAPES`, so the boundary is recorded
rather than discovered. Each takes a deliberate act to write, which is the line the rule
is drawn on: it stops the accident, not the determined author. An earlier draft of this
paragraph claimed the sweep caught the defect "in any of its syntactic forms" — that was
false, and false in the place a reader checks instead of testing.

A call-site deletion does not generalise: #136 removed this shape from `agent/base.py`
and it reappeared here within a day.

---

## 12. Unverified claims

Everything asserted above is cited to §2.2. The following could not be confirmed from a
primary source and is therefore **not** asserted anywhere in this document:

| # | Claim | What was tried | Status |
| :--- | :--- | :--- | :--- |
| U1 | An official A2A conformance test suite exists that UClone-X could run for step 5 of [§9](#9-conformance-verification-plan). | Searched the specification (§12.8 recommends interop testing but names no suite) and the project site's SDK page. Found SDKs, no conformance suite. | **Unverified** — step 5 is written against "an off-the-shelf A2A client", not a named suite. |
| U2 | The AAIF transfer date. | The project's own announcement is dated 2026-08-27; press coverage from 2026-08-17; finding 2026-09-02-004 says 2026-08-20. No AAIF/LF press release naming a transfer date was located. | **Partially verified** — the transfer is confirmed by a primary source; the exact effective date is not. Recorded in [§2.3](#23-governance-and-attribution) with the announcement date rather than an effective date. |
| U3 | The Technical Steering Committee roster. | <https://a2a-protocol.org/latest/> lists members (AWS, Cisco, Google, IBM Research, Microsoft, Salesforce, SAP, ServiceNow); no governance charter page was fetched to confirm currency. | **Unverified** — not relied on; only the AAIF/LF hosting relationship is asserted. |
| U4 | "Sub-millisecond inter-agent communication" for the fastpath (asserted by P2 and the README). | No benchmark, no harness and no fastpath code exist. | **Unverified** — no latency figure appears in this document. See `2026-09-02-009`. |
| U5 | Whether A2A `main` (post-v1.0.1) changes affect UClone-X. | Diffed `main` against tag `v1.0.1`: differences are editorial plus a `security` → `securityRequirements` sample fix, a `PushNotificationConfig` → `TaskPushNotificationConfig` rename in prose anchors, and a new §7.6.4 on in-task authorization scope. No released version carries them. | **Verified.** Immaterial to the pinned target, with one exception noted inline: §7.6.4 is cited as a `main`-only addition that UClone-X adopts anyway. |

---

## 13. Corrections owed by other documents

Found while verifying this file, and **outside its scope to change**. Each is a false
externally-checkable claim.

### 13.1 `README.md`

| Line | Current text | Why false |
| :--- | :--- | :--- |
| 8 | `![Standard](https://img.shields.io/badge/A2A_Protocol-Google_Standard-brightgreen.svg)` | A2A is not a Google standard; it is an AAIF / Linux Foundation project ([§2.3](#23-governance-and-attribution)). The green badge also asserts a conformance status that does not exist. Suggested: `A2A_Protocol-v1.0.1_target-lightgrey`. |
| 53 | "🤝 **Google A2A Protocol Standard:** Full compliance with Google's Agent-to-Agent (A2A) protocol for standard agent discovery and structured task streaming." | Three errors: "Google's" (governance), "Full compliance" (nothing is implemented — and the README's own "What does not exist yet" section says so eight lines earlier), and no version pinned. Suggested: "🤝 **A2A Protocol (AAIF/Linux Foundation) target:** specification written against A2A v1.0.1; no A2A code exists yet — see `docs/a2a-protocol-spec.md` §1." |
| 95 | "🌐 **Google A2A Protocol Specification**: Inter-agent discovery, capability negotiation, and task streaming." | "Google" (governance) and "capability negotiation", which implies a negotiation phase A2A does not have ([§4](#4-there-is-no-handshake)). Suggested: "🌐 **A2A Protocol Conformance Specification**: Agent Card discovery, transport binding selection, and task streaming." |

The word "handshake" quoted by finding 2026-09-02-004 is no longer present in
`README.md`; the badge and the compliance claim are.

### 13.2 `docs/PRD.md`

| Line | Current text | Required |
| :--- | :--- | :--- |
| 85 (FR-2.1) | "Discovery endpoint `/.well-known/agent.json` exposing capabilities, input/output schemas, and endpoints." | `/.well-known/agent-card.json`; an Agent Card exposes `capabilities` (flags), `skills`, and `supportedInterfaces`. There are no input/output schemas ([§3.2](#32-required-structure)). |
| 88 (FR-2.2) | "Remote Wire Protocol: REST `POST /a2a/v1/tasks` and Server-Sent Events (SSE) streaming for distributed agents." | No such endpoint and no create-task operation. JSON-RPC `SendMessage` / `SendStreamingMessage` at the URL from the Agent Card, streaming over SSE ([§6](#6-operations), [§7](#7-transport-bindings-and-streaming)). |
| 24, 32, 68, 84, 171 | "Google A2A" / "Google A2A Adapter" | Governance ([§2.3](#23-governance-and-attribution)). |

### 13.3 P2 (`docs/principles/details/p2-google-a2a-interoperability.md`)

Principle text is not editable here. Two defects, one of them the reason finding
2026-09-02-004 is approval-gated:

1. **Line 7 hard-codes a wrong path** — `Discovery /.well-known/agent.json`. It is
   factually wrong *and* it is a volatile external detail embedded in a document that
   is expensive to amend. The fix is to remove the concrete path, not to correct it.
2. **Line 5 misattributes governance** — "the open **Google A2A Specification**".

Proposed replacement wording for the two bullets, offered for the Builder Manager /
the project owner and asserting nothing this document has not verified:


> * **Core Law**: All external Agent-to-Agent interfaces must adhere to the open
>   **A2A (Agent2Agent) Specification** — an Agentic AI Foundation / Linux Foundation
>   project — implemented via an optimized Dual-Transport model.
> * **Strict Rule**:
>   - Cross-agent communication must logically conform to the A2A specification at the
>     version and in the binding pinned by
>     `docs/a2a-protocol-spec.md`. Concrete discovery
>     paths, endpoints, method names and object shapes are defined there and MUST NOT be
>     restated in this principle.
>   - **In-Memory Fastpath**: For co-located agents in the same process, transport uses
>     zero-copy direct memory dispatch without network or serialization overhead. The
>     fastpath is a non-standard local binding and MUST remain observationally
>     equivalent to the pinned standard binding.
>   - **Remote Wire Protocol**: For distributed or remote agents, transport uses a
>     standard A2A protocol binding as pinned by the specification document.

Notes for the reviewer: dropping the path is a **Tier B** repair under
`docs/governance/principle-amendment-policy.md` §2 — a factual correction plus moving a
volatile constant out of the law, with the requirement itself intact. Fixing the
attribution is likewise factual. Adding the equivalence sentence to the fastpath bullet
adds a requirement and is therefore **Tier A**; it may be dropped without affecting the
other two repairs. The existing "Why" line's "sub-millisecond" figure is unverified
(U4) and belongs to `2026-09-02-009`,
not to this finding.

### 13.4 `AGENTS.md`

Line 36 describes P2 as "**Strict Protocol Interoperability & Dual-Transport A2A**
(Google A2A logical standard with zero-copy in-memory fastpath)". "Google" needs the
same governance correction; the filename
`docs/principles/details/p2-google-a2a-interoperability.md` carries it too, and renaming
that file is a `docs/principles/` change.

---

## 14. Related

* `2026-09-02-004` — the finding this document resolves
* `2026-09-02-009` — volatile constants inside principles; same pattern
* `2026-09-02-016` — security documentation gap
* `2026-09-02-021` — the internal event contract the fastpath will ride on
* `2026-09-02-025` — documents describing code that does not exist
* [`docs/local-collaboration-engine.md`](local-collaboration-engine.md) — in-memory bus the fastpath builds on
* [`docs/principles/details/p2-google-a2a-interoperability.md`](principles/details/p2-google-a2a-interoperability.md) — P2

---

## 15. Verification record

| Date | Checked against | By | Result |
| :--- | :--- | :--- | :--- |
| 2026-09-02 | A2A v1.0.1 (prose + `specification/a2a.proto` @ `v1.0.1`), sources in [§2.2](#22-sources-checked) | a reviewer, task 2026-09-02-014 | Document rewritten; every normative claim cited or labelled. Implementation status: **none**. |
