# Pluggable & Selective Sandbox Execution Architecture

> [!NOTE]
> **Implementation status — partly shipped.** The isolation model
> (`src/uclone_x/sandbox/models.py`), the path validator
> (`src/uclone_x/sandbox/path_validator.py`) and the workspace runner
> (`src/uclone_x/sandbox/workspace_runner.py`) exist and are exercised by
> `tests/unit/test_isolation_policy.py`, `tests/unit/test_sandbox_runner.py` and
> `tests/unit/test_tools_isolation.py`. The container and WASM runners do **not** exist.
> Every section below is marked **Implemented** or **Planned**; this document describes
> the shipped shapes, not the originally proposed ones.

## 1. Overview & Motivation

When executing on local developer machines or self-hosted servers, running commands and manipulating files directly on the host delivers the lowest possible latency and seamless developer ergonomics.

However, when agents execute untrusted scripts, perform destructive file edits, or collaborate with unknown remote agents over A2A, isolation is required.

UClone-X implements **Pluggable & Selective Sandboxing**, allowing developers or agents to configure the execution boundary per agent, per tool, or per task.

Two things this document deliberately does *not* claim, both carried forward from
[`security-threat-model.md`](security-threat-model.md) D1:

* An isolation level is **not** the credential boundary. `workspace` bounds writes and
  closes no credential path at all. The environment allowlist and egress
  deny-by-default (§5) are the controls that close credential exfiltration, and they
  apply at *every* level including `none`.
* No isolation level stops prompt injection from reaching tool execution. Isolation
  caps what a call may touch; it never decides whether the call is made.

---

## 2. Four Isolation Levels

```mermaid
flowchart TD
    Task["Agent Task Execution"] --> LevelSelect{"IsolationPolicy<br/>(discriminated on `level`)"}

    LevelSelect -->|"`NoIsolation()` — explicit opt-in"| DirectHost["Direct Host Execution<br/>- Bare-metal process execution<br/>- No boundary of any kind<br/>- Planned: no runner implemented"]

    LevelSelect -->|"`WorkspaceIsolation()` — the default"| WorkspaceIso["Workspace / Git Worktree Sandbox<br/>- Cwd boundary & env scrubbing (P7)<br/>- No shell write jail, no read/egress control<br/>- Runner: WorkspaceSandboxRunner (direct exec)"]

    LevelSelect -->|"`ContainerIsolation(image=...)`"| Container["Container Sandbox (Docker / Podman)<br/>- CPU, memory, network limits enforceable<br/>- Planned: no runner implemented"]

    LevelSelect -->|"`WasmIsolation()`"| Wasm["WASM Sandbox (WASI)<br/>- No ambient FS or network<br/>- Planned: no runner implemented"]
```

> [!IMPORTANT]
> **Shell Execution Boundary Under `WorkspaceIsolation` (#684)**:
> In the agent runtime tool path, `BashRunTool` (and `run_command`) executes commands via `asyncio.create_subprocess_shell`.
> It enforces that `cwd` stays within `workspace_root` (traversals rejected) and scrubs credentials/secrets from the environment,
> but does **not** mediate or sandbox arbitrary shell command strings (e.g. `touch ../outside.txt`, shell redirects, or `rm -rf`).
> `WorkspaceSandboxRunner` provides argument path validation for structured `ExecutionRequest` calls, but is not wired into `BashRunTool`
> and does not provide an OS-level filesystem jail. True write-path containment requires `ContainerIsolation` (Docker/Podman) or disposable checkouts.
> The one exception is the story library: on macOS the command runs under `sandbox-exec`, which refuses writes there (#1589; §10 item 10).

`IsolationLevel` is a `StrEnum` with members `NONE` / `WORKSPACE` / `CONTAINER` /
`WASM` (values `"none"` / `"workspace"` / `"container"` / `"wasm"`).

---

## 3. Isolation is a Discriminated Union, Not a Mode String — *Implemented*

The isolation configuration is **not** a flat record with a level field and a bag of
limits. It is a Pydantic discriminated union over `level`:

```python
# src/uclone_x/sandbox/models.py
IsolationPolicy = Annotated[
    NoIsolation | WorkspaceIsolation | ContainerIsolation | WasmIsolation,
    Field(discriminator="level"),
]
```

### 3.1 Why the shape is the security property

This is the design principle, not an implementation detail: **a control that a level
cannot enforce is not a field of that level.** Under the previous flat
`SandboxConfig`, `memory_limit_mb`, `cpu_shares` and `allow_network` were accepted at
every level and silently discarded at `none` and `workspace`. As
[`security-threat-model.md`](security-threat-model.md) D1 puts it, "a security control
that is accepted and silently ignored is precisely what P6 forbids".

Under a union, asking a level for a boundary it cannot draw is **unexpressible** rather
than accepted-and-ignored. Requesting network denial from `workspace` is a
`ValidationError` at construction, not a field that quietly does nothing — so the
caller learns at the call site that the guarantee they wanted does not exist here.

| Model | `level` | Fields it carries | Fields it deliberately lacks |
| :--- | :--- | :--- | :--- |
| `NoIsolation` | `none` | *(none)* | every resource and network field — none of them is enforceable here |
| `WorkspaceIsolation` | `workspace` | `write_paths: tuple[Path, ...]` | `allow_network`, `memory_limit_mb`, `cpu_shares` — a write boundary cannot restrict egress or resources |
| `ContainerIsolation` | `container` | `image: str` (required), `write_paths`, `allow_network=False`, `egress_allowlist`, `memory_limit_mb=512`, `cpu_shares=1.0` | — |
| `WasmIsolation` | `wasm` | `preopened_dirs`, `allow_network=False`, `memory_limit_mb=256` | `cpu_shares` |

All four are `frozen=True, extra="forbid", strict=True`, so `extra="forbid"` is what
turns "this level has no such field" into a validation error.

*Verified in tests*: `test_unenforceable_controls_are_not_expressible` asserts
`ValidationError` for `WorkspaceIsolation(allow_network=...)`,
`WorkspaceIsolation(cpu_shares=...)` and `NoIsolation(memory_limit_mb=...)`.
`test_isolation_policies_are_frozen` asserts assignment after construction raises.
`test_enforceable_controls_deny_egress_by_default` asserts
`ContainerIsolation(image=...).allow_network is False`, `egress_allowlist == ()` and
`WasmIsolation().allow_network is False`.

---

## 4. The Default is `workspace`, and `none` Is Unreachable by Defaulting — *Implemented*

```python
class ExecutionRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    command: str
    args: tuple[str, ...] = ()
    cwd: Path
    workspace_root: Path  # required — no default
    isolation: IsolationPolicy = Field(default_factory=WorkspaceIsolation)
    env: ImmutableStrMapping = Field(default_factory=dict)
    env_allowlist: tuple[str, ...] = ()
    timeout_seconds: float = 60.0
```

`DEFAULT_ISOLATION_LEVEL = IsolationLevel.WORKSPACE` records the decision; the
enforcement is the `default_factory=WorkspaceIsolation` on every field that holds an
`IsolationPolicy` — `ExecutionRequest.isolation` and `ToolContext.isolation` in
`src/uclone_x/tools/models.py`.

**"Never reached by defaulting" is literal, not conventional.** Because the default
factory constructs a `WorkspaceIsolation`, there is no field, no environment variable
and no configuration file path that yields `none`. The only way to get there is to
write `NoIsolation()` in source at the call site, where it is visible in review and in
a diff.

This implements the project owner's Tier A decision of 2026-09-02, recorded in
[`governance/principle-amendment-policy.md`](governance/principle-amendment-policy.md)
§3 against issue `2026-09-02-001`:
the mandated default moves from `none` to `workspace`; `container`/`wasm` stay
available by user configuration; `none` is retained as an explicit opt-in that must
never be arrived at by defaulting.

> `workspace_root` is also required with no default. Its previous default of `.` bound
> the boundary to whatever directory the process happened to start in — the developer's
> own checkout. It is held on the request rather than on each policy so exactly one
> copy of it exists.

*Verified in tests*: `test_execution_defaults_to_workspace_isolation` asserts a bare
`ExecutionRequest` yields `WorkspaceIsolation`;
`test_no_isolation_is_reachable_only_by_stating_it` asserts `none` requires
`NoIsolation()`.

### 4.1 Clamping: `effective_isolation_level(requested, floor)` — *Implemented*

```python
def effective_isolation_level(
    requested: IsolationLevel, floor: IsolationLevel
) -> IsolationLevel: ...
```

The rule is that **a request can only strengthen isolation, never weaken it below the
floor**. Strength is ordered `none < workspace < container < wasm`; the function
returns `requested` when it is at least as strong as `floor`, and `floor` otherwise.

A synthesized skill, a remote A2A task or an MCP provider may *ask* for a level; the
runtime resolves that request against its own floor. P3 as amended: a requesting
artifact never raises its own ceiling. It is a function so the clause is testable once
rather than restated at each call site.

**On the word "floor".** The auditor exposes `isolation_floor` — a floor, not a
ceiling. The two words point in opposite directions depending on whether one is
measuring *isolation* (more is stronger, so the runtime sets a lower bound) or
*privilege* (more is weaker, so the runtime sets an upper bound). The amendment log
records this exact repair: "clamps that request downward" was replaced by a comparison,
because it read backwards for half of its readers, and that class of confusion is the
kind that produces an escalation. The API therefore states the direction in the
parameter name.

*Verified in tests*: `test_a_request_can_strengthen_isolation_but_never_weaken_it`
covers `NONE`→floor `WORKSPACE` yielding `WORKSPACE`, `CONTAINER` and `WASM` being
honoured against a `WORKSPACE` floor, and a `NONE` floor constraining nothing.

---

## 5. Environment Allowlist and Deny-by-Default Egress — *Partly implemented*

These two controls sit on the **request**, not inside the union, because the threat
model finds they close more of the credential-exfiltration path (T3) than the isolation
level does — and because they must apply at **every** level including `none`.
`workspace` bounds writes only; it does not scrub the environment and closes no
credential path, so making these properties of a level would leave the default
configuration exposed.

### 5.1 `env` and `env_allowlist` — *Implemented*

* `env: ImmutableStrMapping` — the child's environment, **constructed explicitly**.
  Never inherited.
* `env_allowlist: tuple[str, ...]` — host names to copy in. **Deny-by-default**:
  anything not named here is absent from the child.

### 5.2 `SECRET_ENV_PATTERNS` / `is_secret_env_name()` — *Implemented*

One place to ask whether a name is credential-shaped. The patterns are **generated**
from a list of credential-shaped trailing segments rather than written out, because
writing them out is how they drifted: every pattern was once `*_API_KEY`-shaped, and
`fnmatch`'s `*` cannot match a zero-length segment before a literal `_`, so the bare
names the control is named for — `API_KEY`, `TOKEN`, `PASSWORD`, `PRIVATE_KEY` — all
returned `False`.

```python
SECRET_NAME_TAILS: tuple[str, ...] = (
    "ACCESS_KEY",
    "API_KEY",
    "APIKEY",
    "AUTH",
    "AUTH_CONFIG",
    "COOKIE",
    "CREDENTIAL",
    "CREDENTIALS",
    "DATABASE_URL",
    "KUBECONFIG",
    "MONGODB_URI",
    "NETRC",
    "PASSPHRASE",
    "PASSWD",
    "PASSWORD",
    "PRIVATE_KEY",
    "REDIS_URL",
    "SECRET",
    "SECRET_KEY",
    "TOKEN",
)
SECRET_ENV_EXACT_NAMES: tuple[str, ...] = (  # 19 names; locators, glued
    "AWS_ACCESS_KEY_ID",
    "AWS_CONFIG_FILE",  # forms, code-execution vectors,
    "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",  # and vendor spellings
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "DOCKER_CONFIG",
    "GH_CONFIG_DIR",
    "GIT_CONFIG",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_SYSTEM",
    "GNUPGHOME",
    "GPG_AGENT_INFO",
    "MYSQL_PWD",
    "PGPASSWORD",
    "SSH_ASKPASS",
    "SSH_AUTH_SOCK",
)
SECRET_ENV_FAMILY_PATTERNS: tuple[str, ...] = (  # unbounded arity: no finite
    "GIT_CONFIG_KEY_*",
    "GIT_CONFIG_VALUE_*",  # set can express these
)

# Each tail contributes exactly two globs: the bare name and the `*_`-suffixed form.
SECRET_ENV_PATTERNS: tuple[str, ...] = (
    *SECRET_ENV_EXACT_NAMES,
    *SECRET_ENV_FAMILY_PATTERNS,
    *SECRET_NAME_TAILS,
    *(f"*_{tail}" for tail in SECRET_NAME_TAILS),
)


def is_secret_env_name(name: str) -> bool: ...  # case-insensitive fnmatch; `-`/`.` -> `_`
```

Separators are normalised to `_` before matching, so the header and dotted
span-attribute spellings (`x-api-key`, `llm.api_key`) are recognised too.

**Three classes, and a mechanical rule for which one a name lands in.** Tails, exact
names, and unbounded families are not three styles of the same list — they are the three
things a name can be, and every gap found so far has been a name in one class filed under
another.

> A name belongs in `SECRET_NAME_TAILS` **iff** the credential word is the whole trailing
> `_`-delimited segment. Unbounded families go in `SECRET_ENV_FAMILY_PATTERNS`.
> Everything else is an exact name.

`APIKEY`, `KUBECONFIG`, `NETRC` and `PASSWD` are glued yet correctly tails — their
trailing segment *is* the whole credential word, so `OPENAI_APIKEY` matches. `DATABASE_URL`,
`REDIS_URL` and `MONGODB_URI` are tails so their prefixed forms (`PROD_DATABASE_URL`,
`TEST_REDIS_URL`, `STAGING_MONGODB_URI`) match. `PGPASSWORD`'s
trailing segment is `PGPASSWORD`, not `PASSWORD`, so no tail reaches it; widening one to
`*PASSWORD*` would refuse `PASSWORD_STORE_DIR`, and the `AUTH` equivalent would refuse
`GIT_AUTHOR_NAME` — the regression ruled out twice.

**Whether a locator belongs at all: count artifacts.** For a name that *locates* a
credential rather than being one, ask **how many distinct artifacts must be combined
before the holder has a usable credential**. One means it belongs.
`DOCKER_CONFIG` → one (`auths` are plaintext base64). `GNUPGHOME` → one (the keyring is
the secret; passphrase-less keys are common). `GPG_AGENT_INFO` → one (historical;
inert/removed in GnuPG 2.1 which uses a fixed socket under `GNUPGHOME` or standard run
directory, retained for environments using older GnuPG releases). `SSH_AUTH_SOCK` → one
(live signing capability). `PASSWORD_STORE_DIR` → **two** (GPG-encrypted blobs plus a
keyring located elsewhere), so it stays out. This replaced an "encrypted at rest"
formulation that failed in both directions, because it rested on *may* — a cached
passphrase, a present keyring — and so gave different answers on different machines.

**Arbitrary code execution vectors.** Names that cause tooling to invoke arbitrary
executables or shell commands to source or prompt for credentials belong in exact names
(or unbounded families) regardless of artifact count:

* `AWS_CONFIG_FILE` — an AWS config profile can define `credential_process`, which runs
  an arbitrary shell command to retrieve dynamic credentials.
* `GIT_CONFIG_*` (`GIT_CONFIG`, `GIT_CONFIG_GLOBAL`, `GIT_CONFIG_SYSTEM`,
  `GIT_CONFIG_COUNT` / `GIT_CONFIG_KEY_*` / `GIT_CONFIG_VALUE_*`) — a git configuration
  can define `credential.helper` beginning with `!`, which executes arbitrary shell
  commands when resolving credentials.
* `SSH_ASKPASS` — specifies an executable program invoked by ssh / git to query the
  user for a passphrase or credential.

`AWS_ACCESS_KEY_ID` is listed despite failing that count, and the exception is argued
rather than assumed: it is not a locator, so the count does not govern it. It is one half
of a credential pair, and the count exists to adjudicate locators. Independently, denying
it costs nothing — no child needs the key id without the secret, and the secret is
already refused by `*_ACCESS_KEY`.

Asset A3 in [`security-threat-model.md`](security-threat-model.md) names `~/.aws` and
`~/.config/gh` for exactly these reasons.

**Pointer-shaped names need explicit entries, and that is where the misses happen.** A
name ending `_FILE`, `_URI`, `_DIR` or `_SOCK` does not contain a credential, it says
where to find one — so no tail rule can reach it, because the tail describes the pointer.
Every gap found in this list so far has had that shape, including the three
`AWS_CONTAINER_*` container credential providers (ECS task roles, EKS Pod Identity).
Those three are the strongest entries rather than the weakest: unlike `~/.aws/credentials`,
which resolves with no environment variable at all, **nothing resolves a container
credential endpoint without the variable**, and the URI carries the per-task identifier.
`AWS_CONTAINER_AUTHORIZATION_TOKEN` is caught by `*_TOKEN` while its `_FILE` sibling is
not — the same asymmetry `AWS_WEB_IDENTITY_TOKEN_FILE` exists to close, one provider
over. `test_pointer_shaped_vendor_credentials_are_exact_listed` asserts membership in
this tuple rather than merely that the predicate returns `True`, so a future gap cannot
be closed by widening a tail into a substring.

**Why there are no vendor namespace globs.** `AWS_*`, `GH_*` and `GITHUB_*` were removed
in issue #95. They refused roughly 24 benign names at construction — `GITHUB_REPOSITORY`,
`GITHUB_SHA`, `GITHUB_REF`, `AWS_REGION` and most of a GitHub Actions runner's
environment — and, once separator normalisation landed, they also swept dotted *trace*
attributes (`aws.region`, `github.repository`) into wholesale redaction. Every credential
they caught is now caught either by a tail (`AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`,
`GH_TOKEN`, `GITHUB_TOKEN`) or by an exact name, so their removal releases nothing;
`test_namespace_removal_releases_no_credential` pins that.

**`GIT_CONFIG_*` is a family, and it was verified by execution rather than inference.**
A `credential.helper` beginning `!` runs through the shell. `GIT_CONFIG_GLOBAL` and
`GIT_CONFIG_SYSTEM` both redirect the normal config stack and both execute it (`git var
GIT_AUTHOR_IDENT` reports the injected identity; `git credential fill` runs the helper,
3/3 under `env -i`). `GIT_CONFIG_COUNT` with `GIT_CONFIG_KEY_<n>`/`GIT_CONFIG_VALUE_<n>`
executes it **with no file at all**, at unbounded arity — which is the whole reason
`SECRET_ENV_FAMILY_PATTERNS` exists. `GIT_CONFIG` is the weak member: on git 2.50.1 it is
*not* honoured by the normal stack (`git var` ignores it, the helper does not run) and is
read only by `git config` itself, so it is listed at lower severity as a read, not
execution. It is easy to believe otherwise, because `git config --get` does print the
injected value — the documented `git config`-only special case.

Those globs are prefix patterns, which §5.2 removed for `AWS_*`/`GH_*`/`GITHUB_*`, so the
narrowness is the justification: `GIT_CONFIG_KEY_*` can only match a config-injection
slot, and no benign variable is spelled that way. A prefix glob earns its place only when
the namespace exists solely to carry the dangerous value — which is exactly what was not
true of `AWS_*`, where 24 benign names lived.

**Considered and excluded: `AWS_PROFILE`, `AWS_ROLE_ARN`.** They select or name a
credential the child must already be able to obtain; neither locates one. The first argument for excluding them — that they
"locate nothing once the two AWS file variables are denied" — is **wrong**, because the
SDK falls back to `~/.aws/credentials` with no environment variable at all. The reason
that survives is narrower: they confer *no capability the child does not already have*,
since anything able to resolve the default credentials file can read it and select any
profile in it. Recorded rather than dropped, because an unrecorded exclusion is
indistinguishable from an oversight.

**Tails, never substrings.** This predicate is a *hard construction refusal* — the
`ExecutionRequest` and `MCPConnectionConfig` validators raise on a credential-shaped
`env_allowlist` entry — so an over-match here is a tool that no longer constructs, not a
cosmetic loss. Matching `TOKEN` or `AUTH` as substrings would refuse `MAX_TOKENS`,
`TOKENIZERS_PARALLELISM`, `TOKEN_BUDGET`, `GIT_AUTHOR_NAME` and `CARGO_PKG_AUTHORS`; in a Git-native framework, a tool that
shells out to `git commit` needs `GIT_AUTHOR_NAME`. The telemetry exporter's
`SENSITIVE_KEY_SUBSTRINGS` is deliberately broader than this list, because over-matching
a *trace* attribute key costs only a `[REDACTED]` —
see [`telemetry-opentelemetry.md`](telemetry-opentelemetry.md) §7.

A credential-shaped name is **not** copied into a child environment merely because it
appears in `env_allowlist`. It has to be passed explicitly through `env`, so that
granting a tool a credential is a visible act at the call site rather than a
consequence of a wildcard.

> **Known deviation from P6.** `WorkspaceSandboxRunner._build_environment` *silently
> skips* a credential-shaped name found in `env_allowlist` rather than raising. The
> caller asked for something it did not get and is not told. This is a discarded
> request, which is the shape P6 forbids elsewhere in this document; it is recorded
> here rather than described as intended behaviour.

### 5.3 Egress — *Modelled, not enforced*

`allow_network` defaults to `False` on `ContainerIsolation` and `WasmIsolation`, and
`ContainerIsolation.egress_allowlist` defaults to empty — deny-by-default wherever it
can be enforced at all. Since neither runner exists yet, egress control is currently
**declared and unenforced**: the models are correct and no code honours them.

*Verified in tests*: `test_workspace_runner_environment_isolation_and_secret_filtering`
runs a real subprocess and asserts an allowlisted non-secret is present, that
allowlisted `OPENAI_API_KEY` and `GH_TOKEN` are **absent**, that an unallowlisted
ambient host variable is absent, and that names passed via `env` (including
`EXPLICIT_API_KEY`) are present. `test_credential_shaped_names_are_recognised` covers
`is_secret_env_name` including case-insensitivity (`gh_token`) and negatives
(`PATH`, `HOME`). `test_credential_names_are_refused_at_construction` and
`test_benign_names_are_not_refused` pin both directions by name — the bare forms
(`API_KEY`, `TOKEN`, `SSH_AUTH_SOCK`, …) that the suffix-only patterns missed, and the
benign names (`GIT_AUTHOR_NAME`, `MAX_TOKENS`, `CARGO_PKG_AUTHORS`, …) that a substring
list would refuse. The negative direction has to be named explicitly: every other
`env_allowlist=` in the suite uses benign placeholders far from any boundary, so a
green suite is not by itself evidence about this predicate.

---

## 6. Execution Adapters — Real State

| Adapter | Level | Status | Where |
| :--- | :--- | :--- | :--- |
| `WorkspaceSandboxRunner` | `workspace` | **Implemented** | `src/uclone_x/sandbox/workspace_runner.py` |
| `PathValidator` | — | **Implemented** | `src/uclone_x/sandbox/path_validator.py` |
| Direct-host runner | `none` | **Planned** — no class exists | — |
| Container runner | `container` | **Planned** — no class exists | — |
| WASM runner | `wasm` | **Planned** — no class exists | — |

> Earlier revisions of this document named these `WorkspaceRunner`, `DirectHostRunner`,
> `DockerRunner` and `WasmRunner`. The one that exists is called
> **`WorkspaceSandboxRunner`**; the other three names are not bound to anything in
> `src/`.

Consequence of the table: the shipped system can execute **only** at `workspace`.
A request at any other level reaches `WorkspaceSandboxRunner.execute` and raises
`SandboxViolationError` — it does not degrade to a weaker level (P6). An absent
external runtime must fail the same way once those runners land.

`SandboxRunnerProtocol` (`src/uclone_x/sandbox/protocols.py`) is the contract:
a `level` property and `async def execute(request) -> ExecutionResult`. It is
deliberately **not** `@runtime_checkable` — on a protocol carrying a `@property`,
`issubclass()` raises `TypeError` and `isinstance()` invokes the object's getters as a
side effect of the type test, and neither form checks a signature, which is what
actually drifted. Conformance is enforced statically instead, by module-level
annotated bindings in each implementation file and in
`tests/unit/test_protocol_conformance.py`.

---

## 7. What `WorkspaceSandboxRunner` Actually Enforces — *Implemented*

`execute()` performs these checks in order, and each failure is an exception rather
than a downgrade. All of the following are covered by `tests/unit/test_sandbox_runner.py`.

1. **Level match.** If `request.isolation.level` is not `WORKSPACE`, raise
   `SandboxViolationError`. Verified for `NoIsolation`, `ContainerIsolation` and
   `WasmIsolation`.
2. **Write paths.** Every entry in `WorkspaceIsolation.write_paths` is resolved against
   `workspace_root`; an escaping entry raises `PathTraversalError`. Note the current
   semantics: the paths are *validated as in-bounds*, and no write restriction is
   applied to the child beyond that — a `workspace` child can still write anywhere its
   cwd and OS permissions allow. The declared paths are checked, not mounted. Furthermore,
   `BashRunTool` does not invoke `WorkspaceSandboxRunner` and does not validate write paths;
   it validates `cwd` only.
3. **cwd boundary.** `request.cwd` is resolved against `workspace_root`; escaping
   raises `PathTraversalError`. A cwd that does not exist, or that exists but is a
   file, raises `FileNotFoundError`.
4. **Command boundary.** A `command` containing `..` is resolved against the workspace
   and raises `PathTraversalError` if it escapes.
5. **Argument boundary.** Each argument that *looks like* a path — absolute, or
   containing a `..` component — is resolved against the workspace and rejected if it
   escapes. A `--flag=value` form is split on the first `=` and the value is checked;
   an empty value is skipped. Verified for `../outside_file.txt`, `/etc/passwd` and
   `--output=../../secret.key`.
   > Limitation, stated because it cuts both ways: the runner cannot tell a path
   > argument from a non-path argument, so it applies a heuristic. A non-path argument
   > that merely looks absolute (a regex, an in-container path, a literal string
   > beginning with `/`) is rejected even though nothing would be read; and a path
   > expressed in a form the heuristic does not recognise is not checked. This is a
   > defence-in-depth measure, not a boundary to rely on — the cwd resolution in step 3
   > is the boundary.
6. **Environment construction.** The child environment is built from scratch by
   `_build_environment` (§5): allowlisted non-secret host names, then explicit `env`
   entries. `os.environ` is never passed through, so `PATH`, `HOME` and every
   credential are absent unless named.
7. **Execution.** `asyncio.create_subprocess_exec` with `cwd=<resolved safe cwd>` and
   `env=<constructed>`, stdout and stderr piped. No shell is involved, so shell
   metacharacters in arguments are inert. (Contrast with `BashRunTool` in
   `src/uclone_x/tools/builtin/shell.py`, which invokes `asyncio.create_subprocess_shell`
   with the raw command string, where arbitrary subshell operations execute on host without
   an OS-level write jail).
8. **Timeout.** `asyncio.wait_for(proc.communicate(), timeout=request.timeout_seconds)`.
   On expiry the process is killed, whatever output was captured is kept,
   `timed_out=True` is returned and the exit code falls back to `-1` if the process
   reports none. A timeout is a **result, not an exception** — callers must inspect
   `timed_out`.
9. **Non-zero exit.** Returned as `exit_code` with stderr intact; not an exception.
   Verified with `sys.exit(42)`.
10. **Command not found.** `FileNotFoundError` naming the command, re-raised from the
    subprocess failure. This *is* an exception, in contrast to (8) and (9).
11. **Provenance.** Every result carries `Provenance.primary(provider="sandbox.workspace")`
    and `isolation_level=IsolationLevel.WORKSPACE` — the isolation actually applied, so
    a consumer can compare it against what was requested. `ExecutionResult.provenance`
    is explicit with no default, per P6.

### 7.1 `PathValidator` — *Implemented*

```python
class PathValidator:
    def is_within_workspace(self, target_path: Path, workspace_root: Path) -> bool: ...
    def resolve_safe_path(self, target_path: Path, workspace_root: Path) -> Path: ...
```

Both resolve `workspace_root` first, join a relative `target_path` onto it, `.resolve()`
the result and test `is_relative_to`. `resolve_safe_path` raises `PathTraversalError`
(a subclass of `SandboxViolationError`) naming both the requested and the resolved path;
`is_within_workspace` returns a bool.

**Symlinks are followed before the boundary test**, because `Path.resolve()` is fully
resolving. A symlink inside the workspace pointing outside it therefore **fails** the
check, and one pointing to another in-workspace file passes and resolves to its target.
Non-existent paths inside the workspace pass, so the validator can be used to check a
file about to be created. Verified in `test_path_validator_symlink_boundaries`,
`test_path_validator_within_workspace`, `test_path_validator_outside_workspace` and
`test_path_validator_resolve_safe_path`.

`PathValidatorProtocol` *is* `@runtime_checkable` — it has no properties and
`test_path_validator_satisfies_protocol` performs a real `isinstance` check.
`WorkspaceSandboxRunner` takes the validator by injection
(`__init__(self, validator: PathValidatorProtocol | None = None)`), defaulting to
`PathValidator()`.

---

## 8. Configuration Surface

Isolation is passed per execution, not held in a global `SandboxConfig` — that class
does not exist. Two entry points carry an `IsolationPolicy`, both defaulting to
`WorkspaceIsolation`:

```python
from uclone_x.sandbox import ExecutionRequest, NoIsolation, WorkspaceIsolation
from uclone_x.tools import ToolContext

# Per-execution (sandbox layer)
request = ExecutionRequest(
    command="python",
    args=("-m", "pytest"),
    cwd=Path("/srv/work/repo"),
    workspace_root=Path("/srv/work"),
    # isolation omitted -> WorkspaceIsolation()
    env_allowlist=("PATH", "HOME"),
    env={"CI": "1"},
    timeout_seconds=60.0,
)

# Per-tool-invocation (tools layer)
ctx = ToolContext(
    agent_id="agt_coder",
    session_id="sess_1",
    workspace_root=Path("/srv/work"),  # required, no default
    # isolation omitted -> WorkspaceIsolation()
    timeout_seconds=30.0,
)
```

`ToolResult` and `ExecutionResult` both report `isolation_level` and `provenance`, so
the level actually applied travels with the result.

---

## 9. Terminology Note

Earlier revisions of this document named the field above `mode`. It is renamed to
**`isolation_level`** to resolve a vocabulary collision recorded in
`2026-09-02-019`: the persona
and sub-agent interface ([`dynamic-persona-interface.md`](dynamic-persona-interface.md)
§3.1) had an unrelated field also spelled with "mode" — `workspace_mode` — whose
values described *where a sub-agent's filesystem context comes from* (inherit the
parent's, run isolated, or share one), not *how strongly execution is isolated
from the host*. That field is renamed to **`fs_scope`** for the same reason.

| Document | Old name | New name |
| :--- | :--- | :--- |
| `sandbox-execution-architecture.md` (this document) | `SandboxConfig.mode` | the `IsolationPolicy` union; `IsolationLevel` names the axis |
| `dynamic-persona-interface.md` | `SubagentInvocation.workspace_mode` | `SubagentInvocation.fs_scope` |

`IsolationLevel` names the security-boundary axis directly, and is unambiguous read
alone in a signature. The enum values (`none` / `workspace` / `container` / `wasm`) are
unchanged, so `WorkspaceIsolation` still means "worktree/dir sandbox". A reader of an
older commit or an external discussion that still says `mode`, `workspace_mode` or
`SandboxConfig.isolation_level` should read this table.

---

## 10. Gaps and Non-Guarantees

Stated so the shipped state is not mistaken for the specified one.

1. **Only `workspace` can execute.** `none`, `container` and `wasm` are expressible and
   have no runner. Nothing dispatches across runners either — there is no registry
   mapping a level to its runner.
2. **No read boundary at any level.** Neither `workspace` nor `none` restricts file reads.
   A child process can read any file the user can read. `read_paths` does not exist on any model.
   `read_roots` (on `AgentConfig` and `ToolContext`, set from Settings or `UCLONE_READ_ROOTS`)
   is not one: it *widens* what the read-only file tools (`file_read`, `file_search`,
   `directory_list`) accept beyond the workspace, and constrains neither a child process
   nor `bash_run`. A root is checked (absolute, a folder, not holding the storage directory)
   when the list is read, not on every tool call: a root replaced by a symlink afterwards
   widens until the next Settings change or agent build.
3. **Egress is declared, not enforced** (§5.3).
4. **`write_paths` is validated, not applied** (§7 step 2).
5. **Silent secret skip** in `_build_environment` (§5.2 note).
6. **Argument path detection is heuristic** (§7 step 5).
7. **`WorkspaceSandboxRunner.execute` still reads a legacy field name.** It resolves
   the policy with `getattr(request, "isolation", getattr(request, "policy", None))`.
   `ExecutionRequest` has no `policy` field and forbids extras, so the fallback is
   unreachable vestigial code that weakens the static type of the value it produces.
8. **No provenance-derived floor.** The threat model's D1 recommendation (option D — a
   floor derived from where the execution request came from) is *not* implemented. The
   default is a single global constant, which is option B. The distinction matters:
   A2A-originated tasks and synthesized-skill execution get the same `workspace` floor
   as human-interactive local work, and nothing in the request records its origin.
9. **No shell write-path boundary under `workspace` isolation (#684).** In the agent runtime,
   `BashRunTool` and `run_command` execute commands directly through `asyncio.create_subprocess_shell`.
   They validate that `cwd` stays within `workspace_root` and scrub secret environment variables, but
   do not sandbox command strings at the OS level. A command invoking relative path escapes (e.g.
   `touch ../outside.txt`, shell redirection, or `rm -rf`) executes with ambient host permissions.
   `WorkspaceSandboxRunner` validates argument paths for structured `ExecutionRequest` calls, but is
   not used by `BashRunTool` and does not provide an OS filesystem jail. Strong write containment
   requires `ContainerIsolation` or disposable execution checkouts.
10. **The story library is read-only to the shell and local MCP servers on macOS only (#1583,
   #1589).** `file_write`, `file_edit`, both `generate_image` tools and `character_sheet` resolve
   their target with `BaseTool.resolve_write_path`, which refuses a path in `<workspace>/stories`
   (`tools.base.in_story_library`). `bash_run`/`run_command` and stdio MCP servers cannot be checked
   that way (item 9: the command string is not parsed; an MCP proxy cannot tell a read from a write),
   so on macOS they run under `sandbox-exec` with a profile that denies every write in the library
   and the renaming of the workspace or any folder above it (`sandbox.story_jail`). A missing
   `sandbox-exec`, or one that cannot apply the profile, refuses the command rather than running it
   unjailed. `sandbox-exec` is marked DEPRECATED in its macOS manual page (Apple points apps to the
   App Sandbox); it still ships, and if a later macOS drops it, every shell command and local MCP
   server is refused rather than run unprotected. Still open: Linux and Windows (no jail; a shell
   command or MCP filesystem server can write a story without its lease, digest or approval); MCP
   servers reached over HTTP; a local MCP server loaded from `mcp.json` whose `workspace_root` is
   not the app's workspace (its own `workspace_root`, the loader's root, or the current folder),
   whose jail protects that root's library rather than the real one; a hardlink to a story file
   made before the process started; and a jailed process asking an unjailed one (the local API,
   another application) to write for it.

---

## 11. Related

* [`security-threat-model.md`](security-threat-model.md) — D1 (default level), T1–T3, F5, D2
* `2026-09-02-001` — the finding this default answers
* `2026-09-02-019` — the `mode` vocabulary collision
* [`governance/principle-amendment-policy.md`](governance/principle-amendment-policy.md) §5 — the amendment log rows for P3
* [`skill-system-architecture.md`](skill-system-architecture.md) — the auditor's `isolation_floor` and the skill side of clamping
