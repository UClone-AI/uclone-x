"""Credential-name recognition, shared by every layer that must redact one.

This predicate and its pattern lists were defined in `sandbox/models.py`, which made a
cross-cutting concern look like a sandbox concept. Three subsystems consume it —
`tools/` when building a child environment, `sandbox/` when running one, and
`telemetry/` when exporting a span — so it belongs in the kernel proper rather than in
one subsystem's models module that the others reach into
(the core/shell architecture note C7, §5 module table).

Moving it changes no behaviour and no pattern. The lists below are unmodified; their
rationale, including what each pattern is for and what was deliberately excluded, is
preserved verbatim from where it was written.

**And no importer moved.** `sandbox/models.py` re-exports every name, so all three
subsystems still reach through it — including `telemetry/exporter.py`, whose import line is
what C7 names. That is why C7 is a relocation and not an edge repair: §5 classifies
`sandbox/models.py` as kernel, so the edge was never a layering violation and there is
nothing to break. Updating the importers is mechanical and worth doing; it is not done here,
and this file should not be read as evidence that it was.

Write-Time Credential Redaction (Option A, #569):
An append-only log retains whatever is written to it. An API key pasted into chat or appearing
in tool results is written to the log and any derived summary. Under Principle 0 (P0) and
security threat model T3 (credential exfiltration), retaining unmasked credentials without user
remedy is unacceptable. UClone-X applies Option A write-time redaction across logs, session
state, compactor ledgers, and telemetry using recognizable credential shapes and patterns.

Known Limitations of Pattern-Based Redaction (Option A):
Pattern-based redaction is a heuristic mitigation, not an absolute guarantee:
1. Catches known shapes: It matches recognizable patterns such as OpenAI (`sk-...`),
   Anthropic (`sk-ant-...`), GitHub (`ghp_...`, `github_pat_...`), AWS Access Keys (`AKIA...`),
   AWS Secret Keys, Bearer tokens, private keys, and explicit secret assignments.
2. Unstructured arbitrary secrets: Arbitrary high-entropy hex or base64 strings pasted
   without recognizable prefixes or assignment variable names cannot be reliably detected
   without intolerable false-positive rates, and will not be redacted.
3. Obfuscated tokens: Base64-encoded, split, or obfuscated tokens evade pattern matching.
4. Mitigation, not boundary: Per Principle 3 (P3) and threat model T3, write-time redaction
   reduces retention risk in append-only storage, but process isolation and network egress
   controls remain the primary defense against credential exfiltration.
"""

from __future__ import annotations

import fnmatch
import re

__all__ = [
    "REDACTED_PLACEHOLDER",
    "SECRET_ENV_EXACT_NAMES",
    "SECRET_ENV_FAMILY_PATTERNS",
    "SECRET_ENV_PATTERNS",
    "SECRET_NAME_TAILS",
    "contains_credential",
    "is_secret_env_name",
    "redact_credentials",
]


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
"""Credential-shaped *whole trailing segments* of an environment-variable name.

Each tail yields exactly two globs: the bare name (`TOKEN`) and the underscore-suffixed
form (`*_TOKEN`). It is deliberately **not** a substring list. `fnmatch`'s `*` cannot
match a zero-length segment before a literal `_`, so the suffix glob alone silently
missed every bare name this control is named for — `API_KEY`, `TOKEN`, `PASSWORD`,
`PRIVATE_KEY` all returned False.

The bare/suffix pair is also what keeps the false-positive set empty. Matching `TOKEN`
as a *substring* would refuse `MAX_TOKENS`, `TOKENIZERS_PARALLELISM` and `TOKEN_BUDGET`;
matching `AUTH` as a substring would refuse `GIT_AUTHOR_NAME` and `CARGO_PKG_AUTHORS`.
In a Git-native framework a tool that shells out to `git commit` needs `GIT_AUTHOR_NAME`,
and this predicate is a hard construction refusal (`ExecutionRequest` /
`MCPConnectionConfig` validators), not a display filter: over-matching here breaks a
tool outright. The telemetry exporter's `SENSITIVE_KEY_SUBSTRINGS` may be — and is —
broader, because over-matching a *trace* attribute key costs only a `[REDACTED]`.
"""

SECRET_ENV_EXACT_NAMES: tuple[str, ...] = (
    "AWS_ACCESS_KEY_ID",
    "AWS_CONFIG_FILE",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
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
"""Credential-shaped names that no tail rule can reach.

**Which list a name belongs in** — the mechanical criterion, applied before any
judgement about severity:

> A name belongs in `SECRET_NAME_TAILS` **iff** the credential word is the whole
> trailing `_`-delimited segment. Everything else belongs here.

`APIKEY`, `KUBECONFIG`, `NETRC` and `PASSWD` are glued but their trailing segment *is*
the whole credential word, so they are tails and `OPENAI_APIKEY` matches. `DATABASE_URL`,
`REDIS_URL` and `MONGODB_URI` are tails so their prefixed forms (`PROD_DATABASE_URL`,
`TEST_REDIS_URL`, `STAGING_MONGODB_URI`) match. `PGPASSWORD`'s trailing segment is
`PGPASSWORD`, not `PASSWORD`, so no tail reaches it and no tail should be widened to
try — a `*PASSWORD*` substring would refuse `PASSWORD_STORE_DIR`, and the `AUTH`
equivalent would refuse `GIT_AUTHOR_NAME`, the regression ruled out twice (#87, #95).

**Whether a name belongs at all** — for names that *locate* a credential rather than
being one, count artifacts:

> How many distinct artifacts must be combined before the holder has a usable
> credential? One means it belongs here.

`DOCKER_CONFIG` → one (`auths` are plaintext base64). `GNUPGHOME` → one (the keyring is
the secret; passphrase-less keys are common, so this does not rest on agent state).
`GPG_AGENT_INFO` → one (historical; inert/removed in GnuPG 2.1 which uses a fixed
socket under `GNUPGHOME` or standard run directory, retained for environments using
older GnuPG releases). `SSH_AUTH_SOCK` → one (live signing capability). Each
`AWS_CONTAINER_*` → one. By the same count `PASSWORD_STORE_DIR` is **two** — GPG-encrypted
blobs plus a keyring located elsewhere — and stays out. Counting artifacts replaces an
earlier "encrypted at rest" formulation that failed in both directions: it leaned on
*may* (a cached passphrase, a present keyring) and so rescued or condemned the same name
depending on machine state.

**Arbitrary code execution vectors** — names that cause tooling to invoke arbitrary
executables or shell commands to source or prompt for credentials belong here regardless
of artifact count:

* `AWS_CONFIG_FILE` — an AWS config profile can define `credential_process`, which runs
  an arbitrary shell command to retrieve dynamic credentials.
* `GIT_CONFIG_*` (`GIT_CONFIG`, `GIT_CONFIG_GLOBAL`, `GIT_CONFIG_SYSTEM`,
  `GIT_CONFIG_COUNT` / `GIT_CONFIG_KEY_*` / `GIT_CONFIG_VALUE_*`) — a git configuration
  can define `credential.helper` beginning with `!`, which executes arbitrary shell
  commands when resolving credentials.
* `SSH_ASKPASS` — specifies an executable program invoked by ssh / git to query the
  user for a passphrase or credential.

**`AWS_ACCESS_KEY_ID` is here despite failing that count**, and the exception is argued
rather than assumed. It is not a locator, so the count does not govern it — it is one
half of a credential pair, and the count was devised to adjudicate locators
(`DOCKER_CONFIG` against `PASSWORD_STORE_DIR`). Applying it to a credential *component*
is a category error. The independent reason it costs nothing: no child needs the key id
without the secret, and the secret is already refused by `*_ACCESS_KEY`, so a tool that
legitimately needs AWS credentials passes both through `env` regardless. Denying it
adds no refusal a working tool would hit.

**`GIT_CONFIG_*` — a family, verified by execution, not inference.** A `credential.helper`
beginning `!` is run through the shell, so a child inheriting one of these can execute
arbitrary code and harvest whatever the developer's real helper would have supplied:

* `GIT_CONFIG_GLOBAL`, `GIT_CONFIG_SYSTEM` — both redirect the normal config stack and
  both execute the helper (`git var GIT_AUTHOR_IDENT` reports the injected identity;
  `git credential fill` runs the helper, 3/3 runs under `env -i`).
* `GIT_CONFIG_COUNT` with `GIT_CONFIG_KEY_<n>` / `GIT_CONFIG_VALUE_<n>` executes the
  helper **with no file involved at all** — config injected straight from the
  environment. The arity is unbounded, which is why those two need
  `SECRET_ENV_FAMILY_PATTERNS` below; no finite named set expresses them.
* `GIT_CONFIG` is the weak member and is listed at lower severity: it is **not** honoured
  by the normal config stack on git 2.50.1 — `git var` ignores it and `git credential
  fill` does not execute the helper. It is honoured only by `git config` itself, so it
  leaks to tooling that shells out to `git config` to read settings. A read, not
  execution.

Nothing in this repository sets any of them: `GIT_CONFIG*` appears in no source, script,
`Dockerfile`, `docker-compose.yml` or workflow, and there is no `.pre-commit-config.yaml`
at all.

**Considered and excluded**: `AWS_PROFILE` and `AWS_ROLE_ARN` select or name a credential
the child must already be able to obtain; they locate nothing and confer no capability a
child that can read the default credentials file does not already have (#109).
`PASSWORD_STORE_DIR` is excluded by the artifact count above.

**What this list cannot do.** It is only ever as complete as the fixture it was built
from, so it ends the churn over *mechanism* — pointers, glued forms and families each
now have a home — without ever being able to guarantee that no name is missing. No test
in this design can discover a name nobody listed; see
`test_pointer_shaped_vendor_credentials_are_exact_listed`.
"""

SECRET_ENV_FAMILY_PATTERNS: tuple[str, ...] = (
    "GIT_CONFIG_KEY_*",
    "GIT_CONFIG_VALUE_*",
)
"""Unbounded-arity families: a third class that neither tails nor names can express.

`GIT_CONFIG_COUNT=2` with `GIT_CONFIG_KEY_0`/`_1` and `GIT_CONFIG_VALUE_0`/`_1` injects
config — including an executing `credential.helper` — with no file anywhere. The index is
unbounded, so enumeration is impossible and a prefix glob is the only expression.

These are prefix globs, which #95 removed for `AWS_*`/`GH_*`/`GITHUB_*`, so the
narrowness is the whole justification: `GIT_CONFIG_KEY_*` can only match a name that is
already a git config-injection slot, and no benign variable is spelled that way. A glob
earns its place here when the namespace it covers exists solely to carry the dangerous
value — which was precisely what was *not* true of `AWS_*`, where 24 benign names lived.
"""

SECRET_ENV_PATTERNS: tuple[str, ...] = (
    *SECRET_ENV_EXACT_NAMES,
    *SECRET_ENV_FAMILY_PATTERNS,
    *SECRET_NAME_TAILS,
    *(f"*_{tail}" for tail in SECRET_NAME_TAILS),
)
"""Environment-variable name patterns treated as secrets by default.

From `docs/security-threat-model.md` D1 control 1. A name matching one of these is not
copied into a child environment merely because it was allowlisted; it has to be passed
explicitly, so that granting a tool a credential is a visible act at the call site.
"""

_SECRET_NAME_SEPARATORS: tuple[str, ...] = ("-", ".")


def is_secret_env_name(name: str) -> bool:
    """Return True if `name` looks like a credential by the patterns above.

    Separators are normalised to `_` before matching, so the HTTP-header and dotted
    span-attribute spellings of the same credential (`x-api-key`, `llm.api_key`) are
    recognised as well as the environment spelling (`X_API_KEY`).
    """
    upper = name.upper()
    for separator in _SECRET_NAME_SEPARATORS:
        upper = upper.replace(separator, "_")
    return any(fnmatch.fnmatch(upper, pattern) for pattern in SECRET_ENV_PATTERNS)


REDACTED_PLACEHOLDER: str = "[REDACTED]"
"""Standard marker string replacing redacted credentials across logs and telemetry."""

# Compiled regex patterns and replacement templates for credential values.
# Order is deliberate: multi-line and specific vendor prefixes precede generic assignments.
_CREDENTIAL_VALUE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # 1. Multi-line PEM private keys
    (
        re.compile(
            r"-----BEGIN [A-Z0-9_-]+ PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9_-]+ PRIVATE KEY-----"
        ),
        "[REDACTED PRIVATE KEY]",
    ),
    # 2. OpenAI and Anthropic API keys (including sk-proj-, sk-admin-, sk-ant-api03-)
    (
        re.compile(r"\bsk-(?:proj-|admin-|ant-)?[\w-]{20,}\b"),
        REDACTED_PLACEHOLDER,
    ),
    # 3. GitHub personal access, OAuth, fine-grained tokens
    (
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{22,})\b"),
        REDACTED_PLACEHOLDER,
    ),
    # 4. AWS Access Key IDs
    (
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        REDACTED_PLACEHOLDER,
    ),
    # 5. AWS Secret Access Keys in key-value contexts
    (
        re.compile(
            r"(?i)\b((?:aws_secret_access_key|aws_secret_key)\s*[:=]\s*['\"]?)([A-Za-z0-9/+=]{40})(['\"]?)"
        ),
        r"\1" + REDACTED_PLACEHOLDER + r"\3",
    ),
    # 6. Bearer / JWT Authorization tokens
    (
        re.compile(r"\bBearer\s+[A-Za-z0-9_\-\.~+/]+=*"),
        "Bearer " + REDACTED_PLACEHOLDER,
    ),
    # 7. Slack API tokens (bot, user, app, workspace)
    (
        re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"),
        REDACTED_PLACEHOLDER,
    ),
    # 8. Explicit generic secret assignments
    (
        re.compile(
            r"(?i)\b((?:api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?key|client[_-]?secret|token)\s*[:=]\s*['\"]?)([A-Za-z0-9_\-\.~+/]{16,})(['\"]?)"
        ),
        r"\1" + REDACTED_PLACEHOLDER + r"\3",
    ),
)


def contains_credential(text: str) -> bool:
    """Return True if `text` contains any recognized credential pattern."""
    return any(pattern.search(text) is not None for pattern, _ in _CREDENTIAL_VALUE_PATTERNS)


def redact_credentials(text: str, placeholder: str = REDACTED_PLACEHOLDER) -> str:
    """Redact recognized credential shapes and secrets from `text`.

    Option A mitigation from #569: replaces known shapes with `placeholder`
    before strings reach append-only event logs, session state, or telemetry.

    Known Limitations of Pattern-Based Redaction (Option A):
    Pattern-based redaction is a heuristic mitigation, not a complete security guarantee.
    It catches known credential shapes (e.g. OpenAI `sk-...`, GitHub `ghp_...`, Anthropic
    `sk-ant-...`, AWS access keys, Bearer tokens, and explicit secret assignments), but cannot
    detect arbitrary high-entropy strings, bespoke tokens without prefixes, or obfuscated
    secrets without high false-positive rates. Per Principle 3 (P3) and threat model T3,
    redaction on write reduces retention risk, but does not replace process-level isolation
    or host egress boundaries.
    """
    if not text:
        return text
    result = text
    for pattern, repl in _CREDENTIAL_VALUE_PATTERNS:
        if repl == REDACTED_PLACEHOLDER:
            result = pattern.sub(placeholder, result)
        else:
            result = pattern.sub(repl, result)
    return result
