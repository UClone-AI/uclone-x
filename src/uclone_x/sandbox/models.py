"""Data models for pluggable sandbox execution and path safety.

The isolation configuration is a **discriminated union**, not a flat record with a
level field. `docs/security-threat-model.md` D1 requires this: `memory_limit_mb`,
`cpu_shares` and `allow_network` were previously accepted at every level and silently
discarded at `none` and `workspace`, and "a security control that is accepted and
silently ignored is precisely what P6 forbids". Under a union, a limit that cannot be
enforced at a level is not a field of that level, so requesting it is a validation
error rather than a false sense of protection.

Two controls sit on the request itself rather than inside the union, because the threat
model finds they close more of the credential-exfiltration path (T3) than the isolation
level does, and they apply at every level including `none`:

* `env_allowlist` — the child receives an explicitly constructed environment. Nothing is
  inherited from the host unless it is named here.
* egress — denied by default wherever it can be enforced at all.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from uclone_x.core.immutable import ImmutableStrMapping
from uclone_x.core.provenance import Provenance
from uclone_x.core.secrets import (
    SECRET_ENV_EXACT_NAMES,
    SECRET_ENV_FAMILY_PATTERNS,
    SECRET_ENV_PATTERNS,
    SECRET_NAME_TAILS,
    is_secret_env_name,
)

__all__ = [
    "AVAILABLE_ISOLATION_LEVELS",
    "DEFAULT_ISOLATION_LEVEL",
    "SECRET_ENV_EXACT_NAMES",
    "SECRET_ENV_FAMILY_PATTERNS",
    "SECRET_ENV_PATTERNS",
    "SECRET_NAME_TAILS",
    "ContainerIsolation",
    "ExecutionRequest",
    "ExecutionResult",
    "IsolationLevel",
    "IsolationPolicy",
    "NoIsolation",
    "WasmIsolation",
    "WorkspaceIsolation",
    "effective_isolation_level",
    "is_secret_env_name",
    "is_weaker_isolation",
]


class IsolationLevel(StrEnum):
    """Execution isolation levels.

    Named `isolation_level`, not `mode`: issue 2026-09-02-019 separated this security
    axis from the sub-agent filesystem axis (`FsScope`) after `mode` came to name both.
    """

    NONE = "none"
    WORKSPACE = "workspace"
    CONTAINER = "container"
    WASM = "wasm"


class NoIsolation(BaseModel):
    """Direct host execution. No boundary of any kind.

    Deliberately carries no resource or network fields: none of them can be enforced
    here, so under P6 none of them may be expressible.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    level: Literal[IsolationLevel.NONE] = IsolationLevel.NONE


class WorkspaceIsolation(BaseModel):
    """A filesystem **write** boundary, and nothing more.

    The threat model calls this "the most commonly misread cell in the matrix": it does
    not restrict reads, does not scrub the environment and cannot enforce egress. There
    is therefore no `allow_network` field here — requesting network denial at this level
    would be a control that is accepted and ignored.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    level: Literal[IsolationLevel.WORKSPACE] = IsolationLevel.WORKSPACE
    write_paths: tuple[Path, ...] = Field(
        default_factory=tuple,
        description="Writable paths, relative to the request's `workspace_root`. Empty "
        "means the workspace root itself.",
    )


class ContainerIsolation(BaseModel):
    """Container isolation: a real filesystem boundary, plus enforceable limits."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    level: Literal[IsolationLevel.CONTAINER] = IsolationLevel.CONTAINER
    image: str
    write_paths: tuple[Path, ...] = Field(default_factory=tuple)
    allow_network: bool = False
    egress_allowlist: tuple[str, ...] = Field(default_factory=tuple)
    memory_limit_mb: int = 512
    cpu_shares: float = 1.0


class WasmIsolation(BaseModel):
    """WASM isolation: no ambient filesystem or network, capabilities granted explicitly."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    level: Literal[IsolationLevel.WASM] = IsolationLevel.WASM
    preopened_dirs: tuple[Path, ...] = Field(default_factory=tuple)
    allow_network: bool = False
    memory_limit_mb: int = 256


IsolationPolicy = Annotated[
    NoIsolation | WorkspaceIsolation | ContainerIsolation | WasmIsolation,
    Field(discriminator="level"),
]
"""The isolation applied to one execution, discriminated on `level`."""


DEFAULT_ISOLATION_LEVEL = IsolationLevel.WORKSPACE
"""P3's mandated default, decided by the project owner on 2026-09-02 (issue 2026-09-02-001).

Recorded as a Tier A amendment in `docs/governance/principle-amendment-policy.md` §5.
`none` is retained as an explicit opt-in and "must never be reached by defaulting",
which is why every field below defaults to a `WorkspaceIsolation()` and none of them
defaults to `NoIsolation()`.
"""

AVAILABLE_ISOLATION_LEVELS: frozenset[IsolationLevel] = frozenset(
    {IsolationLevel.NONE, IsolationLevel.WORKSPACE}
)
"""The set of isolation levels that have implemented runner backends in the system."""

_LEVEL_STRENGTH: dict[IsolationLevel, int] = {
    IsolationLevel.NONE: 0,
    IsolationLevel.WORKSPACE: 1,
    IsolationLevel.CONTAINER: 2,
    IsolationLevel.WASM: 3,
}


def is_weaker_isolation(level: IsolationLevel, reference: IsolationLevel) -> bool:
    """Return True if `level` provides strictly weaker isolation than `reference`."""
    return _LEVEL_STRENGTH[level] < _LEVEL_STRENGTH[reference]


def effective_isolation_level(
    requested: IsolationLevel,
    floor: IsolationLevel | None,
    *,
    available_levels: frozenset[IsolationLevel] = AVAILABLE_ISOLATION_LEVELS,
) -> IsolationLevel:
    """Return the level actually applied when `requested` meets a runtime `floor`.

    P3 (as amended for issue 2026-09-02-001): "a requesting artifact never raises its
    own ceiling". A synthesized skill, a remote A2A task or an MCP provider may *ask*
    for a level, and the runtime resolves that request against its own floor — a
    request can only strengthen isolation, never weaken it below the floor. Expressed
    as a function so the clause is testable rather than restated at each call site.

    P6 Fail-Fast: Requesting an isolation level with no available backend runner fails
    explicitly with SandboxViolationError rather than silently certifying isolation that
    never ran (issue #30).
    """
    from uclone_x.errors import SandboxViolationError

    if requested not in available_levels:
        raise SandboxViolationError(
            f"Requested isolation level '{requested.value}' has no available runner backend "
            f"(available: {sorted(lvl.value for lvl in available_levels)})"
        )

    if floor is None:
        raise SandboxViolationError("Host provides no execution runner (isolation floor is None)")

    if _LEVEL_STRENGTH[requested] >= _LEVEL_STRENGTH[floor]:
        return requested
    return floor


class ExecutionRequest(BaseModel):
    """Request specification for executing a command or tool in a sandbox."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    command: str
    args: tuple[str, ...] = Field(default_factory=tuple)
    cwd: Path
    workspace_root: Path = Field(
        description="The boundary the isolation policy is expressed relative to. Held "
        "here rather than on each policy so there is exactly one copy of it "
        "(issue 2026-09-02-042), and required, because the previous default of `.` bound "
        "the boundary to whatever directory the process started in.",
    )
    isolation: IsolationPolicy = Field(
        default_factory=WorkspaceIsolation,
        description="Defaults to `workspace` per P3 as amended for issue "
        "2026-09-02-001. Reaching `none` requires constructing `NoIsolation()` "
        "explicitly, which is the point: it may never be arrived at by defaulting.",
    )
    env: ImmutableStrMapping = Field(
        default_factory=dict,
        description="The child's environment, constructed explicitly. Never inherited.",
    )
    env_allowlist: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Host environment names to copy in. Deny-by-default: anything not "
        "named here is absent from the child. Credential-shaped names must be passed "
        "through `env` instead, so granting one is visible at the call site.",
    )
    timeout_seconds: float = 60.0

    @model_validator(mode="after")
    def _validate_env_allowlist(self) -> Self:
        for name in self.env_allowlist:
            if is_secret_env_name(name):
                raise ValueError(
                    f"Credential-shaped environment variable '{name}' in env_allowlist is denied; "
                    "pass explicit credentials via 'env' instead"
                )
        return self


class ExecutionResult(BaseModel):
    """Result of sandboxed execution."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0.0
    timed_out: bool = False
    isolation_level: IsolationLevel = Field(
        description="The isolation actually applied, which a consumer must be able to "
        "compare against what was requested.",
    )
    provenance: Provenance | None = Field(
        description="In-band attribution required by Principle 6. Explicit with no "
        "default: `None` is representable so a non-conformant value can be rejected by "
        "`require_provenance`, but it is never inherited silently.",
    )
