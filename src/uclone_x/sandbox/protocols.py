"""Protocols for sandbox runners and path safety validators.

`@runtime_checkable` is applied only where a runtime `isinstance` check is actually
performed. On a protocol with a `@property`, `issubclass()` raises `TypeError` and
`isinstance()` calls the object's getters as a side effect of the type test, and neither
form checks a signature — which is what actually drifted in issue 2026-09-02-035.
Conformance is enforced statically instead, by the bindings in
`tests/unit/test_protocol_conformance.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from uclone_x.sandbox.models import (
    ExecutionRequest,
    ExecutionResult,
    IsolationLevel,
)


@runtime_checkable
class PathValidatorProtocol(Protocol):
    """Protocol for enforcing workspace bounds and preventing path traversal attacks."""

    def is_within_workspace(self, target_path: Path, workspace_root: Path) -> bool:
        """Return True if path resolves strictly inside the workspace boundary.

        NOTE: Prefer `resolve_safe_path()` over this boolean check to prevent fail-open
        mistakes (P3, P6).
        """
        ...

    def resolve_safe_path(self, target_path: Path, workspace_root: Path) -> Path:
        """Resolve an absolute path, raising `PathTraversalError` if outside the boundary."""
        ...


class SandboxRunnerProtocol(Protocol):
    """Protocol for pluggable sandbox execution runners."""

    @property
    def level(self) -> IsolationLevel:
        """The isolation level handled by this runner."""
        ...

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute the command within the requested isolation.

        Raises:
            SandboxViolationError: If the request asks for a boundary this runner cannot
                enforce. Per P6 a control that cannot be honoured is an error, never a
                discarded field.
        """
        ...
