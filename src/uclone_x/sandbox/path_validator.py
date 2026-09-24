"""Path validator enforcing workspace boundaries and preventing path traversal."""

from __future__ import annotations

from pathlib import Path

from uclone_x.errors import PathTraversalError
from uclone_x.sandbox.protocols import PathValidatorProtocol


class PathValidator:
    """Validator for enforcing workspace bounds and preventing path traversal attacks."""

    def is_within_workspace(self, target_path: Path, workspace_root: Path) -> bool:
        """Return True if path resolves strictly inside the workspace boundary.

        NOTE: Prefer `resolve_safe_path()` over this boolean check to prevent fail-open
        mistakes (P3, P6).
        """
        try:
            self.resolve_safe_path(target_path, workspace_root)
            return True
        except PathTraversalError:
            return False

    def resolve_safe_path(self, target_path: Path, workspace_root: Path) -> Path:
        """Resolve an absolute path, raising `PathTraversalError` if outside the boundary."""
        resolved_root = workspace_root.resolve()
        if target_path.is_absolute():
            resolved_target = target_path.resolve()
        else:
            resolved_target = (resolved_root / target_path).resolve()
        if not resolved_target.is_relative_to(resolved_root):
            raise PathTraversalError(
                f"Path '{target_path}' resolves to '{resolved_target}', "
                f"which escapes workspace root '{resolved_root}'"
            )
        return resolved_target


# Static conformance check
_conformance_check: PathValidatorProtocol = PathValidator()
