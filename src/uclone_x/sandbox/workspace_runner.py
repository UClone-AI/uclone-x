"""Workspace sandbox runner enforcing process isolation and path boundary safety."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from uclone_x.core.provenance import Provenance
from uclone_x.errors import SandboxViolationError
from uclone_x.sandbox.models import (
    ExecutionRequest,
    ExecutionResult,
    IsolationLevel,
    WorkspaceIsolation,
    is_secret_env_name,
)
from uclone_x.sandbox.path_validator import PathValidator
from uclone_x.sandbox.protocols import PathValidatorProtocol, SandboxRunnerProtocol


class WorkspaceSandboxRunner:
    """Pluggable sandbox runner enforcing workspace-level isolation.

    Enforces workspace directory boundary constraints, rejects path traversals,
    constructs an isolated environment filtering out secret variables, and executes
    commands asynchronously with timeout enforcement and in-band provenance.
    """

    def __init__(self, validator: PathValidatorProtocol | None = None) -> None:
        self._validator: PathValidatorProtocol = (
            validator if validator is not None else PathValidator()
        )

    @property
    def level(self) -> IsolationLevel:
        """The isolation level handled by this runner."""
        return IsolationLevel.WORKSPACE

    def _validate_path_arg(self, arg: str, workspace_root: Path) -> None:
        """Validate an argument or sub-argument against workspace boundaries if it represents a path."""
        p = Path(arg)
        if p.is_absolute() or ".." in p.parts:
            self._validator.resolve_safe_path(p, workspace_root)

    def _build_environment(self, request: ExecutionRequest) -> dict[str, str]:
        """Construct isolated environment from allowlist and explicit variables."""
        child_env: dict[str, str] = {}

        # 1. Allowlisted host environment variables (filtering out secret patterns)
        for key in request.env_allowlist:
            if is_secret_env_name(key):
                # Threat model D1: Secrets cannot be implicitly inherited via allowlist
                continue
            if key in os.environ:
                child_env[key] = os.environ[key]

        # 2. Explicitly passed environment variables (explicit grants at call site)
        for key, value in request.env.items():
            child_env[key] = str(value)

        return child_env

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute the command within workspace isolation.

        Raises:
            SandboxViolationError: If the request policy level is not WORKSPACE.
            PathTraversalError: If working directory, write paths, or arguments escape workspace.
            FileNotFoundError: If the working directory or command does not exist.
        """
        # 1. Validate isolation level
        isolation = getattr(request, "isolation", getattr(request, "policy", None))
        if isolation is None or isolation.level != IsolationLevel.WORKSPACE:
            level_name = isolation.level if isolation is not None else "None"
            raise SandboxViolationError(
                f"WorkspaceSandboxRunner only handles IsolationLevel.WORKSPACE, got {level_name}"
            )

        # 2. Validate write paths if specified in WorkspaceIsolation
        if isinstance(isolation, WorkspaceIsolation) and isolation.write_paths:
            for write_path in isolation.write_paths:
                self._validator.resolve_safe_path(write_path, request.workspace_root)

        # 3. Validate working directory
        safe_cwd = self._validator.resolve_safe_path(request.cwd, request.workspace_root)
        if not safe_cwd.exists() or not safe_cwd.is_dir():
            raise FileNotFoundError(f"Working directory does not exist: '{safe_cwd}'")

        # 4. Validate command and arguments
        p_cmd = Path(request.command)
        if ".." in p_cmd.parts:
            self._validator.resolve_safe_path(p_cmd, request.workspace_root)

        for arg in request.args:
            if arg.startswith("-") and "=" in arg:
                _, val = arg.split("=", 1)
                if val:
                    self._validate_path_arg(val, request.workspace_root)
            else:
                self._validate_path_arg(arg, request.workspace_root)

        # 5. Build isolated environment
        child_env = self._build_environment(request)

        # 6. Execute subprocess asynchronously
        start_time = time.monotonic()
        timed_out = False
        stdout_str = ""
        stderr_str = ""
        exit_code = 0

        try:
            proc = await asyncio.create_subprocess_exec(
                request.command,
                *request.args,
                cwd=str(safe_cwd),
                env=child_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=request.timeout_seconds,
                )
                stdout_str = stdout_bytes.decode("utf-8", errors="replace")
                stderr_str = stderr_bytes.decode("utf-8", errors="replace")
                exit_code = proc.returncode if proc.returncode is not None else 0
            except TimeoutError:
                timed_out = True
                try:
                    proc.kill()
                    stdout_bytes, stderr_bytes = await proc.communicate()
                    stdout_str = stdout_bytes.decode("utf-8", errors="replace")
                    stderr_str = stderr_bytes.decode("utf-8", errors="replace")
                except Exception:
                    pass
                exit_code = proc.returncode if proc.returncode is not None else -1
        except FileNotFoundError as err:
            raise FileNotFoundError(f"Command not found: '{request.command}'") from err

        duration_ms = (time.monotonic() - start_time) * 1000.0
        provenance = Provenance.primary(provider="sandbox.workspace")

        return ExecutionResult(
            exit_code=exit_code,
            stdout=stdout_str,
            stderr=stderr_str,
            duration_ms=round(duration_ms, 3),
            timed_out=timed_out,
            isolation_level=IsolationLevel.WORKSPACE,
            provenance=provenance,
        )


# Static conformance check
_conformance_check: SandboxRunnerProtocol = WorkspaceSandboxRunner()
