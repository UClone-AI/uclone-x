"""Unit tests for WorkspaceSandboxRunner and PathValidator."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from uclone_x.core.provenance import ExecutionPath
from uclone_x.errors import PathTraversalError, SandboxViolationError
from uclone_x.sandbox import (
    ContainerIsolation,
    ExecutionRequest,
    ExecutionResult,
    IsolationLevel,
    NoIsolation,
    PathValidator,
    PathValidatorProtocol,
    SandboxRunnerProtocol,
    WasmIsolation,
    WorkspaceIsolation,
    WorkspaceSandboxRunner,
)

# ======================================================================================
# PathValidator Tests
# ======================================================================================


def test_path_validator_satisfies_protocol() -> None:
    validator = PathValidator()
    assert isinstance(validator, PathValidatorProtocol)


def test_path_validator_within_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sub_dir = workspace / "sub"
    sub_dir.mkdir()
    file_path = sub_dir / "test.txt"
    file_path.write_text("hello")

    validator = PathValidator()

    # Root itself
    assert validator.is_within_workspace(workspace, workspace)
    assert validator.is_within_workspace(Path("."), workspace)

    # Relative paths inside workspace
    assert validator.is_within_workspace(Path("sub/test.txt"), workspace)
    assert validator.is_within_workspace(Path("./sub/test.txt"), workspace)
    assert validator.is_within_workspace(Path("sub/../sub/test.txt"), workspace)

    # Absolute paths inside workspace
    assert validator.is_within_workspace(file_path, workspace)
    assert validator.is_within_workspace(sub_dir, workspace)

    # Non-existent paths inside workspace
    assert validator.is_within_workspace(Path("sub/new_file.txt"), workspace)


def test_path_validator_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "secret.txt"
    outside_file.write_text("secret")

    validator = PathValidator()

    # Relative traversal escaping workspace
    assert not validator.is_within_workspace(Path("../outside/secret.txt"), workspace)
    assert not validator.is_within_workspace(Path("../../secret.txt"), workspace)

    # Absolute path escaping workspace
    assert not validator.is_within_workspace(outside_file, workspace)
    assert not validator.is_within_workspace(outside_dir, workspace)
    assert not validator.is_within_workspace(Path("/etc/passwd"), workspace)


def test_path_validator_resolve_safe_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sub_dir = workspace / "sub"
    sub_dir.mkdir()
    target_file = sub_dir / "file.txt"
    target_file.write_text("content")

    validator = PathValidator()

    # Safe resolutions
    resolved = validator.resolve_safe_path(Path("sub/file.txt"), workspace)
    assert resolved == target_file.resolve()

    resolved_root = validator.resolve_safe_path(Path("."), workspace)
    assert resolved_root == workspace.resolve()

    # Traversal attempts raise PathTraversalError
    with pytest.raises(PathTraversalError):
        validator.resolve_safe_path(Path("../outside.txt"), workspace)

    with pytest.raises(PathTraversalError):
        validator.resolve_safe_path(Path("/etc/passwd"), workspace)


def test_path_validator_symlink_boundaries(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_target = outside_dir / "target.txt"
    outside_target.write_text("outside data")

    inside_target = workspace / "inside.txt"
    inside_target.write_text("inside data")

    # Symlink pointing outside workspace
    escaping_symlink = workspace / "escaping_link"
    escaping_symlink.symlink_to(outside_target)

    # Symlink pointing inside workspace
    internal_symlink = workspace / "internal_link"
    internal_symlink.symlink_to(inside_target)

    validator = PathValidator()

    # Internal symlink passes
    assert validator.is_within_workspace(Path("internal_link"), workspace)
    resolved_internal = validator.resolve_safe_path(Path("internal_link"), workspace)
    assert resolved_internal == inside_target.resolve()

    # Escaping symlink fails
    assert not validator.is_within_workspace(Path("escaping_link"), workspace)
    with pytest.raises(PathTraversalError):
        validator.resolve_safe_path(Path("escaping_link"), workspace)


# ======================================================================================
# WorkspaceSandboxRunner Tests
# ======================================================================================


def test_workspace_runner_satisfies_protocol() -> None:
    runner = WorkspaceSandboxRunner()
    _check: SandboxRunnerProtocol = runner
    assert _check.level is IsolationLevel.WORKSPACE
    assert runner.level is IsolationLevel.WORKSPACE


@pytest.mark.asyncio
async def test_workspace_runner_rejects_mismatched_isolation(tmp_path: Path) -> None:
    runner = WorkspaceSandboxRunner()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Reject NoIsolation
    req_none = ExecutionRequest(
        command="python",
        cwd=workspace,
        workspace_root=workspace,
        isolation=NoIsolation(),
    )
    with pytest.raises(SandboxViolationError, match="IsolationLevel.WORKSPACE"):
        await runner.execute(req_none)

    # Reject ContainerIsolation
    req_container = ExecutionRequest(
        command="python",
        cwd=workspace,
        workspace_root=workspace,
        isolation=ContainerIsolation(image="python:3.11"),
    )
    with pytest.raises(SandboxViolationError, match="IsolationLevel.WORKSPACE"):
        await runner.execute(req_container)

    # Reject WasmIsolation
    req_wasm = ExecutionRequest(
        command="python",
        cwd=workspace,
        workspace_root=workspace,
        isolation=WasmIsolation(),
    )
    with pytest.raises(SandboxViolationError, match="IsolationLevel.WORKSPACE"):
        await runner.execute(req_wasm)


@pytest.mark.asyncio
async def test_workspace_runner_enforces_cwd_boundary(tmp_path: Path) -> None:
    runner = WorkspaceSandboxRunner()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    # Cwd escaping workspace root
    req_escape_cwd = ExecutionRequest(
        command="python",
        cwd=outside,
        workspace_root=workspace,
    )
    with pytest.raises(PathTraversalError):
        await runner.execute(req_escape_cwd)

    # Cwd does not exist on disk
    req_missing_cwd = ExecutionRequest(
        command="python",
        cwd=workspace / "non_existent_subdir",
        workspace_root=workspace,
    )
    with pytest.raises(FileNotFoundError):
        await runner.execute(req_missing_cwd)

    # Cwd is a file, not a directory
    cwd_file = workspace / "not_a_dir.txt"
    cwd_file.write_text("hello")
    req_file_cwd = ExecutionRequest(
        command="python",
        cwd=cwd_file,
        workspace_root=workspace,
    )
    with pytest.raises(FileNotFoundError, match="Working directory does not exist"):
        await runner.execute(req_file_cwd)


@pytest.mark.asyncio
async def test_workspace_runner_enforces_write_paths(tmp_path: Path) -> None:
    runner = WorkspaceSandboxRunner()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Write paths escaping workspace
    req_bad_write = ExecutionRequest(
        command="python",
        cwd=workspace,
        workspace_root=workspace,
        isolation=WorkspaceIsolation(write_paths=(Path("../outside_write"),)),
    )
    with pytest.raises(PathTraversalError):
        await runner.execute(req_bad_write)

    # Valid write paths inside workspace
    req_good_write = ExecutionRequest(
        command=sys.executable,
        args=("-c", "print('ok')"),
        cwd=workspace,
        workspace_root=workspace,
        isolation=WorkspaceIsolation(write_paths=(Path("build"), Path("dist"))),
    )
    res = await runner.execute(req_good_write)
    assert res.exit_code == 0
    assert res.stdout.strip() == "ok"


@pytest.mark.asyncio
async def test_workspace_runner_enforces_command_and_arg_boundaries(tmp_path: Path) -> None:
    runner = WorkspaceSandboxRunner()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Command with path traversal escaping workspace
    req_bad_cmd = ExecutionRequest(
        command="../escape.sh",
        cwd=workspace,
        workspace_root=workspace,
    )
    with pytest.raises(PathTraversalError):
        await runner.execute(req_bad_cmd)

    # Argument with relative traversal escaping workspace
    req_bad_arg = ExecutionRequest(
        command=sys.executable,
        args=("-f", "../outside_file.txt"),
        cwd=workspace,
        workspace_root=workspace,
    )
    with pytest.raises(PathTraversalError):
        await runner.execute(req_bad_arg)

    # Argument with absolute path escaping workspace
    req_bad_abs_arg = ExecutionRequest(
        command=sys.executable,
        args=("-f", "/etc/passwd"),
        cwd=workspace,
        workspace_root=workspace,
    )
    with pytest.raises(PathTraversalError):
        await runner.execute(req_bad_abs_arg)

    # Flag with '=' containing path traversal escaping workspace
    req_bad_flag = ExecutionRequest(
        command=sys.executable,
        args=("--output=../../secret.key",),
        cwd=workspace,
        workspace_root=workspace,
    )
    with pytest.raises(PathTraversalError):
        await runner.execute(req_bad_flag)

    # Flag with empty '=' value should pass without error
    req_empty_flag = ExecutionRequest(
        command=sys.executable,
        args=("-c", "print('flag_ok')", "--output="),
        cwd=workspace,
        workspace_root=workspace,
    )
    res_flag = await runner.execute(req_empty_flag)
    assert res_flag.exit_code == 0
    assert res_flag.stdout.strip() == "flag_ok"


@pytest.mark.asyncio
async def test_workspace_runner_environment_isolation_and_secret_filtering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = WorkspaceSandboxRunner()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Set mock host environment variables
    monkeypatch.setenv("HOST_SAFE_VAR", "safe_value")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-from-host")
    monkeypatch.setenv("GH_TOKEN", "ghp_secret_from_host")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws_secret_from_host")
    monkeypatch.setenv("UNALLOWLISTED_HOST_VAR", "should_not_leak")

    # Script prints env vars as JSON
    script = (
        "import os, json; "
        "print(json.dumps({"
        "'SAFE': os.getenv('HOST_SAFE_VAR'), "
        "'KEY': os.getenv('OPENAI_API_KEY'), "
        "'GH': os.getenv('GH_TOKEN'), "
        "'UNALLOWLISTED': os.getenv('UNALLOWLISTED_HOST_VAR'), "
        "'EXPLICIT_CUSTOM': os.getenv('EXPLICIT_VAR'), "
        "'EXPLICIT_KEY': os.getenv('EXPLICIT_API_KEY')"
        "}))"
    )

    req = ExecutionRequest(
        command=sys.executable,
        args=("-c", script),
        cwd=workspace,
        workspace_root=workspace,
        env_allowlist=("HOST_SAFE_VAR", "NON_EXISTENT_VAR"),
        env={"EXPLICIT_VAR": "custom_val", "EXPLICIT_API_KEY": "sk-explicit-key"},
    )

    res = await runner.execute(req)
    assert res.exit_code == 0

    import json

    data = json.loads(res.stdout)

    # Safe allowlisted variable is copied
    assert data["SAFE"] == "safe_value"

    # Unallowlisted ambient host variable is not copied
    assert data["UNALLOWLISTED"] is None

    # Explicit variables passed via `env` are present
    assert data["EXPLICIT_CUSTOM"] == "custom_val"
    assert data["EXPLICIT_KEY"] == "sk-explicit-key"

    # Issue #38: Credential-shaped env_allowlist entries are rejected on validation
    from pydantic import ValidationError

    with pytest.raises(
        ValidationError,
        match="Credential-shaped environment variable 'OPENAI_API_KEY' in env_allowlist is denied",
    ):
        ExecutionRequest(
            command=sys.executable,
            cwd=workspace,
            workspace_root=workspace,
            env_allowlist=("OPENAI_API_KEY",),
        )


@pytest.mark.asyncio
async def test_workspace_runner_successful_execution(tmp_path: Path) -> None:
    runner = WorkspaceSandboxRunner()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    req = ExecutionRequest(
        command=sys.executable,
        args=("-c", "import sys; print('stdout test'); sys.stderr.write('stderr test\\n')"),
        cwd=workspace,
        workspace_root=workspace,
    )

    result = await runner.execute(req)

    assert isinstance(result, ExecutionResult)
    assert result.exit_code == 0
    assert result.stdout.strip() == "stdout test"
    assert result.stderr.strip() == "stderr test"
    assert result.duration_ms > 0.0
    assert result.timed_out is False
    assert result.isolation_level is IsolationLevel.WORKSPACE

    # Check provenance
    prov = result.provenance
    assert prov is not None
    assert prov.path is ExecutionPath.PRIMARY
    assert prov.requested.provider == "sandbox.workspace"
    assert prov.served_by.provider == "sandbox.workspace"
    assert prov.degraded is False
    assert prov.attempts == ()


@pytest.mark.asyncio
async def test_workspace_runner_non_zero_exit(tmp_path: Path) -> None:
    runner = WorkspaceSandboxRunner()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    req = ExecutionRequest(
        command=sys.executable,
        args=("-c", "import sys; sys.stderr.write('fatal error\\n'); sys.exit(42)"),
        cwd=workspace,
        workspace_root=workspace,
    )

    result = await runner.execute(req)

    assert result.exit_code == 42
    assert "fatal error" in result.stderr
    assert result.timed_out is False
    assert result.provenance is not None
    assert result.provenance.served_by.provider == "sandbox.workspace"


@pytest.mark.asyncio
async def test_workspace_runner_command_not_found(tmp_path: Path) -> None:
    runner = WorkspaceSandboxRunner()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    req = ExecutionRequest(
        command="non_existent_command_executable_12345",
        cwd=workspace,
        workspace_root=workspace,
    )

    with pytest.raises(FileNotFoundError, match="non_existent_command_executable_12345"):
        await runner.execute(req)


@pytest.mark.asyncio
async def test_workspace_runner_timeout_handling(tmp_path: Path) -> None:
    runner = WorkspaceSandboxRunner()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    req = ExecutionRequest(
        command=sys.executable,
        args=("-c", "import time; time.sleep(5)"),
        cwd=workspace,
        workspace_root=workspace,
        timeout_seconds=0.2,
    )

    result = await runner.execute(req)

    assert result.timed_out is True
    assert result.duration_ms >= 150.0
    assert result.isolation_level is IsolationLevel.WORKSPACE
    assert result.provenance is not None
