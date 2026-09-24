"""Unit tests for sandboxed shell tool execution (BashRunTool) with P7 credential scrubbing."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import cast

import pytest

from uclone_x.core.provenance import ExecutionPath
from uclone_x.errors import PathTraversalError
from uclone_x.sandbox.models import IsolationLevel
from uclone_x.tools import (
    BashRunTool,
    ToolContext,
    ToolRegistry,
    create_default_tool_registry,
)


@pytest.fixture
def workspace_ctx(tmp_path: Path) -> ToolContext:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return ToolContext(
        agent_id="agent-tester",
        session_id="session-123",
        workspace_root=workspace,
    )


# ======================================================================================
# 1. Properties, Schema & Default Tool Registry Tests
# ======================================================================================


def test_bash_run_tool_properties_and_schema() -> None:
    """BashRunTool conforms to ToolProtocol properties and declares valid JSON Schema."""
    tool = BashRunTool()
    assert tool.name == "bash_run"
    assert "shell command" in tool.description.lower()
    schema = tool.parameters_schema
    assert schema["type"] == "object"
    assert "command" in schema["properties"]
    assert "cwd" in schema["properties"]
    assert "timeout_seconds" in schema["properties"]
    assert "max_output_bytes" in schema["properties"]

    custom_tool = BashRunTool(name="run_command", description="Custom runner")
    assert custom_tool.name == "run_command"
    assert custom_tool.description == "Custom runner"


def test_default_tool_registry_contains_bash_run() -> None:
    """Default tool registry includes bash_run and run_command tools."""
    registry = create_default_tool_registry()
    tools = registry.list_tools()
    tool_names = [t.name for t in tools]
    assert "bash_run" in tool_names
    assert "run_command" in tool_names

    bash_tool = registry.get("bash_run")
    assert isinstance(bash_tool, BashRunTool)

    cmd_tool = registry.get("run_command")
    assert isinstance(cmd_tool, BashRunTool)

    class_default_registry = ToolRegistry.default()
    assert class_default_registry.get("bash_run") is not None


# ======================================================================================
# 2. Execution & Exit Code Tests
# ======================================================================================


@pytest.mark.asyncio
async def test_bash_run_basic_execution(workspace_ctx: ToolContext) -> None:
    """Basic command execution captures stdout and exit code 0."""
    tool = BashRunTool()
    res = await tool.execute({"command": "echo 'hello world'"}, workspace_ctx)

    assert res.success is True
    assert res.error is None
    assert isinstance(res.output, dict)
    stdout = res.output.get("stdout")
    assert isinstance(stdout, str)
    assert stdout.strip() == "hello world"
    assert res.output.get("stderr") == ""
    assert res.output.get("exit_code") == 0
    assert res.isolation_level == IsolationLevel.WORKSPACE
    assert res.execution_time_ms > 0

    # In-band provenance verification (P6)
    assert res.provenance is not None
    assert res.provenance.path is ExecutionPath.PRIMARY
    assert res.provenance.requested.provider == "tool.bash_run"
    assert res.provenance.requested.model == "bash_run"
    assert res.provenance.served_by == res.provenance.requested
    assert res.provenance.degraded is False


@pytest.mark.asyncio
async def test_bash_run_non_zero_exit_code(workspace_ctx: ToolContext) -> None:
    """Non-zero exit code is cleanly captured with success=False and error message."""
    tool = BashRunTool()
    res = await tool.execute({"command": "exit 42"}, workspace_ctx)

    assert res.success is False
    assert isinstance(res.output, dict)
    assert res.output.get("exit_code") == 42
    assert res.output.get("stdout") == ""
    assert res.error is not None
    assert "42" in res.error


@pytest.mark.asyncio
async def test_bash_run_stderr_capture(workspace_ctx: ToolContext) -> None:
    """Stderr output is captured in both output and error message on failure."""
    tool = BashRunTool()
    res = await tool.execute(
        {
            "command": f"{sys.executable} -c \"import sys; sys.stderr.write('stderr alert\\n'); sys.exit(1)\""
        },
        workspace_ctx,
    )

    assert res.success is False
    assert isinstance(res.output, dict)
    assert res.output.get("exit_code") == 1
    stderr = res.output.get("stderr")
    assert isinstance(stderr, str)
    assert "stderr alert" in stderr
    assert res.error is not None
    assert "stderr alert" in res.error


@pytest.mark.asyncio
async def test_bash_run_missing_or_invalid_command(workspace_ctx: ToolContext) -> None:
    """Empty or missing command parameter returns ToolResult failure."""
    tool = BashRunTool()

    res_empty = await tool.execute({"command": "   "}, workspace_ctx)
    assert res_empty.success is False
    assert "required" in str(res_empty.error)

    res_missing = await tool.execute({}, workspace_ctx)
    assert res_missing.success is False
    assert "required" in str(res_missing.error)

    res_none = await tool.execute({"command": None}, workspace_ctx)
    assert res_none.success is False
    assert "required" in str(res_none.error)


# ======================================================================================
# 3. Working Directory & Path Traversal Enforcement Tests
# ======================================================================================


@pytest.mark.asyncio
async def test_bash_run_cwd_within_workspace(workspace_ctx: ToolContext) -> None:
    """Valid cwd within workspace root is honored."""
    sub_dir = workspace_ctx.require_workspace() / "subdir"
    sub_dir.mkdir()

    tool = BashRunTool()
    res = await tool.execute(
        {"command": "pwd", "cwd": "subdir"},
        workspace_ctx,
    )

    assert res.success is True
    assert isinstance(res.output, dict)
    stdout = res.output.get("stdout")
    assert isinstance(stdout, str)
    resolved_pwd = stdout.strip()
    assert str(sub_dir.resolve()) == resolved_pwd or sub_dir.name in resolved_pwd


@pytest.mark.asyncio
async def test_bash_run_cwd_directory_traversal_rejection(
    workspace_ctx: ToolContext, tmp_path: Path
) -> None:
    """Cwd attempting path traversal outside workspace root raises PathTraversalError."""
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()

    tool = BashRunTool()

    # Relative traversal escaping root
    with pytest.raises(PathTraversalError):
        await tool.execute({"command": "pwd", "cwd": "../outside_dir"}, workspace_ctx)

    # Absolute path escaping root
    with pytest.raises(PathTraversalError):
        await tool.execute({"command": "pwd", "cwd": str(outside_dir)}, workspace_ctx)


@pytest.mark.asyncio
async def test_bash_run_cwd_non_existent_directory(workspace_ctx: ToolContext) -> None:
    """Non-existent cwd within workspace raises FileNotFoundError."""
    tool = BashRunTool()
    with pytest.raises(FileNotFoundError, match="Working directory does not exist"):
        await tool.execute({"command": "pwd", "cwd": "non_existent_folder_xyz"}, workspace_ctx)


# ======================================================================================
# 4. Timeout & Process Tree Termination Tests
# ======================================================================================


@pytest.mark.asyncio
async def test_bash_run_timeout_and_process_tree_cleanup(workspace_ctx: ToolContext) -> None:
    """Commands exceeding timeout_seconds are killed and return clean timeout failure."""
    tool = BashRunTool()
    start_time = time.monotonic()

    res = await tool.execute(
        {
            "command": f'{sys.executable} -c "import time; time.sleep(10)"',
            "timeout_seconds": 0.3,
        },
        workspace_ctx,
    )
    elapsed = time.monotonic() - start_time

    assert res.success is False
    assert isinstance(res.output, dict)
    assert res.output.get("timed_out") is True
    assert res.output.get("exit_code") != 0
    assert res.error is not None
    assert "timed out after 0.3" in res.error
    assert elapsed < 3.0  # Must not hang for 10 seconds


@pytest.mark.asyncio
async def test_bash_run_process_tree_killpg(workspace_ctx: ToolContext) -> None:
    """Timeout kills child process groups, preventing orphan background processes."""
    tool = BashRunTool()
    cmd = (
        f'{sys.executable} -c "'
        "import subprocess, sys, time; "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(15)']); "
        "p.wait()"
        '"'
    )

    res = await tool.execute(
        {"command": cmd, "timeout_seconds": 0.3},
        workspace_ctx,
    )
    assert res.success is False
    assert isinstance(res.output, dict)
    assert res.output.get("timed_out") is True


# ======================================================================================
# 5. Output Buffer Truncation Tests
# ======================================================================================


@pytest.mark.asyncio
async def test_bash_run_stdout_truncation(workspace_ctx: ToolContext) -> None:
    """Stdout exceeding max_output_bytes is truncated with a clear notice."""
    tool = BashRunTool()
    cmd = f"{sys.executable} -c \"print('X' * 5000)\""

    res = await tool.execute(
        {"command": cmd, "max_output_bytes": 1000},
        workspace_ctx,
    )

    assert res.success is True
    assert isinstance(res.output, dict)
    assert res.output.get("truncated") is True
    stdout = res.output.get("stdout")
    assert isinstance(stdout, str)
    assert "[Output truncated: exceeded limit of 1000 bytes]" in stdout
    content_before_notice = stdout.split("\n[Output truncated:")[0]
    assert len(content_before_notice.encode("utf-8")) <= 1000


@pytest.mark.asyncio
async def test_bash_run_stderr_truncation(workspace_ctx: ToolContext) -> None:
    """Stderr exceeding max_output_bytes is truncated with a clear notice."""
    tool = BashRunTool()
    cmd = f"{sys.executable} -c \"import sys; sys.stderr.write('E' * 4000); sys.exit(1)\""

    res = await tool.execute(
        {"command": cmd, "max_output_bytes": 500},
        workspace_ctx,
    )

    assert res.success is False
    assert isinstance(res.output, dict)
    assert res.output.get("truncated") is True
    stderr = res.output.get("stderr")
    assert isinstance(stderr, str)
    assert "[Output truncated: exceeded limit of 500 bytes]" in stderr


@pytest.mark.asyncio
async def test_bash_run_truncation_preserves_tail(workspace_ctx: ToolContext) -> None:
    """Truncation preserves tail content, not just the head (#1207).

    A traceback's actionable line (a `SyntaxError`, a test runner's failure summary) is
    almost always at the *tail* of long output. Head-only truncation silently discarded
    it. The two tests above use a homogeneous payload (`'X' * 5000`) that can't tell
    head-only truncation apart from head+tail truncation — this uses distinct,
    non-repeating head and tail markers so it fails under head-only truncation and
    passes once both ends are preserved.

    Killed by: src/uclone_x/tools/builtin/shell.py :: raw_bytes[:head_budget] + separator + raw_bytes[len(raw_bytes) - tail_budget :]
    Becomes: raw_bytes[:max_bytes]
    """
    tool = BashRunTool()
    cmd = (
        f'{sys.executable} -c "import sys; '
        "sys.stdout.write('HEAD_MARKER_START' + 'x' * 5000 + 'TAIL_MARKER_SyntaxError_final_line')\""
    )

    res = await tool.execute(
        {"command": cmd, "max_output_bytes": 1000},
        workspace_ctx,
    )

    assert res.success is True
    assert isinstance(res.output, dict)
    assert res.output.get("truncated") is True
    stdout = res.output.get("stdout")
    assert isinstance(stdout, str)
    assert "HEAD_MARKER_START" in stdout
    assert "TAIL_MARKER_SyntaxError_final_line" in stdout
    assert "[Output truncated: exceeded limit of 1000 bytes]" in stdout


# ======================================================================================
# 6. P7 Credential Scrubbing & Explicit Environment Passing Tests
# ======================================================================================


@pytest.mark.asyncio
async def test_bash_run_credential_scrubbing(
    workspace_ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Host credentials matching secret naming patterns are scrubbed from the child environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-secret-test-openai")
    monkeypatch.setenv("GH_TOKEN", "ghp_live_secret_test_gh")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_live_secret_test_github")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws_secret_live_access_key")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
    monkeypatch.setenv("DATABASE_URL", "postgres://admin:secretpass@localhost:5432/db")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/private/tmp/com.apple.launchd.secret/Listeners")
    monkeypatch.setenv("SAFE_CUSTOM_CONFIG_VAR", "safe_public_configuration")

    tool = BashRunTool()
    cmd = f'{sys.executable} -c "import os, json; print(json.dumps(dict(os.environ)))"'

    res = await tool.execute({"command": cmd, "max_output_bytes": 500_000}, workspace_ctx)
    assert res.success is True
    assert isinstance(res.output, dict)
    stdout = res.output.get("stdout")
    assert isinstance(stdout, str)

    raw_env = json.loads(stdout)
    assert isinstance(raw_env, dict)
    child_env = cast(dict[str, str], raw_env)

    # Safe variable must be present
    assert child_env.get("SAFE_CUSTOM_CONFIG_VAR") == "safe_public_configuration"

    # All secret variables must be completely scrubbed
    assert "OPENAI_API_KEY" not in child_env
    assert "GH_TOKEN" not in child_env
    assert "GITHUB_TOKEN" not in child_env
    assert "AWS_SECRET_ACCESS_KEY" not in child_env
    assert "AWS_ACCESS_KEY_ID" not in child_env
    assert "DATABASE_URL" not in child_env
    assert "SSH_AUTH_SOCK" not in child_env


@pytest.mark.asyncio
async def test_bash_run_explicit_env_passed(workspace_ctx: ToolContext) -> None:
    """Explicit environment variables passed in params are provided to the child."""
    tool = BashRunTool()
    cmd = f"{sys.executable} -c \"import os; print(os.environ.get('EXPLICIT_TEST_VAR', ''))\""

    res = await tool.execute(
        {"command": cmd, "env": {"EXPLICIT_TEST_VAR": "explicitly_granted_value"}},
        workspace_ctx,
    )
    assert res.success is True
    assert isinstance(res.output, dict)
    stdout = res.output.get("stdout")
    assert isinstance(stdout, str)
    assert stdout.strip() == "explicitly_granted_value"


@pytest.fixture
def unrestricted_ctx(tmp_path: Path) -> ToolContext:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    from uclone_x.sandbox.models import NoIsolation

    return ToolContext(
        agent_id="agent-unrestricted",
        session_id="session-124",
        workspace_root=workspace,
        isolation=NoIsolation(),
    )


@pytest.mark.asyncio
async def test_bash_run_unrestricted_execution(
    unrestricted_ctx: ToolContext, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Unrestricted host execution allows arbitrary CWD and does not scrub environment.
    Killed by: src/uclone_x/tools/builtin/shell.py :: if context.isolation.level == "none":
    Becomes: if context.isolation.level == "workspace":
    """
    monkeypatch.setenv("SECRET_VAR", "supersecret")
    tool = BashRunTool()

    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()

    res = await tool.execute(
        {
            "command": f"{sys.executable} -c \"import os; print(os.environ.get('SECRET_VAR'))\"",
            "cwd": str(outside_dir),
        },
        unrestricted_ctx,
    )

    assert res.success is True
    out1 = res.output
    assert isinstance(out1, dict)
    assert "supersecret" in str(out1.get("stdout"))
    assert res.isolation_level == IsolationLevel.NONE


@pytest.mark.asyncio
async def test_bash_run_daemon_execution(workspace_ctx: ToolContext) -> None:
    """
    Tests daemon background execution and status checking.
    Killed by: src/uclone_x/tools/builtin/shell.py :: self._daemons[proc.pid] = proc
    Becomes: pass
    """
    tool = BashRunTool()
    cmd = f'{sys.executable} -c "import time; time.sleep(2)"'

    res = await tool.execute({"action": "run", "command": cmd, "is_daemon": True}, workspace_ctx)

    assert res.success is True
    out2 = res.output
    assert isinstance(out2, dict)
    pid = out2.get("pid")
    assert isinstance(pid, int)

    status_res = await tool.execute({"action": "status", "daemon_pid": pid}, workspace_ctx)

    assert status_res.success is True
    out3 = status_res.output
    assert isinstance(out3, dict)
    assert out3.get("status") == "running"

    kill_res = await tool.execute({"action": "kill", "daemon_pid": pid}, workspace_ctx)

    assert kill_res.success is True
    out4 = kill_res.output
    assert isinstance(out4, dict)
    assert out4.get("status") == "killed"


@pytest.mark.asyncio
async def test_bash_run_workspace_isolation_bounds_cwd_not_subshell_writes(
    workspace_ctx: ToolContext, tmp_path: Path
) -> None:
    """`BashRunTool` validates `cwd` against workspace root, but subshell commands have no OS write jail (#684).

    `cwd` escaping the workspace root is rejected with `PathTraversalError`. However, once
    inside a safe `cwd`, arbitrary shell command strings execute through `create_subprocess_shell`
    without OS filesystem virtualization. Callers requiring true write boundaries must use
    `ContainerIsolation` or disposable checkouts.

    Killed by: src/uclone_x/tools/builtin/shell.py :: target_cwd, context.require_workspace()
    Becomes: target_cwd, target_cwd
    """
    tool = BashRunTool()
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()

    # 1. Cwd outside workspace is strictly rejected (the boundary that IS enforced)
    with pytest.raises(PathTraversalError):
        await tool.execute({"command": "pwd", "cwd": str(outside_dir)}, workspace_ctx)

    # 2. Inside workspace, subshell write outside workspace succeeds (the boundary that is NOT enforced)
    target_outside_file = outside_dir / "unbounded_write.txt"
    assert not target_outside_file.exists()

    res = await tool.execute(
        {"command": f"touch '{target_outside_file}'"},
        workspace_ctx,
    )
    assert res.success is True
    assert target_outside_file.exists()
    target_outside_file.unlink()
