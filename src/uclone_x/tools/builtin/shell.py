"""Sandboxed shell execution tool with P7 credential scrubbing and timeout management."""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from uclone_x.core.provenance import Provenance
from uclone_x.errors import PlainRefusalError
from uclone_x.sandbox.models import is_secret_env_name
from uclone_x.sandbox.path_validator import PathValidator
from uclone_x.sandbox.protocols import PathValidatorProtocol
from uclone_x.sandbox.story_jail import (
    JAIL_SETUP_REFUSAL,
    jailed_shell,
    story_library_jail,
)
from uclone_x.tools.models import ToolContext, ToolResult
from uclone_x.tools.protocols import ToolProtocol

__all__ = [
    "STORY_LIBRARY_SHELL_NOTE",
    "BashRunTool",
]

#: What macOS prints for a write the story-library jail refused, among other refusals.
_NOT_PERMITTED = "Operation not permitted"

#: Added to a failed command's error when the system refused something, so the model
#: knows one likely reason and what to use instead (#1589).
STORY_LIBRARY_SHELL_NOTE = (
    "(The shell cannot change files in the story library, 'stories/'. Stories are changed "
    "with the story tools: story_manuscript, story_outline and story_codex.)"
)


class BashRunTool:
    """Tool for executing shell commands with P7 credential scrubbing and process isolation.

    Features:
    - Enforces workspace bounds for `cwd` via `PathValidator`.
    - P7 Credential scrubbing: filters out secrets from environment variables.
    - Process group isolation: creates child in an isolated session/process group.
    - Process tree termination: kills the entire process group if execution times out.
    - Output buffer truncation: caps stdout and stderr to `max_output_bytes`.
    - In-band provenance tracking conforming to Principle 6.
    - On macOS, the command cannot write the workspace's story library
      (`sandbox.story_jail`, #1589). Elsewhere nothing stops it.
    """

    #: A shell can write any file the process can (`echo > f`, `rm f`), so
    #: `enable_write_tools: false` refuses it under both names it is registered as
    #: (`bash_run`, and the unadvertised alias `run_command`) (#1167, #1424).
    writes_files: ClassVar[bool] = True

    def __init__(
        self,
        name: str = "bash_run",
        description: str = "Execute a shell command with credential scrubbing (unless unrestricted), timeout management, and daemon support.",
        validator: PathValidatorProtocol | None = None,
        alias_of: str | None = None,
    ) -> None:
        #: The canonical tool this registration is a compatibility name for, or `None`.
        #: Read by `drop_shadowed_aliases`, which keeps an alias out of a request that
        #: already carries its canonical tool (#1424).
        self.alias_of = alias_of
        self._name = name
        self._description = description
        self._validator: PathValidatorProtocol = (
            validator if validator is not None else PathValidator()
        )
        self._daemons: dict[int, asyncio.subprocess.Process] = {}
        self._parameters_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["run", "status", "kill"],
                    "description": "The action to perform: 'run' to execute a command, 'status' to check a daemon, 'kill' to stop one.",
                    "default": "run",
                },
                "command": {
                    "type": ["string", "null"],
                    "description": "The shell command to execute. Required if action is 'run'.",
                },
                "cwd": {
                    "type": ["string", "null"],
                    "description": "Optional working directory relative to or within the workspace root.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Maximum execution time in seconds before terminating the command.",
                    "default": 30,
                },
                "max_output_bytes": {
                    "type": "integer",
                    "description": "Maximum allowed output bytes before truncation.",
                    "default": 30000,
                },
                "is_daemon": {
                    "type": "boolean",
                    "description": "If true and action is 'run', run the command in the background and return its PID immediately.",
                    "default": False,
                },
                "daemon_pid": {
                    "type": ["integer", "null"],
                    "description": "The PID of the daemon process to check or kill.",
                },
            },
        }

    @property
    def name(self) -> str:
        """Unique tool identifier."""
        return self._name

    @property
    def description(self) -> str:
        """Detailed functional description for LLM reasoning."""
        return self._description

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema defining tool input arguments."""
        return self._parameters_schema

    def _build_sanitized_environment(
        self,
        extra_env: Mapping[str, str] | None = None,
        isolation_level: str = "workspace",
    ) -> dict[str, str]:
        """Construct isolated child environment with host secrets scrubbed."""
        child_env: dict[str, str] = {}
        for key, value in os.environ.items():
            if isolation_level == "none" or not is_secret_env_name(key):
                child_env[key] = value

        if extra_env is not None:
            for key, value in extra_env.items():
                child_env[str(key)] = str(value)

        return child_env

    def _truncate_output(self, raw_bytes: bytes, max_bytes: int) -> tuple[str, bool]:
        """Truncate raw bytes to max_bytes, keeping both head and tail context.

        A traceback's actionable line (the final `SyntaxError:`, a test runner's failure
        summary) is almost always at the *tail* of long output, not the head. Keeping
        only the head silently discarded exactly the content a model needs to diagnose
        its own failure (#1207), so the budget is split between the start and the end of
        the output instead of being spent entirely on the start.
        """
        if len(raw_bytes) <= max_bytes:
            return raw_bytes.decode("utf-8", errors="replace"), False

        separator = b"\n...\n"
        budget = max(max_bytes - len(separator), 0)
        head_budget = budget // 2
        tail_budget = budget - head_budget
        truncated_bytes = (
            raw_bytes[:head_budget] + separator + raw_bytes[len(raw_bytes) - tail_budget :]
        )
        text = (
            truncated_bytes.decode("utf-8", errors="replace")
            + f"\n[Output truncated: exceeded limit of {max_bytes} bytes]"
        )
        return text, True

    async def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        """Execute the command in a sandboxed subshell."""
        start_time = time.monotonic()
        prov = Provenance.primary(provider=f"tool.{self._name}", model=self._name)

        action = params.get("action", "run")

        if action == "status":
            daemon_pid = params.get("daemon_pid")
            if not isinstance(daemon_pid, int):
                return ToolResult(
                    success=False,
                    output={"exit_code": -1},
                    error="daemon_pid required for status",
                    execution_time_ms=round((time.monotonic() - start_time) * 1000, 3),
                    isolation_level=context.isolation.level,
                    provenance=prov,
                )
            proc = self._daemons.get(daemon_pid)
            if not proc:
                return ToolResult(
                    success=False,
                    output={"exit_code": -1},
                    error=f"No daemon found with PID {daemon_pid}",
                    execution_time_ms=round((time.monotonic() - start_time) * 1000, 3),
                    isolation_level=context.isolation.level,
                    provenance=prov,
                )
            # Check if running
            if proc.returncode is None:
                return ToolResult(
                    success=True,
                    output={"pid": proc.pid, "status": "running"},
                    execution_time_ms=round((time.monotonic() - start_time) * 1000, 3),
                    isolation_level=context.isolation.level,
                    provenance=prov,
                )
            else:
                return ToolResult(
                    success=True,
                    output={"pid": proc.pid, "status": "exited", "exit_code": proc.returncode},
                    execution_time_ms=round((time.monotonic() - start_time) * 1000, 3),
                    isolation_level=context.isolation.level,
                    provenance=prov,
                )

        if action == "kill":
            daemon_pid = params.get("daemon_pid")
            if not isinstance(daemon_pid, int):
                return ToolResult(
                    success=False,
                    output={"exit_code": -1},
                    error="daemon_pid required for kill",
                    execution_time_ms=round((time.monotonic() - start_time) * 1000, 3),
                    isolation_level=context.isolation.level,
                    provenance=prov,
                )
            proc = self._daemons.get(daemon_pid)
            if not proc:
                return ToolResult(
                    success=False,
                    output={"exit_code": -1},
                    error=f"No daemon found with PID {daemon_pid}",
                    execution_time_ms=round((time.monotonic() - start_time) * 1000, 3),
                    isolation_level=context.isolation.level,
                    provenance=prov,
                )
            if proc.returncode is None:
                try:
                    if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                        try:
                            pgid = os.getpgid(proc.pid)
                            os.killpg(pgid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            proc.kill()
                    else:
                        proc.kill()
                except Exception:
                    pass
            return ToolResult(
                success=True,
                output={"pid": proc.pid, "status": "killed"},
                execution_time_ms=round((time.monotonic() - start_time) * 1000, 3),
                isolation_level=context.isolation.level,
                provenance=prov,
            )

        # 1. Validate command parameter
        raw_command = params.get("command")
        if not isinstance(raw_command, str) or not raw_command.strip():
            duration_ms = (time.monotonic() - start_time) * 1000.0
            return ToolResult(
                success=False,
                output={"stdout": "", "stderr": "", "exit_code": -1},
                error="Parameter 'command' is required and must be a non-empty string",
                execution_time_ms=round(duration_ms, 3),
                isolation_level=context.isolation.level,
                provenance=prov,
            )

        command = raw_command

        # 2. Validate working directory
        raw_cwd = params.get("cwd")
        if context.isolation.level == "none":
            if raw_cwd is not None and str(raw_cwd).strip():
                safe_cwd = Path(str(raw_cwd)).resolve()
            else:
                safe_cwd = Path(".").resolve()
        else:
            if raw_cwd is not None and str(raw_cwd).strip():
                target_cwd = Path(str(raw_cwd))
                safe_cwd = self._validator.resolve_safe_path(
                    target_cwd, context.require_workspace()
                )
            else:
                safe_cwd = self._validator.resolve_safe_path(Path("."), context.require_workspace())

        if not safe_cwd.exists() or not safe_cwd.is_dir():
            raise FileNotFoundError(f"Working directory does not exist: '{safe_cwd}'")

        # 3. Timeout & buffer limit configuration
        raw_timeout = params.get("timeout_seconds", context.timeout_seconds)
        try:
            timeout_seconds = float(raw_timeout) if raw_timeout is not None else 30.0
        except (ValueError, TypeError):
            timeout_seconds = 30.0

        raw_max_bytes = params.get("max_output_bytes", 30000)
        try:
            max_output_bytes = int(raw_max_bytes) if raw_max_bytes is not None else 30000
        except (ValueError, TypeError):
            max_output_bytes = 30000

        # 4. Scrub environment variables
        extra_env = params.get("env") if isinstance(params.get("env"), Mapping) else None
        child_env = self._build_sanitized_environment(
            extra_env=extra_env, isolation_level=context.isolation.level
        )

        is_daemon = bool(params.get("is_daemon"))

        # The story library is changed only by the story tools (#1589). Where the system
        # has a jail for it, the command runs inside one; `story_library_jail` refuses
        # rather than run without it on a system that should have one.
        try:
            jail = story_library_jail(context.workspace_root)
        except PlainRefusalError as refusal:
            return ToolResult(
                success=False,
                output={"stdout": "", "stderr": "", "exit_code": -1},
                error=str(refusal),
                execution_time_ms=round((time.monotonic() - start_time) * 1000, 3),
                isolation_level=context.isolation.level,
                provenance=prov,
            )

        # 5. Spawn subprocess with isolated process group
        preexec = getattr(os, "setsid", None)
        timed_out = False
        # Created by the jailed shell before it runs the command, so a missing file means
        # the jail never started it (`jailed_shell`).
        started_dir: Path | None = None
        started = False
        exit_code = 0
        stdout_str = ""
        stderr_str = ""
        truncated = False

        try:
            stdout_target = asyncio.subprocess.PIPE if not is_daemon else asyncio.subprocess.DEVNULL
            stderr_target = asyncio.subprocess.PIPE if not is_daemon else asyncio.subprocess.DEVNULL
            if jail:
                if not is_daemon:
                    started_dir = Path(tempfile.mkdtemp(prefix="ucx-jail-"))
                proc = await asyncio.create_subprocess_exec(
                    *jailed_shell(jail, command, started_dir / "started" if started_dir else None),
                    cwd=str(safe_cwd),
                    env=child_env,
                    stdout=stdout_target,
                    stderr=stderr_target,
                    preexec_fn=preexec,
                )
            else:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    cwd=str(safe_cwd),
                    env=child_env,
                    stdout=stdout_target,
                    stderr=stderr_target,
                    preexec_fn=preexec,
                )

            if is_daemon:
                self._daemons[proc.pid] = proc
                duration_ms = (time.monotonic() - start_time) * 1000.0
                return ToolResult(
                    success=True,
                    output={"pid": proc.pid, "message": "Daemon started in background."},
                    execution_time_ms=round(duration_ms, 3),
                    isolation_level=context.isolation.level,
                    provenance=prov,
                )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout_seconds,
                )
                exit_code = proc.returncode
            except TimeoutError:
                timed_out = True
                # Terminate entire process tree / process group
                try:
                    if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                        try:
                            pgid = os.getpgid(proc.pid)
                            os.killpg(pgid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            proc.kill()
                    else:
                        proc.kill()
                except Exception:
                    pass

                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=2.0,
                    )
                except Exception:
                    stdout_bytes, stderr_bytes = b"", b""

                exit_code = proc.returncode

            if started_dir is not None:
                started = (started_dir / "started").exists()

            # 6. Output buffer truncation
            stdout_text, out_trunc = self._truncate_output(stdout_bytes, max_output_bytes)
            stderr_text, err_trunc = self._truncate_output(stderr_bytes, max_output_bytes)
            stdout_str = stdout_text
            stderr_str = stderr_text
            truncated = out_trunc or err_trunc

        except FileNotFoundError as err:
            duration_ms = (time.monotonic() - start_time) * 1000.0
            return ToolResult(
                success=False,
                output={"stdout": "", "stderr": str(err), "exit_code": -1},
                error=f"Command not found or failed to execute: {err}",
                execution_time_ms=round(duration_ms, 3),
                isolation_level=context.isolation.level,
                provenance=prov,
            )
        except Exception as exc:
            duration_ms = (time.monotonic() - start_time) * 1000.0
            return ToolResult(
                success=False,
                output={"stdout": "", "stderr": str(exc), "exit_code": -1},
                error=f"Subprocess execution error: {exc}",
                execution_time_ms=round(duration_ms, 3),
                isolation_level=context.isolation.level,
                provenance=prov,
            )
        finally:
            if started_dir is not None:
                # Only the marker the shell may have created is in it.
                shutil.rmtree(started_dir, ignore_errors=True)

        duration_ms = (time.monotonic() - start_time) * 1000.0

        output_data: dict[str, Any] = {
            "stdout": stdout_str,
            "stderr": stderr_str,
            "exit_code": exit_code,
        }
        if truncated:
            output_data["truncated"] = True
        if timed_out:
            output_data["timed_out"] = True

        if timed_out:
            return ToolResult(
                success=False,
                output=output_data,
                error=f"Command timed out after {timeout_seconds} seconds",
                execution_time_ms=round(duration_ms, 3),
                isolation_level=context.isolation.level,
                provenance=prov,
            )

        if jail and not started:
            return ToolResult(
                success=False,
                output=output_data,
                error=JAIL_SETUP_REFUSAL,
                execution_time_ms=round(duration_ms, 3),
                isolation_level=context.isolation.level,
                provenance=prov,
            )

        if exit_code != 0:
            err_msg = f"Command failed with exit code {exit_code}"
            if stderr_str.strip():
                err_msg += f": {stderr_str.strip()}"
            if jail and _NOT_PERMITTED in stderr_str:
                err_msg += f" {STORY_LIBRARY_SHELL_NOTE}"
            return ToolResult(
                success=False,
                output=output_data,
                error=err_msg,
                execution_time_ms=round(duration_ms, 3),
                isolation_level=context.isolation.level,
                provenance=prov,
            )

        return ToolResult(
            success=True,
            output=output_data,
            error=None,
            execution_time_ms=round(duration_ms, 3),
            isolation_level=context.isolation.level,
            provenance=prov,
        )


# Static protocol conformance check
_bash_tool_conformance: ToolProtocol = BashRunTool()
