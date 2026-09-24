"""Subprocess script hook runner with JSON IPC, timeout protection, and failure policies."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, cast

from uclone_x.agent.hooks.models import (
    FailurePolicy,
    HookAction,
    HookContext,
    HookDecision,
)
from uclone_x.agent.hooks.protocols import BaseHook

logger = logging.getLogger(__name__)


class ScriptHook(BaseHook):
    """Executes an external executable hook script via async subprocess with JSON IPC."""

    def __init__(
        self,
        script_path: str | Path,
        name: str | None = None,
        failure_policy: FailurePolicy = FailurePolicy.FAIL_OPEN,
        timeout_seconds: float = 3.0,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        path = Path(script_path)
        hook_name = name or path.name
        super().__init__(name=hook_name, failure_policy=failure_policy)
        self._script_path = path
        self._timeout_seconds = timeout_seconds
        self._cwd = Path(cwd) if cwd is not None else None
        self._env = dict(env) if env is not None else None

    @property
    def script_path(self) -> Path:
        """Path to the executable hook script."""
        return self._script_path

    @property
    def timeout_seconds(self) -> float:
        """Execution timeout in seconds for this script hook."""
        return self._timeout_seconds

    def _resolve_command(self) -> list[str]:
        """Determine command arguments to execute the script."""
        resolved = self._script_path.resolve()
        if resolved.suffix == ".py":
            return [sys.executable, str(resolved)]
        elif resolved.suffix == ".sh":
            return ["bash", str(resolved)]
        return [str(resolved)]

    async def _run_script(self, context: HookContext) -> HookDecision:
        """Execute the external script subprocess with JSON IPC over stdin/stdout."""
        cmd = self._resolve_command()
        payload_data = context.model_dump()
        stdin_bytes = json.dumps(payload_data).encode("utf-8")

        proc_env = os.environ.copy()
        if self._env:
            proc_env.update(self._env)

        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._cwd) if self._cwd else None,
                env=proc_env,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input=stdin_bytes),
                timeout=self._timeout_seconds,
            )
        except TimeoutError:
            if proc is not None:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
            err_msg = f"Hook '{self.name}' timed out after {self._timeout_seconds}s"
            if self._failure_policy == FailurePolicy.FAIL_CLOSED:
                logger.warning("%s (fail_closed)", err_msg)
                return HookDecision(action=HookAction.BLOCK, reason=f"{err_msg} (fail_closed)")
            logger.warning("%s (ignored under fail_open)", err_msg)
            return HookDecision(action=HookAction.ALLOW, reason=f"{err_msg} (fail_open)")
        except Exception as exc:
            if proc is not None:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
            err_msg = f"Hook '{self.name}' execution error: {exc}"
            if self._failure_policy == FailurePolicy.FAIL_CLOSED:
                logger.warning("%s (fail_closed)", err_msg)
                return HookDecision(action=HookAction.BLOCK, reason=f"{err_msg} (fail_closed)")
            logger.warning("%s (ignored under fail_open)", err_msg)
            return HookDecision(action=HookAction.ALLOW, reason=f"{err_msg} (fail_open)")

        exit_code = proc.returncode if proc.returncode is not None else 0
        stdout_str = stdout_bytes.decode("utf-8", errors="replace").strip()
        stderr_str = stderr_bytes.decode("utf-8", errors="replace").strip()

        if exit_code == 0:
            if not stdout_str:
                return HookDecision(action=HookAction.ALLOW)
            try:
                parsed: object = json.loads(stdout_str)
                if not isinstance(parsed, dict):
                    if self._failure_policy == FailurePolicy.FAIL_CLOSED:
                        return HookDecision(
                            action=HookAction.BLOCK,
                            reason=f"Hook '{self.name}' returned invalid JSON (expected object)",
                        )
                    return HookDecision(action=HookAction.ALLOW)

                data_dict: dict[str, Any] = cast(dict[str, Any], parsed)
                raw_action: object = data_dict.get("action", "allow")
                action = HookAction(str(raw_action))
                reason_val: object = data_dict.get("reason")
                reason: str | None = str(reason_val) if reason_val is not None else None
                raw_mod: object = data_dict.get("modified_payload") or data_dict.get("payload")
                modified_payload: dict[str, Any] | None = None
                if isinstance(raw_mod, dict):
                    modified_payload = cast(dict[str, Any], raw_mod)

                return HookDecision(
                    action=action,
                    reason=reason,
                    modified_payload=modified_payload,
                )
            except Exception as exc:
                if self._failure_policy == FailurePolicy.FAIL_CLOSED:
                    return HookDecision(
                        action=HookAction.BLOCK,
                        reason=f"Hook '{self.name}' returned malformed JSON: {exc} (fail_closed)",
                    )
                logger.warning("Hook '%s' JSON parse error: %s (fail_open)", self.name, exc)
                return HookDecision(action=HookAction.ALLOW)

        elif exit_code == 2:
            # Exit code 2: Policy BLOCK
            reason = stderr_str if stderr_str else "Blocked by hook"
            return HookDecision(action=HookAction.BLOCK, reason=reason)

        else:
            # Non-zero exit code (other than 2)
            err_msg = stderr_str or f"Script exited with code {exit_code}"
            if self._failure_policy == FailurePolicy.FAIL_CLOSED:
                return HookDecision(
                    action=HookAction.BLOCK,
                    reason=f"Hook '{self.name}' failed with exit code {exit_code}: {err_msg} (fail_closed)",
                )
            logger.warning(
                "Hook '%s' failed with exit code %s: %s (ignored under fail_open)",
                self.name,
                exit_code,
                err_msg,
            )
            return HookDecision(
                action=HookAction.ALLOW,
                reason=f"Hook '{self.name}' exited with {exit_code} (fail_open)",
            )

    async def on_pre_turn(self, context: HookContext) -> HookDecision:
        return await self._run_script(context)

    async def on_post_turn(self, context: HookContext) -> HookDecision:
        return await self._run_script(context)

    async def on_pre_tool_use(self, context: HookContext) -> HookDecision:
        return await self._run_script(context)

    async def on_post_tool_use(self, context: HookContext) -> HookDecision:
        return await self._run_script(context)

    async def on_error(self, context: HookContext) -> HookDecision:
        return await self._run_script(context)
