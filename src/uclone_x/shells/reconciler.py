"""Terraform-style declarative cluster stack reconciler (`plan`, `apply`, `status`, `down`).

Manages the 5-component UClone-X Track 1 topology incrementally:
  1. `docker_infra`    — Kafka, PostgreSQL, Valkey cluster (6 nodes), Centrifugo + schema/topics
  2. `frontend_bundle` — `frontend/` -> `src/uclone_x/ui_static` production bundle
  3. `model_gateway`   — `ucx gateway serve` (single model API, quotas, remote GPU SSH tunnel)
  4. `agent_workers`   — `ucx worker` × N stateless background workers (zero provider secrets)
  5. `web_ui_server`   — cluster-backed Web UI & Room ingress (`UCLONE_CLUSTER_MODE=docker`)

Each resource computes a deterministic SHA-256 input fingerprint and inspects live OS/HTTP
reality before choosing `+ create`, `~ update`, `-/+ replace`, or `= unchanged`. Unchanged
resources are left untouched — preserving active SSH tunnels, warm connection pools, and running
containers across iterative code reloads.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field

from uclone_x import __version__
from uclone_x.adapters.cluster.config import (
    REQUIRED_STACK_SECRETS,
    STACK_ENV_FILE_VAR,
    ClusterConfigError,
    ClusterRuntimeConfig,
    read_stack_env,
    stack_env_path,
)
from uclone_x.errors import (
    DashboardNotIdentifiedError,
    DashboardStopUnconfirmedError,
    ListeningProcessLookupError,
    UCloneXError,
)
from uclone_x.shells import daemon_supervisor, ui_process

__all__ = [
    "CLOUD_PROVIDER_SECRET_VARS",
    "DOCKER_CLUSTER_SERVICES",
    "AppliedResourceResult",
    "PlanAction",
    "ReconcileError",
    "ResourceName",
    "ResourcePlan",
    "StackApplyResult",
    "StackDownResult",
    "StackPlan",
    "StackReconciler",
    "StackSpec",
    "build_child_env",
    "combine_fingerprint",
    "compute_python_tree_digest",
    "write_deploy_state",
]

PlanAction = Literal["create", "update", "replace", "unchanged"]
ResourceName = Literal[
    "docker_infra",
    "frontend_bundle",
    "model_gateway",
    "agent_workers",
    "web_ui_server",
]

DOCKER_CLUSTER_SERVICES: Final[tuple[str, ...]] = (
    "kafka",
    "postgres",
    "valkey-1",
    "valkey-2",
    "valkey-3",
    "valkey-4",
    "valkey-5",
    "valkey-6",
    "valkey-init",
    "centrifugo",
)

#: Cloud provider variables stripped from stateless workers so every turn traverses `model_gateway`.
CLOUD_PROVIDER_SECRET_VARS: Final[tuple[str, ...]] = (
    "LLM_PROVIDER",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "VLLM_API_KEY",
)

_IGNORED_DIR_NAMES: Final[frozenset[str]] = frozenset(
    {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules"}
)
_IGNORED_SUFFIXES: Final[tuple[str, ...]] = (".pyc", ".pyo")


class ReconcileError(UCloneXError):
    """Raised when a resource in `StackReconciler.apply` fails to converge or verify."""


class StackSpec(BaseModel):
    """Desired topology specification for the declarative cluster stack."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    repo_root: Path
    compose_file: Path
    stack_env_file: Path
    state_file: Path
    ui_host: str = "127.0.0.1"
    ui_port: int = Field(default=5180, ge=1, le=65535)
    gateway_host: str = "127.0.0.1"
    gateway_port: int = Field(default=8765, ge=1, le=65535)
    remote_host: str | None = None
    workers: int = Field(default=2, ge=0, le=64)
    workspace_dir: Path | None = None
    wait_seconds: int = Field(default=60, ge=1)
    source_checkout: bool = True

    @property
    def ui_url(self) -> str:
        return f"http://{self.ui_host}:{self.ui_port}"

    @property
    def gateway_url(self) -> str:
        return f"http://{self.gateway_host}:{self.gateway_port}"


class ResourcePlan(BaseModel):
    """Planned action and drift inspection for one stack resource."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: ResourceName
    action: PlanAction
    desired_fingerprint: str
    applied_fingerprint: str | None = None
    live_healthy: bool
    reason: str
    endpoint: str = ""
    pids: tuple[int, ...] = ()
    inputs: dict[str, str] = Field(default_factory=dict[str, str])


class StackPlan(BaseModel):
    """Aggregate execution plan across all 5 stack resources."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    resources: tuple[ResourcePlan, ...]
    inspection_ms: float

    @property
    def has_changes(self) -> bool:
        return any(r.action != "unchanged" for r in self.resources)

    def counts(self) -> dict[PlanAction, int]:
        out: dict[PlanAction, int] = {"create": 0, "update": 0, "replace": 0, "unchanged": 0}
        for res in self.resources:
            out[res.action] += 1
        return out


class AppliedResourceResult(BaseModel):
    """Outcome of reconciling a single stack resource."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    name: ResourceName
    action: PlanAction
    healthy: bool
    duration_ms: float
    detail: str
    endpoint: str = ""
    pids: tuple[int, ...] = ()


class StackApplyResult(BaseModel):
    """Aggregate result of `StackReconciler.apply`."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    plan: StackPlan
    results: tuple[AppliedResourceResult, ...]
    verification_checks: tuple[dict[str, Any], ...] = ()
    all_healthy: bool
    ui_url: str
    gateway_url: str


class StackDownResult(BaseModel):
    """Aggregate result of `StackReconciler.down`."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    ui_stopped_pids: tuple[int, ...]
    worker_stopped_pids: tuple[int, ...]
    gateway_stopped_pids: tuple[int, ...]
    docker_stopped: bool
    volumes_removed: bool
    warnings: tuple[str, ...] = ()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_digest(path: Path) -> str:
    if path.is_symlink():
        return f"symlink->{os.readlink(path)}"
    if path.is_file():
        return _sha256_bytes(path.read_bytes())
    if path.exists():
        return "non-regular"
    return "absent"


def compute_python_tree_digest(
    roots: Sequence[Path],
    *,
    base_dir: Path,
    exclude_rel_prefixes: Sequence[str] = (),
) -> str:
    """Compute a deterministic SHA-256 digest over files under `roots` relative to `base_dir`."""
    entries: dict[str, str] = {}
    for target in roots:
        if not target.exists() and not target.is_symlink():
            rel = _safe_rel(target, base_dir)
            entries[rel] = "absent"
            continue
        if target.is_file() or target.is_symlink():
            rel = _safe_rel(target, base_dir)
            if not _excluded(rel, exclude_rel_prefixes):
                entries[rel] = _file_digest(target)
            continue
        for item in sorted(target.rglob("*")):
            if any(part in _IGNORED_DIR_NAMES for part in item.parts):
                continue
            if item.name.endswith(_IGNORED_SUFFIXES) or item.name == ".DS_Store":
                continue
            if item.is_dir() and not item.is_symlink():
                continue
            rel = _safe_rel(item, base_dir)
            if _excluded(rel, exclude_rel_prefixes):
                continue
            entries[rel] = _file_digest(item)
    hasher = hashlib.sha256()
    for name in sorted(entries):
        hasher.update(f"{name}\0{entries[name]}\n".encode())
    return hasher.hexdigest()


def _safe_rel(path: Path, base_dir: Path) -> str:
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _excluded(rel_posix: str, prefixes: Sequence[str]) -> bool:
    return any(rel_posix == p or rel_posix.startswith(f"{p}/") for p in prefixes)


def combine_fingerprint(inputs: Mapping[str, str]) -> str:
    """Combine sorted `key=value` input digests into a single `sha256:...` resource fingerprint."""
    hasher = hashlib.sha256()
    for key in sorted(inputs):
        hasher.update(f"{key}={inputs[key]}\n".encode())
    return f"sha256:{hasher.hexdigest()}"


_combine_fingerprint = combine_fingerprint


def _diff_changed_keys(
    prev_inputs: Mapping[str, str] | None, current_inputs: Mapping[str, str]
) -> tuple[str, ...]:
    if not prev_inputs:
        return tuple(sorted(current_inputs))
    changed = [k for k in sorted(current_inputs) if prev_inputs.get(k) != current_inputs[k]]
    for k in sorted(prev_inputs):
        if k not in current_inputs and k not in changed:
            changed.append(k)
    return tuple(changed)


def build_child_env(
    spec: StackSpec,
    *,
    base_env: Mapping[str, str] | None = None,
    strip_provider_secrets: bool = False,
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Construct an isolated child daemon environment pinning the worktree's `PYTHONPATH` (#718)."""
    env = dict(os.environ if base_env is None else base_env)
    if strip_provider_secrets:
        for secret_var in CLOUD_PROVIDER_SECRET_VARS:
            env.pop(secret_var, None)
    if spec.source_checkout:
        py_paths: list[str] = []
        src_dir = spec.repo_root / "src"
        if src_dir.is_dir():
            py_paths.append(str(src_dir.resolve()))
        story_src = spec.repo_root / "examples" / "story" / "src"
        if story_src.is_dir():
            py_paths.append(str(story_src.resolve()))
        existing_py = env.get("PYTHONPATH", "").strip()
        if existing_py:
            py_paths.extend(p for p in existing_py.split(os.pathsep) if p and p not in py_paths)
        if py_paths:
            env["PYTHONPATH"] = os.pathsep.join(py_paths)
    if extra_env:
        env.update(extra_env)
    return env


def _read_state_file(path: Path) -> dict[str, dict[str, Any]]:
    """Read `.ucx-cluster/deploy-state.json`; return empty mapping when absent or unreadable."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return {}
    try:
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            raw: object = json.loads(handle.read())
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    resources = cast(dict[str, object], raw).get("resources")
    if not isinstance(resources, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, val in cast(dict[str, object], resources).items():
        if isinstance(val, dict):
            out[key] = cast(dict[str, Any], val)
    return out


def write_deploy_state(path: Path, resources: Mapping[str, dict[str, Any]]) -> None:
    """Atomically write `.ucx-cluster/deploy-state.json` with mode `0600`."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    body = json.dumps(
        {
            "schema_version": 1,
            "updated_at": datetime.now(UTC).isoformat(),
            "resources": dict(resources),
        },
        indent=2,
        sort_keys=True,
    )
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(f"{body}\n")
    os.replace(tmp, path)


_write_state_file = write_deploy_state


def _ensure_stack_secrets(env_file: Path) -> tuple[str, ...]:
    """Ensure `.ucx-cluster/stack.env` holds every required cluster secret without rotating."""
    from uclone_x.cli.commands.cluster import ensure_stack_env

    return ensure_stack_env(env_file)


class StackReconciler:
    """Terraform-style planner and incremental applier for the 5-component cluster stack."""

    def __init__(self, spec: StackSpec) -> None:
        self.spec = spec

    @classmethod
    def from_environment(
        cls,
        *,
        ui_host: str = "127.0.0.1",
        ui_port: int = 5180,
        gateway_host: str = "127.0.0.1",
        gateway_port: int = 8765,
        remote_host: str | None = None,
        workers: int = 2,
        workspace_dir: Path | None = None,
        wait_seconds: int = 60,
    ) -> StackReconciler:
        """Build a `StackReconciler` resolving repository and compose paths from env/cwd."""
        from uclone_x.cli.main import running_from_source_checkout

        is_checkout = running_from_source_checkout()
        pkg_repo_root = Path(__file__).resolve().parents[3]
        compose_override = os.environ.get("UCX_CLUSTER_COMPOSE_FILE", "").strip()
        if compose_override:
            compose_file = Path(compose_override).expanduser().resolve()
            repo_root = pkg_repo_root if is_checkout else compose_file.parent
        else:
            cwd_compose = (Path.cwd() / "docker-compose.yml").resolve()
            if cwd_compose.is_file():
                compose_file = cwd_compose
                repo_root = (
                    Path.cwd().resolve()
                    if (Path.cwd() / "src" / "uclone_x").is_dir()
                    else pkg_repo_root
                )
            elif is_checkout and (pkg_repo_root / "docker-compose.yml").is_file():
                compose_file = (pkg_repo_root / "docker-compose.yml").resolve()
                repo_root = pkg_repo_root
            else:
                raise ClusterConfigError(
                    f"No cluster compose file at {cwd_compose}. Run from the repository root or "
                    "set UCX_CLUSTER_COMPOSE_FILE."
                )

        env_override = os.environ.get(STACK_ENV_FILE_VAR, "").strip()
        env_file = (
            Path(env_override).expanduser().resolve()
            if env_override
            else stack_env_path(compose_file.parent).resolve()
        )
        state_file = env_file.parent / "deploy-state.json"
        resolved_remote_host = remote_host.strip() if remote_host and remote_host.strip() else ""
        if not resolved_remote_host:
            resolved_remote_host = os.environ.get("UCX_REMOTE_HOST", "").strip()
        if not resolved_remote_host and env_file.is_file():
            resolved_remote_host = read_stack_env(env_file).get("UCX_REMOTE_HOST", "").strip()
        return cls(
            StackSpec(
                repo_root=repo_root,
                compose_file=compose_file,
                stack_env_file=env_file,
                state_file=state_file,
                ui_host=ui_host,
                ui_port=ui_port,
                gateway_host=gateway_host,
                gateway_port=gateway_port,
                remote_host=resolved_remote_host or None,
                workers=workers,
                workspace_dir=workspace_dir.resolve() if workspace_dir is not None else None,
                wait_seconds=wait_seconds,
                source_checkout=is_checkout,
            )
        )

    # -- Fingerprinting -----------------------------------------------------

    def _docker_inputs(self) -> dict[str, str]:
        src = self.spec.repo_root / "src" / "uclone_x"
        centrifugo_cfg = self.spec.compose_file.parent / "deploy" / "centrifugo" / "config.json"
        schema_sha = (
            compute_python_tree_digest(
                [
                    src / "adapters" / "cluster" / "postgres_pool.py",
                    src / "adapters" / "cluster" / "kafka_broker.py",
                ],
                base_dir=self.spec.repo_root,
            )
            if self.spec.source_checkout
            else __version__
        )
        return {
            "compose_sha": _file_digest(self.spec.compose_file),
            "centrifugo_cfg_sha": _file_digest(centrifugo_cfg),
            "stack_env_sha": _file_digest(self.spec.stack_env_file),
            "schema_sha": schema_sha,
        }

    def _frontend_inputs(self) -> tuple[dict[str, str], str | None]:
        """Return `(inputs_map, recorded_bundle_digest)`."""
        ui_dir = Path(__file__).resolve().parents[1] / "ui"
        static_dir = Path(__file__).resolve().parents[1] / "ui_static"
        frontend_dir = self.spec.repo_root / "frontend"
        if self.spec.source_checkout and (frontend_dir / "package.json").is_file():
            from uclone_x.ui import build_inputs

            src_digest = build_inputs.build_input_digest(frontend_dir)
            recorded = build_inputs.recorded_digest(ui_dir.parent)
            return {"frontend_inputs_sha": src_digest}, recorded
        index_digest = _file_digest(static_dir / "index.html")
        return {"wheel_version": __version__, "index_sha": index_digest}, index_digest

    def _gateway_inputs(self) -> dict[str, str]:
        src = self.spec.repo_root / "src" / "uclone_x"
        code_sha = (
            compute_python_tree_digest(
                [
                    src / "llm",
                    src / "core" / "remote_worker.py",
                    src / "shells" / "gateway_server.py",
                    src / "ui" / "usage.py",
                ],
                base_dir=self.spec.repo_root,
            )
            if self.spec.source_checkout
            else __version__
        )
        return {
            "code_sha": code_sha,
            "bind": f"{self.spec.gateway_host}:{self.spec.gateway_port}",
            "remote_host": self.spec.remote_host or "",
        }

    def _worker_inputs(self, *, stack_env_sha: str) -> dict[str, str]:
        src = self.spec.repo_root / "src" / "uclone_x"
        story_src = self.spec.repo_root / "examples" / "story" / "src"
        roots: list[Path] = [
            src / "agent",
            src / "core",
            src / "engine",
            src / "tools",
            src / "memory",
            src / "skills",
            src / "room",
            src / "knowledge",
            src / "ontology",
            src / "adapters" / "cluster",
            src / "shells" / "cluster",
            src / "cli" / "commands" / "worker.py",
            src / "llm" / "connectors" / "gateway_client.py",
        ]
        if story_src.is_dir():
            roots.append(story_src)
        code_sha = (
            compute_python_tree_digest(roots, base_dir=self.spec.repo_root)
            if self.spec.source_checkout
            else __version__
        )
        return {
            "code_sha": code_sha,
            "replicas": str(self.spec.workers),
            "gateway_url": self.spec.gateway_url,
            "workspace_dir": str(self.spec.workspace_dir) if self.spec.workspace_dir else "",
            "stack_env_sha": stack_env_sha,
        }

    def _ui_inputs(self, *, stack_env_sha: str) -> dict[str, str]:
        src = self.spec.repo_root / "src" / "uclone_x"
        story_src = self.spec.repo_root / "examples" / "story" / "src"
        roots: list[Path] = [
            src / "ui",
            src / "room",
            src / "shells" / "cluster",
            src / "llm" / "connectors" / "gateway_client.py",
        ]
        if story_src.is_dir():
            roots.append(story_src)
        # Exclude `ui_static` and `ui_build_inputs.sha256` so rebuilding `frontend_bundle`
        # serves updated static assets immediately without restarting `web_ui_server`.
        code_sha = (
            compute_python_tree_digest(
                roots,
                base_dir=self.spec.repo_root,
                exclude_rel_prefixes=(
                    "src/uclone_x/ui_static",
                    "src/uclone_x/ui_build_inputs.sha256",
                ),
            )
            if self.spec.source_checkout
            else __version__
        )
        return {
            "code_sha": code_sha,
            "bind": f"{self.spec.ui_host}:{self.spec.ui_port}",
            "gateway_url": self.spec.gateway_url,
            "workspace_dir": str(self.spec.workspace_dir) if self.spec.workspace_dir else "",
            "stack_env_sha": stack_env_sha,
        }

    def snapshot_desired_state(self) -> dict[str, dict[str, Any]]:
        """Return `{resource_name: {'fingerprint': ..., 'inputs': ...}}` for current sources."""
        d_in = self._docker_inputs()
        env_sha = (
            d_in["stack_env_sha"]
            if d_in["stack_env_sha"] != "absent"
            else "pending-stack-env-generation"
        )
        fe_in, _ = self._frontend_inputs()
        gw_in = self._gateway_inputs()
        wk_in = self._worker_inputs(stack_env_sha=env_sha)
        ui_in = self._ui_inputs(stack_env_sha=env_sha)
        return {
            "docker_infra": {"fingerprint": combine_fingerprint(d_in), "inputs": d_in},
            "frontend_bundle": {"fingerprint": combine_fingerprint(fe_in), "inputs": fe_in},
            "model_gateway": {"fingerprint": combine_fingerprint(gw_in), "inputs": gw_in},
            "agent_workers": {"fingerprint": combine_fingerprint(wk_in), "inputs": wk_in},
            "web_ui_server": {"fingerprint": combine_fingerprint(ui_in), "inputs": ui_in},
        }

    # -- Live Reality Probes ------------------------------------------------

    def _probe_docker_infra(self) -> tuple[bool, bool, str]:
        """Return `(any_present, all_healthy, detail)`."""
        if not self.spec.stack_env_file.is_file():
            return False, False, "stack secrets not initialized yet"
        env_values = read_stack_env(self.spec.stack_env_file)
        if any(not env_values.get(k) for k in REQUIRED_STACK_SECRETS):
            return False, False, "stack secrets incomplete"

        from uclone_x.shells.cluster.runtime_stack import ClusterRuntimeStack

        try:
            cfg = ClusterRuntimeConfig.from_stack_env(self.spec.stack_env_file)
            stack = ClusterRuntimeStack.create_docker_stack(config=cfg)
        except Exception as exc:
            return False, False, f"invalid cluster config: {exc}"

        async def _check() -> Any:
            try:
                return await stack.check_health()
            finally:
                await stack.close()

        try:
            report = asyncio.run(_check())
        except Exception as exc:
            return False, False, f"health check error: {exc}"

        healthy_count = sum(1 for c in report.components if c.healthy)
        total = len(report.components)
        if report.all_healthy and total > 0:
            return True, True, f"{healthy_count}/{total} cluster services healthy"
        any_up = healthy_count > 0
        unhealthy_names = [c.name for c in report.components if not c.healthy]
        return (
            any_up,
            False,
            f"unhealthy: {', '.join(unhealthy_names)}" if unhealthy_names else "cluster offline",
        )

    def _probe_frontend_bundle(
        self, desired_inputs: Mapping[str, str], recorded_digest: str | None
    ) -> tuple[bool, bool, str]:
        """Return `(exists, fresh, detail)`."""
        static_dir = Path(__file__).resolve().parents[1] / "ui_static"
        index_ok = (static_dir / "index.html").is_file()
        assets_ok = (static_dir / "assets").is_dir()
        if not (index_ok and assets_ok):
            return False, False, "ui_static bundle missing"
        if "frontend_inputs_sha" in desired_inputs:
            if recorded_digest == desired_inputs["frontend_inputs_sha"]:
                return True, True, "ui_static matches frontend sources"
            return True, False, "frontend sources changed since last ui_static build"
        return True, True, "prebuilt wheel ui_static verified"

    def _probe_model_gateway(self) -> tuple[tuple[int, ...], bool, str]:
        """Return `(alive_pids, healthy, detail)`."""
        slot = str(self.spec.gateway_port)
        try:
            rec = daemon_supervisor.read_daemon_record("gateway", slot)
        except daemon_supervisor.DaemonNotIdentifiedError as exc:
            return (), False, str(exc)
        if rec is None:
            return (), False, "not running"
        try:
            stale = daemon_supervisor.daemon_staleness(rec)
        except Exception as exc:
            return (), False, f"process identity check failed: {exc}"
        if stale is not None:
            return (), False, f"recorded pid {rec.pid} {stale}"

        try:
            resp = httpx.get(f"{self.spec.gateway_url}/v1/health", timeout=2.0)
            if resp.status_code != 200:
                return (rec.pid,), False, f"HTTP {resp.status_code} on /v1/health"
            body: object = resp.json()
        except Exception as exc:
            return (rec.pid,), False, f"health probe failed: {exc}"

        tunnel_note = ""
        if isinstance(body, dict):
            body_map = cast(dict[str, object], body)
            rgpu = body_map.get("remote_tunnel") or body_map.get("remote_gpu")
            if isinstance(rgpu, dict):
                rgpu_map = cast(dict[str, object], rgpu)
                connected = bool(rgpu_map.get("connected"))
                reconnecting = bool(rgpu_map.get("reconnecting"))
                thost = str(rgpu_map.get("host") or rgpu_map.get("desired_host") or "").strip()
                terr = str(rgpu_map.get("error") or "").strip()
                if connected:
                    label_host = thost or "remote"
                    tunnel_note = f", tunnel={label_host} preserved"
                    if self.spec.remote_host and label_host != self.spec.remote_host:
                        return (
                            (rec.pid,),
                            False,
                            f"connected tunnel={label_host} != requested {self.spec.remote_host}",
                        )
                elif self.spec.remote_host:
                    if reconnecting:
                        return (
                            (rec.pid,),
                            False,
                            f"remote_host={self.spec.remote_host} connecting...",
                        )
                    reason = (
                        f"tunnel failed: {terr}"
                        if terr and terr != "No active remote tunnel"
                        else "requested but tunnel not active"
                    )
                    return (
                        (rec.pid,),
                        False,
                        f"remote_host={self.spec.remote_host} {reason}",
                    )
                elif thost:
                    if reconnecting:
                        tunnel_note = f", tunnel={thost} connecting"
                    elif terr and terr != "No active remote tunnel":
                        tunnel_note = f", tunnel={thost} disconnected ({terr})"
                    else:
                        tunnel_note = f", tunnel={thost} disconnected"
            elif self.spec.remote_host:
                return (
                    (rec.pid,),
                    False,
                    f"remote_host={self.spec.remote_host} requested but tunnel not active",
                )
        return (rec.pid,), True, f"pid={rec.pid}{tunnel_note}"

    def _probe_agent_workers(self) -> tuple[tuple[int, ...], bool, str]:
        """Return `(alive_pids, matches_desired_replicas, detail)`."""
        records = daemon_supervisor.list_daemon_records("worker")
        alive_by_slot: dict[str, int] = {}
        for rec in records:
            with contextlib.suppress(Exception):
                if daemon_supervisor.daemon_staleness(rec) is None:
                    alive_by_slot[rec.slot] = rec.pid

        desired_slots = {str(i) for i in range(self.spec.workers)}
        alive_pids = tuple(alive_by_slot[s] for s in sorted(alive_by_slot))
        if set(alive_by_slot) == desired_slots:
            if self.spec.workers == 0:
                return (), True, "0 replicas configured"
            pids_str = ", ".join(map(str, alive_pids))
            return alive_pids, True, f"{len(alive_pids)} worker(s) running (pid={pids_str})"
        if not alive_by_slot and self.spec.workers > 0:
            return (), False, "not running"
        return (
            alive_pids,
            False,
            f"running {len(alive_by_slot)}/{self.spec.workers} worker slot(s)",
        )

    def _probe_web_ui(self) -> tuple[tuple[int, ...], bool, str]:
        """Return `(alive_pids, healthy_cluster_mode, detail)`."""
        try:
            rec = ui_process.read_dashboard_record(self.spec.ui_port)
        except DashboardNotIdentifiedError as exc:
            return (), False, str(exc)
        if rec is None:
            return (), False, "not running"
        try:
            stale = ui_process._staleness(rec)  # pyright: ignore[reportPrivateUsage]
        except Exception as exc:
            return (), False, f"process identity check failed: {exc}"
        if stale is not None:
            return (), False, f"recorded pid {rec.pid} {stale}"

        try:
            resp = httpx.get(f"{self.spec.ui_url}/api/health", timeout=2.0)
            if resp.status_code != 200:
                return (rec.pid,), False, f"HTTP {resp.status_code} on /api/health"
            raw: object = resp.json()
        except Exception as exc:
            return (rec.pid,), False, f"health probe failed: {exc}"

        data = cast(dict[str, object], raw) if isinstance(raw, dict) else {}
        mode = str(data.get("cluster_mode", "standalone"))
        if mode != "docker":
            return (rec.pid,), False, f"running in cluster_mode={mode} (expected docker)"
        return (rec.pid,), True, f"pid={rec.pid}, cluster_mode=docker"

    # -- Plan ---------------------------------------------------------------

    def plan(self) -> StackPlan:
        """Inspect desired inputs, persisted deploy state, and live health to build a plan."""
        t0 = time.perf_counter()
        saved = _read_state_file(self.spec.state_file)
        plans: list[ResourcePlan] = []

        # 1. docker_infra
        docker_in = self._docker_inputs()
        docker_fp = _combine_fingerprint(docker_in)
        docker_prev = saved.get("docker_infra")
        docker_prev_fp = (
            str(docker_prev["fingerprint"])
            if docker_prev and isinstance(docker_prev.get("fingerprint"), str)
            else None
        )
        docker_any, docker_ok, docker_detail = self._probe_docker_infra()
        if docker_ok and docker_prev_fp == docker_fp:
            docker_action: PlanAction = "unchanged"
            docker_reason = docker_detail
        elif not docker_any and docker_prev_fp is None:
            docker_action = "create"
            docker_reason = docker_detail
        else:
            docker_action = "update"
            prev_in = cast(
                dict[str, str] | None, docker_prev.get("inputs") if docker_prev else None
            )
            changed = _diff_changed_keys(prev_in, docker_in)
            docker_reason = (
                f"changed: {', '.join(changed)}" if changed and docker_ok else docker_detail
            )
        plans.append(
            ResourcePlan(
                name="docker_infra",
                action=docker_action,
                desired_fingerprint=docker_fp,
                applied_fingerprint=docker_prev_fp,
                live_healthy=docker_ok,
                reason=docker_reason,
                endpoint=str(self.spec.compose_file),
                inputs=docker_in,
            )
        )

        # 2. frontend_bundle
        fe_in, fe_recorded = self._frontend_inputs()
        fe_fp = _combine_fingerprint(fe_in)
        fe_prev = saved.get("frontend_bundle")
        fe_prev_fp = (
            str(fe_prev["fingerprint"])
            if fe_prev and isinstance(fe_prev.get("fingerprint"), str)
            else None
        )
        fe_exists, fe_fresh, fe_detail = self._probe_frontend_bundle(fe_in, fe_recorded)
        if fe_fresh:
            fe_action: PlanAction = "unchanged"
        elif not fe_exists:
            fe_action = "create"
        else:
            fe_action = "update"
        plans.append(
            ResourcePlan(
                name="frontend_bundle",
                action=fe_action,
                desired_fingerprint=fe_fp,
                applied_fingerprint=fe_prev_fp,
                live_healthy=fe_fresh,
                reason=fe_detail,
                endpoint=str(Path(__file__).resolve().parents[1] / "ui_static"),
                inputs=fe_in,
            )
        )

        # 3. model_gateway
        gw_in = self._gateway_inputs()
        gw_fp = _combine_fingerprint(gw_in)
        gw_prev = saved.get("model_gateway")
        gw_prev_fp = (
            str(gw_prev["fingerprint"])
            if gw_prev and isinstance(gw_prev.get("fingerprint"), str)
            else None
        )
        gw_pids, gw_ok, gw_detail = self._probe_model_gateway()
        if gw_ok and gw_prev_fp == gw_fp:
            gw_action: PlanAction = "unchanged"
            gw_reason = gw_detail
        elif not gw_pids:
            gw_action = "create"
            gw_reason = gw_detail
        else:
            gw_action = "replace"
            prev_in = cast(dict[str, str] | None, gw_prev.get("inputs") if gw_prev else None)
            changed = _diff_changed_keys(prev_in, gw_in)
            gw_reason = (
                f"changed: {', '.join(changed)}"
                if changed and gw_ok
                else ("unapplied fingerprint; restart required" if gw_ok else gw_detail)
            )
        plans.append(
            ResourcePlan(
                name="model_gateway",
                action=gw_action,
                desired_fingerprint=gw_fp,
                applied_fingerprint=gw_prev_fp,
                live_healthy=gw_ok,
                reason=gw_reason,
                endpoint=self.spec.gateway_url,
                pids=gw_pids,
                inputs=gw_in,
            )
        )

        # If `stack.env` does not exist yet (`docker_infra` will generate it on create), mark the
        # env hash as pending so `agent_workers` and `web_ui_server` plan create/replace cleanly.
        effective_env_sha = (
            docker_in["stack_env_sha"]
            if docker_in["stack_env_sha"] != "absent"
            else "pending-stack-env-generation"
        )

        # 4. agent_workers
        wk_in = self._worker_inputs(stack_env_sha=effective_env_sha)
        wk_fp = _combine_fingerprint(wk_in)
        wk_prev = saved.get("agent_workers")
        wk_prev_fp = (
            str(wk_prev["fingerprint"])
            if wk_prev and isinstance(wk_prev.get("fingerprint"), str)
            else None
        )
        wk_pids, wk_ok, wk_detail = self._probe_agent_workers()
        if self.spec.workers == 0 and not wk_pids:
            wk_action: PlanAction = "unchanged"
            wk_reason = wk_detail
        elif wk_ok and wk_prev_fp == wk_fp:
            wk_action = "unchanged"
            wk_reason = wk_detail
        elif not wk_pids:
            wk_action = "create"
            wk_reason = f"{self.spec.workers} worker replica(s) to start"
        else:
            wk_action = "replace"
            prev_in = cast(dict[str, str] | None, wk_prev.get("inputs") if wk_prev else None)
            changed = _diff_changed_keys(prev_in, wk_in)
            wk_reason = (
                f"changed: {', '.join(changed)}"
                if changed and wk_ok
                else ("unapplied fingerprint; restart required" if wk_ok else wk_detail)
            )
        plans.append(
            ResourcePlan(
                name="agent_workers",
                action=wk_action,
                desired_fingerprint=wk_fp,
                applied_fingerprint=wk_prev_fp,
                live_healthy=wk_ok,
                reason=wk_reason,
                endpoint=f"kafka-group:ucx-worker (x{self.spec.workers})",
                pids=wk_pids,
                inputs=wk_in,
            )
        )

        # 5. web_ui_server
        ui_in = self._ui_inputs(stack_env_sha=effective_env_sha)
        ui_fp = _combine_fingerprint(ui_in)
        ui_prev = saved.get("web_ui_server")
        ui_prev_fp = (
            str(ui_prev["fingerprint"])
            if ui_prev and isinstance(ui_prev.get("fingerprint"), str)
            else None
        )
        ui_pids, ui_ok, ui_detail = self._probe_web_ui()
        if ui_ok and ui_prev_fp == ui_fp:
            ui_action: PlanAction = "unchanged"
            ui_reason = ui_detail
        elif not ui_pids:
            ui_action = "create"
            ui_reason = ui_detail
        else:
            ui_action = "replace"
            prev_in = cast(dict[str, str] | None, ui_prev.get("inputs") if ui_prev else None)
            changed = _diff_changed_keys(prev_in, ui_in)
            ui_reason = (
                f"changed: {', '.join(changed)}"
                if changed and ui_ok
                else ("unapplied fingerprint; restart required" if ui_ok else ui_detail)
            )
        plans.append(
            ResourcePlan(
                name="web_ui_server",
                action=ui_action,
                desired_fingerprint=ui_fp,
                applied_fingerprint=ui_prev_fp,
                live_healthy=ui_ok,
                reason=ui_reason,
                endpoint=self.spec.ui_url,
                pids=ui_pids,
                inputs=ui_in,
            )
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return StackPlan(resources=tuple(plans), inspection_ms=elapsed_ms)

    # -- Apply --------------------------------------------------------------

    def apply(
        self,
        *,
        precomputed_plan: StackPlan | None = None,
        verify: bool = True,
    ) -> StackApplyResult:
        """Execute only the planned actions (`create`, `update`, `replace`) in DAG order."""
        active_plan = precomputed_plan if precomputed_plan is not None else self.plan()
        saved = _read_state_file(self.spec.state_file)
        results: list[AppliedResourceResult] = []
        plan_by_name = {r.name: r for r in active_plan.resources}

        # 1. docker_infra
        docker_res = self._apply_docker_infra(plan_by_name["docker_infra"], saved)
        results.append(docker_res)
        if not docker_res.healthy:
            _write_state_file(self.spec.state_file, saved)
            raise ReconcileError(f"docker_infra failed: {docker_res.detail}")

        # Recompute stack_env_sha now that `ensure_stack_env` has guaranteed `stack.env` exists.
        live_env_sha = _file_digest(self.spec.stack_env_file)

        # 2. frontend_bundle
        fe_res = self._apply_frontend_bundle(plan_by_name["frontend_bundle"], saved)
        results.append(fe_res)
        if not fe_res.healthy:
            _write_state_file(self.spec.state_file, saved)
            raise ReconcileError(f"frontend_bundle failed: {fe_res.detail}")

        # 3. model_gateway
        gw_res = self._apply_model_gateway(plan_by_name["model_gateway"], saved)
        results.append(gw_res)
        if not gw_res.healthy:
            _write_state_file(self.spec.state_file, saved)
            raise ReconcileError(f"model_gateway failed: {gw_res.detail}")

        # 4. agent_workers (depends on docker_infra + model_gateway)
        wk_res = self._apply_agent_workers(
            plan_by_name["agent_workers"], saved, stack_env_sha=live_env_sha
        )
        results.append(wk_res)
        if not wk_res.healthy:
            _write_state_file(self.spec.state_file, saved)
            raise ReconcileError(f"agent_workers failed: {wk_res.detail}")

        # 5. web_ui_server (depends on docker_infra + frontend_bundle + model_gateway)
        ui_res = self._apply_web_ui(
            plan_by_name["web_ui_server"], saved, stack_env_sha=live_env_sha
        )
        results.append(ui_res)
        _write_state_file(self.spec.state_file, saved)
        if not ui_res.healthy:
            raise ReconcileError(f"web_ui_server failed: {ui_res.detail}")

        checks: tuple[dict[str, Any], ...] = ()
        checks_ok = True
        if verify:
            checks, checks_ok = self._run_post_apply_verification()

        all_ok = all(r.healthy for r in results) and checks_ok
        return StackApplyResult(
            plan=active_plan,
            results=tuple(results),
            verification_checks=checks,
            all_healthy=all_ok,
            ui_url=self.spec.ui_url,
            gateway_url=self.spec.gateway_url,
        )

    def _apply_docker_infra(
        self, plan: ResourcePlan, saved: dict[str, dict[str, Any]]
    ) -> AppliedResourceResult:
        t0 = time.perf_counter()
        if plan.action == "unchanged":
            return AppliedResourceResult(
                name="docker_infra",
                action="unchanged",
                healthy=True,
                duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                detail=plan.reason,
                endpoint=plan.endpoint,
            )

        _ensure_stack_secrets(self.spec.stack_env_file)
        compose_cmd = [
            "docker",
            "compose",
            "-f",
            str(self.spec.compose_file),
            "--env-file",
            str(self.spec.stack_env_file),
            "up",
            "-d",
            "--wait",
            *DOCKER_CLUSTER_SERVICES,
        ]
        completed = subprocess.run(compose_cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            err = completed.stderr.strip() or completed.stdout.strip() or "compose exited non-zero"
            return AppliedResourceResult(
                name="docker_infra",
                action=plan.action,
                healthy=False,
                duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                detail=err,
                endpoint=plan.endpoint,
            )

        from uclone_x.shells.cluster.runtime_stack import ClusterRuntimeStack

        cfg = ClusterRuntimeConfig.from_stack_env(self.spec.stack_env_file)
        stack = ClusterRuntimeStack.create_docker_stack(config=cfg)

        async def _bootstrap_and_wait() -> tuple[bool, str]:
            deadline = time.monotonic() + max(1.0, float(self.spec.wait_seconds))
            last_err = ""
            try:
                while True:
                    try:
                        await stack.bootstrap_schema()
                        last_err = ""
                    except Exception as exc:
                        last_err = f"{type(exc).__name__}: {exc}"
                    report = await stack.check_health()
                    if (report.all_healthy and not last_err) or time.monotonic() >= deadline:
                        break
                    await asyncio.sleep(1.0)
                if last_err:
                    return False, f"schema/topic bootstrap failed: {last_err}"
                if not report.all_healthy:
                    bad = [c.name for c in report.components if not c.healthy]
                    return False, f"unhealthy after wait: {', '.join(bad)}"
                return True, f"{len(report.components)} cluster services ready & bootstrapped"
            finally:
                await stack.close()

        ok, detail = asyncio.run(_bootstrap_and_wait())
        if ok:
            fresh_in = self._docker_inputs()
            saved["docker_infra"] = {
                "fingerprint": _combine_fingerprint(fresh_in),
                "inputs": fresh_in,
            }
        return AppliedResourceResult(
            name="docker_infra",
            action=plan.action,
            healthy=ok,
            duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            detail=detail,
            endpoint=plan.endpoint,
        )

    def _apply_frontend_bundle(
        self, plan: ResourcePlan, saved: dict[str, dict[str, Any]]
    ) -> AppliedResourceResult:
        t0 = time.perf_counter()
        if plan.action == "unchanged":
            saved["frontend_bundle"] = {
                "fingerprint": plan.desired_fingerprint,
                "inputs": dict(plan.inputs),
            }
            return AppliedResourceResult(
                name="frontend_bundle",
                action="unchanged",
                healthy=True,
                duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                detail=plan.reason,
                endpoint=plan.endpoint,
            )

        frontend_dir = self.spec.repo_root / "frontend"
        if not (frontend_dir / "package.json").is_file():
            return AppliedResourceResult(
                name="frontend_bundle",
                action=plan.action,
                healthy=False,
                duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                detail=f"missing frontend/package.json at {frontend_dir}",
                endpoint=plan.endpoint,
            )

        # Pin `NODE_ENV=production` and `BROWSERSLIST=defaults` to match quality gate stage 6b.
        build_env = {
            **os.environ,
            "NODE_ENV": "production",
            "BROWSERSLIST": "defaults",
        }
        completed = subprocess.run(
            ["npm", "run", "build"],
            cwd=str(frontend_dir),
            env=build_env,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            err = completed.stderr.strip() or completed.stdout.strip() or "npm run build failed"
            return AppliedResourceResult(
                name="frontend_bundle",
                action=plan.action,
                healthy=False,
                duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                detail=err,
                endpoint=plan.endpoint,
            )

        from uclone_x.ui import build_inputs

        static_parent = Path(__file__).resolve().parents[1]
        digest = build_inputs.write_record(frontend_dir, static_parent)
        fresh_in = {"frontend_inputs_sha": digest}
        saved["frontend_bundle"] = {
            "fingerprint": _combine_fingerprint(fresh_in),
            "inputs": fresh_in,
        }
        return AppliedResourceResult(
            name="frontend_bundle",
            action=plan.action,
            healthy=True,
            duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            detail="rebuilt ui_static production bundle",
            endpoint=plan.endpoint,
        )

    def _apply_model_gateway(
        self, plan: ResourcePlan, saved: dict[str, dict[str, Any]]
    ) -> AppliedResourceResult:
        t0 = time.perf_counter()
        if plan.action == "unchanged":
            return AppliedResourceResult(
                name="model_gateway",
                action="unchanged",
                healthy=True,
                duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                detail=plan.reason,
                endpoint=plan.endpoint,
                pids=plan.pids,
            )

        slot = str(self.spec.gateway_port)
        if plan.action == "replace":
            daemon_supervisor.stop_daemon("gateway", slot, expected_port=self.spec.gateway_port)

        cmd = [
            sys.executable,
            "-m",
            "uclone_x.cli.main",
            "gateway",
            "serve",
            "--host",
            self.spec.gateway_host,
            "--port",
            slot,
        ]
        if self.spec.remote_host:
            cmd.extend(["--remote-host", self.spec.remote_host])

        child_env = build_child_env(self.spec, strip_provider_secrets=False)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=child_env,
        )

        last_detail = ""
        deadline = time.monotonic() + max(2.0, float(self.spec.wait_seconds))
        while time.monotonic() < deadline:
            pids, ok, detail = self._probe_model_gateway()
            last_detail = detail
            if ok and pids:
                saved["model_gateway"] = {
                    "fingerprint": plan.desired_fingerprint,
                    "inputs": dict(plan.inputs),
                    "pids": list(pids),
                    "port": self.spec.gateway_port,
                }
                return AppliedResourceResult(
                    name="model_gateway",
                    action=plan.action,
                    healthy=True,
                    duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    detail=f"started gateway ({detail})",
                    endpoint=self.spec.gateway_url,
                    pids=pids,
                )
            if (
                pids
                and self.spec.remote_host
                and "tunnel failed:" in detail
                and "connecting..." not in detail
            ):
                return AppliedResourceResult(
                    name="model_gateway",
                    action=plan.action,
                    healthy=False,
                    duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    detail=f"gateway running (pid={pids[0]}), {detail}",
                    endpoint=self.spec.gateway_url,
                    pids=pids,
                )
            if proc.poll() is not None:
                break
            time.sleep(0.2)

        fail_suffix = f" ({last_detail})" if last_detail else ""
        return AppliedResourceResult(
            name="model_gateway",
            action=plan.action,
            healthy=False,
            duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            detail=f"gateway daemon failed to become healthy on {self.spec.gateway_url}{fail_suffix}",
            endpoint=self.spec.gateway_url,
        )

    def _apply_agent_workers(
        self,
        plan: ResourcePlan,
        saved: dict[str, dict[str, Any]],
        *,
        stack_env_sha: str,
    ) -> AppliedResourceResult:
        t0 = time.perf_counter()
        fresh_in = self._worker_inputs(stack_env_sha=stack_env_sha)
        fresh_fp = _combine_fingerprint(fresh_in)
        if plan.action == "unchanged" and fresh_fp == plan.desired_fingerprint:
            return AppliedResourceResult(
                name="agent_workers",
                action="unchanged",
                healthy=True,
                duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                detail=plan.reason,
                endpoint=plan.endpoint,
                pids=plan.pids,
            )

        effective_action: PlanAction = "replace" if plan.action == "unchanged" else plan.action
        for existing in daemon_supervisor.list_daemon_records("worker"):
            with contextlib.suppress(daemon_supervisor.DaemonNotIdentifiedError):
                daemon_supervisor.stop_daemon("worker", existing.slot)

        if self.spec.workers == 0:
            saved["agent_workers"] = {
                "fingerprint": fresh_fp,
                "inputs": fresh_in,
                "pids": [],
                "replicas": 0,
            }
            return AppliedResourceResult(
                name="agent_workers",
                action=effective_action,
                healthy=True,
                duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                detail="scaled workers to 0",
                endpoint=plan.endpoint,
            )

        extra = {
            "UCX_GATEWAY_URL": self.spec.gateway_url,
            STACK_ENV_FILE_VAR: str(self.spec.stack_env_file),
        }
        worker_env = build_child_env(
            self.spec,
            strip_provider_secrets=True,
            extra_env=extra,
        )
        procs: list[subprocess.Popen[bytes]] = []
        for idx in range(self.spec.workers):
            slot = str(idx)
            cmd = [
                sys.executable,
                "-m",
                "uclone_x.cli.main",
                "worker",
                "--slot",
                slot,
                "--worker-id",
                f"ucx-worker-{idx}",
            ]
            if self.spec.workspace_dir is not None:
                cmd.extend(["--workspace", str(self.spec.workspace_dir)])
            procs.append(
                subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    env=worker_env,
                )
            )

        deadline = time.monotonic() + min(15.0, max(2.0, float(self.spec.wait_seconds)))
        while time.monotonic() < deadline:
            pids, ok, detail = self._probe_agent_workers()
            if ok and len(pids) == self.spec.workers:
                # Wait briefly to verify none immediately crash on Kafka/Postgres connect.
                time.sleep(0.25)
                if all(p.poll() is None for p in procs):
                    saved["agent_workers"] = {
                        "fingerprint": fresh_fp,
                        "inputs": fresh_in,
                        "pids": list(pids),
                        "replicas": self.spec.workers,
                    }
                    return AppliedResourceResult(
                        name="agent_workers",
                        action=effective_action,
                        healthy=True,
                        duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                        detail=detail,
                        endpoint=plan.endpoint,
                        pids=pids,
                    )
            if any(p.poll() is not None for p in procs):
                break
            time.sleep(0.15)

        return AppliedResourceResult(
            name="agent_workers",
            action=effective_action,
            healthy=False,
            duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            detail="one or more stateless worker daemons exited during startup",
            endpoint=plan.endpoint,
        )

    def _apply_web_ui(
        self,
        plan: ResourcePlan,
        saved: dict[str, dict[str, Any]],
        *,
        stack_env_sha: str,
    ) -> AppliedResourceResult:
        t0 = time.perf_counter()
        fresh_in = self._ui_inputs(stack_env_sha=stack_env_sha)
        fresh_fp = _combine_fingerprint(fresh_in)
        if plan.action == "unchanged" and fresh_fp == plan.desired_fingerprint:
            return AppliedResourceResult(
                name="web_ui_server",
                action="unchanged",
                healthy=True,
                duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                detail=plan.reason,
                endpoint=plan.endpoint,
                pids=plan.pids,
            )

        effective_action: PlanAction = "replace" if plan.action == "unchanged" else plan.action
        with contextlib.suppress(
            ListeningProcessLookupError,
            DashboardNotIdentifiedError,
            DashboardStopUnconfirmedError,
        ):
            ui_process.stop_ui_server(port=self.spec.ui_port)

        ui_env = build_child_env(
            self.spec,
            strip_provider_secrets=True,
            extra_env={
                "UCLONE_CLUSTER_MODE": "docker",
                STACK_ENV_FILE_VAR: str(self.spec.stack_env_file),
                "UCX_GATEWAY_URL": self.spec.gateway_url,
            },
        )
        cmd = [
            sys.executable,
            "-m",
            "uclone_x.cli.main",
            "ui",
            "--host",
            self.spec.ui_host,
            "--port",
            str(self.spec.ui_port),
        ]
        if self.spec.workspace_dir is not None:
            cmd.extend(["--cwd", str(self.spec.workspace_dir)])

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=ui_env,
        )

        deadline = time.monotonic() + max(2.0, float(self.spec.wait_seconds))
        while time.monotonic() < deadline:
            pids, ok, detail = self._probe_web_ui()
            if ok and pids:
                saved["web_ui_server"] = {
                    "fingerprint": fresh_fp,
                    "inputs": fresh_in,
                    "pids": list(pids),
                    "port": self.spec.ui_port,
                }
                return AppliedResourceResult(
                    name="web_ui_server",
                    action=effective_action,
                    healthy=True,
                    duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                    detail=f"started cluster Web UI ({detail})",
                    endpoint=self.spec.ui_url,
                    pids=pids,
                )
            if proc.poll() is not None:
                break
            time.sleep(0.25)

        return AppliedResourceResult(
            name="web_ui_server",
            action=effective_action,
            healthy=False,
            duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            detail=f"Web UI daemon failed to reach cluster_mode=docker on {self.spec.ui_url}",
            endpoint=self.spec.ui_url,
        )

    def _run_post_apply_verification(self) -> tuple[tuple[dict[str, Any], ...], bool]:
        """Verify live Gateway, Cluster E2E readiness, and Web UI endpoints."""
        checks: list[dict[str, Any]] = []

        # 1. Model Gateway `/v1/health`
        t0 = time.perf_counter()
        try:
            gw_resp = httpx.get(f"{self.spec.gateway_url}/v1/health", timeout=3.0)
            gw_ok = gw_resp.status_code == 200
            gw_detail = (
                "Model Gateway /v1/health HTTP 200" if gw_ok else f"HTTP {gw_resp.status_code}"
            )
        except Exception as exc:
            gw_ok = False
            gw_detail = str(exc)
        checks.append(
            {
                "name": "Model Gateway (/v1/health)",
                "endpoint": f"{self.spec.gateway_url}/v1/health",
                "healthy": gw_ok,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "detail": gw_detail,
            }
        )

        # 2. Web UI + Cluster Health (`/api/health`, `/api/rooms`)
        t1 = time.perf_counter()
        try:
            ui_resp = httpx.get(f"{self.spec.ui_url}/api/health", timeout=3.0)
            rooms_resp = httpx.get(f"{self.spec.ui_url}/api/rooms", timeout=3.0)
            ui_body: object = ui_resp.json() if ui_resp.status_code == 200 else {}
            mode = (
                str(cast(dict[str, object], ui_body).get("cluster_mode", ""))
                if isinstance(ui_body, dict)
                else ""
            )
            ui_ok = (
                ui_resp.status_code == 200 and rooms_resp.status_code == 200 and mode == "docker"
            )
            ui_detail = f"cluster_mode={mode}, /api/rooms HTTP {rooms_resp.status_code}"
        except Exception as exc:
            ui_ok = False
            ui_detail = str(exc)
        checks.append(
            {
                "name": "Cluster Web UI (/api/health, /api/rooms)",
                "endpoint": f"{self.spec.ui_url}/api/health",
                "healthy": ui_ok,
                "latency_ms": round((time.perf_counter() - t1) * 1000.0, 2),
                "detail": ui_detail,
            }
        )

        return tuple(checks), all(bool(c["healthy"]) for c in checks)

    # -- Down ---------------------------------------------------------------

    def down(self, *, volumes: bool = False) -> StackDownResult:
        """Stop Web UI, stateless workers, Model Gateway, and Docker cluster infrastructure."""
        warnings: list[str] = []

        # 1. Stop Web UI
        ui_pids: tuple[int, ...] = ()
        try:
            ui_out = ui_process.stop_ui_server(port=self.spec.ui_port)
            ui_pids = ui_out.stopped_pids
        except (
            ListeningProcessLookupError,
            DashboardNotIdentifiedError,
            DashboardStopUnconfirmedError,
        ) as exc:
            warnings.append(f"web_ui_server: {exc}")

        # 2. Stop all recorded workers
        worker_pids: list[int] = []
        for rec in daemon_supervisor.list_daemon_records("worker"):
            try:
                wk_out = daemon_supervisor.stop_daemon("worker", rec.slot)
                worker_pids.extend(wk_out.stopped_pids)
            except (
                daemon_supervisor.DaemonNotIdentifiedError,
                daemon_supervisor.DaemonStopUnconfirmedError,
            ) as exc:
                warnings.append(str(exc))

        # 3. Stop Model Gateway
        gw_pids: tuple[int, ...] = ()
        try:
            gw_out = daemon_supervisor.stop_daemon(
                "gateway",
                str(self.spec.gateway_port),
                expected_port=self.spec.gateway_port,
            )
            gw_pids = gw_out.stopped_pids
        except (
            daemon_supervisor.DaemonNotIdentifiedError,
            daemon_supervisor.DaemonStopUnconfirmedError,
        ) as exc:
            warnings.append(str(exc))

        # 4. Stop Docker cluster services
        docker_stopped = False
        if self.spec.stack_env_file.is_file():
            compose_args = ["down", "--volumes"] if volumes else ["stop", *DOCKER_CLUSTER_SERVICES]
            cmd = [
                "docker",
                "compose",
                "-f",
                str(self.spec.compose_file),
                "--env-file",
                str(self.spec.stack_env_file),
                *compose_args,
            ]
            completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
            docker_stopped = completed.returncode == 0
            if completed.returncode != 0:
                err = completed.stderr.strip() or completed.stdout.strip()
                if err:
                    warnings.append(f"docker_infra: {err}")

        if volumes:
            self.spec.state_file.unlink(missing_ok=True)

        return StackDownResult(
            ui_stopped_pids=ui_pids,
            worker_stopped_pids=tuple(worker_pids),
            gateway_stopped_pids=gw_pids,
            docker_stopped=docker_stopped,
            volumes_removed=volumes,
            warnings=tuple(warnings),
        )
