"""Standalone High-Speed Model Gateway Server (`ucx gateway serve`, #2475).

Exposes a single unified inference and streaming Data Plane (`/v1/chat/generate`,
`/v1/chat/stream`, `/v1/chat/completions`, `/v1/embeddings`) alongside the Control Plane
for connections, live model catalogs, API keys, default slots, and rolling token quotas
under both `/v1/*` and `/api/*` for Frontend compatibility.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Final, cast

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError as exc:
    pkg = "fastapi" if "fastapi" in str(exc) else "uvicorn"
    from uclone_x.errors import MissingDependencyError

    raise MissingDependencyError(
        extra="http",
        package=pkg,
        feature="Model Gateway HTTP/SSE server",
    ) from exc

from uclone_x.agent.persona_registry import get_default_persona_registry
from uclone_x.core.immutable import unwrap_immutable
from uclone_x.core.remote_worker import (
    COMFYUI_SERVICE_NAME,
    SSHTunnelManager,
    find_configured_ssh_hosts,
    is_valid_host,
    probe_remote_host,
)
from uclone_x.errors import (
    EmbeddingDimensionError,
    EmbeddingError,
    LLMModelNotConfiguredError,
    PlainRefusalError,
    UsageLimitsUnreadableError,
)
from uclone_x.llm.catalog import CatalogCache
from uclone_x.llm.connections import (
    Connection,
    ConnectionError_,
    ModelRef,
    ModelRefError,
    kind_rows,
    saved_default_models,
)
from uclone_x.llm.connectors.base import named_model
from uclone_x.llm.connectors.gateway_client import (
    TURN_ADMITTED_HEADER,
    deserialize_llm_request,
    serialize_gateway_error,
)
from uclone_x.llm.connectors.ollama_embedder import OllamaEmbedder
from uclone_x.llm.connectors.saved_choice import (
    add_connection,
    remove_connection,
    save_default_models,
    settings_data,
    settings_file,
    update_connection,
)
from uclone_x.llm.gateway import (
    Capability,
    ConnectorFactory,
    DefaultBinding,
    ModelGateway,
    connection_paid,
    unsupported_listing,
)
from uclone_x.llm.models import (
    ChatMessage,
    LLMRequest,
    MessageRole,
    ToolCallRequest,
    ToolDefinition,
)
from uclone_x.llm.protocols import (
    EmbedderProtocol,
    ImageInputProbe,
    LLMProviderProtocol,
    request_for_model,
)
from uclone_x.llm.usage.admission import leave_admitted_turn, turn_admission
from uclone_x.llm.usage.gate import UsageGate, shared_store
from uclone_x.llm.usage.limits import env_limits, save_limits
from uclone_x.llm.usage.store import default_usage_file
from uclone_x.tools.base import replace_file
from uclone_x.ui.usage import (
    NOT_AN_OBJECT_DETAIL,
    SETTINGS_UNREADABLE_DETAIL,
    SETTINGS_UNWRITABLE_DETAIL,
    _limits_from_body,  # pyright: ignore[reportPrivateUsage]
    usage_report,
)

__all__ = [
    "GatewayService",
    "REMOTE_GPU_CONNECTION_ID",
    "REMOTE_GPU_CONNECTION_LABEL",
    "REMOTE_GPU_PICTURES_ID",
    "REMOTE_GPU_PICTURES_LABEL",
    "create_gateway_app",
    "run_gateway_server",
]

logger = logging.getLogger(__name__)

_MASK_BULLETS: Final = "••••••••"

#: The connection the remote-GPU tunnel's Ollama is added as (model-gateway G1).
REMOTE_GPU_CONNECTION_LABEL: Final = "Remote GPU"
REMOTE_GPU_CONNECTION_ID: Final = "remote-gpu"
#: The connection the remote-GPU tunnel's ComfyUI is added as (model-gateway §3.5).
REMOTE_GPU_PICTURES_LABEL: Final = "Remote GPU pictures"
REMOTE_GPU_PICTURES_ID: Final = "remote-gpu-pictures"


def _str_map(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    items = cast("dict[object, object]", raw).items()
    return {k: v for k, v in items if isinstance(k, str) and isinstance(v, str)}


def _mask_key(key: str) -> str:
    clean = key.strip()
    if len(clean) <= 8:
        return _MASK_BULLETS
    return f"{clean[:3]}{_MASK_BULLETS}{clean[-4:]}"


def _optional_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


class GatewayService:
    """Core service backing the standalone Model Gateway HTTP/SSE server."""

    def __init__(
        self,
        *,
        settings_path: Path | None = None,
        usage_path: Path | None = None,
        workspace_dir: Path | None = None,
        gateway: ModelGateway | None = None,
        usage_gate: UsageGate | None = None,
        catalog: CatalogCache | None = None,
        default_binding: DefaultBinding | None = None,
        connector_factory: ConnectorFactory | None = None,
        embedder: EmbedderProtocol | None = None,
        tunnel_manager: SSHTunnelManager | None = None,
        ssh_config_path: Path | None = None,
        on_models_changed: Callable[[], None] | None = None,
    ) -> None:
        self._settings_file = settings_path if settings_path is not None else settings_file()
        self._storage_dir = self._settings_file.parent
        self._usage_file = (
            usage_path
            if usage_path is not None
            else (
                self._storage_dir / "usage.sqlite3"
                if settings_path is not None
                else default_usage_file()
            )
        )
        self._workspace_dir = workspace_dir if workspace_dir is not None else Path.cwd()
        self._usage_gate = (
            usage_gate
            if usage_gate is not None
            else UsageGate.for_storage(self._settings_file, self._usage_file)
        )
        self._gateway = (
            gateway
            if gateway is not None
            else ModelGateway(
                self._settings_file,
                usage_gate=self._usage_gate,
                catalog=catalog,
                default_binding=default_binding,
                connector_factory=connector_factory,
            )
        )
        self._embedder = embedder
        self._on_models_changed = on_models_changed
        self._ssh_config_path = ssh_config_path
        self._remote_restore_path = self._storage_dir / "remote_gpu_restore.json"
        self._remote_session_path = self._storage_dir / "remote_gpu_session.json"
        self._reconnect_lock = asyncio.Lock()
        self._reconnecting = False
        self._last_reconnect_mono: float = 0.0
        self._configured_remote_host: str | None = None
        self._drop_recovery_delays: tuple[float, ...] = (1.5, 5.0, 15.0, 30.0, 60.0)
        self._drop_recovery_task: asyncio.Task[None] | None = None
        self._tunnel_manager = tunnel_manager if tunnel_manager is not None else SSHTunnelManager()
        self._tunnel_manager.add_disconnect_listener(self._on_tunnel_disconnected)

    @property
    def gateway(self) -> ModelGateway:
        return self._gateway

    @property
    def settings_path(self) -> Path:
        return self._settings_file

    @property
    def usage_path(self) -> Path:
        return self._usage_file

    @property
    def usage_gate(self) -> UsageGate:
        return self._usage_gate

    @property
    def tunnel_manager(self) -> SSHTunnelManager:
        return self._tunnel_manager

    def _notify_models_changed(self) -> None:
        if self._on_models_changed is not None:
            self._on_models_changed()

    # -- Remote Machine Auto-Tunneling & Auto-Connection (Dell / Mac mini) --

    def set_configured_remote_host(self, host: str | None) -> None:
        """Record an explicit `--remote-host` target for this gateway lifecycle."""
        clean = _optional_text(host)
        self._configured_remote_host = clean if clean and is_valid_host(clean) else None

    def read_remote_session(self) -> dict[str, Any] | None:
        """Return validated `remote_gpu_session.json` record if present."""
        try:
            raw: object = json.loads(self._remote_session_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(raw, dict):
            return None
        record = cast("dict[str, object]", raw)
        host = record.get("host")
        if not isinstance(host, str) or not is_valid_host(host):
            return None
        return dict(record)

    def desired_remote_host(self) -> str | None:
        """Return the remote SSH host the user/stack intends to keep connected, if any."""
        rec = self.read_remote_session()
        if rec is not None:
            host_val = rec.get("host")
            if isinstance(host_val, str) and host_val.strip():
                return host_val.strip()
        if self._configured_remote_host:
            return self._configured_remote_host
        env_host = _optional_text(os.getenv("UCX_REMOTE_HOST"))
        return env_host if env_host and is_valid_host(env_host) else None

    def _save_remote_session(
        self,
        *,
        host: str,
        apply_settings: bool,
        auto_start_comfyui: bool,
        sync_llm: bool,
        ports: Mapping[str, int],
    ) -> None:
        session_record = {
            "host": host,
            "apply_settings": apply_settings,
            "auto_start_comfyui": auto_start_comfyui,
            "sync_llm": sync_llm,
            "ports": dict(ports),
        }
        self._remote_session_path.parent.mkdir(parents=True, exist_ok=True)
        replace_file(self._remote_session_path, json.dumps(session_record, indent=2).encode())

    def _schedule_background_recovery(self, delays: Sequence[float]) -> bool:
        """Schedule `_recover_dropped_tunnel` asynchronously without blocking the caller."""
        if self._tunnel_manager.is_connected or self.desired_remote_host() is None:
            return False
        if self._drop_recovery_task is not None and not self._drop_recovery_task.done():
            return False
        with contextlib.suppress(RuntimeError):
            loop = asyncio.get_running_loop()
            self._drop_recovery_task = loop.create_task(self._recover_dropped_tunnel(tuple(delays)))
            return True
        return False

    def trigger_background_reconnect(self, *, cooldown_seconds: float = 15.0) -> bool:
        """Non-blocking reconnect nudge (`0 ms` API latency) when desired remote GPU is offline."""
        if self._tunnel_manager.is_connected or self._reconnecting:
            return False
        if self.desired_remote_host() is None:
            return False
        if self._drop_recovery_task is not None and not self._drop_recovery_task.done():
            return False
        if (
            self._last_reconnect_mono > 0.0
            and (time.monotonic() - self._last_reconnect_mono) < cooldown_seconds
        ):
            return False
        return self._schedule_background_recovery((0.0, *self._drop_recovery_delays))

    def _on_tunnel_disconnected(self) -> dict[str, str]:
        """Restore tunnel connections and trigger bounded drop-recovery if session intent remains."""
        restored = self.restore_remote_settings()
        if self.read_remote_session() is not None:
            self._schedule_background_recovery(self._drop_recovery_delays)
        return restored

    async def _recover_dropped_tunnel(self, delays: Sequence[float]) -> None:
        """Attempt bounded event-driven recovery after an unexpected SSH tunnel exit."""
        for delay in delays:
            if delay > 0.0:
                await asyncio.sleep(delay)
            if self._tunnel_manager.is_connected or self.desired_remote_host() is None:
                return
            res = await self.reconnect_remote_gpu()
            if res is not None and bool(res.get("connected")):
                logger.info("Gateway recovered dropped remote-GPU SSH tunnel")
                return

    async def close(self) -> None:
        """Cancel background drop-recovery tasks and disconnect active SSH tunnels."""
        if self._drop_recovery_task is not None and not self._drop_recovery_task.done():
            self._drop_recovery_task.cancel()
            with contextlib.suppress(BaseException):
                await self._drop_recovery_task
        self._drop_recovery_task = None
        await self._tunnel_manager.disconnect()

    def read_remote_restore(self) -> tuple[dict[str, str], dict[str, str]]:
        """Return `(original, applied)` from `remote_gpu_restore.json`."""
        try:
            raw: object = json.loads(self._remote_restore_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}, {}
        if not isinstance(raw, dict):
            return {}, {}
        record = cast("dict[object, object]", raw)
        return _str_map(record.get("original")), _str_map(record.get("applied"))

    def restore_remote_settings(self) -> dict[str, str]:
        """Remove ephemeral tunnel connections (`remote-gpu`, `remote-gpu-pictures`) on disconnect."""
        _original, applied = self.read_remote_restore()
        restored: dict[str, str] = {}
        for conn_id, field in (
            (REMOTE_GPU_CONNECTION_ID, "connection"),
            (REMOTE_GPU_PICTURES_ID, "picture_connection"),
        ):
            tunnel_conn = self._gateway.connection(conn_id)
            if (
                conn_id in applied
                and tunnel_conn is not None
                and tunnel_conn.source == "settings"
                and tunnel_conn.base_url == applied[conn_id]
            ):
                remove_connection(conn_id, path=self._settings_file)
                self._notify_models_changed()
                restored[field] = conn_id
        with contextlib.suppress(FileNotFoundError):
            self._remote_restore_path.unlink()
        return restored

    async def probe_remote_gpu(self, host: str, *, timeout: float = 6.0) -> dict[str, Any]:
        """Probe an SSH remote host (e.g. `dell`, `mac-mini`) for GPU/Metal and AI services."""
        inspection = await probe_remote_host(host.strip(), timeout=timeout)
        return inspection.to_dict()

    async def connect_remote_gpu(
        self,
        host: str,
        *,
        apply_settings: bool = True,
        auto_start_comfyui: bool = True,
        sync_llm: bool = True,
        timeout: float = 15.0,
        ports: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Establish an SSH tunnel to the remote host, auto-start Ollama/ComfyUI, and register connections."""
        clean_host = host.strip()
        existing_rec = self.read_remote_session()
        preserved_ports: dict[str, int] = {}
        if existing_rec is not None and existing_rec.get("host") == clean_host:
            raw_existing_ports = existing_rec.get("ports")
            if isinstance(raw_existing_ports, dict):
                for pk, pv in cast("dict[object, object]", raw_existing_ports).items():
                    if isinstance(pk, str) and isinstance(pv, int):
                        preserved_ports[pk] = pv
        effective_ports = dict(preserved_ports)
        if ports:
            effective_ports.update(ports)

        # Persist target intent immediately so temporary offline hosts still reconnect later.
        if is_valid_host(clean_host):
            self._save_remote_session(
                host=clean_host,
                apply_settings=apply_settings,
                auto_start_comfyui=auto_start_comfyui,
                sync_llm=sync_llm,
                ports=effective_ports,
            )

        self._reconnecting = True
        try:
            tunnel_status = await self._tunnel_manager.connect(
                host=clean_host,
                preferred_local_ollama_port=effective_ports.get("ollama"),
                preferred_local_comfyui_port=effective_ports.get("comfyui"),
                auto_start_comfyui=auto_start_comfyui,
                timeout=timeout,
            )
        finally:
            self._reconnecting = False
            self._last_reconnect_mono = time.monotonic()

        if not tunnel_status.connected:
            return {
                "status": "error",
                "connected": False,
                "error": tunnel_status.error or "Failed to connect tunnel",
                "tunnel": tunnel_status.to_dict(),
                "llm_on_remote": False,
                "images_on_remote": False,
            }

        changes: dict[str, Any] = {}
        llm_skipped: str | None = None
        if apply_settings:
            original, applied = self.read_remote_restore()
            for m in tunnel_status.mappings:
                if m.service_name == "ollama":
                    if not sync_llm:
                        continue
                    address = f"http://127.0.0.1:{m.local_port}"
                    existing = self._gateway.connection(REMOTE_GPU_CONNECTION_ID)
                    if existing is None:
                        add_connection(
                            "ollama",
                            label=REMOTE_GPU_CONNECTION_LABEL,
                            base_url=address,
                            path=self._settings_file,
                        )
                    elif existing.source == "settings":
                        update_connection(
                            REMOTE_GPU_CONNECTION_ID,
                            base_url=address,
                            path=self._settings_file,
                        )
                    self._notify_models_changed()
                    applied[REMOTE_GPU_CONNECTION_ID] = address
                    original.setdefault(REMOTE_GPU_CONNECTION_ID, "")
                    changes["connection"] = REMOTE_GPU_CONNECTION_ID
                elif m.service_name == COMFYUI_SERVICE_NAME:
                    address = f"http://127.0.0.1:{m.local_port}"
                    existing = self._gateway.connection(REMOTE_GPU_PICTURES_ID)
                    if existing is None or existing.source == "settings":
                        update_connection(
                            REMOTE_GPU_PICTURES_ID,
                            kind="comfyui",
                            create=True,
                            label=REMOTE_GPU_PICTURES_LABEL,
                            base_url=address,
                            path=self._settings_file,
                        )
                    self._notify_models_changed()
                    applied[REMOTE_GPU_PICTURES_ID] = address
                    original.setdefault(REMOTE_GPU_PICTURES_ID, "")
                    changes["picture_connection"] = REMOTE_GPU_PICTURES_ID
            if original:
                self._remote_restore_path.parent.mkdir(parents=True, exist_ok=True)
                record = {"original": original, "applied": applied}
                replace_file(self._remote_restore_path, json.dumps(record, indent=2).encode())

        self._save_remote_session(
            host=clean_host,
            apply_settings=apply_settings,
            auto_start_comfyui=auto_start_comfyui,
            sync_llm=sync_llm,
            ports={m.service_name: m.local_port for m in tunnel_status.mappings},
        )

        images_on_remote = bool(
            tunnel_status.connected
            and any(m.service_name == COMFYUI_SERVICE_NAME for m in tunnel_status.mappings)
        )
        llm_on_remote = bool(
            tunnel_status.connected
            and any(m.service_name == "ollama" for m in tunnel_status.mappings)
            and changes.get("connection")
        )
        return {
            "status": "ok",
            "connected": True,
            "tunnel": tunnel_status.to_dict(),
            "applied_changes": changes,
            "llm_skipped": llm_skipped,
            "llm_on_remote": llm_on_remote,
            "images_on_remote": images_on_remote,
        }

    async def disconnect_remote_gpu(self) -> dict[str, Any]:
        """Disconnect active SSH tunnel, clear desired session, and restore connections."""
        self._configured_remote_host = None
        if self._drop_recovery_task is not None and not self._drop_recovery_task.done():
            self._drop_recovery_task.cancel()
            with contextlib.suppress(BaseException):
                await self._drop_recovery_task
        self._drop_recovery_task = None
        with contextlib.suppress(FileNotFoundError):
            self._remote_session_path.unlink()
        await self._tunnel_manager.disconnect()
        restored = dict(self.restore_remote_settings())
        return {
            "status": "ok",
            "connected": False,
            "restored_settings": restored,
            "tunnel": self._tunnel_manager.get_status().to_dict(),
        }

    def remote_gpu_status(self) -> dict[str, Any]:
        """Return live SSH tunnel status, desired target metadata, and routing flags."""
        self.trigger_background_reconnect()
        status = self._tunnel_manager.get_status()
        payload = status.to_dict()
        desired = self.desired_remote_host()
        if not status.connected and not payload.get("host") and desired:
            payload["host"] = desired
            if payload.get("error") == "No active remote tunnel":
                payload["error"] = None
        payload["desired_host"] = desired
        payload["auto_reconnect"] = desired is not None
        payload["reconnecting"] = self._reconnecting or bool(
            self._drop_recovery_task is not None and not self._drop_recovery_task.done()
        )
        if not status.connected and self.read_remote_restore()[0]:
            payload["restored_settings"] = self.restore_remote_settings()
        _, applied_changes = self.read_remote_restore()
        payload["images_on_remote"] = bool(
            status.connected
            and any(m.service_name == COMFYUI_SERVICE_NAME for m in status.mappings)
        )
        payload["llm_on_remote"] = bool(
            status.connected
            and any(m.service_name == "ollama" for m in status.mappings)
            and applied_changes.get(REMOTE_GPU_CONNECTION_ID)
        )
        return payload

    def remote_gpu_hosts(self) -> dict[str, Any]:
        """Return configured SSH hosts (`~/.ssh/config`) ordered with the last-connected host first."""
        last_host = self.desired_remote_host()
        cfg_hosts = find_configured_ssh_hosts(self._ssh_config_path)
        hosts: list[str] = []
        if last_host:
            hosts.append(last_host)
        for h in cfg_hosts:
            if h not in hosts:
                hosts.append(h)
        return {"hosts": hosts}

    async def reconnect_remote_gpu(
        self, *, explicit_host: str | None = None
    ) -> dict[str, Any] | None:
        """Auto-connect using `explicit_host`, `UCX_REMOTE_HOST`, or `remote_gpu_session.json`."""
        async with self._reconnect_lock:
            if explicit_host and explicit_host.strip():
                self.set_configured_remote_host(explicit_host)
            if self._tunnel_manager.is_connected and explicit_host is None:
                return {
                    "status": "ok",
                    "connected": True,
                    "tunnel": self._tunnel_manager.get_status().to_dict(),
                }

            target_host = _optional_text(explicit_host) or _optional_text(
                os.getenv("UCX_REMOTE_HOST")
            )
            record = self.read_remote_session()
            if target_host and is_valid_host(target_host):
                if record is None or record.get("host") != target_host:
                    try:
                        result = await self.connect_remote_gpu(target_host)
                        if result.get("connected"):
                            logger.info("Gateway auto-connected SSH tunnel to %s", target_host)
                        else:
                            logger.warning(
                                "Gateway auto-connect to %s failed: %s",
                                target_host,
                                result.get("error"),
                            )
                        return result
                    except Exception:
                        logger.warning(
                            "Gateway auto-connect to %s failed", target_host, exc_info=True
                        )
                        return None

            if record is None:
                return None
            host = str(record["host"])
            raw_ports = record.get("ports")
            ports: dict[str, int] = {}
            if isinstance(raw_ports, dict):
                for k, v in cast("dict[object, object]", raw_ports).items():
                    if isinstance(k, str) and isinstance(v, int):
                        ports[k] = v
            try:
                result = await self.connect_remote_gpu(
                    host,
                    apply_settings=record.get("apply_settings") is not False,
                    auto_start_comfyui=record.get("auto_start_comfyui") is not False,
                    sync_llm=record.get("sync_llm") is not False,
                    timeout=15.0,
                    ports=ports,
                )
            except Exception:
                logger.warning("Remote-GPU reconnect to %s failed", host, exc_info=True)
                return None
            if result["connected"]:
                logger.info("Remote-GPU tunnel to %s reconnected", host)
            else:
                logger.warning("Remote-GPU reconnect to %s failed: %s", host, result["error"])
            return result

    def _model_targets_remote_gpu(self, requested_model: str | None) -> bool:
        """Return `True` when `requested_model` (or its default slot) routes via `remote-gpu`."""
        model_key = named_model(requested_model)
        candidate: str | None = None
        if model_key is None or model_key == "deep":
            candidate = self._gateway.defaults().deep
        elif model_key == "fast":
            defaults = self._gateway.defaults()
            candidate = defaults.fast or defaults.deep
        elif "/" in model_key:
            candidate = model_key
        if not candidate:
            return False
        return candidate.startswith(f"{REMOTE_GPU_CONNECTION_ID}/")

    async def ensure_remote_gpu_for_model(
        self, requested_model: str | None, *, cooldown_seconds: float = 15.0
    ) -> None:
        """Nudge background SSH reconnect without blocking normal API requests (`0 ms` wait).

        Only when `requested_model` explicitly names `remote-gpu/...` and that connection is
        currently absent does the turn await the single-flight reconnect to avoid an immediate
        missing-connection error.
        """
        if self._tunnel_manager.is_connected:
            return
        if self.desired_remote_host() is None:
            return
        self.trigger_background_reconnect(cooldown_seconds=cooldown_seconds)
        if not self._model_targets_remote_gpu(requested_model):
            return
        if self._reconnecting:
            async with self._reconnect_lock:
                return
        if (
            self._last_reconnect_mono > 0.0
            and (time.monotonic() - self._last_reconnect_mono) < cooldown_seconds
        ):
            return
        await self.reconnect_remote_gpu()

    # -- Connections & Metadata --

    def connection_payload(self, conn: Connection) -> dict[str, Any]:
        listing = self._gateway.last_listing(conn)
        if listing is None and conn.unsupported:
            listing = unsupported_listing(conn)
        status = "unchecked" if listing is None else listing.status
        if listing is None and conn.kind in ("gemini", "openai", "anthropic") and not conn.key:
            status = "no_key"
        return {
            "id": conn.id,
            "kind": conn.kind,
            "label": conn.display_label,
            "base_url": conn.base_url,
            "key_set": bool(conn.key),
            "key_masked": _mask_key(conn.key) if conn.key else None,
            "source": conn.source,
            "env_var": conn.env_var,
            "key_env_var": conn.key_env_var,
            "paid": connection_paid(conn),
            "status": status,
            "detail": None if listing is None else listing.detail,
            "model_count": None
            if listing is None or listing.status != "connected"
            else len([e for e in listing.entries if e.chat_capable]),
        }

    def connections_payload(self) -> dict[str, Any]:
        self.trigger_background_reconnect()
        return {
            "connections": [self.connection_payload(c) for c in self._gateway.connections()],
            "kinds": kind_rows(),
        }

    def _writable_connection(self, conn_id: str) -> Connection:
        conn = self._gateway.connection(conn_id)
        if conn is None:
            raise ConnectionError_(f"There is no connection called {conn_id!r}.")
        if conn.source == "env":
            variable = conn.env_var or "an environment variable"
            raise ConnectionError_(
                f"The connection {conn.id} is set by {variable}, so it cannot be changed "
                f"here. Change or unset the variable instead."
            )
        return conn

    async def check_connection(self, conn_id: str) -> dict[str, Any]:
        conn = self._gateway.connection(conn_id)
        if conn is None:
            raise ConnectionError_(f"There is no connection called {conn_id!r}.")
        await self._gateway.listing(conn, refresh=True)
        return self.connection_payload(conn)

    async def add_connection(
        self,
        kind: str,
        *,
        label: str | None,
        base_url: str | None,
        key: str | None,
    ) -> dict[str, Any]:
        conn = add_connection(
            kind,
            label=label,
            base_url=base_url,
            key=key,
            path=self._settings_file,
        )
        self._notify_models_changed()
        return await self.check_connection(conn.id)

    async def patch_connection(self, conn_id: str, changes: Mapping[str, Any]) -> dict[str, Any]:
        self._writable_connection(conn_id)
        fields = {name: changes[name] for name in ("label", "base_url", "key") if name in changes}
        update_connection(conn_id, path=self._settings_file, **fields)
        self._notify_models_changed()
        return await self.check_connection(conn_id)

    def remove_connection(self, conn_id: str) -> None:
        self._writable_connection(conn_id)
        remove_connection(conn_id, path=self._settings_file)
        self._notify_models_changed()

    def connection_dependents(self, conn_id: str) -> dict[str, Any]:
        prefix = f"{conn_id}/"
        registry = get_default_persona_registry(self._workspace_dir)
        clones: list[dict[str, Any]] = []
        for persona in registry.list_personas():
            cfg = persona.llm_config
            slots = [
                slot
                for slot, value in (
                    ("model_name", cfg.model_name),
                    ("fast_model", cfg.fast_model),
                    ("image_model", cfg.image_model),
                )
                if value and value.startswith(prefix)
            ]
            if slots:
                clones.append(
                    {
                        "id": registry.id_of(persona.name) or persona.name,
                        "name": persona.name,
                        "slots": slots,
                    }
                )
        saved = saved_default_models(settings_data(self._settings_file))
        defaults = [
            slot
            for slot, value in (
                ("deep", saved.deep),
                ("fast", saved.fast),
                ("image", saved.image),
            )
            if value and value.startswith(prefix)
        ]
        return {"clones": clones, "defaults": defaults}

    def defaults_payload(self) -> dict[str, Any]:
        defaults = self._gateway.defaults()
        return {
            "deep": defaults.deep,
            "fast": defaults.fast,
            "image": defaults.image,
            "env_vars": dict(defaults.env_vars),
        }

    def update_defaults(self, changes: Mapping[str, str | None]) -> dict[str, Any]:
        save_default_models(changes, path=self._settings_file)
        self._notify_models_changed()
        return self.defaults_payload()

    async def models_payload(
        self, capability: Capability = "chat", *, refresh: bool = False
    ) -> dict[str, Any]:
        self.trigger_background_reconnect()
        models = await self._gateway.model_set(capability, refresh=refresh)
        return {
            "groups": [group.as_json() for group in models.groups],
            "defaults": {
                "deep": self._gateway.defaults().deep,
                "fast": self._gateway.defaults().fast,
                "image": self._gateway.defaults().image,
            },
            "recommended": self._gateway.recommended(models)
            if capability == "chat"
            else {"deep": None, "fast": None},
        }

    # -- Model & Connector Resolution for Inference --

    def resolve_connector(self, requested_model: str | None) -> tuple[LLMProviderProtocol, str]:
        """Resolve `(connector, bare_model_id)` for `requested_model`.

        * `None`, `""`, `"default"`, or `"deep"` -> resolves the default `deep` slot.
        * `"fast"` -> resolves the default `fast` slot (which falls back to `deep`).
        * `"<connection_id>/<model_id>"` -> resolves `gateway.connector_for(ref)`.
        * Bare model id -> uses `default_binding` when set, else checks connected listings,
          else resolves on the default `deep` connection.
        """
        clean = named_model(requested_model)
        if clean is None or clean == "deep":
            connector, model = self._gateway.default_deep()
            if connector is None or model is None:
                raise LLMModelNotConfiguredError("Gateway")
            return connector, model
        if clean == "fast":
            connector, model = self._gateway.default_fast()
            if connector is None or model is None:
                raise LLMModelNotConfiguredError("Gateway")
            return connector, model
        if "/" in clean:
            ref = ModelRef.parse(clean)
            return self._gateway.connector_for(ref), ref.model

        # Bare model name: check if any connection's last listing contains it, else use deep connection
        for conn in self._gateway.connections():
            listing = self._gateway.last_listing(conn)
            if listing is not None and any(e.id == clean for e in listing.entries):
                ref = ModelRef(connection_id=conn.id, model=clean)
                return self._gateway.connector_for(ref), clean

        binding = self._gateway.default_binding
        if binding is not None and binding.connector is not None:
            return binding.connector, clean

        deep_ref = self._gateway.resolve(None, "deep")
        if deep_ref is not None:
            ref = ModelRef(connection_id=deep_ref.connection_id, model=clean)
            return self._gateway.connector_for(ref), clean

        raise LLMModelNotConfiguredError("Gateway")

    async def accepts_images(self, requested_model: str | None = None) -> bool:
        try:
            connector, bare_model = self.resolve_connector(requested_model)
        except Exception:
            return False
        if isinstance(connector, ImageInputProbe):
            return await connector.accepts_images(bare_model)
        return False

    def resolve_embedder(
        self, requested_model: str | None = None, dimensions: int = 768
    ) -> EmbedderProtocol:
        if self._embedder is not None:
            return self._embedder
        clean = named_model(requested_model) or "nomic-embed-text"
        conn_id = "ollama"
        model_id = clean
        if "/" in clean:
            ref = ModelRef.parse(clean)
            conn_id = ref.connection_id
            model_id = ref.model
        conn = self._gateway.connection(conn_id)
        base_url = conn.base_url if conn is not None else None
        return OllamaEmbedder(
            model=model_id,
            dimensions=dimensions,
            base_url=base_url,
        )


def _openai_messages_to_ucx(raw_messages: object) -> tuple[ChatMessage, ...]:
    if not isinstance(raw_messages, list):
        raise PlainRefusalError("messages must be a list of chat message objects.")
    converted: list[ChatMessage] = []
    for item in cast(list[object], raw_messages):
        if not isinstance(item, Mapping):
            continue
        msg = cast(Mapping[str, Any], item)
        role_str = str(msg.get("role") or "user").lower()
        try:
            role = MessageRole(role_str)
        except ValueError:
            role = MessageRole.USER
        raw_content = msg.get("content")
        content = str(raw_content) if isinstance(raw_content, str) else None
        tool_calls: list[ToolCallRequest] = []
        raw_tcs = msg.get("tool_calls")
        if isinstance(raw_tcs, list):
            for tc_obj in cast(list[object], raw_tcs):
                if not isinstance(tc_obj, Mapping):
                    continue
                tc = cast(Mapping[str, Any], tc_obj)
                fn = tc.get("function")
                if isinstance(fn, Mapping):
                    fn_map = cast(Mapping[str, Any], fn)
                    raw_args = fn_map.get("arguments")
                    args_dict: dict[str, Any] = {}
                    if isinstance(raw_args, str) and raw_args.strip():
                        try:
                            parsed: object = json.loads(raw_args)
                            if isinstance(parsed, Mapping):
                                args_dict = dict(cast(Mapping[str, Any], parsed))
                        except ValueError:
                            args_dict = {}
                    elif isinstance(raw_args, Mapping):
                        args_dict = dict(cast(Mapping[str, Any], raw_args))
                    tool_calls.append(
                        ToolCallRequest(
                            id=str(tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"),
                            name=str(fn_map.get("name") or "tool"),
                            arguments=args_dict,
                        )
                    )
        converted.append(
            ChatMessage(
                role=role,
                content=content,
                name=_optional_text(msg.get("name")),
                tool_call_id=_optional_text(msg.get("tool_call_id")),
                tool_calls=tuple(tool_calls),
            )
        )
    return tuple(converted)


def _openai_tools_to_ucx(raw_tools: object) -> tuple[ToolDefinition, ...]:
    if not isinstance(raw_tools, list):
        return ()
    tools: list[ToolDefinition] = []
    for item in cast(list[object], raw_tools):
        if not isinstance(item, Mapping):
            continue
        entry = cast(Mapping[str, Any], item)
        fn = entry.get("function") if entry.get("type") == "function" else entry
        if not isinstance(fn, Mapping):
            continue
        fn_map = cast(Mapping[str, Any], fn)
        name = _optional_text(fn_map.get("name"))
        if not name:
            continue
        params = fn_map.get("parameters")
        tools.append(
            ToolDefinition(
                name=name,
                description=str(fn_map.get("description") or ""),
                parameters=dict(cast(Mapping[str, Any], params))
                if isinstance(params, Mapping)
                else {},
            )
        )
    return tuple(tools)


def create_gateway_app(
    *,
    service: GatewayService | None = None,
    settings_path: Path | None = None,
    usage_path: Path | None = None,
    workspace_dir: Path | None = None,
    api_key: str | None = None,
    remote_host: str | None = None,
    auto_reconnect_remote: bool = True,
) -> FastAPI:
    """Create the standalone FastAPI application for the Model Gateway server."""
    svc = (
        service
        if service is not None
        else GatewayService(
            settings_path=settings_path,
            usage_path=usage_path,
            workspace_dir=workspace_dir,
        )
    )
    expected_key = api_key.strip() if api_key and api_key.strip() else None

    @asynccontextmanager
    async def _gateway_lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        if remote_host and remote_host.strip():
            svc.set_configured_remote_host(remote_host.strip())
        if svc.tunnel_manager.get_status().connected is False and svc.read_remote_restore()[0]:
            with contextlib.suppress(Exception):
                restored = svc.restore_remote_settings()
                logger.info("Gateway restored stale remote-GPU tunnel settings: %s", restored)
        reconnect_task: asyncio.Task[Any] | None = None
        if auto_reconnect_remote:
            reconnect_task = asyncio.create_task(
                svc.reconnect_remote_gpu(explicit_host=remote_host)
            )
        try:
            yield
        finally:
            if reconnect_task is not None and not reconnect_task.done():
                reconnect_task.cancel()
                with contextlib.suppress(BaseException):
                    await reconnect_task
            await svc.close()

    app = FastAPI(
        title="UClone-X Model Gateway",
        version="1.0.0",
        description="Standalone High-Speed Model Gateway Server for UClone-X Agents & Frontend",
        lifespan=_gateway_lifespan,
    )
    app.state.gateway_service = svc

    def _authenticate(request: Request) -> None:
        if expected_key is None:
            return
        auth_header = request.headers.get("Authorization") or ""
        token = ""
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
        elif "X-UCX-Gateway-Key" in request.headers:
            token = request.headers["X-UCX-Gateway-Key"].strip()
        if not token or not hmac.compare_digest(token, expected_key):
            raise HTTPException(status_code=401, detail="Invalid or missing gateway API key.")

    def _connection_error_response(exc: Exception, status_code: int = 400) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=status_code)

    # -- Health --

    @app.get("/health")
    @app.get("/v1/health")
    async def health_check() -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        conns = svc.gateway.connections()
        return {
            "status": "ok",
            "connections_count": len(conns),
            "defaults": svc.defaults_payload(),
            "remote_tunnel": svc.remote_gpu_status(),
        }

    # -- Remote Machine SSH Tunneling & Auto-Connection (Dell / Mac mini) --

    @app.get("/v1/remote-gpu/hosts")
    @app.get("/api/settings/remote-gpu/hosts")
    async def get_remote_gpu_hosts_route(request: Request) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        _authenticate(request)
        return svc.remote_gpu_hosts()

    @app.get("/v1/remote-gpu/status")
    @app.get("/api/settings/remote-gpu/status")
    async def get_remote_gpu_status_route(request: Request) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        _authenticate(request)
        return svc.remote_gpu_status()

    @app.post("/v1/remote-gpu/probe")
    @app.post("/api/settings/remote-gpu/probe")
    async def probe_remote_gpu_route(  # pyright: ignore[reportUnusedFunction]
        request: Request, req: dict[str, Any]
    ) -> dict[str, Any]:
        _authenticate(request)
        host = str(req.get("host") or "").strip()
        timeout = float(req.get("timeout", 6.0))
        if not host:
            raise HTTPException(status_code=400, detail="Field 'host' is required")
        return await svc.probe_remote_gpu(host, timeout=timeout)

    @app.post("/v1/remote-gpu/connect")
    @app.post("/api/settings/remote-gpu/connect")
    async def connect_remote_gpu_route(  # pyright: ignore[reportUnusedFunction]
        request: Request, req: dict[str, Any]
    ) -> dict[str, Any]:
        _authenticate(request)
        host = str(req.get("host") or "").strip()
        if not host:
            raise HTTPException(status_code=400, detail="Field 'host' is required")
        apply_settings = bool(req.get("apply_settings", True))
        auto_start_comfyui = bool(req.get("auto_start_comfyui", True))
        sync_llm = req.get("sync_llm") is not False
        timeout = float(req.get("timeout", 15.0))
        return await svc.connect_remote_gpu(
            host,
            apply_settings=apply_settings,
            auto_start_comfyui=auto_start_comfyui,
            sync_llm=sync_llm,
            timeout=timeout,
        )

    @app.post("/v1/remote-gpu/disconnect")
    @app.post("/api/settings/remote-gpu/disconnect")
    async def disconnect_remote_gpu_route(request: Request) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        _authenticate(request)
        return await svc.disconnect_remote_gpu()

    # -- Connections & API Key Management (/v1/* and /api/*) --

    @app.get("/v1/connections")
    @app.get("/api/connections")
    async def list_connections(request: Request) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        _authenticate(request)
        return svc.connections_payload()

    @app.post("/v1/connections")
    @app.post("/api/connections")
    async def add_connection_route(request: Request, req: dict[str, Any]) -> Response:  # pyright: ignore[reportUnusedFunction]
        _authenticate(request)
        try:
            added = await svc.add_connection(
                str(req.get("kind") or ""),
                label=_optional_text(req.get("label")),
                base_url=_optional_text(req.get("base_url")),
                key=_optional_text(req.get("key")),
            )
        except (ConnectionError_, ValueError, OSError) as exc:
            return _connection_error_response(exc)
        return JSONResponse(added)

    @app.patch("/v1/connections/{conn_id}")
    @app.patch("/api/connections/{conn_id}")
    async def patch_connection_route(  # pyright: ignore[reportUnusedFunction]
        request: Request, conn_id: str, req: dict[str, Any]
    ) -> Response:
        _authenticate(request)
        try:
            changed = await svc.patch_connection(conn_id, req)
        except (ConnectionError_, ValueError, OSError) as exc:
            return _connection_error_response(exc)
        return JSONResponse(changed)

    @app.post("/v1/connections/{conn_id}/check")
    @app.post("/api/connections/{conn_id}/check")
    async def check_connection_route(  # pyright: ignore[reportUnusedFunction]
        request: Request, conn_id: str
    ) -> Response:
        _authenticate(request)
        try:
            checked = await svc.check_connection(conn_id)
        except ConnectionError_ as exc:
            return _connection_error_response(exc, 404)
        return JSONResponse(checked)

    @app.get("/v1/connections/{conn_id}/dependents")
    @app.get("/api/connections/{conn_id}/dependents")
    async def connection_dependents_route(  # pyright: ignore[reportUnusedFunction]
        request: Request, conn_id: str
    ) -> dict[str, Any]:
        _authenticate(request)
        return svc.connection_dependents(conn_id)

    @app.delete("/v1/connections/{conn_id}")
    @app.delete("/api/connections/{conn_id}")
    async def remove_connection_route(  # pyright: ignore[reportUnusedFunction]
        request: Request, conn_id: str
    ) -> Response:
        _authenticate(request)
        try:
            svc.remove_connection(conn_id)
        except (ConnectionError_, ValueError, OSError) as exc:
            return _connection_error_response(exc)
        return JSONResponse({"removed": conn_id})

    # -- Live Model Set & Defaults (/v1/* and /api/*) --

    @app.get("/v1/models")
    @app.get("/api/models")
    async def get_models_route(  # pyright: ignore[reportUnusedFunction]
        request: Request, capability: str = "chat", refresh: bool = False
    ) -> Response:
        _authenticate(request)
        wanted: Capability
        if capability == "chat":
            wanted = "chat"
        elif capability == "image":
            wanted = "image"
        else:
            return JSONResponse({"detail": "Choose chat or image models."}, status_code=400)
        payload = await svc.models_payload(wanted, refresh=refresh)
        return JSONResponse(payload)

    @app.get("/v1/models/accepts-images")
    async def get_model_accepts_images(  # pyright: ignore[reportUnusedFunction]
        request: Request, model: str | None = None
    ) -> dict[str, Any]:
        _authenticate(request)
        supported = await svc.accepts_images(model)
        return {"model": model or "deep", "accepts_images": supported}

    @app.get("/v1/defaults")
    @app.get("/api/defaults")
    async def get_defaults_route(request: Request) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        _authenticate(request)
        return svc.defaults_payload()

    @app.put("/v1/defaults")
    @app.put("/api/defaults")
    async def put_defaults_route(  # pyright: ignore[reportUnusedFunction]
        request: Request, req: dict[str, str | None]
    ) -> Response:
        _authenticate(request)
        try:
            updated = svc.update_defaults(req)
        except (ModelRefError, ValueError, OSError) as exc:
            return _connection_error_response(exc)
        return JSONResponse(updated)

    # -- Quota & Usage Measurement (/v1/usage and /api/usage) --

    def _usage_report() -> dict[str, Any]:
        try:
            return usage_report(svc.settings_path, shared_store(svc.usage_path))
        except UsageLimitsUnreadableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/usage")
    @app.get("/api/usage")
    async def get_usage_route(request: Request) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        _authenticate(request)
        return _usage_report()

    @app.put("/v1/usage/limits")
    @app.put("/api/usage/limits")
    async def put_usage_limits_route(request: Request) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        _authenticate(request)
        try:
            body: object = await request.json()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=NOT_AN_OBJECT_DETAIL) from exc
        limits = _limits_from_body(body)
        try:
            env_limits()
        except UsageLimitsUnreadableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        try:
            save_limits(limits, path=svc.settings_path)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=SETTINGS_UNREADABLE_DETAIL) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail=SETTINGS_UNWRITABLE_DETAIL) from exc
        return _usage_report()

    @app.post("/v1/usage/admit")
    async def post_usage_admit(request: Request) -> Response:  # pyright: ignore[reportUnusedFunction]
        _authenticate(request)
        try:
            svc.usage_gate.admit()
        except Exception as exc:
            status, payload = serialize_gateway_error(exc)
            return JSONResponse(payload, status_code=status)
        return JSONResponse(
            {
                "admitted": True,
                "turn_ceiling": svc.usage_gate.turn_ceiling(),
            }
        )

    # -- Data Plane: Normalized Inference & Streaming --

    @app.post("/v1/chat/generate")
    async def chat_generate_route(  # pyright: ignore[reportUnusedFunction]
        request: Request, payload: dict[str, Any]
    ) -> Response:
        _authenticate(request)
        admitted_header = request.headers.get(TURN_ADMITTED_HEADER, "").strip() == "1"
        saved_admission = turn_admission()
        leave_admitted_turn((admitted_header, None))
        try:
            llm_request = deserialize_llm_request(payload)
            await svc.ensure_remote_gpu_for_model(llm_request.model)
            connector, bare_model = svc.resolve_connector(llm_request.model)
            bound_request = llm_request.model_copy(update={"model": bare_model})
            bound_request = await request_for_model(connector, bound_request)
            response = await connector.generate(bound_request)
            out = response.model_dump(mode="json")
            out["paid"] = bool(getattr(connector, "paid", False))
            return JSONResponse(out)
        except Exception as exc:
            status, err_payload = serialize_gateway_error(exc)
            return JSONResponse(err_payload, status_code=status)
        finally:
            leave_admitted_turn(saved_admission)

    @app.post("/v1/chat/stream")
    async def chat_stream_route(  # pyright: ignore[reportUnusedFunction]
        request: Request, payload: dict[str, Any]
    ) -> Response:
        _authenticate(request)
        admitted_header = request.headers.get(TURN_ADMITTED_HEADER, "").strip() == "1"
        try:
            stream_request = deserialize_llm_request(payload)
            await svc.ensure_remote_gpu_for_model(stream_request.model)
            connector, bare_model = svc.resolve_connector(stream_request.model)
            bound_request = stream_request.model_copy(update={"model": bare_model})
            bound_request = await request_for_model(connector, bound_request)
        except Exception as exc:
            status, err_payload = serialize_gateway_error(exc)
            return JSONResponse(err_payload, status_code=status)

        is_paid = bool(getattr(connector, "paid", False))

        async def _event_generator() -> AsyncIterator[str]:
            saved_admission = turn_admission()
            leave_admitted_turn((admitted_header, None))
            try:
                async for chunk in connector.stream(bound_request):
                    yield f"data: {chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                _, err_dict = serialize_gateway_error(exc)
                yield f"event: error\ndata: {json.dumps(err_dict)}\n\n"
            finally:
                leave_admitted_turn(saved_admission)

        return StreamingResponse(
            _event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-UCX-Paid": "1" if is_paid else "0",
            },
        )

    # -- Data Plane: OpenAI-Compatible Chat Completions (`/v1/chat/completions`) --

    @app.post("/v1/chat/completions")
    async def openai_chat_completions_route(  # pyright: ignore[reportUnusedFunction]
        request: Request, payload: dict[str, Any]
    ) -> Response:
        _authenticate(request)
        try:
            messages = _openai_messages_to_ucx(payload.get("messages"))
            tools = _openai_tools_to_ucx(payload.get("tools"))
            raw_model = _optional_text(payload.get("model"))
            temperature = float(payload.get("temperature", 0.7))
            raw_max_tokens = payload.get("max_tokens")
            max_tokens = int(raw_max_tokens) if isinstance(raw_max_tokens, int) else None
            stream = bool(payload.get("stream", False))

            llm_req = LLMRequest(
                model=raw_model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            await svc.ensure_remote_gpu_for_model(llm_req.model)
            connector, bare_model = svc.resolve_connector(llm_req.model)
            bound_request = llm_req.model_copy(update={"model": bare_model})
            bound_request = await request_for_model(connector, bound_request)
        except Exception as exc:
            status, err_payload = serialize_gateway_error(exc)
            return JSONResponse(
                {
                    "error": {
                        "message": err_payload.get("detail"),
                        "type": err_payload.get("error_type"),
                    }
                },
                status_code=status,
            )

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if not stream:
            try:
                res = await connector.generate(bound_request)
            except Exception as exc:
                status, err_payload = serialize_gateway_error(exc)
                return JSONResponse(
                    {
                        "error": {
                            "message": err_payload.get("detail"),
                            "type": err_payload.get("error_type"),
                        }
                    },
                    status_code=status,
                )
            msg_out: dict[str, Any] = {
                "role": "assistant",
                "content": res.content,
            }
            if res.tool_calls:
                msg_out["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(unwrap_immutable(tc.arguments)),
                        },
                    }
                    for tc in res.tool_calls
                ]
            return JSONResponse(
                {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": res.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": msg_out,
                            "finish_reason": res.finish_reason.value
                            if res.finish_reason is not None
                            else "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": res.usage.input_tokens,
                        "completion_tokens": res.usage.output_tokens,
                        "total_tokens": res.usage.total_tokens,
                    },
                    "uclone_provenance": res.provenance.model_dump(mode="json")
                    if res.provenance is not None
                    else None,
                }
            )

        async def _openai_sse() -> AsyncIterator[str]:
            try:
                async for chunk in connector.stream(bound_request):
                    delta: dict[str, Any] = {}
                    if chunk.delta_content is not None:
                        delta["content"] = chunk.delta_content
                    if chunk.tool_calls:
                        delta["tool_calls"] = [
                            {
                                "index": idx,
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(unwrap_immutable(tc.arguments)),
                                },
                            }
                            for idx, tc in enumerate(chunk.tool_calls)
                        ]
                    frame: dict[str, Any] = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": chunk.model or bare_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": chunk.finish_reason.value
                                if chunk.finish_reason is not None
                                else None,
                            }
                        ],
                    }
                    if chunk.usage is not None:
                        frame["usage"] = {
                            "prompt_tokens": chunk.usage.input_tokens,
                            "completion_tokens": chunk.usage.output_tokens,
                            "total_tokens": chunk.usage.total_tokens,
                        }
                    yield f"data: {json.dumps(frame)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                _, err_payload = serialize_gateway_error(exc)
                yield f"data: {json.dumps({'error': err_payload})}\n\n"

        return StreamingResponse(
            _openai_sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # -- Data Plane: Embeddings (`/v1/embeddings`) --

    @app.post("/v1/embeddings")
    async def embeddings_route(  # pyright: ignore[reportUnusedFunction]
        request: Request, payload: dict[str, Any]
    ) -> Response:
        _authenticate(request)
        raw_texts = payload.get("texts")
        if raw_texts is None and "input" in payload:
            raw_input = payload.get("input")
            raw_texts = [raw_input] if isinstance(raw_input, str) else raw_input
        if not isinstance(raw_texts, list) or not all(
            isinstance(t, str) for t in cast(list[object], raw_texts)
        ):
            return JSONResponse(
                {"error_type": "EmbeddingError", "detail": "texts must be a list of strings."},
                status_code=400,
            )
        texts = [str(t) for t in cast(list[object], raw_texts)]
        model_arg = _optional_text(payload.get("model"))
        raw_dims = payload.get("dimensions")
        dims = int(raw_dims) if isinstance(raw_dims, int) and raw_dims > 0 else 768
        try:
            embedder = svc.resolve_embedder(model_arg, dimensions=dims)
            vectors = await embedder.embed(texts)
        except (EmbeddingDimensionError, EmbeddingError) as exc:
            status, err_payload = serialize_gateway_error(exc)
            return JSONResponse(err_payload, status_code=status)
        except Exception as exc:
            return JSONResponse(
                {"error_type": "EmbeddingError", "detail": str(exc)},
                status_code=502,
            )
        return JSONResponse(
            {
                "model_name": embedder.model_name,
                "dimensions": embedder.dimensions,
                "vectors": [list(v) for v in vectors],
            }
        )

    return app


def run_gateway_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    settings_path: Path | None = None,
    usage_path: Path | None = None,
    workspace_dir: Path | None = None,
    api_key: str | None = None,
    remote_host: str | None = None,
    log_level: str = "info",
) -> None:
    """Start the standalone Uvicorn server for the Model Gateway."""
    from uclone_x.shells import daemon_supervisor

    app = create_gateway_app(
        settings_path=settings_path,
        usage_path=usage_path,
        workspace_dir=workspace_dir,
        api_key=api_key,
        remote_host=remote_host,
    )
    record, _ = daemon_supervisor.new_daemon_record("gateway", str(port), port=port, host=host)
    with contextlib.suppress(daemon_supervisor.DaemonNotIdentifiedError):
        daemon_supervisor.write_daemon_record(record)
    try:
        uvicorn.run(app, host=host, port=port, log_level=log_level)
    finally:
        daemon_supervisor.remove_daemon_record(record)
