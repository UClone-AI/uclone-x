"""Remote GPU worker inspector and SSH auto-tunnel manager.

Enables zero-config connection to remote GPU machines (such as Dell workstation or
dedicated inference nodes) via SSH port-forwarding without exposing remote ports to
the public internet.
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import json
import logging
import re
import shutil
import socket
from dataclasses import asdict, dataclass, field
from typing import Any, cast

logger = logging.getLogger(__name__)

DEFAULT_REMOTE_OLLAMA_PORT = 11434
DEFAULT_REMOTE_COMFYUI_PORT = 8188

_VALID_HOST_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.\@\:]+$")


def is_valid_host(host: str) -> bool:
    """Validate that the given host string is safe for SSH argument usage."""
    if not host:
        return False
    stripped = host.strip()
    if stripped.startswith("-"):
        return False
    return bool(_VALID_HOST_REGEX.match(stripped))


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a local TCP port is already open and accepting connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        try:
            return s.connect_ex((host, port)) == 0
        except OSError:
            return False


def find_free_port(preferred: int | None = None) -> int:
    """Find an available TCP port on localhost, attempting `preferred` first."""
    if preferred is not None and 1024 <= preferred <= 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", preferred))
                return preferred
            except OSError:
                pass

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@dataclass(frozen=True)
class RemoteGPUInfo:
    """Metadata about a remote GPU detected via nvidia-smi."""

    name: str
    total_mb: int
    used_mb: int
    driver: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RemoteServiceStatus:
    """Status of an AI service running on the remote host."""

    name: str
    port: int
    listening: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RemoteHostInspection:
    """Complete inspection result for an SSH host."""

    host: str
    reachable: bool
    error: str | None = None
    gpu: RemoteGPUInfo | None = None
    ports: dict[str, RemoteServiceStatus] = field(default_factory=dict[str, RemoteServiceStatus])

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "reachable": self.reachable,
            "error": self.error,
            "gpu": self.gpu.to_dict() if self.gpu else None,
            "ports": {k: v.to_dict() for k, v in self.ports.items()},
        }


@dataclass(frozen=True)
class PortMapping:
    """Local-to-remote port forwarding specification."""

    service_name: str
    remote_port: int
    local_port: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TunnelSessionStatus:
    """Live status of an active SSH tunnel session."""

    host: str
    connected: bool
    pid: int | None = None
    mappings: list[PortMapping] = field(default_factory=list[PortMapping])
    gpu: RemoteGPUInfo | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "connected": self.connected,
            "pid": self.pid,
            "mappings": [m.to_dict() for m in self.mappings],
            "gpu": self.gpu.to_dict() if self.gpu else None,
            "error": self.error,
        }


_REMOTE_PROBE_PYTHON_CMD = (
    "python3 -c '"
    "import json, subprocess, shutil, socket\n"
    'res = {"gpu": None, "ports": {}}\n'
    'nv = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"\n'
    "try:\n"
    '    out = subprocess.check_output([nv, "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL).strip()\n'
    "    if out:\n"
    '        parts = [p.strip() for p in out.splitlines()[0].split(",")]\n'
    '        res["gpu"] = {"name": parts[0], "total_mb": int(parts[1]), "used_mb": int(parts[2]), "driver": parts[3]}\n'
    "except Exception:\n"
    "    pass\n"
    'for port, name in [(11434, "ollama"), (8188, "comfyui")]:\n'
    "    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
    "    s.settimeout(0.5)\n"
    '    r = s.connect_ex(("127.0.0.1", port))\n'
    "    s.close()\n"
    '    res["ports"][name] = {"port": port, "listening": (r == 0)}\n'
    "print(json.dumps(res))\n"
    "'"
)


async def probe_remote_host(host: str, timeout: float = 6.0) -> RemoteHostInspection:
    """Probe an SSH host for GPU availability and active AI services."""
    clean_host = host.strip()
    if not is_valid_host(clean_host):
        return RemoteHostInspection(
            host=clean_host,
            reachable=False,
            error=f"Invalid SSH host format: {clean_host!r}",
        )

    if not shutil.which("ssh"):
        return RemoteHostInspection(
            host=clean_host,
            reachable=False,
            error="'ssh' executable not found in PATH",
        )

    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={max(1, int(timeout))}",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "--",
        clean_host,
        _REMOTE_PROBE_PYTHON_CMD,
    ]

    proc: asyncio.subprocess.Process | None = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout + 2.0)
    except TimeoutError:
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(Exception):
                proc.kill()
                await proc.wait()
        return RemoteHostInspection(
            host=clean_host,
            reachable=False,
            error=f"SSH probe timed out after {timeout:g}s",
        )
    except Exception as exc:
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(Exception):
                proc.kill()
                await proc.wait()
        return RemoteHostInspection(
            host=clean_host,
            reachable=False,
            error=f"Failed to execute SSH probe: {exc}",
        )

    if proc.returncode != 0:
        err_msg = stderr.decode().strip() or f"SSH exited with code {proc.returncode}"
        return RemoteHostInspection(
            host=clean_host,
            reachable=False,
            error=err_msg,
        )

    try:
        data = json.loads(stdout.decode().strip())
    except Exception as exc:
        return RemoteHostInspection(
            host=clean_host,
            reachable=True,
            error=f"Failed to parse remote probe response: {exc}",
        )

    gpu_info: RemoteGPUInfo | None = None
    if isinstance(data.get("gpu"), dict):
        g = data["gpu"]
        gpu_info = RemoteGPUInfo(
            name=str(g.get("name", "Unknown GPU")),
            total_mb=int(g.get("total_mb", 0)),
            used_mb=int(g.get("used_mb", 0)),
            driver=str(g.get("driver", "")),
        )

    ports: dict[str, RemoteServiceStatus] = {}
    raw_ports = data.get("ports")
    if isinstance(raw_ports, dict):
        raw_dict = cast(dict[str, Any], raw_ports)
        for name, pinfo in raw_dict.items():
            if isinstance(pinfo, dict):
                pdict = cast(dict[str, Any], pinfo)
                ports[str(name)] = RemoteServiceStatus(
                    name=str(name),
                    port=int(pdict.get("port", 0)),
                    listening=bool(pdict.get("listening", False)),
                )

    return RemoteHostInspection(
        host=clean_host,
        reachable=True,
        gpu=gpu_info,
        ports=ports,
    )


class SSHTunnelManager:
    """Manages an active background SSH port-forwarding session."""

    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None
        self._session: TunnelSessionStatus | None = None
        self._drain_task: asyncio.Task[None] | None = None
        self._stderr_lines: list[str] = []
        self._lock = asyncio.Lock()
        atexit.register(self._sync_cleanup)

    def _sync_cleanup(self) -> None:
        """Synchronous cleanup handler registered with atexit."""
        if self._proc is not None and self._proc.returncode is None:
            with contextlib.suppress(Exception):
                self._proc.kill()
            self._proc = None
        if self._drain_task is not None:
            self._drain_task.cancel()
            self._drain_task = None

    @property
    def is_connected(self) -> bool:
        if self._proc is None or self._session is None:
            return False
        if self._proc.returncode is not None:
            self._session.connected = False
            return False
        return self._session.connected

    def get_status(self) -> TunnelSessionStatus:
        if self._session is None or not self.is_connected:
            return TunnelSessionStatus(
                host="",
                connected=False,
                error="No active remote tunnel",
            )
        return self._session

    async def connect(
        self,
        host: str,
        preferred_local_ollama_port: int | None = None,
        preferred_local_comfyui_port: int | None = None,
        timeout: float = 10.0,
    ) -> TunnelSessionStatus:
        """Probe remote host and establish an SSH port-forwarding tunnel."""
        async with self._lock:
            # Clean up existing session if any
            await self._disconnect_unlocked()

            clean_host = host.strip()
            inspection = await probe_remote_host(clean_host, timeout=min(6.0, timeout))
            if not inspection.reachable:
                status = TunnelSessionStatus(
                    host=clean_host,
                    connected=False,
                    error=inspection.error or "Host unreachable",
                )
                self._session = status
                return status

            # Determine port mappings based on remote confirmed services
            mappings: list[PortMapping] = []

            ollama_listening = bool(
                inspection.ports.get("ollama") and inspection.ports["ollama"].listening
            )
            comfy_listening = bool(
                inspection.ports.get("comfyui") and inspection.ports["comfyui"].listening
            )

            # 1. Ollama mapping
            if ollama_listening or not comfy_listening:
                ollama_target_local = find_free_port(
                    preferred=preferred_local_ollama_port
                    if preferred_local_ollama_port is not None
                    else (11434 if not is_port_in_use(11434) else 11435)
                )
                mappings.append(
                    PortMapping(
                        service_name="ollama",
                        remote_port=DEFAULT_REMOTE_OLLAMA_PORT,
                        local_port=ollama_target_local,
                    )
                )

            # 2. ComfyUI mapping
            if comfy_listening:
                comfy_target_local = find_free_port(
                    preferred=preferred_local_comfyui_port
                    if preferred_local_comfyui_port is not None
                    else (8188 if not is_port_in_use(8188) else 8189)
                )
                mappings.append(
                    PortMapping(
                        service_name="comfyui",
                        remote_port=DEFAULT_REMOTE_COMFYUI_PORT,
                        local_port=comfy_target_local,
                    )
                )

            ssh_cmd = [
                "ssh",
                "-N",
                "-o",
                "BatchMode=yes",
                "-o",
                "ExitOnForwardFailure=yes",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=3",
                "-o",
                "StrictHostKeyChecking=accept-new",
            ]
            for m in mappings:
                ssh_cmd.extend(["-L", f"{m.local_port}:127.0.0.1:{m.remote_port}"])
            ssh_cmd.extend(["--", clean_host])

            try:
                proc = await asyncio.create_subprocess_exec(
                    *ssh_cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                self._proc = proc
            except Exception as exc:
                err_msg = f"Failed to spawn ssh tunnel process: {exc}"
                status = TunnelSessionStatus(host=clean_host, connected=False, error=err_msg)
                self._session = status
                return status

            # Asynchronously drain stderr in the background to avoid OS pipe deadlock
            self._stderr_lines.clear()

            async def _drain_stderr(stream: asyncio.StreamReader) -> None:
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    self._stderr_lines.append(line.decode(errors="replace").strip())
                    if len(self._stderr_lines) > 100:
                        self._stderr_lines.pop(0)

            if proc.stderr is not None:
                self._drain_task = asyncio.create_task(_drain_stderr(proc.stderr))

            # Wait for ports to be bound locally
            deadline = asyncio.get_event_loop().time() + timeout
            all_ready = False
            while asyncio.get_event_loop().time() < deadline:
                if proc.returncode is not None:
                    # Subprocess terminated prematurely
                    await asyncio.sleep(0.05)
                    err = (
                        "\n".join(self._stderr_lines).strip()
                        or f"ssh exited code {proc.returncode}"
                    )
                    status = TunnelSessionStatus(
                        host=clean_host,
                        connected=False,
                        error=f"Tunnel process died: {err}",
                    )
                    self._session = status
                    self._proc = None
                    return status

                # Check if all local ports accept connections
                ready_count = 0
                for m in mappings:
                    if is_port_in_use(m.local_port):
                        ready_count += 1
                if ready_count == len(mappings):
                    all_ready = True
                    break

                await asyncio.sleep(0.15)

            if not all_ready:
                await self._disconnect_unlocked()
                status = TunnelSessionStatus(
                    host=clean_host,
                    connected=False,
                    error=f"Ports failed to open locally within {timeout:g}s",
                )
                self._session = status
                return status

            status = TunnelSessionStatus(
                host=clean_host,
                connected=True,
                pid=proc.pid,
                mappings=mappings,
                gpu=inspection.gpu,
            )
            self._session = status
            logger.info(
                "SSH tunnel established to %s (pid=%s): %s",
                clean_host,
                proc.pid,
                [(m.service_name, m.local_port, m.remote_port) for m in mappings],
            )
            return status

    async def _disconnect_unlocked(self) -> None:
        """Internal disconnect without acquiring lock."""
        if self._drain_task is not None:
            self._drain_task.cancel()
            with contextlib.suppress(BaseException):
                await self._drain_task
            self._drain_task = None

        proc = self._proc
        self._proc = None
        if proc is not None and proc.returncode is None:
            try:
                proc.terminate()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
            except Exception:
                pass
            if proc.returncode is None:
                try:
                    proc.kill()
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(proc.wait(), timeout=1.0)
                except Exception:
                    pass
        if self._session is not None:
            self._session.connected = False
            self._session.pid = None

    async def disconnect(self) -> None:
        """Gracefully disconnect active SSH tunnel session."""
        async with self._lock:
            await self._disconnect_unlocked()

    async def close(self) -> None:
        """Alias for disconnect for lifecycle conformity."""
        await self.disconnect()
