# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateUsage=false, reportUntypedFunctionDecorator=false, reportUnknownParameterType=false, reportMissingParameterType=false
"""Unit tests for remote GPU worker inspector and SSH auto-tunnel manager."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from uclone_x.core.remote_worker import (
    DEFAULT_REMOTE_COMFYUI_PORT,
    DEFAULT_REMOTE_OLLAMA_PORT,
    PortMapping,
    RemoteGPUInfo,
    RemoteHostInspection,
    RemoteServiceStatus,
    SSHTunnelManager,
    TunnelSessionStatus,
    find_configured_ssh_hosts,
    find_free_port,
    is_port_in_use,
    is_valid_host,
    launch_remote_comfyui,
    launch_remote_ollama,
    probe_remote_host,
)
from uclone_x.ui.app import create_ui_app


def test_is_valid_host() -> None:
    """Test SSH host validation against command injection and invalid formats."""
    assert is_valid_host("dell")
    assert is_valid_host("10.202.1.121")
    assert is_valid_host("user@10.202.1.121")
    assert is_valid_host("my-gpu-worker_01.local")
    assert is_valid_host("user@dell:22")

    assert not is_valid_host("")
    assert not is_valid_host("-oProxyCommand=touch /tmp/pwn")
    assert not is_valid_host("-v")
    assert not is_valid_host("--help")
    assert not is_valid_host("dell; rm -rf /")
    assert not is_valid_host("dell`id`")
    assert not is_valid_host("dell $(whoami)")
    assert not is_valid_host("dell|cat")


def test_find_free_port_and_in_use() -> None:
    """Test TCP port availability probing and free port acquisition."""
    port = find_free_port()
    assert 1024 <= port <= 65535
    # The allocated port should not be in use yet
    assert not is_port_in_use(port)


@pytest.mark.asyncio
async def test_probe_remote_host_success() -> None:
    """Test parsing of successful remote GPU and port probe output."""
    mock_payload = {
        "gpu": {
            "name": "NVIDIA GeForce RTX 5070 Ti",
            "total_mb": 16303,
            "used_mb": 932,
            "driver": "610.88",
        },
        "ports": {
            "ollama": {"port": 11434, "listening": True},
            "comfyui": {"port": 8188, "listening": False},
        },
    }

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(json.dumps(mock_payload).encode("utf-8"), b""))

    with patch("shutil.which", return_value="/usr/bin/ssh"):
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            res = await probe_remote_host("dell", timeout=5.0)

    assert res.reachable is True
    assert res.host == "dell"
    assert res.error is None
    assert res.gpu is not None
    assert res.gpu.name == "NVIDIA GeForce RTX 5070 Ti"
    assert res.gpu.total_mb == 16303
    assert res.ports["ollama"].listening is True
    assert res.ports["comfyui"].listening is False


@pytest.mark.asyncio
async def test_probe_remote_host_apple_silicon() -> None:
    """Test remote host probe parsing for Apple Silicon (macOS) hardware."""
    mock_payload = {
        "gpu": {
            "name": "Apple M4 Pro",
            "total_mb": 49152,
            "used_mb": 0,
            "driver": "Apple Metal",
        },
        "ports": {
            "ollama": {"port": 11434, "listening": True},
            "comfyui": {"port": 8188, "listening": False},
        },
        "comfyui_dir": None,
        "ollama_models": ["qwen3:8b"],
    }

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(json.dumps(mock_payload).encode("utf-8"), b""))

    with patch("shutil.which", return_value="/usr/bin/ssh"):
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            res = await probe_remote_host("mac-mini", timeout=5.0)

    assert res.reachable is True
    assert res.host == "mac-mini"
    assert res.error is None
    assert res.gpu is not None
    assert res.gpu.name == "Apple M4 Pro"
    assert res.gpu.total_mb == 49152
    assert res.gpu.used_mb == 0
    assert res.gpu.driver == "Apple Metal"
    assert res.comfyui_dir is None
    assert res.ollama_models == ["qwen3:8b"]


@pytest.mark.asyncio
async def test_probe_remote_host_failure() -> None:
    """Test SSH probe failure handling when host is unreachable."""
    mock_proc = MagicMock()
    mock_proc.returncode = 255
    mock_proc.communicate = AsyncMock(
        return_value=(b"", b"ssh: connect to host unreachable-host port 22: Connection refused")
    )

    with patch("shutil.which", return_value="/usr/bin/ssh"):
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            res = await probe_remote_host("unreachable-host", timeout=2.0)

    assert res.reachable is False
    assert "Connection refused" in (res.error or "")


@pytest.mark.asyncio
async def test_ssh_tunnel_manager_connect_and_disconnect() -> None:
    """Test SSHTunnelManager connect, status, and disconnect lifecycle."""
    mgr = SSHTunnelManager()

    mock_inspection = RemoteHostInspection(
        host="dell",
        reachable=True,
        gpu=RemoteGPUInfo(name="RTX 5070 Ti", total_mb=16303, used_mb=500, driver="610.88"),
        ports={
            "ollama": RemoteServiceStatus("ollama", 11434, True),
            "comfyui": RemoteServiceStatus("comfyui", 8188, True),
        },
    )

    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.pid = 99999
    mock_proc.terminate = MagicMock()
    mock_proc.kill = MagicMock()
    mock_proc.wait = AsyncMock(return_value=0)

    with patch("uclone_x.core.remote_worker.probe_remote_host", return_value=mock_inspection):
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("uclone_x.core.remote_worker.is_port_in_use", return_value=True):
                status = await mgr.connect("dell", timeout=2.0)

    assert status.connected is True
    assert status.host == "dell"
    assert status.pid == 99999
    assert len(status.mappings) == 2
    assert mgr.is_connected is True

    # Check status method
    current = mgr.get_status()
    assert current.connected is True
    assert current.host == "dell"

    # Disconnect
    await mgr.disconnect()
    assert mgr.is_connected is False
    mock_proc.terminate.assert_called_once()


@pytest.mark.asyncio
async def test_ui_remote_gpu_endpoints(tmp_path: Path) -> None:
    """Test /api/settings/remote-gpu API endpoints."""
    app = create_ui_app(storage_dir=tmp_path / "uclone_storage")

    mock_inspection = RemoteHostInspection(
        host="dell",
        reachable=True,
        gpu=RemoteGPUInfo(name="RTX 5070 Ti", total_mb=16303, used_mb=500, driver="610.88"),
        ports={
            "ollama": RemoteServiceStatus("ollama", 11434, True),
            "comfyui": RemoteServiceStatus("comfyui", 8188, False),
        },
    )

    mock_tunnel_status = TunnelSessionStatus(
        host="dell",
        connected=True,
        pid=12345,
        mappings=[
            PortMapping("ollama", DEFAULT_REMOTE_OLLAMA_PORT, 11435),
            PortMapping("comfyui", DEFAULT_REMOTE_COMFYUI_PORT, 8189),
        ],
        gpu=mock_inspection.gpu,
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        # 1. Probe endpoint
        with patch("uclone_x.ui.app.probe_remote_host", return_value=mock_inspection):
            resp = await client.post("/api/settings/remote-gpu/probe", json={"host": "dell"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["reachable"] is True
            assert data["gpu"]["name"] == "RTX 5070 Ti"

        # 2. Connect endpoint
        tunnel_mgr = app.state.tunnel_manager
        with patch.object(tunnel_mgr, "connect", return_value=mock_tunnel_status):
            resp = await client.post(
                "/api/settings/remote-gpu/connect",
                json={"host": "dell", "apply_settings": True, "sync_llm": True},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["connected"] is True
            # The tunnel's Ollama joins the model set as its own connection (model-gateway G1).
            assert data["applied_changes"]["connection"] == "remote-gpu"
            assert _connection(app, "remote-gpu") == "http://127.0.0.1:11435"

        # 3. Status endpoint
        with patch.object(tunnel_mgr, "get_status", return_value=mock_tunnel_status):
            resp = await client.get("/api/settings/remote-gpu/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["connected"] is True
            assert data["host"] == "dell"

        # 4. Disconnect endpoint restores previous settings
        with patch.object(tunnel_mgr, "disconnect", new_callable=AsyncMock):
            with patch.object(
                tunnel_mgr,
                "get_status",
                return_value=TunnelSessionStatus(host="", connected=False),
            ):
                resp = await client.post("/api/settings/remote-gpu/disconnect")
                assert resp.status_code == 200
                data = resp.json()
                assert data["status"] == "ok"
                assert data["connected"] is False
                assert "restored_settings" in data

        # 5. Cross-origin request rejection (CSRF protection via _refuse_unless_local)
        resp = await client.post(
            "/api/settings/remote-gpu/probe",
            json={"host": "dell"},
            headers={"Origin": "https://malicious-attacker.com"},
        )
        assert resp.status_code == 403

        resp = await client.post(
            "/api/settings/remote-gpu/connect",
            json={"host": "dell"},
            headers={"Origin": "https://malicious-attacker.com"},
        )
        assert resp.status_code == 403

        # 6. Empty host validation
        resp = await client.post("/api/settings/remote-gpu/probe", json={"host": ""})
        assert resp.status_code == 400
        resp = await client.post("/api/settings/remote-gpu/connect", json={"host": " "})
        assert resp.status_code == 400


@pytest.mark.asyncio
async def test_probe_remote_host_timeout_kills_process() -> None:
    """Test that subprocess is killed and reaped when probe times out."""
    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.kill = MagicMock()
    mock_proc.wait = AsyncMock(return_value=0)
    mock_proc.communicate = AsyncMock(side_effect=TimeoutError())

    with patch("shutil.which", return_value="/usr/bin/ssh"):
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            res = await probe_remote_host("dell", timeout=0.1)

    assert res.reachable is False
    assert "timed out" in (res.error or "")
    mock_proc.kill.assert_called_once()
    mock_proc.wait.assert_called_once()


@pytest.mark.asyncio
async def test_probe_remote_host_invalid_json() -> None:
    """Test SSH probe handling when remote stdout is not valid JSON."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"not-json-output", b""))

    with patch("shutil.which", return_value="/usr/bin/ssh"):
        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            res = await probe_remote_host("dell", timeout=2.0)

    assert res.reachable is True
    assert "Failed to parse remote probe response" in (res.error or "")


@pytest.mark.asyncio
async def test_probe_remote_host_no_ssh() -> None:
    """Test SSH probe handling when ssh is not in PATH."""
    with patch("shutil.which", return_value=None):
        res = await probe_remote_host("dell", timeout=2.0)
    assert res.reachable is False
    assert "'ssh' executable not found" in (res.error or "")


@pytest.mark.asyncio
async def test_ssh_tunnel_manager_connect_failure_cases() -> None:
    """Test failure cases during SSH tunnel establishment."""
    mgr = SSHTunnelManager()

    # 1. Unreachable host
    mock_inspection = RemoteHostInspection(
        host="bad-host",
        reachable=False,
        error="Host unreachable",
    )
    with patch("uclone_x.core.remote_worker.probe_remote_host", return_value=mock_inspection):
        status = await mgr.connect("bad-host")
        assert status.connected is False
        assert "Host unreachable" in (status.error or "")

    # 2. Subprocess terminates prematurely
    mock_good_inspection = RemoteHostInspection(
        host="dell",
        reachable=True,
    )
    mock_failed_proc = MagicMock()
    mock_failed_proc.returncode = 1
    mock_failed_proc.stderr = AsyncMock()
    mock_failed_proc.stderr.readline = AsyncMock(
        side_effect=[b"bind: Address already in use\n", b""]
    )

    with patch("uclone_x.core.remote_worker.probe_remote_host", return_value=mock_good_inspection):
        with patch("asyncio.create_subprocess_exec", return_value=mock_failed_proc):
            status = await mgr.connect("dell")
            assert status.connected is False
            assert "Address already in use" in (status.error or "")


@pytest.mark.asyncio
async def test_launch_remote_comfyui_success() -> None:
    """Test remote ComfyUI launch and polling success."""
    mock_spawn = MagicMock()
    mock_spawn.returncode = 0
    mock_spawn.communicate = AsyncMock(return_value=(b"", b""))

    mock_poll = MagicMock()
    mock_poll.returncode = 0
    mock_poll.wait = AsyncMock(return_value=0)

    with patch("asyncio.create_subprocess_exec", side_effect=[mock_spawn, mock_poll]):
        ok, err = await launch_remote_comfyui("dell", "/home/kennylim/ComfyUI", timeout=3.0)
    assert ok is True
    assert err is None


@pytest.mark.asyncio
async def test_launch_remote_comfyui_failure_with_tail() -> None:
    """Test remote ComfyUI launch failure reading tail of comfy.log."""
    mock_spawn = MagicMock()
    mock_spawn.returncode = 0
    mock_spawn.communicate = AsyncMock(return_value=(b"", b""))

    mock_poll = MagicMock()
    mock_poll.returncode = 1
    mock_poll.wait = AsyncMock(return_value=1)

    mock_tail = MagicMock()
    mock_tail.returncode = 0
    mock_tail.communicate = AsyncMock(
        return_value=(b"ModuleNotFoundError: No module named 'onnxruntime'", b"")
    )

    with patch("asyncio.create_subprocess_exec", side_effect=[mock_spawn, mock_poll, mock_tail]):
        ok, err = await launch_remote_comfyui("dell", "/home/kennylim/ComfyUI", timeout=0.2)
    assert ok is False
    assert err is not None
    assert "onnxruntime" in err


@pytest.mark.asyncio
async def test_ssh_tunnel_manager_auto_start_comfyui() -> None:
    """Test that connect() automatically launches ComfyUI when not listening."""
    mgr = SSHTunnelManager()

    mock_inspection = RemoteHostInspection(
        host="dell",
        reachable=True,
        ports={
            "ollama": RemoteServiceStatus("ollama", 11434, True),
            "comfyui": RemoteServiceStatus("comfyui", 8188, False),
        },
        comfyui_dir="/home/kennylim/ComfyUI",
        ollama_models=["qwen3:8b"],
    )

    mock_proc = MagicMock()
    mock_proc.pid = 9999
    mock_proc.returncode = None
    mock_proc.stderr = AsyncMock()
    mock_proc.stderr.readline = AsyncMock(return_value=b"")

    with patch("uclone_x.core.remote_worker.probe_remote_host", return_value=mock_inspection):
        with patch(
            "uclone_x.core.remote_worker.launch_remote_comfyui", return_value=(True, None)
        ) as mock_launch:
            with patch("uclone_x.core.remote_worker.is_port_in_use", return_value=True):
                with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                    status = await mgr.connect("dell", auto_start_comfyui=True)
                    assert status.connected is True
                    assert status.comfyui_autostarted is True
                    assert any(m.service_name == "comfyui" for m in status.mappings)
                    mock_launch.assert_called_once_with(
                        "dell", "/home/kennylim/ComfyUI", timeout=10.0
                    )


@pytest.mark.asyncio
async def test_launch_remote_ollama_success() -> None:
    """Test remote Ollama launch and polling success."""
    mock_spawn = MagicMock()
    mock_spawn.returncode = 0
    mock_spawn.communicate = AsyncMock(return_value=(b"", b""))

    mock_poll = MagicMock()
    mock_poll.returncode = 0
    mock_poll.wait = AsyncMock(return_value=0)

    with patch("asyncio.create_subprocess_exec", side_effect=[mock_spawn, mock_poll]):
        ok, err = await launch_remote_ollama("dell", timeout=3.0)
    assert ok is True
    assert err is None


@pytest.mark.asyncio
async def test_launch_remote_ollama_spawn_failure() -> None:
    """Test remote Ollama launch failure when spawn command returns non-zero."""
    mock_spawn = MagicMock()
    mock_spawn.returncode = 127
    mock_spawn.communicate = AsyncMock(return_value=(b"", b"ollama: command not found"))

    with patch("asyncio.create_subprocess_exec", return_value=mock_spawn):
        ok, err = await launch_remote_ollama("dell", timeout=3.0)
    assert ok is False
    assert err is not None
    assert "command not found" in err


@pytest.mark.asyncio
async def test_launch_remote_ollama_timeout_with_tail() -> None:
    """Test remote Ollama launch failure reading tail of ollama.log."""
    mock_spawn = MagicMock()
    mock_spawn.returncode = 0
    mock_spawn.communicate = AsyncMock(return_value=(b"", b""))

    mock_poll = MagicMock()
    mock_poll.returncode = 1
    mock_poll.wait = AsyncMock(return_value=1)

    mock_tail = MagicMock()
    mock_tail.returncode = 0
    mock_tail.communicate = AsyncMock(return_value=(b"bind: address already in use", b""))

    with patch("asyncio.create_subprocess_exec", side_effect=[mock_spawn, mock_poll, mock_tail]):
        ok, err = await launch_remote_ollama("dell", timeout=0.2)
    assert ok is False
    assert err is not None
    assert "address already in use" in err


@pytest.mark.asyncio
async def test_launch_remote_ollama_invalid_host() -> None:
    """Test remote Ollama launch validation with invalid host."""
    ok, err = await launch_remote_ollama("-badhost")
    assert ok is False
    assert err is not None
    assert "Invalid host" in err


@pytest.mark.asyncio
async def test_ssh_tunnel_manager_auto_start_ollama() -> None:
    """Test that connect() automatically launches Ollama when not listening."""
    mgr = SSHTunnelManager()

    mock_inspection = RemoteHostInspection(
        host="dell",
        reachable=True,
        ports={
            "ollama": RemoteServiceStatus("ollama", 11434, False),
            "comfyui": RemoteServiceStatus("comfyui", 8188, False),
        },
        comfyui_dir=None,
    )

    mock_proc = MagicMock()
    mock_proc.pid = 9999
    mock_proc.returncode = None
    mock_proc.stderr = AsyncMock()
    mock_proc.stderr.readline = AsyncMock(return_value=b"")

    with patch("uclone_x.core.remote_worker.probe_remote_host", return_value=mock_inspection):
        with patch(
            "uclone_x.core.remote_worker.launch_remote_ollama", return_value=(True, None)
        ) as mock_launch:
            with patch("uclone_x.core.remote_worker.is_port_in_use", return_value=True):
                with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                    status = await mgr.connect("dell", auto_start_ollama=True)
                    assert status.connected is True
                    assert status.ollama_autostarted is True
                    assert any(m.service_name == "ollama" for m in status.mappings)
                    mock_launch.assert_called_once_with("dell", timeout=8.0)


@pytest.mark.asyncio
async def test_ssh_tunnel_manager_auto_start_ollama_disabled() -> None:
    """Test that connect() respects auto_start_ollama=False."""
    mgr = SSHTunnelManager()

    mock_inspection = RemoteHostInspection(
        host="dell",
        reachable=True,
        ports={
            "ollama": RemoteServiceStatus("ollama", 11434, False),
            "comfyui": RemoteServiceStatus("comfyui", 8188, False),
        },
        comfyui_dir=None,
    )

    mock_proc = MagicMock()
    mock_proc.pid = 9999
    mock_proc.returncode = None
    mock_proc.stderr = AsyncMock()
    mock_proc.stderr.readline = AsyncMock(return_value=b"")

    with patch("uclone_x.core.remote_worker.probe_remote_host", return_value=mock_inspection):
        with patch(
            "uclone_x.core.remote_worker.launch_remote_ollama", return_value=(True, None)
        ) as mock_launch:
            with patch("uclone_x.core.remote_worker.is_port_in_use", return_value=True):
                with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                    status = await mgr.connect("dell", auto_start_ollama=False)
                    assert status.connected is True
                    assert status.ollama_autostarted is False
                    mock_launch.assert_not_called()


@pytest.mark.asyncio
async def test_ssh_tunnel_manager_skip_comfyui_when_dir_none() -> None:
    """Verify that when comfyui_dir is None, ComfyUI launch is skipped immediately without error."""
    mgr = SSHTunnelManager()

    mock_inspection = RemoteHostInspection(
        host="mac-mini",
        reachable=True,
        ports={
            "ollama": RemoteServiceStatus("ollama", 11434, True),
            "comfyui": RemoteServiceStatus("comfyui", 8188, False),
        },
        comfyui_dir=None,
    )

    mock_proc = MagicMock()
    mock_proc.pid = 8888
    mock_proc.returncode = None
    mock_proc.stderr = AsyncMock()
    mock_proc.stderr.readline = AsyncMock(return_value=b"")

    with patch("uclone_x.core.remote_worker.probe_remote_host", return_value=mock_inspection):
        with patch(
            "uclone_x.core.remote_worker.launch_remote_comfyui",
            return_value=(False, "Should not run"),
        ) as mock_launch:
            with patch("uclone_x.core.remote_worker.is_port_in_use", return_value=True):
                with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                    status = await mgr.connect("mac-mini", auto_start_comfyui=True)
                    assert status.connected is True
                    assert status.comfyui_autostarted is False
                    # ComfyUI launch should NOT be attempted when comfyui_dir is None
                    mock_launch.assert_not_called()
                    # Only ollama mapping should exist, comfyui mapping should not exist
                    assert any(m.service_name == "ollama" for m in status.mappings)
                    assert not any(m.service_name == "comfyui" for m in status.mappings)


@pytest.mark.asyncio
async def test_ui_connect_adds_the_tunnel_beside_the_local_ollama(tmp_path: Path) -> None:
    """Connecting one source never disconnects another, nor changes a default (G1, G7).

    The tunnel's Ollama used to replace the LLM address, and was skipped when the saved
    model was not on the worker. It is now a connection of its own, so the local one and
    the default model stay exactly as they were.

    Killed by: src/uclone_x/ui/app.py :: changes["connection"] = REMOTE_GPU_CONNECTION_ID
    Becomes: pass
    """
    app = create_ui_app(storage_dir=tmp_path / "uclone_storage")
    tunnel_mgr = app.state.tunnel_manager
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        await _save_local_ollama(client, app)
        with patch.object(tunnel_mgr, "connect", return_value=_dell_tunnel()):
            data = (
                await client.post(
                    "/api/settings/remote-gpu/connect",
                    json={"host": "dell", "apply_settings": True, "sync_llm": True},
                )
            ).json()
    assert data["applied_changes"]["connection"] == "remote-gpu"
    assert _connection(app, "ollama") == "http://127.0.0.1:11434"
    assert _connection(app, "remote-gpu") == "http://127.0.0.1:11435"
    defaults = app.state.session_manager.gateway.defaults()
    assert defaults.deep == "ollama/qwen3:8b"


def _dell_tunnel() -> TunnelSessionStatus:
    return TunnelSessionStatus(
        host="dell",
        connected=True,
        pid=12345,
        mappings=[
            PortMapping("ollama", DEFAULT_REMOTE_OLLAMA_PORT, 11435),
            PortMapping("comfyui", DEFAULT_REMOTE_COMFYUI_PORT, 8189),
        ],
        ollama_models=["qwen3:8b"],
    )


async def _save_local_ollama(client: AsyncClient, app: Any) -> None:
    """A local Ollama connection with a default model on it, and a local ComfyUI one."""
    from uclone_x.llm.connectors.saved_choice import add_connection, save_default_models

    path = app.state.session_manager.settings_file
    add_connection("ollama", base_url="http://127.0.0.1:11434", path=path)
    add_connection("comfyui", base_url="http://127.0.0.1:8188", path=path)
    save_default_models({"deep": "ollama/qwen3:8b"}, path=path)


def _connection(app: Any, conn_id: str) -> str | None:
    """The address of connection ``conn_id``, or `None` when there is no such connection."""
    conn = app.state.session_manager.gateway.connection(conn_id)
    return None if conn is None else conn.base_url


@pytest.mark.asyncio
async def test_ui_connect_leaves_llm_local_unless_asked(tmp_path: Path) -> None:
    """A worker connected for images does not take the LLM, even with the same model on it.

    Its ComfyUI becomes a picture connection of its own (model-gateway §3.5).

    Killed by: src/uclone_x/ui/app.py :: changes["picture_connection"] = REMOTE_GPU_PICTURES_ID
    Becomes: pass
    """
    app = create_ui_app(storage_dir=tmp_path / "uclone_storage")
    tunnel_mgr = app.state.tunnel_manager
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        await _save_local_ollama(client, app)
        with patch.object(tunnel_mgr, "connect", return_value=_dell_tunnel()):
            resp = await client.post(
                "/api/settings/remote-gpu/connect", json={"host": "dell", "apply_settings": True}
            )
        data = resp.json()
        assert data["applied_changes"] == {"picture_connection": "remote-gpu-pictures"}
        assert data["llm_skipped"] is None
        assert _connection(app, "remote-gpu") is None
        # The tunnel's ComfyUI is a picture connection beside the local one (§3.5).
        assert _connection(app, "remote-gpu-pictures") == "http://127.0.0.1:8189"
        assert _connection(app, "comfyui") == "http://127.0.0.1:8188"


@pytest.mark.asyncio
async def test_ui_dead_tunnel_restores_saved_addresses_from_disk(tmp_path: Path) -> None:
    """A tunnel that died -- or a restart -- puts back what connect replaced, from disk."""
    storage = tmp_path / "uclone_storage"
    app = create_ui_app(storage_dir=storage)
    tunnel_mgr = app.state.tunnel_manager
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        await _save_local_ollama(client, app)
        with patch.object(tunnel_mgr, "connect", return_value=_dell_tunnel()):
            await client.post(
                "/api/settings/remote-gpu/connect",
                json={"host": "dell", "apply_settings": True, "sync_llm": True},
            )
            # A second connect must not record the tunnel's own addresses as "previous".
            await client.post(
                "/api/settings/remote-gpu/connect",
                json={"host": "dell", "apply_settings": True, "sync_llm": True},
            )
        record = json.loads((storage / "remote_gpu_restore.json").read_text())
        assert record == {
            "original": {"remote-gpu": "", "remote-gpu-pictures": ""},
            "applied": {
                "remote-gpu": "http://127.0.0.1:11435",
                "remote-gpu-pictures": "http://127.0.0.1:8189",
            },
        }
        # The ssh process is gone; nobody pressed Disconnect.
        resp = await client.get("/api/settings/remote-gpu/status")
        data = resp.json()
        assert data["connected"] is False
        assert data["restored_settings"] == {
            "connection": "remote-gpu",
            "picture_connection": "remote-gpu-pictures",
        }
        assert _connection(app, "remote-gpu") is None
        assert _connection(app, "remote-gpu-pictures") is None
        assert _connection(app, "ollama") == "http://127.0.0.1:11434"
        assert _connection(app, "comfyui") == "http://127.0.0.1:8188"
        assert not (storage / "remote_gpu_restore.json").exists()
        # Nothing left to restore: a later status read changes nothing.
        assert (
            "restored_settings" not in (await client.get("/api/settings/remote-gpu/status")).json()
        )


@pytest.mark.asyncio
async def test_ui_restore_leaves_an_address_changed_by_hand(tmp_path: Path) -> None:
    """Only a field still holding the tunnel's address is put back."""
    app = create_ui_app(storage_dir=tmp_path / "uclone_storage")
    tunnel_mgr = app.state.tunnel_manager
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        await _save_local_ollama(client, app)
        with patch.object(tunnel_mgr, "connect", return_value=_dell_tunnel()):
            await client.post(
                "/api/settings/remote-gpu/connect",
                json={"host": "dell", "apply_settings": True, "sync_llm": True},
            )
        # While connected, the user points the tunnel's connection at another server by hand.
        from uclone_x.llm.connectors.saved_choice import update_connection

        update_connection(
            "remote-gpu",
            base_url="http://10.0.0.7:11434",
            path=app.state.session_manager.settings_file,
        )
        data = (await client.get("/api/settings/remote-gpu/status")).json()
        assert data["restored_settings"] == {"picture_connection": "remote-gpu-pictures"}
        assert _connection(app, "remote-gpu") == "http://10.0.0.7:11434"
        assert _connection(app, "remote-gpu-pictures") is None


@pytest.mark.asyncio
async def test_ui_startup_restores_a_record_the_last_run_left(tmp_path: Path) -> None:
    """A run stopped while connected: the next start puts the local addresses back."""
    storage = tmp_path / "uclone_storage"
    app = create_ui_app(storage_dir=storage)
    tunnel_mgr = app.state.tunnel_manager
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        await _save_local_ollama(client, app)
        with patch.object(tunnel_mgr, "connect", return_value=_dell_tunnel()):
            await client.post(
                "/api/settings/remote-gpu/connect",
                json={"host": "dell", "apply_settings": True, "sync_llm": True},
            )
    restarted = create_ui_app(storage_dir=storage)
    # The worker does not answer this time.
    unreachable = TunnelSessionStatus(host="dell", connected=False, error="Host unreachable")
    reconnect = AsyncMock(return_value=unreachable)
    with patch.object(restarted.state.tunnel_manager, "connect", reconnect):
        async with restarted.router.lifespan_context(restarted):
            await _until(lambda: reconnect.await_count > 0)
    assert _connection(restarted, "remote-gpu") is None
    assert _connection(restarted, "remote-gpu-pictures") is None
    assert _connection(restarted, "comfyui") == "http://127.0.0.1:8188"
    assert not (storage / "remote_gpu_restore.json").exists()
    # A worker that did not answer is tried again at the next start.
    assert (storage / "remote_gpu_session.json").exists()


async def _until(cond: Callable[[], bool]) -> None:
    for _ in range(200):
        if cond():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never held")


@pytest.mark.asyncio
async def test_ui_startup_reconnects_a_tunnel_left_connected(tmp_path: Path) -> None:
    """A restart makes the same connect again, asking for the ports the last run had."""
    storage = tmp_path / "uclone_storage"
    app = create_ui_app(storage_dir=storage)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        await _save_local_ollama(client, app)
        with patch.object(app.state.tunnel_manager, "connect", return_value=_dell_tunnel()):
            await client.post(
                "/api/settings/remote-gpu/connect",
                json={"host": "dell", "apply_settings": True, "sync_llm": True},
            )
    restarted = create_ui_app(storage_dir=storage)
    reconnect = AsyncMock(return_value=_dell_tunnel())
    with patch.object(restarted.state.tunnel_manager, "connect", reconnect):
        async with restarted.router.lifespan_context(restarted):
            await _until(lambda: (storage / "remote_gpu_restore.json").exists())
    reconnect.assert_awaited_once()
    assert reconnect.await_args is not None
    kwargs = reconnect.await_args.kwargs
    assert kwargs["host"] == "dell"
    assert kwargs["preferred_local_ollama_port"] == 11435
    assert kwargs["preferred_local_comfyui_port"] == 8189
    assert _connection(restarted, "remote-gpu") == "http://127.0.0.1:11435"
    assert _connection(restarted, "remote-gpu-pictures") == "http://127.0.0.1:8189"
    # Still what Disconnect would take away again.
    record = json.loads((storage / "remote_gpu_restore.json").read_text())
    assert record["applied"]["remote-gpu-pictures"] == "http://127.0.0.1:8189"


@pytest.mark.asyncio
async def test_ui_disconnect_stops_the_next_start_reconnecting(tmp_path: Path) -> None:
    """Disconnect pressed by hand: the next start leaves the worker alone."""
    storage = tmp_path / "uclone_storage"
    app = create_ui_app(storage_dir=storage)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        with patch.object(app.state.tunnel_manager, "connect", return_value=_dell_tunnel()):
            await client.post("/api/settings/remote-gpu/connect", json={"host": "dell"})
        assert (storage / "remote_gpu_session.json").exists()
        await client.post("/api/settings/remote-gpu/disconnect")
    assert not (storage / "remote_gpu_session.json").exists()
    restarted = create_ui_app(storage_dir=storage)
    reconnect = AsyncMock(return_value=_dell_tunnel())
    with patch.object(restarted.state.tunnel_manager, "connect", reconnect):
        async with restarted.router.lifespan_context(restarted):
            await asyncio.sleep(0.05)
    reconnect.assert_not_awaited()


def test_find_configured_ssh_hosts(tmp_path: Path) -> None:
    """find_configured_ssh_hosts reads ~/.ssh/config and skips wildcards and invalid patterns."""
    # Killed by: src/uclone_x/core/remote_worker.py :: if tokens[0].lower() == "host":
    # Becomes: if False:
    config_file = tmp_path / "ssh_config"
    # Missing file returns empty list
    assert find_configured_ssh_hosts(config_file) == []

    content = """
    # SSH Configuration sample
    Host macmini dell
        HostName 192.168.1.50
        User dev

    Host *
        ServerAliveInterval 60

    Host *.local
        StrictHostKeyChecking no

    Host ?wildcard
        Port 2222

    Host=server1
        User admin

    Host dell
        Port 22
    """
    config_file.write_text(content, encoding="utf-8")
    hosts = find_configured_ssh_hosts(config_file)
    assert hosts == ["dell", "macmini", "server1"]


@pytest.mark.asyncio
async def test_ssh_tunnel_disconnect_listener_fires_on_termination() -> None:
    """Disconnect listeners registered on SSHTunnelManager are invoked when process drops."""
    mgr = SSHTunnelManager()
    called = False

    def on_drop() -> None:
        nonlocal called
        called = True

    mgr.add_disconnect_listener(on_drop)

    mock_proc = MagicMock()
    wait_future: asyncio.Future[int] = asyncio.get_event_loop().create_future()
    mock_proc.wait = AsyncMock(side_effect=lambda: wait_future)
    mock_proc.returncode = None
    mock_proc.stderr = None
    mock_proc.pid = 99999

    mgr._proc = mock_proc
    mgr._session = TunnelSessionStatus(host="dell", connected=True, pid=99999)
    mgr._drain_task = asyncio.create_task(mgr._watch_and_drain(mock_proc))

    assert mgr.is_connected is True
    assert called is False

    # Simulate unexpected process termination
    mock_proc.returncode = 1
    wait_future.set_result(1)
    await asyncio.sleep(0.02)

    assert called is True
    assert mgr.is_connected is False
    assert mgr.get_status().connected is False


@pytest.mark.asyncio
async def test_tunnel_drop_immediately_restores_ui_settings(tmp_path: Path) -> None:
    """When the SSH tunnel process terminates, disconnect listener immediately restores settings."""
    storage = tmp_path / "uclone_storage"
    app = create_ui_app(storage_dir=storage)
    tunnel_mgr = app.state.tunnel_manager

    wait_future: asyncio.Future[int] = asyncio.get_event_loop().create_future()
    mock_proc = MagicMock()
    mock_proc.wait = AsyncMock(side_effect=lambda: wait_future)
    mock_proc.returncode = None
    mock_proc.stderr = None
    mock_proc.pid = 12345

    tunnel_status = TunnelSessionStatus(
        host="dell",
        connected=True,
        pid=12345,
        mappings=[
            PortMapping(service_name="ollama", remote_port=11434, local_port=11435),
            PortMapping(service_name="comfyui", remote_port=8188, local_port=8189),
        ],
        ollama_models=["qwen3:8b"],
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        await _save_local_ollama(client, app)
        with patch.object(tunnel_mgr, "connect", return_value=tunnel_status):
            resp = await client.post(
                "/api/settings/remote-gpu/connect",
                json={"host": "dell", "apply_settings": True, "sync_llm": True},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["llm_on_remote"] is True
            assert data["images_on_remote"] is True

        # Hook watcher
        tunnel_mgr._proc = mock_proc
        tunnel_mgr._session = tunnel_status
        tunnel_mgr._drain_task = asyncio.create_task(tunnel_mgr._watch_and_drain(mock_proc))

        # Check settings point to tunnel
        assert _connection(app, "remote-gpu") == "http://127.0.0.1:11435"
        assert _connection(app, "remote-gpu-pictures") == "http://127.0.0.1:8189"

        # Tunnel drops!
        mock_proc.returncode = 255
        wait_future.set_result(255)
        await asyncio.sleep(0.02)

        # Settings are restored immediately without polling status or calling disconnect
        assert _connection(app, "remote-gpu") is None
        assert _connection(app, "remote-gpu-pictures") is None
        assert _connection(app, "comfyui") == "http://127.0.0.1:8188"


@pytest.mark.asyncio
async def test_remote_gpu_hosts_endpoint(tmp_path: Path) -> None:
    """GET /api/settings/remote-gpu/hosts combines configured SSH hosts with last connected host."""
    storage = tmp_path / "uclone_storage"
    app = create_ui_app(storage_dir=storage)

    session_file = storage / "remote_gpu_session.json"
    session_file.write_text(json.dumps({"host": "macmini"}), encoding="utf-8")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        with patch(
            "uclone_x.ui.app.find_configured_ssh_hosts", return_value=["dell", "macmini", "server"]
        ):
            resp = await client.get("/api/settings/remote-gpu/hosts")
            assert resp.status_code == 200
            data = resp.json()
            assert data["hosts"] == ["macmini", "dell", "server"]


@pytest.mark.asyncio
async def test_remote_gpu_status_and_connect_routing_flags(tmp_path: Path) -> None:
    """Connect and status responses include honest llm_on_remote and images_on_remote flags."""
    storage = tmp_path / "uclone_storage"
    app = create_ui_app(storage_dir=storage)
    tunnel_mgr = app.state.tunnel_manager

    tunnel = _dell_tunnel()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        await _save_local_ollama(client, app)

        # 1. Connect with sync_llm=False -> only images on remote
        with patch.object(tunnel_mgr, "connect", return_value=tunnel):
            with patch.object(tunnel_mgr, "get_status", return_value=tunnel):
                resp = await client.post(
                    "/api/settings/remote-gpu/connect",
                    json={"host": "dell", "apply_settings": True, "sync_llm": False},
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["llm_on_remote"] is False
                assert data["images_on_remote"] is True

                status = (await client.get("/api/settings/remote-gpu/status")).json()
                assert status["llm_on_remote"] is False
                assert status["images_on_remote"] is True

        # 2. Connect with sync_llm=True -> both on remote
        with patch.object(tunnel_mgr, "connect", return_value=tunnel):
            with patch.object(tunnel_mgr, "get_status", return_value=tunnel):
                resp = await client.post(
                    "/api/settings/remote-gpu/connect",
                    json={"host": "dell", "apply_settings": True, "sync_llm": True},
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data["llm_on_remote"] is True
                assert data["images_on_remote"] is True

                status = (await client.get("/api/settings/remote-gpu/status")).json()
                assert status["llm_on_remote"] is True
                assert status["images_on_remote"] is True
