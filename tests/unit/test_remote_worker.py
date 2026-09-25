# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportPrivateUsage=false, reportUntypedFunctionDecorator=false, reportUnknownParameterType=false, reportMissingParameterType=false
"""Unit tests for remote GPU worker inspector and SSH auto-tunnel manager."""

from __future__ import annotations

import json
from pathlib import Path
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
    find_free_port,
    is_port_in_use,
    is_valid_host,
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
                json={"host": "dell", "apply_settings": True},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["connected"] is True
            assert "http://127.0.0.1:11435" in data["applied_changes"]["llm_base_url"]

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
