"""Unit tests for the cluster stack's secrets, config, CLI wiring and HTTP routes.

Nothing here talks to docker or a server: the stack's behaviour against the real compose
topology is `tests/integration/cluster_live/`. These tests cover what the shell decides on its
own — which secrets exist, which commands it runs, which routes it mounts.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from typer.testing import CliRunner

from tests.support.cluster_fakes import FakeBroker
from uclone_x.adapters.cluster.config import (
    REQUIRED_STACK_SECRETS,
    ClusterConfigError,
    ClusterRuntimeConfig,
    read_stack_env,
)
from uclone_x.adapters.cluster.valkey_cache import ValkeyClusterCache
from uclone_x.cli.commands import cluster as cluster_cli
from uclone_x.cli.commands import worker as worker_cli
from uclone_x.cli.main import app as ucx_app
from uclone_x.core.cluster_protocols import (
    RoomCommandEnvelope,
    SeatTurnRef,
    TurnCommandEnvelope,
    TurnLeaseProtocol,
    TurnLeaseToken,
)
from uclone_x.errors import LLMProviderNotConfiguredError
from uclone_x.shells.cluster.ingress_routes import ClusterIngressService
from uclone_x.shells.cluster.runtime_stack import (
    ClusterHealthReport,
    ClusterRuntimeStack,
    InfraComponentStatus,
    create_cluster_http_app,
)
from uclone_x.ui.app import create_ui_app

runner = CliRunner()


def _write_env(path: Path, **values: str) -> Path:
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8")
    return path


def _full_env(path: Path) -> Path:
    return _write_env(path, **{name: f"{name.lower()}-value" for name in REQUIRED_STACK_SECRETS})


def test_ensure_stack_env_generates_each_secret_once_and_keeps_existing_ones(
    tmp_path: Path,
) -> None:
    """`ucx cluster up` generates missing secrets into a 0600 file and never rotates existing ones.

    Killed by: src/uclone_x/cli/commands/cluster.py :: generated = tuple(name for name in REQUIRED_STACK_SECRETS if not current.get(name))
    Becomes: generated = tuple(REQUIRED_STACK_SECRETS)
    """
    env_file = tmp_path / ".ucx-cluster" / "stack.env"
    env_file.parent.mkdir(parents=True)
    _write_env(env_file, UCX_POSTGRES_PASSWORD="kept")

    generated = cluster_cli.ensure_stack_env(env_file)
    values = read_stack_env(env_file)

    assert set(generated) == set(REQUIRED_STACK_SECRETS) - {"UCX_POSTGRES_PASSWORD"}
    assert values["UCX_POSTGRES_PASSWORD"] == "kept"
    for name in generated:
        assert len(values[name]) == 64  # 32 random bytes, hex
    assert len({values[name] for name in generated}) == len(generated)
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600

    assert cluster_cli.ensure_stack_env(env_file) == ()
    assert read_stack_env(env_file) == values


def test_config_refuses_a_stack_env_missing_a_secret(tmp_path: Path) -> None:
    """There is no default secret: a missing one raises and names itself.

    Killed by: src/uclone_x/adapters/cluster/config.py :: missing = [name for name in REQUIRED_STACK_SECRETS if not env.get(name)]
    Becomes: missing = []
    """
    env_file = _full_env(tmp_path / "stack.env")
    text = env_file.read_text(encoding="utf-8").replace(
        "UCX_CENTRIFUGO_HMAC_SECRET=ucx_centrifugo_hmac_secret-value\n", ""
    )
    env_file.write_text(text, encoding="utf-8")

    with pytest.raises(ClusterConfigError, match="UCX_CENTRIFUGO_HMAC_SECRET"):
        ClusterRuntimeConfig.from_stack_env(env_file)
    with pytest.raises(ClusterConfigError, match="ucx cluster up"):
        ClusterRuntimeConfig.from_stack_env(tmp_path / "absent.env")


def test_config_maps_announced_valkey_hostnames_to_published_ports(tmp_path: Path) -> None:
    """Valkey nodes announce `valkey-N:6379`; a host-side client reaches them on the stack's ports.

    Killed by: src/uclone_x/adapters/cluster/config.py :: f"valkey-{i}:6379": f"{host}:{p}" for i, p in enumerate(valkey_ports, start=1)
    Becomes: f"valkey-{i}:6379": f"{host}:{p}" for i, p in enumerate(valkey_ports, start=0)
    """
    env_file = _full_env(tmp_path / "stack.env")
    with env_file.open("a", encoding="utf-8") as handle:
        handle.write("UCX_VALKEY_PORT_2=27002\nUCX_POSTGRES_PORT=25432\n")

    config = ClusterRuntimeConfig.from_stack_env(env_file)

    assert config.valkey_address_map["valkey-1:6379"] == "127.0.0.1:17001"
    assert config.valkey_address_map["valkey-2:6379"] == "127.0.0.1:27002"
    assert len(config.valkey_address_map) == 6
    assert config.postgres_dsn.endswith("@127.0.0.1:25432/uclone_cluster")
    assert "ucx_postgres_password-value" in config.postgres_dsn


def test_compose_commands_pass_the_stack_env_and_scope_volume_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every compose call reads the stack env file; only `--volumes` removes data.

    Killed by: src/uclone_x/cli/commands/cluster.py :: return _compose("down", "--volumes")
    Becomes: return _compose("stop", *_DOCKER_SERVICES)
    """
    compose = tmp_path / "docker-compose.yml"
    compose.write_text("services: {}\n", encoding="utf-8")
    monkeypatch.setenv("UCX_CLUSTER_COMPOSE_FILE", str(compose))
    monkeypatch.delenv("UCX_CLUSTER_ENV_FILE", raising=False)

    stop = cluster_cli._compose_down_command(False)  # pyright: ignore[reportPrivateUsage]
    down = cluster_cli._compose_down_command(True)  # pyright: ignore[reportPrivateUsage]

    env_args = ["--env-file", str(tmp_path / ".ucx-cluster" / "stack.env")]
    for cmd in (stop, down):
        assert cmd[:4] == ["docker", "compose", "-f", str(compose.resolve())]
        assert cmd[4:6] == env_args
    assert stop[6] == "stop" and "valkey-6" in stop
    assert down[6:] == ["down", "--volumes"]


def test_ui_app_refuses_the_removed_memory_cluster_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`UCLONE_CLUSTER_MODE=memory` (the in-process stand-in) fails instead of silently running local.

    Killed by: src/uclone_x/ui/app.py :: elif cluster_mode_env:
    Becomes: elif False:
    """
    monkeypatch.setenv("UCLONE_CLUSTER_MODE", "memory")
    with pytest.raises(ValueError, match="memory"):
        create_ui_app(storage_dir=tmp_path / "sessions", fallback_to_mock=True)


class _RecordingLease(TurnLeaseProtocol):
    def __init__(self) -> None:
        self.tombstones: list[tuple[str, str]] = []

    async def acquire(
        self, conversation_id: str, turn_id: str, worker_id: str, ttl_seconds: int | None = None
    ) -> TurnLeaseToken:
        raise AssertionError("ingress never takes a lease")

    async def renew(self, token: TurnLeaseToken) -> bool:
        return False

    async def release(self, token: TurnLeaseToken) -> None:
        return None

    async def write_cancel_tombstone(
        self, conversation_id: str, turn_id: str, ttl_seconds: int = 300
    ) -> None:
        self.tombstones.append((conversation_id, turn_id))

    async def is_cancelled(self, conversation_id: str, turn_id: str) -> bool:
        return (conversation_id, turn_id) in self.tombstones


async def test_cluster_http_routes_publish_commands_and_mint_channel_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ingress routes answer 202 with a token Centrifugo accepts for exactly that channel.

    Killed by: src/uclone_x/shells/cluster/runtime_stack.py :: token = _mint_subscription_jwt(sub=cmd.user_id, channel=accepted.sse_channel)
    Becomes: token = _mint_subscription_jwt(sub=cmd.user_id, channel="ucx:session:other")
    """
    config = ClusterRuntimeConfig.from_stack_env(_full_env(tmp_path / "stack.env"))
    stack = ClusterRuntimeStack.create_docker_stack(config=config)
    broker, lease = FakeBroker(), _RecordingLease()
    stack.ingress = ClusterIngressService(broker, lease)
    executed: list[Any] = []

    async def record_execute(**fields: Any) -> Any:
        executed.append(fields)
        raise AssertionError("a worker's id reached the turn this route runs")

    monkeypatch.setattr(stack, "execute_turn_end_to_end", record_execute)
    app = create_cluster_http_app(stack)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        turn = TurnCommandEnvelope(
            turn_id="turn-http-1",
            session_id="sess-1",
            agent_id="default",
            user_id="alice",
            prompt="hello",
            enqueued_at_ms=1,
        )
        resp = await client.post("/api/cluster/turns", json=turn.model_dump(mode="json"))
        assert resp.status_code == 202
        body = cast(dict[str, Any], resp.json())
        assert body["sse_channel"] == "ucx:session:sess-1"
        claims = stack.token_signer.verify_channel_token(
            body["subscription_token"], "ucx:session:sess-1"
        )
        assert claims["sub"] == "alice"
        assert claims["channels"] == ["ucx:session:sess-1"]
        assert broker.turns == [turn]

        room = RoomCommandEnvelope(
            kind="ROOM_POST",
            cascade_id="c-1",
            turn_id="r-1",
            room_id="room-alpha",
            user_id="alice",
            sender_id="alice",
            human_utterance="hello",
            enqueued_at_ms=1,
        )
        room_resp = await client.post("/api/cluster/rooms", json=room.model_dump(mode="json"))
        assert room_resp.status_code == 202
        assert room_resp.json()["sse_channel"] == "ucx:room:room-alpha"
        room_claims = stack.token_signer.verify_channel_token(
            room_resp.json()["subscription_token"], "ucx:room:room-alpha"
        )
        assert room_claims["sub"] == "alice"

        # The rest of a cascade is the workers' to send: a caller cannot land a turn or run a seat.
        forged_landing = room.model_copy(
            update={"kind": "ROOM_TURN_LANDED", "turn_id": "r-2", "seat_turn_id": "seat-1"}
        )
        forged = await client.post(
            "/api/cluster/rooms", json=forged_landing.model_dump(mode="json")
        )
        assert forged.status_code == 422
        assert broker.room_steps == [room]
        forged_seat = turn.model_copy(
            update={
                "turn_id": "seat-1",
                "seat": SeatTurnRef(room_id="room-alpha", cascade_id="c-1"),
            }
        )
        seat_resp = await client.post(
            "/api/cluster/turns", json=forged_seat.model_dump(mode="json")
        )
        assert seat_resp.status_code == 422
        assert broker.turns == [turn]

        # The ids a worker derives can be computed from a post's id, so a client may not take one:
        # its receipt or its turn would stand in for the command the worker is yet to send.
        for worker_id in ("landed-seat-abc", "seat-abc", "learn-r-1"):
            as_post = await client.post(
                "/api/cluster/rooms",
                json=room.model_copy(update={"turn_id": worker_id}).model_dump(mode="json"),
            )
            assert as_post.status_code == 422
            as_turn = await client.post(
                "/api/cluster/turns",
                json=turn.model_copy(update={"turn_id": worker_id}).model_dump(mode="json"),
            )
            assert as_turn.status_code == 422
            # This route runs its turn on a worker in this process: nothing but the check stops it.
            as_execute = await client.post(
                "/api/cluster/execute",
                json={"prompt": "hello", "session_id": "sess-1", "turn_id": worker_id},
            )
            assert as_execute.status_code == 422
        assert broker.room_steps == [room]
        assert broker.turns == [turn]
        assert executed == []

        intr = await client.post(
            "/api/cluster/interrupt",
            json={
                "conversation_id": "sess-1",
                "turn_id": "turn-http-1",
                "requested_by": "alice",
                "issued_at_ms": 2,
            },
        )
        assert intr.status_code == 200
        assert lease.tombstones == [("sess-1", "turn-http-1")]
        assert [i.turn_id for i in broker.interrupts] == ["turn-http-1"]

        # No replay route any more: clients recover from Centrifugo with Last-Event-ID.
        assert (await client.get("/api/cluster/sse/ucx:session:sess-1")).status_code == 404


def test_ucx_ui_and_cluster_up_down_report_mode_and_components(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`ucx ui up` stays standalone; `ucx cluster up` reports the compose cluster and its checks.

    Killed by: src/uclone_x/cli/commands/cluster.py :: "mode": "cluster",
    Becomes: "mode": "standalone",
    """
    compose = tmp_path / "docker-compose.yml"
    compose.write_text("services: {}\n", encoding="utf-8")
    monkeypatch.setenv("UCX_CLUSTER_COMPOSE_FILE", str(compose))
    monkeypatch.delenv("UCX_CLUSTER_ENV_FILE", raising=False)

    def _ok(name: str) -> InfraComponentStatus:
        return InfraComponentStatus(name=name, endpoint="x", healthy=True, latency_ms=1.0)

    def _fake_start(wait_seconds: int = 30) -> tuple[Any, ClusterHealthReport]:
        _ = wait_seconds
        cluster_cli.ensure_stack_env(tmp_path / ".ucx-cluster" / "stack.env")
        components = tuple(_ok(n) for n in ("valkey", "postgres", "kafka", "centrifugo"))
        return object(), ClusterHealthReport(all_healthy=True, components=components)

    async def _fake_verify(stack: Any) -> tuple[InfraComponentStatus, ...]:
        _ = stack
        return tuple(_ok(f"{i}. check") for i in range(1, 6))

    ran: list[list[str]] = []

    def _fake_run(cmd: list[str], *a: Any, **kw: Any) -> subprocess.CompletedProcess[str]:
        ran.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(cluster_cli, "_start_and_bootstrap_docker_cluster", _fake_start)
    monkeypatch.setattr(cluster_cli, "_verify_and_close", _fake_verify)

    def _fake_ollama() -> InfraComponentStatus:
        return _ok("local_llm_ollama")

    def _fake_ui(**kw: Any) -> InfraComponentStatus:
        _ = kw
        return _ok("web_ui_server")

    def _fake_ui_checks(
        host: str, port: int, cluster_mode: str | None = None
    ) -> tuple[InfraComponentStatus, ...]:
        _ = (host, port, cluster_mode)
        return (_ok("ui check"),)

    def _fake_stop(port: int) -> tuple[tuple[int, ...], str | None]:
        _ = port
        return (4242,), None

    monkeypatch.setattr(cluster_cli, "_probe_local_ollama", _fake_ollama)
    monkeypatch.setattr(cluster_cli, "_ensure_ui_daemon_running", _fake_ui)
    monkeypatch.setattr(cluster_cli, "_verify_ui_endpoints", _fake_ui_checks)
    monkeypatch.setattr(cluster_cli, "_stop_managed_port", _fake_stop)
    monkeypatch.setattr(cluster_cli.subprocess, "run", _fake_run)

    standalone = runner.invoke(ucx_app, ["ui", "up", "--json"])
    assert standalone.exit_code == 0, standalone.output
    assert json.loads(standalone.stdout.strip())["mode"] == "standalone"

    cluster = runner.invoke(ucx_app, ["cluster", "up", "--json"])
    assert cluster.exit_code == 0, cluster.output
    data = cast(dict[str, Any], json.loads(cluster.stdout.strip()))
    assert data["mode"] == "cluster"
    assert data["backend"] == "docker"
    assert data["all_healthy"] is True
    assert [c["name"] for c in data["components"]][:4] == [
        "valkey",
        "postgres",
        "kafka",
        "centrifugo",
    ]
    assert len(data["checks"]) == 6

    assert "--backend" not in runner.invoke(ucx_app, ["cluster", "up", "--help"]).output

    down = runner.invoke(ucx_app, ["cluster", "down", "--json"])
    assert down.exit_code == 0, down.output
    assert json.loads(down.stdout.strip())["docker_cluster_stopped"] is True
    assert ran[-1][:2] == ["docker", "compose"] and "stop" in ran[-1]
    assert os.environ.get("UCLONE_CLUSTER_MODE") is None


def test_generating_secrets_keeps_stdout_free_for_the_json_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The first `ucx cluster up --json` generates secrets; its notice must not corrupt stdout.

    Killed by: src/uclone_x/cli/commands/cluster.py :: Console(stderr=True).print(f"[dim]• Generated stack secrets into {_stack_env_file()}[/dim]")
    Becomes: console.print(f"[dim]• Generated stack secrets into {_stack_env_file()}[/dim]")
    """
    compose = tmp_path / "docker-compose.yml"
    compose.write_text("services: {}\n", encoding="utf-8")
    monkeypatch.setenv("UCX_CLUSTER_COMPOSE_FILE", str(compose))
    monkeypatch.delenv("UCX_CLUSTER_ENV_FILE", raising=False)

    class _ReadyStack:
        async def bootstrap_schema(self) -> None:
            return None

        async def check_health(self) -> ClusterHealthReport:
            return ClusterHealthReport(all_healthy=True, components=())

        async def close(self) -> None:
            return None

    def _fake_run(cmd: list[str], *a: Any, **kw: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    def _fake_build(num_workers: int = 2) -> Any:
        _ = num_workers
        return _ReadyStack()

    monkeypatch.setattr(cluster_cli.subprocess, "run", _fake_run)
    monkeypatch.setattr(cluster_cli, "_build_stack", _fake_build)

    _, report = cluster_cli._start_and_bootstrap_docker_cluster(wait_seconds=1)  # pyright: ignore[reportPrivateUsage]

    captured = capsys.readouterr()
    assert report.all_healthy is True
    assert captured.out == ""
    assert "Generated stack secrets" in captured.err
    assert (tmp_path / ".ucx-cluster" / "stack.env").is_file()


class _ReplicaReport(ValkeyClusterCache):
    def __init__(self, primaries: int, ready: int) -> None:
        self._report = (primaries, ready)

    async def ping(self) -> bool:
        return True

    async def replication_status(self) -> tuple[int, int]:
        return self._report


@pytest.mark.parametrize(
    ("primaries", "ready", "healthy"),
    [(3, 3, True), (3, 2, False), (0, 0, False)],
)
async def test_valkey_health_needs_a_replica_behind_every_primary(
    tmp_path: Path, primaries: int, ready: int, healthy: bool
) -> None:
    """The replica guard (#2468): `ucx cluster up` waits on it for failover readiness.

    Killed by: src/uclone_x/shells/cluster/runtime_stack.py :: if primaries == 0 or ready < primaries:
    Becomes: if False:
    """
    path = tmp_path / "stack.env"
    cluster_cli.ensure_stack_env(path)
    stack = ClusterRuntimeStack.create_docker_stack(
        config=ClusterRuntimeConfig.from_stack_env(path)
    )
    stack.cache = _ReplicaReport(primaries, ready)
    status = await stack._probe("valkey", "nodes", stack._probe_valkey)  # pyright: ignore[reportPrivateUsage]
    assert status.healthy is healthy
    assert (f"{ready}/{primaries}" in status.detail) is True


def test_ucx_worker_refuses_to_start_on_a_workspace_it_cannot_write_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/cli/commands/worker.py :: _refuse_an_unwritable_workspace(workspace.resolve())
    Becomes: pass
    """
    served: list[object] = []

    def _serve(**fields: object) -> None:
        served.append(fields)

    def _run(_: object) -> None:
        return None

    monkeypatch.setattr(worker_cli, "_serve", _serve)
    monkeypatch.setattr(worker_cli.asyncio, "run", _run)
    env_file = tmp_path / "stack.env"
    cluster_cli.ensure_stack_env(env_file)
    monkeypatch.setenv("UCX_CLUSTER_ENV_FILE", str(env_file))
    mount = tmp_path / "mount"
    mount.mkdir()

    mount.chmod(0o500)
    try:
        refused = CliRunner().invoke(ucx_app, ["worker", "--workspace", str(mount)])
    finally:
        mount.chmod(0o700)
    assert refused.exit_code == 1
    assert f"cannot write to --workspace {mount.resolve()}" in refused.output
    assert served == []

    # A `users` folder already there is no proof of anything: the file has to go in.
    (mount / "users").mkdir()
    (mount / "users").chmod(0o500)
    try:
        refused = CliRunner().invoke(ucx_app, ["worker", "--workspace", str(mount)])
    finally:
        (mount / "users").chmod(0o700)
    assert refused.exit_code == 1
    assert served == []

    # A folder that is really a file is refused the same way, and a good one is not.
    occupied = tmp_path / "occupied"
    occupied.write_text("not a folder", encoding="utf-8")
    assert CliRunner().invoke(ucx_app, ["worker", "--workspace", str(occupied)]).exit_code == 1
    assert served == []

    fresh = tmp_path / "not-yet" / "there"
    assert CliRunner().invoke(ucx_app, ["worker", "--workspace", str(fresh)]).exit_code == 0
    assert len(served) == 1
    assert list((fresh / "users").iterdir()) == []  # nothing left behind


def test_ucx_worker_refuses_to_start_with_no_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No stand-in model: a worker with no provider stops before it joins the consumer group."""
    env_file = tmp_path / "stack.env"
    cluster_cli.ensure_stack_env(env_file)
    monkeypatch.setenv("UCX_CLUSTER_ENV_FILE", str(env_file))
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in (
        "LLM_PROVIDER",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OLLAMA_BASE_URL",
        "OLLAMA_FAST_BASE_URL",
        "LOCAL_LLM_BASE_URL",
        "OLLAMA_HOST",
        "VLLM_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)

    result = CliRunner().invoke(ucx_app, ["worker"])

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMProviderNotConfiguredError)
