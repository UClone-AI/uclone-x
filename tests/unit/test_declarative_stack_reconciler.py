"""Unit and regression tests for `daemon_supervisor`, `StackReconciler`, and `ucx up`/`down` (#2478)."""

from __future__ import annotations

import json
import os
import socket
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from uclone_x.adapters.cluster.config import REQUIRED_STACK_SECRETS
from uclone_x.cli.main import app as ucx_app
from uclone_x.shells import daemon_supervisor, reconciler, ui_process
from uclone_x.shells.daemon_supervisor import (
    DaemonNotIdentifiedError,
    DaemonRecord,
    daemon_record_path,
    daemon_staleness,
    list_daemon_records,
    new_daemon_record,
    read_daemon_record,
    remove_daemon_record,
    stop_daemon,
    write_daemon_record,
)
from uclone_x.shells.reconciler import (
    CLOUD_PROVIDER_SECRET_VARS,
    DOCKER_CLUSTER_SERVICES,
    AppliedResourceResult,
    ResourcePlan,
    StackApplyResult,
    StackDownResult,
    StackPlan,
    StackReconciler,
    StackSpec,
    build_child_env,
    compute_python_tree_digest,
    write_deploy_state,
)
from uclone_x.shells.ui_process import ProcessIdentity
from uclone_x.ui import build_inputs


def _make_mini_repo(root: Path) -> StackSpec:
    """Create a minimal synthetic source tree exercising all 5 StackReconciler resource roots."""
    repo = root / "repo"
    (repo / "src" / "uclone_x" / "adapters" / "cluster").mkdir(parents=True)
    (repo / "src" / "uclone_x" / "shells" / "cluster").mkdir(parents=True)
    (repo / "src" / "uclone_x" / "llm" / "connectors").mkdir(parents=True)
    (repo / "src" / "uclone_x" / "core").mkdir(parents=True)
    (repo / "src" / "uclone_x" / "agent").mkdir(parents=True)
    (repo / "src" / "uclone_x" / "ui").mkdir(parents=True)
    (repo / "src" / "uclone_x" / "room").mkdir(parents=True)
    (repo / "src" / "uclone_x" / "cli" / "commands").mkdir(parents=True)
    (repo / "src" / "uclone_x" / "ui_static" / "assets").mkdir(parents=True)
    (repo / "examples" / "story" / "src").mkdir(parents=True)
    (repo / "deploy" / "centrifugo").mkdir(parents=True)
    (repo / "frontend" / "src").mkdir(parents=True)

    compose_file = repo / "docker-compose.yml"
    compose_file.write_text("services:\n  kafka: {}\n", encoding="utf-8")
    (repo / "deploy" / "centrifugo" / "config.json").write_text("{}", encoding="utf-8")
    (repo / "src" / "uclone_x" / "adapters" / "cluster" / "postgres_pool.py").write_text(
        "# schema v1\n", encoding="utf-8"
    )
    (repo / "src" / "uclone_x" / "adapters" / "cluster" / "kafka_broker.py").write_text(
        "# topics v1\n", encoding="utf-8"
    )
    (repo / "src" / "uclone_x" / "llm" / "gateway.py").write_text(
        "# gateway v1\n", encoding="utf-8"
    )
    (repo / "src" / "uclone_x" / "core" / "remote_worker.py").write_text(
        "# remote v1\n", encoding="utf-8"
    )
    (repo / "src" / "uclone_x" / "shells" / "gateway_server.py").write_text(
        "# server v1\n", encoding="utf-8"
    )
    (repo / "src" / "uclone_x" / "ui" / "usage.py").write_text("# usage v1\n", encoding="utf-8")
    (repo / "src" / "uclone_x" / "agent" / "turn.py").write_text("# agent v1\n", encoding="utf-8")
    (repo / "src" / "uclone_x" / "ui" / "app.py").write_text("# ui v1\n", encoding="utf-8")
    (repo / "src" / "uclone_x" / "room" / "service.py").write_text("# room v1\n", encoding="utf-8")
    (repo / "src" / "uclone_x" / "cli" / "commands" / "worker.py").write_text(
        "# worker v1\n", encoding="utf-8"
    )
    (repo / "src" / "uclone_x" / "llm" / "connectors" / "gateway_client.py").write_text(
        "# client v1\n", encoding="utf-8"
    )
    (repo / "examples" / "story" / "src" / "ext.py").write_text("# story v1\n", encoding="utf-8")

    # Frontend sources + matching recorded digest
    (repo / "frontend" / "package.json").write_text('{"name": "ucx-fe"}\n', encoding="utf-8")
    (repo / "frontend" / "src" / "App.tsx").write_text("export const v = 1;\n", encoding="utf-8")
    (repo / "src" / "uclone_x" / "ui_static" / "index.html").write_text(
        "<html></html>\n", encoding="utf-8"
    )
    build_inputs.write_record(repo / "frontend", repo / "src" / "uclone_x")

    stack_env = repo / ".ucx-cluster" / "stack.env"
    state_file = repo / ".ucx-cluster" / "deploy-state.json"
    return StackSpec(
        repo_root=repo,
        compose_file=compose_file,
        stack_env_file=stack_env,
        state_file=state_file,
        ui_host="127.0.0.1",
        ui_port=15180,
        gateway_host="127.0.0.1",
        gateway_port=18765,
        remote_host="dell",
        workers=2,
        workspace_dir=repo,
        wait_seconds=5,
        source_checkout=True,
    )


def _install_healthy_probes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gw_detail: str = "pid=5100, tunnel=dell preserved",
) -> None:
    def probe_docker(_self: StackReconciler) -> tuple[bool, bool, str]:
        return True, True, "healthy"

    def probe_gw(_self: StackReconciler) -> tuple[tuple[int, ...], bool, str]:
        return (5100,), True, gw_detail

    def probe_wk(_self: StackReconciler) -> tuple[tuple[int, ...], bool, str]:
        return (5201, 5202), True, "2 worker(s) running"

    def probe_ui(_self: StackReconciler) -> tuple[tuple[int, ...], bool, str]:
        return (5300,), True, "pid=5300, cluster_mode=docker"

    monkeypatch.setattr(StackReconciler, "_probe_docker_infra", probe_docker)
    monkeypatch.setattr(StackReconciler, "_probe_model_gateway", probe_gw)
    monkeypatch.setattr(StackReconciler, "_probe_agent_workers", probe_wk)
    monkeypatch.setattr(StackReconciler, "_probe_web_ui", probe_ui)


def test_daemon_supervisor_write_read_list_and_instance_guarded_remove(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`daemon_supervisor` writes `0600` records, lists by kind, guards `instance`, and rejects symlinks.

    Killed by: src/uclone_x/shells/daemon_supervisor.py :: if current is not None and current.instance == record.instance:
    Becomes:   if current is not None and current.instance != record.instance:
    """
    state_dir = tmp_path / "ui-state"
    monkeypatch.setenv("UCLONE_UI_STATE_DIR", str(state_dir))

    rec0, problem0 = new_daemon_record("worker", "0", pid=os.getpid())
    assert problem0 is None
    path0 = write_daemon_record(rec0)
    assert stat.S_IMODE(path0.stat().st_mode) == 0o600
    assert stat.S_IMODE(state_dir.stat().st_mode) == 0o700

    rec1, _ = new_daemon_record("worker", "1", pid=os.getpid())
    write_daemon_record(rec1)
    gw_rec, _ = new_daemon_record("gateway", "8765", pid=os.getpid(), port=8765, host="127.0.0.1")
    write_daemon_record(gw_rec)

    listed_workers = list_daemon_records("worker")
    assert [r.slot for r in listed_workers] == ["0", "1"]
    loaded_gw = read_daemon_record("gateway", "8765")
    assert loaded_gw is not None and loaded_gw.port == 8765

    # Removing with a stale `instance` token MUST keep the newer slot record intact.
    stale_copy = DaemonRecord(
        kind="worker",
        slot="0",
        pid=rec0.pid,
        instance="different_instance_token",
        process=rec0.process,
    )
    remove_daemon_record(stale_copy)
    assert read_daemon_record("worker", "0") is not None

    # Removing with the matching `instance` deletes the record.
    remove_daemon_record(rec0)
    assert read_daemon_record("worker", "0") is None

    # Symlink record is refused with DaemonNotIdentifiedError.
    symlink_target = tmp_path / "forged.json"
    symlink_target.write_text("{}", encoding="utf-8")
    sym_path = daemon_record_path("gateway", "9999")
    sym_path.symlink_to(symlink_target)
    with pytest.raises(DaemonNotIdentifiedError, match="symlink"):
        read_daemon_record("gateway", "9999")


def test_daemon_supervisor_detects_pid_reuse_and_refuses_to_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`daemon_staleness` and `stop_daemon` never signal a live PID whose `lstart`/`command` drifted.

    Killed by: src/uclone_x/shells/daemon_supervisor.py :: if current.started != record.process.started or current.command != record.process.command:
    Becomes:   if current.started == record.process.started and current.command == record.process.command:
    """
    monkeypatch.setenv("UCLONE_UI_STATE_DIR", str(tmp_path / "state"))

    reused_record = DaemonRecord(
        kind="worker",
        slot="0",
        pid=os.getpid(),
        instance="inst-reused",
        process=ProcessIdentity(
            started="Mon Jan  1 00:00:00 1999",
            command="/usr/bin/completely-different-binary",
        ),
    )
    rec_path = write_daemon_record(reused_record)
    assert daemon_staleness(reused_record) == "now belongs to another process"

    signalled: list[tuple[int, int]] = []
    real_kill = os.kill

    def spy_kill(pid: int, sig: int) -> None:
        if sig != 0:
            signalled.append((pid, sig))
            return
        real_kill(pid, sig)

    monkeypatch.setattr(daemon_supervisor.os, "kill", spy_kill)
    outcome = stop_daemon("worker", "0")
    assert outcome.stopped_pids == ()
    assert outcome.stale_pid == os.getpid()
    assert outcome.stale_reason == "now belongs to another process"
    assert signalled == []
    assert not rec_path.exists()


def test_daemon_supervisor_refuses_unrecorded_port_listener_and_stops_owned_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`stop_daemon` refuses an unrecorded port listener and cleanly terminates a verified child.

    Killed by: src/uclone_x/shells/daemon_supervisor.py :: if check_port is not None and _loopback_accepts_connections(check_port):
    Becomes:   if check_port is None and _loopback_accepts_connections(check_port):
    """
    monkeypatch.setenv("UCLONE_UI_STATE_DIR", str(tmp_path / "state"))

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        occupied_port = int(listener.getsockname()[1])
        with pytest.raises(DaemonNotIdentifiedError, match="refusing to signal an unknown process"):
            stop_daemon("gateway", str(occupied_port), expected_port=occupied_port)

    # Spawn a real sleeper subprocess, record its verified OS identity, and stop it cleanly.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        rec, problem = new_daemon_record("worker", "7", pid=child.pid)
        assert problem is None
        write_daemon_record(rec)
        assert daemon_staleness(rec) is None

        out = stop_daemon("worker", "7", terminate_wait_s=3.0)
        child.wait(timeout=3.0)
        assert out.stopped_pids == (child.pid,)
        assert read_daemon_record("worker", "7") is None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=2.0)


def test_build_child_env_strips_cloud_secrets_and_pins_worktree_pythonpath(tmp_path: Path) -> None:
    """`build_child_env` strips provider secrets for workers/UI while retaining them for gateway.

    Killed by: src/uclone_x/shells/reconciler.py :: if strip_provider_secrets:
    Becomes:   if not strip_provider_secrets:
    """
    spec = _make_mini_repo(tmp_path)
    base_env = {
        "PATH": "/usr/bin",
        "PYTHONPATH": "/existing/path",
        "OPENAI_API_KEY": "sk-openai-secret",
        "GEMINI_API_KEY": "gem-secret",
        "ANTHROPIC_API_KEY": "ant-secret",
        "LLM_PROVIDER": "openai",
    }

    worker_env = build_child_env(
        spec,
        base_env=base_env,
        strip_provider_secrets=True,
        extra_env={"UCX_GATEWAY_URL": spec.gateway_url},
    )
    for var in CLOUD_PROVIDER_SECRET_VARS:
        assert var not in worker_env
    assert worker_env["UCX_GATEWAY_URL"] == spec.gateway_url
    py_parts = worker_env["PYTHONPATH"].split(os.pathsep)
    assert py_parts[0] == str((spec.repo_root / "src").resolve())
    assert py_parts[1] == str((spec.repo_root / "examples" / "story" / "src").resolve())
    assert "/existing/path" in py_parts

    gw_env = build_child_env(spec, base_env=base_env, strip_provider_secrets=False)
    assert gw_env["OPENAI_API_KEY"] == "sk-openai-secret"
    assert gw_env["GEMINI_API_KEY"] == "gem-secret"


@dataclass
class _ClusterSimState:
    docker_up: bool = False
    gw_pid: int = 0
    worker_pids: tuple[int, ...] = ()
    ui_pid: int = 0
    compose_calls: int = 0
    gw_starts: int = 0
    worker_starts: int = 0
    ui_starts: int = 0


def test_reconciler_cold_start_converges_then_second_plan_is_all_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cold start applies needed resources once; an immediate re-run marks all 5 `= unchanged`.

    Killed by: src/uclone_x/shells/reconciler.py :: if docker_ok and docker_prev_fp == docker_fp:
    Becomes:   if docker_ok and docker_prev_fp != docker_fp:
    """
    spec = _make_mini_repo(tmp_path)
    fake_pkg_init = spec.repo_root / "src" / "uclone_x" / "shells" / "reconciler.py"
    fake_pkg_init.parent.mkdir(parents=True, exist_ok=True)
    fake_pkg_init.write_text("# stub\n", encoding="utf-8")
    monkeypatch.setattr(reconciler, "__file__", str(fake_pkg_init))

    sim = _ClusterSimState()

    def fake_probe_docker(_self: StackReconciler) -> tuple[bool, bool, str]:
        if sim.docker_up:
            return True, True, "5/5 cluster services healthy"
        return False, False, "stack secrets not initialized yet"

    def fake_probe_gw(_self: StackReconciler) -> tuple[tuple[int, ...], bool, str]:
        if sim.gw_pid:
            return (sim.gw_pid,), True, f"pid={sim.gw_pid}, tunnel=dell preserved"
        return (), False, "not running"

    def fake_probe_wk(self: StackReconciler) -> tuple[tuple[int, ...], bool, str]:
        if len(sim.worker_pids) == self.spec.workers:
            return sim.worker_pids, True, "2 worker(s) running"
        return (), False, "not running"

    def fake_probe_ui(_self: StackReconciler) -> tuple[tuple[int, ...], bool, str]:
        if sim.ui_pid:
            return (sim.ui_pid,), True, f"pid={sim.ui_pid}, cluster_mode=docker"
        return (), False, "not running"

    def fake_apply_docker(
        self: StackReconciler, plan: ResourcePlan, saved: dict[str, dict[str, Any]]
    ) -> AppliedResourceResult:
        if plan.action == "unchanged":
            return AppliedResourceResult(
                name="docker_infra",
                action="unchanged",
                healthy=True,
                duration_ms=1.0,
                detail=plan.reason,
            )
        sim.compose_calls += 1
        spec.stack_env_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        spec.stack_env_file.write_text(
            "".join(f"{k}=secretval\n" for k in REQUIRED_STACK_SECRETS), encoding="utf-8"
        )
        sim.docker_up = True
        saved["docker_infra"] = self.snapshot_desired_state()["docker_infra"]
        return AppliedResourceResult(
            name="docker_infra",
            action=plan.action,
            healthy=True,
            duration_ms=10.0,
            detail="bootstrapped",
        )

    def fake_apply_gw(
        _self: StackReconciler, plan: ResourcePlan, saved: dict[str, dict[str, Any]]
    ) -> AppliedResourceResult:
        if plan.action == "unchanged":
            return AppliedResourceResult(
                name="model_gateway",
                action="unchanged",
                healthy=True,
                duration_ms=0.5,
                detail=plan.reason,
                pids=plan.pids,
            )
        sim.gw_starts += 1
        sim.gw_pid = 4100
        saved["model_gateway"] = {
            "fingerprint": plan.desired_fingerprint,
            "inputs": dict(plan.inputs),
            "pids": [4100],
        }
        return AppliedResourceResult(
            name="model_gateway",
            action=plan.action,
            healthy=True,
            duration_ms=5.0,
            detail="started",
            pids=(4100,),
        )

    def fake_apply_wk(
        self: StackReconciler,
        plan: ResourcePlan,
        saved: dict[str, dict[str, Any]],
        *,
        stack_env_sha: str,
    ) -> AppliedResourceResult:
        _ = stack_env_sha
        snap = self.snapshot_desired_state()["agent_workers"]
        if plan.action == "unchanged" and snap["fingerprint"] == plan.desired_fingerprint:
            return AppliedResourceResult(
                name="agent_workers",
                action="unchanged",
                healthy=True,
                duration_ms=0.5,
                detail=plan.reason,
                pids=plan.pids,
            )
        sim.worker_starts += 1
        sim.worker_pids = (4201, 4202)
        saved["agent_workers"] = {**snap, "pids": [4201, 4202]}
        return AppliedResourceResult(
            name="agent_workers",
            action=plan.action,
            healthy=True,
            duration_ms=5.0,
            detail="2 workers started",
            pids=(4201, 4202),
        )

    def fake_apply_ui(
        self: StackReconciler,
        plan: ResourcePlan,
        saved: dict[str, dict[str, Any]],
        *,
        stack_env_sha: str,
    ) -> AppliedResourceResult:
        _ = stack_env_sha
        snap = self.snapshot_desired_state()["web_ui_server"]
        if plan.action == "unchanged" and snap["fingerprint"] == plan.desired_fingerprint:
            return AppliedResourceResult(
                name="web_ui_server",
                action="unchanged",
                healthy=True,
                duration_ms=0.5,
                detail=plan.reason,
                pids=plan.pids,
            )
        sim.ui_starts += 1
        sim.ui_pid = 4300
        saved["web_ui_server"] = {**snap, "pids": [4300]}
        return AppliedResourceResult(
            name="web_ui_server",
            action=plan.action,
            healthy=True,
            duration_ms=5.0,
            detail="ui started",
            pids=(4300,),
        )

    monkeypatch.setattr(StackReconciler, "_probe_docker_infra", fake_probe_docker)
    monkeypatch.setattr(StackReconciler, "_probe_model_gateway", fake_probe_gw)
    monkeypatch.setattr(StackReconciler, "_probe_agent_workers", fake_probe_wk)
    monkeypatch.setattr(StackReconciler, "_probe_web_ui", fake_probe_ui)
    monkeypatch.setattr(StackReconciler, "_apply_docker_infra", fake_apply_docker)
    monkeypatch.setattr(StackReconciler, "_apply_model_gateway", fake_apply_gw)
    monkeypatch.setattr(StackReconciler, "_apply_agent_workers", fake_apply_wk)
    monkeypatch.setattr(StackReconciler, "_apply_web_ui", fake_apply_ui)

    rec = StackReconciler(spec)
    first_plan = rec.plan()
    assert {r.name: r.action for r in first_plan.resources} == {
        "docker_infra": "create",
        "frontend_bundle": "unchanged",
        "model_gateway": "create",
        "agent_workers": "create",
        "web_ui_server": "create",
    }

    first_apply = rec.apply(precomputed_plan=first_plan, verify=False)
    assert first_apply.all_healthy is True
    assert stat.S_IMODE(spec.state_file.stat().st_mode) == 0o600
    assert sim.compose_calls == 1
    assert sim.gw_starts == 1
    assert sim.worker_starts == 1
    assert sim.ui_starts == 1

    # Second plan right away: ZERO changes across all 5 resources!
    second_plan = rec.plan()
    assert second_plan.has_changes is False
    assert {r.name: r.action for r in second_plan.resources} == {
        "docker_infra": "unchanged",
        "frontend_bundle": "unchanged",
        "model_gateway": "unchanged",
        "agent_workers": "unchanged",
        "web_ui_server": "unchanged",
    }
    second_apply = rec.apply(precomputed_plan=second_plan, verify=False)
    assert second_apply.all_healthy is True
    assert sim.compose_calls == 1
    assert sim.gw_starts == 1
    assert sim.worker_starts == 1
    assert sim.ui_starts == 1


def test_reconciler_partial_reload_replaces_only_agent_workers_preserving_gateway_tunnel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Editing `agent/` marks ONLY `agent_workers` as `replace`, keeping `model_gateway` untouched.

    Killed by: src/uclone_x/shells/reconciler.py :: elif wk_ok and wk_prev_fp == wk_fp:
    Becomes:   elif wk_ok and wk_prev_fp != wk_fp:
    """
    spec = _make_mini_repo(tmp_path)
    fake_reconciler_file = spec.repo_root / "src" / "uclone_x" / "shells" / "reconciler.py"
    fake_reconciler_file.parent.mkdir(parents=True, exist_ok=True)
    fake_reconciler_file.write_text("# stub\n", encoding="utf-8")
    monkeypatch.setattr(reconciler, "__file__", str(fake_reconciler_file))

    spec.stack_env_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    spec.stack_env_file.write_text(
        "".join(f"{k}=secret\n" for k in REQUIRED_STACK_SECRETS), encoding="utf-8"
    )
    _install_healthy_probes(monkeypatch)

    rec = StackReconciler(spec)
    write_deploy_state(spec.state_file, rec.snapshot_desired_state())
    assert rec.plan().has_changes is False

    # Modify an agent file -> ONLY `agent_workers` transitions to `replace`!
    (spec.repo_root / "src" / "uclone_x" / "agent" / "turn.py").write_text(
        "# agent v2 modified!\n", encoding="utf-8"
    )
    drift_plan = rec.plan()
    actions = {r.name: r.action for r in drift_plan.resources}
    assert actions == {
        "docker_infra": "unchanged",
        "frontend_bundle": "unchanged",
        "model_gateway": "unchanged",
        "agent_workers": "replace",
        "web_ui_server": "unchanged",
    }
    by_name = {r.name: r for r in drift_plan.resources}
    assert "changed: code_sha" in by_name["agent_workers"].reason
    assert "tunnel=dell preserved" in by_name["model_gateway"].reason


def test_reconciler_frontend_only_change_rebuilds_bundle_without_restarting_ui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Editing `frontend/src/` updates `frontend_bundle` with pinned env and keeps `web_ui_server` unchanged.

    Killed by: src/uclone_x/shells/reconciler.py :: if recorded_digest == desired_inputs["frontend_inputs_sha"]:
    Becomes:   if recorded_digest != desired_inputs["frontend_inputs_sha"]:
    """
    spec = _make_mini_repo(tmp_path)
    fake_reconciler_file = spec.repo_root / "src" / "uclone_x" / "shells" / "reconciler.py"
    fake_reconciler_file.parent.mkdir(parents=True, exist_ok=True)
    fake_reconciler_file.write_text("# stub\n", encoding="utf-8")
    monkeypatch.setattr(reconciler, "__file__", str(fake_reconciler_file))

    spec.stack_env_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    spec.stack_env_file.write_text(
        "".join(f"{k}=secret\n" for k in REQUIRED_STACK_SECRETS), encoding="utf-8"
    )
    _install_healthy_probes(monkeypatch, gw_detail="pid=6100")

    rec = StackReconciler(spec)
    write_deploy_state(spec.state_file, rec.snapshot_desired_state())
    assert rec.plan().has_changes is False

    # Edit frontend source -> `frontend_bundle` becomes `update`, `web_ui_server` stays `unchanged`!
    (spec.repo_root / "frontend" / "src" / "App.tsx").write_text(
        "export const v = 2;\n", encoding="utf-8"
    )
    plan_after_fe_edit = rec.plan()
    assert {r.name: r.action for r in plan_after_fe_edit.resources} == {
        "docker_infra": "unchanged",
        "frontend_bundle": "update",
        "model_gateway": "unchanged",
        "agent_workers": "unchanged",
        "web_ui_server": "unchanged",
    }

    captured_npm: list[tuple[list[str], str, dict[str, str]]] = []

    def fake_run(
        cmd: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        capture_output: bool = False,
        text: bool = False,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        _ = (capture_output, text, check)
        captured_npm.append((cmd, cwd or "", dict(env or {})))
        (spec.repo_root / "src" / "uclone_x" / "ui_static" / "index.html").write_text(
            "<html>v2</html>\n", encoding="utf-8"
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="built", stderr="")

    monkeypatch.setattr(reconciler.subprocess, "run", fake_run)
    applied = rec.apply(precomputed_plan=plan_after_fe_edit, verify=False)
    assert applied.all_healthy is True
    assert len(captured_npm) == 1
    npm_cmd, npm_cwd, npm_env = captured_npm[0]
    assert npm_cmd == ["npm", "run", "build"]
    assert Path(npm_cwd) == spec.repo_root / "frontend"
    assert npm_env["NODE_ENV"] == "production"
    assert npm_env["BROWSERSLIST"] == "defaults"

    # Even though `ui_static/index.html` and `ui_build_inputs.sha256` changed on disk,
    # `web_ui_server` excludes both and remains `unchanged` on the next plan!
    assert rec.plan().has_changes is False


def test_reconciler_down_stops_all_daemons_and_respects_volumes_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`StackReconciler.down` stops UI, workers, gateway, and runs `stop` vs `down --volumes`.

    Killed by: src/uclone_x/shells/reconciler.py :: ["down", "--volumes"] if volumes else ["stop", *DOCKER_CLUSTER_SERVICES]
    Becomes:   ["down", "--volumes"] if not volumes else ["stop", *DOCKER_CLUSTER_SERVICES]
    """
    spec = _make_mini_repo(tmp_path)
    spec.stack_env_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    spec.stack_env_file.write_text("POSTGRES_PASSWORD=x\n", encoding="utf-8")
    spec.state_file.write_text('{"resources": {}}\n', encoding="utf-8")

    def fake_stop_ui(port: int) -> ui_process.StopOutcome:
        return ui_process.StopOutcome(port=port, stopped_pids=(7001,))

    def fake_list_records(kind: str) -> tuple[DaemonRecord, ...]:
        if kind == "worker":
            return (
                DaemonRecord(kind="worker", slot="0", pid=7002, instance="i0", process=None),
                DaemonRecord(kind="worker", slot="1", pid=7003, instance="i1", process=None),
            )
        return ()

    def fake_stop_daemon(
        kind: str, slot: str, expected_port: int | None = None
    ) -> daemon_supervisor.DaemonStopOutcome:
        _ = expected_port
        pid = 7002 if slot == "0" else (7003 if slot == "1" else 7004)
        return daemon_supervisor.DaemonStopOutcome(kind=kind, slot=slot, stopped_pids=(pid,))

    monkeypatch.setattr(ui_process, "stop_ui_server", fake_stop_ui)
    monkeypatch.setattr(daemon_supervisor, "list_daemon_records", fake_list_records)
    monkeypatch.setattr(daemon_supervisor, "stop_daemon", fake_stop_daemon)

    seen_compose_cmds: list[list[str]] = []

    def fake_compose_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        seen_compose_cmds.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(reconciler.subprocess, "run", fake_compose_run)

    rec = StackReconciler(spec)
    out_keep = rec.down(volumes=False)
    assert out_keep.ui_stopped_pids == (7001,)
    assert out_keep.worker_stopped_pids == (7002, 7003)
    assert out_keep.gateway_stopped_pids == (7004,)
    assert out_keep.docker_stopped is True
    assert out_keep.volumes_removed is False
    assert seen_compose_cmds[-1][-len(DOCKER_CLUSTER_SERVICES) - 1 :] == [
        "stop",
        *DOCKER_CLUSTER_SERVICES,
    ]
    assert spec.state_file.exists()

    out_vol = rec.down(volumes=True)
    assert out_vol.volumes_removed is True
    assert seen_compose_cmds[-1][-2:] == ["down", "--volumes"]
    assert not spec.state_file.exists()


def test_cli_up_plan_detailed_exitcode_and_status_stack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ucx up --plan --detailed-exitcode`, `ucx up --json`, `ucx down --json`, and `ucx status --stack`.

    Killed by: src/uclone_x/cli/commands/up.py :: if detailed_exitcode and computed_plan.has_changes:
    Becomes:   if detailed_exitcode and not computed_plan.has_changes:
    """
    spec = _make_mini_repo(tmp_path)
    runner = CliRunner()

    drifted_plan = StackPlan(
        resources=(
            ResourcePlan(
                name="docker_infra",
                action="unchanged",
                desired_fingerprint="sha256:aaa",
                applied_fingerprint="sha256:aaa",
                live_healthy=True,
                reason="healthy",
                endpoint=str(spec.compose_file),
            ),
            ResourcePlan(
                name="frontend_bundle",
                action="unchanged",
                desired_fingerprint="sha256:bbb",
                applied_fingerprint="sha256:bbb",
                live_healthy=True,
                reason="fresh",
                endpoint="ui_static",
            ),
            ResourcePlan(
                name="model_gateway",
                action="unchanged",
                desired_fingerprint="sha256:ccc",
                applied_fingerprint="sha256:ccc",
                live_healthy=True,
                reason="pid=100, tunnel=dell preserved",
                endpoint=spec.gateway_url,
                pids=(100,),
            ),
            ResourcePlan(
                name="agent_workers",
                action="replace",
                desired_fingerprint="sha256:ddd2",
                applied_fingerprint="sha256:ddd1",
                live_healthy=True,
                reason="changed: code_sha",
                endpoint="kafka-group:ucx-worker (x2)",
                pids=(201, 202),
            ),
            ResourcePlan(
                name="web_ui_server",
                action="unchanged",
                desired_fingerprint="sha256:eee",
                applied_fingerprint="sha256:eee",
                live_healthy=True,
                reason="pid=300, cluster_mode=docker",
                endpoint=spec.ui_url,
                pids=(300,),
            ),
        ),
        inspection_ms=12.5,
    )

    converged_plan = StackPlan(
        resources=tuple(
            r.model_copy(
                update={"action": "unchanged", "applied_fingerprint": r.desired_fingerprint}
            )
            for r in drifted_plan.resources
        ),
        inspection_ms=8.2,
    )

    active_plan_box = [drifted_plan]

    class _StubReconciler:
        def __init__(self) -> None:
            self.spec = spec

        def plan(self) -> StackPlan:
            return active_plan_box[0]

        def apply(
            self, *, precomputed_plan: StackPlan | None = None, verify: bool = True
        ) -> StackApplyResult:
            _ = verify
            p = precomputed_plan or active_plan_box[0]
            return StackApplyResult(
                plan=p,
                results=tuple(
                    AppliedResourceResult(
                        name=r.name,
                        action=r.action,
                        healthy=True,
                        duration_ms=2.0,
                        detail=r.reason,
                        endpoint=r.endpoint,
                        pids=r.pids,
                    )
                    for r in p.resources
                ),
                verification_checks=({"name": "Gateway", "healthy": True},),
                all_healthy=True,
                ui_url=spec.ui_url,
                gateway_url=spec.gateway_url,
            )

        def down(self, *, volumes: bool = False) -> StackDownResult:
            return StackDownResult(
                ui_stopped_pids=(300,),
                worker_stopped_pids=(201, 202),
                gateway_stopped_pids=(100,),
                docker_stopped=True,
                volumes_removed=volumes,
            )

        def _run_post_apply_verification(self) -> tuple[tuple[dict[str, Any], ...], bool]:
            return (({"name": "Gateway", "healthy": True},), True)

    def fake_from_env(_cls: type[StackReconciler], **kwargs: Any) -> _StubReconciler:
        _ = kwargs
        return _StubReconciler()

    monkeypatch.setattr(StackReconciler, "from_environment", classmethod(fake_from_env))

    # 1. `--plan --detailed-exitcode` with pending changes -> exit code 2!
    res_drift = runner.invoke(ucx_app, ["up", "--plan", "--detailed-exitcode", "--json"])
    assert res_drift.exit_code == 2
    parsed_drift = json.loads(res_drift.stdout)
    assert parsed_drift["has_changes"] is True
    assert parsed_drift["counts"]["replace"] == 1
    assert parsed_drift["counts"]["unchanged"] == 4

    # 2. `ucx status --stack --json` reports `drifted:replace` on `agent_workers`
    res_status = runner.invoke(ucx_app, ["status", "--stack", "--json"])
    assert res_status.exit_code == 0
    parsed_status = json.loads(res_status.stdout)
    assert parsed_status["in_sync"] is False
    by_res = {r["name"]: r for r in parsed_status["resources"]}
    assert by_res["agent_workers"]["drift_state"] == "drifted:replace"
    assert by_res["model_gateway"]["drift_state"] == "in-sync"

    # 3. `ucx up --json` applies the drifted plan cleanly
    res_apply = runner.invoke(ucx_app, ["up", "--json"])
    assert res_apply.exit_code == 0
    parsed_apply = json.loads(res_apply.stdout)
    assert parsed_apply["changed"] is True
    assert parsed_apply["all_healthy"] is True

    # 4. `--plan --detailed-exitcode` when converged -> exit code 0!
    active_plan_box[0] = converged_plan
    res_clean = runner.invoke(ucx_app, ["up", "--plan", "--detailed-exitcode", "--json"])
    assert res_clean.exit_code == 0
    assert json.loads(res_clean.stdout)["has_changes"] is False

    # 5. `ucx down --volumes --json`
    res_down = runner.invoke(ucx_app, ["down", "--volumes", "--json"])
    assert res_down.exit_code == 0
    parsed_down = json.loads(res_down.stdout)
    assert parsed_down["volumes_removed"] is True
    assert parsed_down["worker_stopped_pids"] == [201, 202]


def test_ui_app_delegates_remote_gpu_tunnel_ownership_to_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When `UCX_GATEWAY_URL` is configured, `create_ui_app` neither wipes `remote_gpu_restore.json` nor owns SSH tunnels, and proxies `/api/settings/remote-gpu/status` to the gateway.

    Killed by: src/uclone_x/ui/app.py :: gateway_owns_tunnel = has_configured_gateway_endpoint()
    Becomes:   gateway_owns_tunnel = not has_configured_gateway_endpoint()
    """
    from uclone_x.ui.app import AgentSessionManager, create_ui_app

    storage_dir = tmp_path / "ui-storage"
    storage_dir.mkdir(parents=True)
    restore_file = storage_dir / "remote_gpu_restore.json"
    restore_file.write_text(
        json.dumps(
            {
                "original": {"remote-gpu": ""},
                "applied": {"remote-gpu": "http://127.0.0.1:11435"},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("UCX_GATEWAY_URL", "http://127.0.0.1:18765")
    monkeypatch.setenv("UCX_GATEWAY_KEY", "gw-secret-token")

    seen_proxy_calls: list[tuple[str, str, str]] = []

    class _MockAsyncClient:
        def __init__(self, **kwargs: Any) -> None:
            _ = kwargs

        async def __aenter__(self) -> _MockAsyncClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            _ = args

        async def request(
            self,
            method: str,
            url: str,
            json: dict[str, Any] | None = None,
            headers: dict[str, str] | None = None,
        ) -> httpx.Response:
            _ = json
            seen_proxy_calls.append((method, url, (headers or {}).get("Authorization", "")))
            return httpx.Response(
                200,
                json={
                    "connected": True,
                    "host": "dell",
                    "pid": 99999,
                    "forwarded_ports": [{"local_port": 11435, "remote_port": 11434}],
                },
            )

    monkeypatch.setattr(httpx, "AsyncClient", _MockAsyncClient)

    session_mgr = AgentSessionManager(storage_dir=storage_dir, workspace_dir=tmp_path)
    ui_app = create_ui_app(session_manager=session_mgr, workspace_dir=tmp_path)
    with TestClient(ui_app, base_url="http://127.0.0.1:5180") as client:
        assert restore_file.is_file()
        resp = client.get("/api/settings/remote-gpu/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["connected"] is True
        assert body["host"] == "dell"

    assert restore_file.is_file()
    assert seen_proxy_calls == [
        ("GET", "http://127.0.0.1:18765/api/settings/remote-gpu/status", "Bearer gw-secret-token")
    ]


def test_compute_python_tree_digest_ignores_pycache_and_honors_exclusions(tmp_path: Path) -> None:
    """`compute_python_tree_digest` ignores `__pycache__`/`.pyc` and honors `exclude_rel_prefixes`."""
    pkg = tmp_path / "src" / "pkg"
    (pkg / "__pycache__").mkdir(parents=True)
    (pkg / "ui_static").mkdir(parents=True)
    (pkg / "core.py").write_text("x = 1\n", encoding="utf-8")
    (pkg / "__pycache__" / "core.cpython-312.pyc").write_bytes(b"\x00\x01")
    (pkg / "ui_static" / "app.js").write_text("console.log(1);\n", encoding="utf-8")

    d1 = compute_python_tree_digest(
        [pkg], base_dir=tmp_path, exclude_rel_prefixes=("src/pkg/ui_static",)
    )
    (pkg / "__pycache__" / "core.cpython-312.pyc").write_bytes(b"\xff\xfe\xfd")
    (pkg / "ui_static" / "app.js").write_text("console.log(999);\n", encoding="utf-8")
    d2 = compute_python_tree_digest(
        [pkg], base_dir=tmp_path, exclude_rel_prefixes=("src/pkg/ui_static",)
    )
    assert d1 == d2

    (pkg / "core.py").write_text("x = 2\n", encoding="utf-8")
    d3 = compute_python_tree_digest(
        [pkg], base_dir=tmp_path, exclude_rel_prefixes=("src/pkg/ui_static",)
    )
    assert d3 != d1


def test_probe_model_gateway_reads_remote_tunnel_and_env_remote_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_probe_model_gateway` reads `/v1/health`'s `remote_tunnel` payload and `StackReconciler.from_environment` honors `UCX_REMOTE_HOST`.

    Killed by: src/uclone_x/shells/reconciler.py :: rgpu = body_map.get("remote_tunnel") or body_map.get("remote_gpu")
    Becomes:   rgpu = body_map.get("remote_gpu")
    """
    spec = _make_mini_repo(tmp_path)
    state_dir = tmp_path / "ui-state"
    monkeypatch.setenv("UCLONE_UI_STATE_DIR", str(state_dir))

    rec, _ = new_daemon_record(
        "gateway", str(spec.gateway_port), pid=os.getpid(), port=spec.gateway_port
    )
    write_daemon_record(rec)

    health_payload: dict[str, Any] = {
        "status": "ok",
        "remote_tunnel": {
            "host": "dell",
            "desired_host": "dell",
            "connected": True,
            "reconnecting": False,
            "error": None,
        },
    }

    def _fake_get(url: str, timeout: float = 2.0) -> httpx.Response:
        _ = (url, timeout)
        return httpx.Response(200, json=health_payload)

    monkeypatch.setattr(httpx, "get", _fake_get)

    reconciler = StackReconciler(spec)
    pids, ok, detail = reconciler._probe_model_gateway()  # pyright: ignore[reportPrivateUsage]
    assert ok is True
    assert pids == (os.getpid(),)
    assert "tunnel=dell preserved" in detail

    # Failed tunnel surfaces the exact SSH error instead of generic "tunnel not active"
    health_payload["remote_tunnel"] = {
        "host": "dell",
        "desired_host": "dell",
        "connected": False,
        "reconnecting": False,
        "error": "Operation timed out",
    }
    pids2, ok2, detail2 = reconciler._probe_model_gateway()  # pyright: ignore[reportPrivateUsage]
    assert ok2 is False
    assert pids2 == (os.getpid(),)
    assert "Operation timed out" in detail2

    # When `remote_host` is not pinned on StackSpec, gateway remains healthy while surfacing disconnected note
    spec_no_pin = StackSpec(
        repo_root=spec.repo_root,
        compose_file=spec.compose_file,
        stack_env_file=spec.stack_env_file,
        state_file=spec.state_file,
        gateway_port=spec.gateway_port,
        remote_host=None,
    )
    _, ok3, detail3 = StackReconciler(spec_no_pin)._probe_model_gateway()  # pyright: ignore[reportPrivateUsage]
    assert ok3 is True
    assert "tunnel=dell disconnected (Operation timed out)" in detail3

    # `from_environment` resolves `UCX_REMOTE_HOST` from stack.env and environment
    spec.stack_env_file.parent.mkdir(parents=True, exist_ok=True)
    spec.stack_env_file.write_text("UCX_REMOTE_HOST=mac-mini\n", encoding="utf-8")
    monkeypatch.setenv("UCX_CLUSTER_COMPOSE_FILE", str(spec.compose_file))
    monkeypatch.setenv("UCX_CLUSTER_ENV_FILE", str(spec.stack_env_file))
    monkeypatch.delenv("UCX_REMOTE_HOST", raising=False)

    from_stack_env = StackReconciler.from_environment()
    assert from_stack_env.spec.remote_host == "mac-mini"

    monkeypatch.setenv("UCX_REMOTE_HOST", "dell")
    from_proc_env = StackReconciler.from_environment()
    assert from_proc_env.spec.remote_host == "dell"
