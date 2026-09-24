"""Unit tests for bootstrap pre-flight hardware check and local Ollama onboarding."""

from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from uclone_x.cli.commands import bootstrap
from uclone_x.core.environment_install import install_into_running_environment
from uclone_x.tools.builtin.image import DependencyProblem


def test_get_system_specs_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_sysctl(*args: Any, **kwargs: Any) -> str:
        return "17179869184\n"

    def fake_disk_usage(path: str) -> Any:
        class Usage:
            free = 50 * (1024**3)

        return Usage()

    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("subprocess.check_output", fake_sysctl)
    monkeypatch.setattr("shutil.disk_usage", fake_disk_usage)

    specs = bootstrap.get_system_specs()
    assert specs.total_ram_gb == 16.0
    assert specs.free_disk_gb == 50.0
    assert specs.recommended_model == bootstrap.RECOMMENDED_8B_MODEL


def test_get_system_specs_low_ram(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_sysctl(*args: Any, **kwargs: Any) -> str:
        return "4294967296\n"

    def fake_disk_usage(path: str) -> Any:
        class Usage:
            free = 20 * (1024**3)

        return Usage()

    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("subprocess.check_output", fake_sysctl)
    monkeypatch.setattr("shutil.disk_usage", fake_disk_usage)

    specs = bootstrap.get_system_specs()
    assert specs.total_ram_gb == 4.0
    assert specs.recommended_model == bootstrap.LIGHTWEIGHT_MODEL


def test_get_system_specs_recommends_the_light_model_on_an_8gb_mac(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_sysctl(*args: Any, **kwargs: Any) -> str:
        return str(8 * 1024**3) + "\n"

    def fake_disk_usage(path: str) -> Any:
        class Usage:
            free = 50 * (1024**3)

        return Usage()

    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("subprocess.check_output", fake_sysctl)
    monkeypatch.setattr("shutil.disk_usage", fake_disk_usage)

    specs = bootstrap.get_system_specs()
    assert specs.total_ram_gb == 8.0
    assert specs.recommended_model == bootstrap.LIGHTWEIGHT_MODEL


def test_get_system_specs_counts_a_16gb_linux_box_as_16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A 16 GB machine's /proc/meminfo reads about 15.6 GiB.
    def fake_disk_usage(path: str) -> Any:
        class Usage:
            free = 50 * (1024**3)

        return Usage()

    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("builtins.open", mock_open(read_data="MemTotal:       16357000 kB\n"))
    monkeypatch.setattr("shutil.disk_usage", fake_disk_usage)

    specs = bootstrap.get_system_specs()
    assert specs.total_ram_gb < 16.0
    assert specs.recommended_model == bootstrap.RECOMMENDED_8B_MODEL


def test_get_system_specs_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_disk_usage(path: str) -> Any:
        class Usage:
            free = 100 * (1024**3)

        return Usage()

    fake_meminfo = "MemTotal:       32800000 kB\nMemFree:        10000000 kB\n"
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("builtins.open", mock_open(read_data=fake_meminfo))
    monkeypatch.setattr("shutil.disk_usage", fake_disk_usage)

    specs = bootstrap.get_system_specs()
    assert specs.total_ram_gb >= 30.0
    assert specs.recommended_model == bootstrap.RECOMMENDED_8B_MODEL


def test_is_ollama_reachable_success() -> None:
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        assert bootstrap.is_ollama_reachable("http://localhost:11434") is True


def test_is_ollama_reachable_failure() -> None:
    with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        assert bootstrap.is_ollama_reachable("http://localhost:11434") is False


def test_get_installed_ollama_models() -> None:
    mock_resp = MagicMock()
    mock_resp.status = 200
    payload = json.dumps({"models": [{"name": "qwen3:8b"}, {"name": "llama3:8b"}]}).encode("utf-8")
    mock_resp.read.return_value = payload
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        models = bootstrap.get_installed_ollama_models("http://localhost:11434")
        assert "qwen3:8b" in models
        assert "llama3:8b" in models


def test_is_ollama_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """PATH decides when it can; the app-bundle list is the only other place looked.

    The bundle list is emptied here rather than trusted to be absent: the developer Mac
    this suite runs on *has* `/Applications/Ollama.app`, so a test that only stubbed
    `shutil.which` would read the real filesystem and assert nothing.
    """
    monkeypatch.setattr(bootstrap, "OLLAMA_APP_BINARIES", ())
    with patch("shutil.which", return_value="/usr/local/bin/ollama"):
        assert bootstrap.is_ollama_installed() is True

    with patch("shutil.which", return_value=None):
        assert bootstrap.is_ollama_installed() is False


def test_ensure_local_profile_already_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bootstrap,
        "get_system_specs",
        lambda: bootstrap.SystemSpecs(16.0, 50.0, "qwen3:8b"),
    )

    def fake_reachable(url: str = bootstrap.DEFAULT_OLLAMA_URL, timeout: float = 2.0) -> bool:
        return True

    def fake_models(url: str = bootstrap.DEFAULT_OLLAMA_URL) -> list[str]:
        return ["qwen3:8b", "llama3:latest"]

    monkeypatch.setattr(bootstrap, "is_ollama_reachable", fake_reachable)
    monkeypatch.setattr(bootstrap, "get_installed_ollama_models", fake_models)

    success, model = bootstrap.ensure_local_profile(interactive=False)
    assert success is True
    assert model == "qwen3:8b"


def test_ensure_local_profile_user_declines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bootstrap,
        "get_system_specs",
        lambda: bootstrap.SystemSpecs(16.0, 50.0, "qwen3:8b"),
    )

    def fake_reachable(url: str = bootstrap.DEFAULT_OLLAMA_URL, timeout: float = 2.0) -> bool:
        return False

    def fake_confirm(*args: Any, **kwargs: Any) -> bool:
        return False

    monkeypatch.setattr(bootstrap, "is_ollama_reachable", fake_reachable)
    monkeypatch.setattr(bootstrap, "stdin_is_interactive", lambda: True)
    monkeypatch.setattr("rich.prompt.Confirm.ask", fake_confirm)

    success, model = bootstrap.ensure_local_profile(interactive=True)
    assert success is False
    assert model == "qwen3:8b"


def test_ensure_local_profile_full_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bootstrap,
        "get_system_specs",
        lambda: bootstrap.SystemSpecs(16.0, 50.0, "qwen3:8b"),
    )
    reachable_calls = [False, True, True]

    def fake_reachable(url: str = bootstrap.DEFAULT_OLLAMA_URL, timeout: float = 2.0) -> bool:
        return reachable_calls.pop(0) if reachable_calls else True

    def fake_models(url: str = bootstrap.DEFAULT_OLLAMA_URL) -> list[str]:
        return []

    def fake_pull(model: str) -> bool:
        return True

    monkeypatch.setattr(bootstrap, "is_ollama_reachable", fake_reachable)
    monkeypatch.setattr(bootstrap, "is_ollama_installed", lambda: True)
    monkeypatch.setattr(bootstrap, "start_ollama_daemon", lambda: True)
    monkeypatch.setattr(bootstrap, "get_installed_ollama_models", fake_models)
    monkeypatch.setattr(bootstrap, "pull_ollama_model", fake_pull)

    # `assume_yes`, not `interactive=False`: consent is now required rather than assumed
    # in the absence of a prompt. See `test_a_non_interactive_llm_setup_declines_without_yes`.
    success, model = bootstrap.ensure_local_profile(interactive=False, assume_yes=True)
    assert success is True
    assert model == "qwen3:8b"


def _report(
    *,
    remote: str | None = None,
    remote_alive: bool = False,
    comfy_alive: bool = False,
    deps_ok: bool = False,
    checkpoint: str | None = None,
) -> bootstrap.ImageEngineReport:
    return bootstrap.ImageEngineReport(
        remote_url=remote,
        remote_alive=remote_alive,
        comfy_url="http://127.0.0.1:8188",
        comfy_alive=comfy_alive,
        dependency_problems=() if deps_ok else ("'torch' is not installed (needs torch>=2.2.0)",),
        checkpoint=checkpoint,
    )


def test_image_engine_report_is_not_ready_on_a_package_without_a_checkpoint() -> None:
    """`diffusers` importing is half the requirement; a load with no file would fail (P6).

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: return self.dependencies_ok and self.checkpoint is not None
    Becomes: return self.dependencies_ok
    """
    report = _report(deps_ok=True, checkpoint=None)

    assert report.in_process_ready is False
    assert report.ready is False
    assert report.engine == "none"


def test_image_engine_report_engine_order_matches_the_dispatcher() -> None:
    """A detected daemon outranks the in-process engine, and a remote worker outranks both.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: if self.comfy_alive:
    Becomes: if False:
    """
    both = _report(comfy_alive=True, deps_ok=True, checkpoint="/m/c.safetensors")
    assert both.engine == "comfyui-local"

    all_three = _report(
        remote="http://gpu:9000",
        remote_alive=True,
        comfy_alive=True,
        deps_ok=True,
        checkpoint="/m/c.safetensors",
    )
    assert all_three.engine == "remote-cuda"

    in_process = _report(deps_ok=True, checkpoint="/m/c.safetensors")
    assert in_process.engine == "diffusers-sdxl"


def test_image_engine_report_does_not_count_a_remote_worker_that_never_answered() -> None:
    """`UCX_IMAGE_REMOTE_URL` is an address, not a worker; readiness needs the probe (P6).

    `ucx media status` reported "Ready -- 'remote-cuda' would run" and exited 0 for a port
    with nothing listening, while the dispatcher -- which probes `/health` through
    `RemoteCudaImageEngine.is_available` -- refused the very same environment with
    `ImageGenerationError` (reviewer, PR #1096).

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: return bool(self.remote_url) and self.remote_alive
    Becomes: return bool(self.remote_url)
    """
    unreachable = _report(remote="http://127.0.0.1:59999", remote_alive=False)
    assert unreachable.remote_ready is False
    assert unreachable.ready is False
    assert unreachable.engine == "none"

    answering = _report(remote="http://127.0.0.1:59999", remote_alive=True)
    assert answering.remote_ready is True
    assert answering.ready is True
    assert answering.engine == "remote-cuda"

    # A live probe with no address configured is not a worker either.
    assert _report(remote=None, remote_alive=True).remote_ready is False


def test_image_engine_report_is_ready_on_the_in_process_engine_alone() -> None:
    """The daemon-free baseline is an engine too, so `ready` must count it on its own.

    `ready` is a three-arm disjunction and this arm was pinned by no assertion in the suite
    (#1134): the engine-order case reads `engine`, which reaches the in-process engine
    through a branch of its own, and every `ready is True` case ran through the remote
    worker. Dropping `or self.in_process_ready` therefore left the suite green while
    `ucx media status` would have reported "not ready" on a machine that can generate --
    the promise-the-wrong-way-round half of P6.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: return self.remote_ready or self.comfy_alive or self.in_process_ready
    Becomes: return self.remote_ready or self.comfy_alive
    """
    alone = _report(deps_ok=True, checkpoint="/m/c.safetensors")

    assert alone.remote_ready is False
    assert alone.comfy_alive is False
    assert alone.in_process_ready is True
    assert alone.ready is True
    assert alone.engine == "diffusers-sdxl"


def test_probe_image_engines_reads_the_environment_and_the_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The configured remote worker is probed, not assumed present from its variable.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: asyncio.run(RemoteCudaImageEngine(base_url=remote_url).is_available())
    Becomes: True
    """
    checkpoint = tmp_path / "c.safetensors"
    checkpoint.write_bytes(b"x")
    monkeypatch.setenv("UCX_IMAGE_REMOTE_URL", "http://10.0.0.99:8000")
    monkeypatch.setenv("UCX_IMAGE_CHECKPOINT", str(checkpoint))

    client = AsyncMock()
    client.alive.return_value = False

    remote = AsyncMock()
    remote.is_available.return_value = False

    with (
        patch("uclone_x.tools.builtin.image.ComfyClient", return_value=client),
        patch("uclone_x.tools.builtin.image.RemoteCudaImageEngine", return_value=remote),
        # This test's own checkpoint file makes `in_process_ready` depend on whether the
        # in-process engine's real dependencies (torch, diffusers, ...) happen to be
        # importable in whatever venv runs the suite -- true on a machine that also does
        # image-generation work, false on a lean one. Forcing a problem here keeps the
        # assertion about the *remote* and *comfy* engines below from swinging on that.
        patch(
            "uclone_x.tools.builtin.image.in_process_dependency_problems",
            return_value=(DependencyProblem("torch", "torch>=2.2.0", installed=None),),
        ),
    ):
        report = bootstrap.probe_image_engines()

    assert report.remote_url == "http://10.0.0.99:8000"
    assert report.remote_alive is False
    assert report.remote_ready is False
    # `ready` again, now that all three engines are stubbed: with the in-process probe
    # forced to report a missing package above, every arm of the disjunction is decided by
    # this test's own inputs, so the whole report can be asserted rather than the one
    # property the host cannot reach into (#1134).
    assert report.ready is False
    assert report.engine == "none"
    assert report.comfy_alive is False
    assert report.checkpoint == str(checkpoint)


def test_probe_image_engines_takes_the_in_process_verdict_from_the_probe_not_the_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Both host states, in one run: `torch`/`diffusers` present and absent agree here.

    The claim this module needs to make is an equivalence -- that its result is the same on
    a machine with the in-process stack installed and on one without -- and until #1134 that
    was argued in a comment rather than asserted. It is asserted directly by simulating both
    machines: the only difference allowed between the two reports is the dependency field
    that was stubbed, and every other field, including `ready` and `engine`, follows the
    stub rather than this interpreter's site-packages.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: dependency_problems=tuple(
    Becomes: dependency_problems=("'torch' is not installed (needs torch>=2.2.0)",) or tuple(
    """
    checkpoint = tmp_path / "c.safetensors"
    checkpoint.write_bytes(b"x")
    monkeypatch.delenv("UCX_IMAGE_REMOTE_URL", raising=False)
    monkeypatch.delenv("UCX_MEDIA_REMOTE_URL", raising=False)
    monkeypatch.setenv("UCX_IMAGE_CHECKPOINT", str(checkpoint))

    client = AsyncMock()
    client.alive.return_value = False

    def probe(problems: tuple[DependencyProblem, ...]) -> bootstrap.ImageEngineReport:
        with (
            patch("uclone_x.tools.builtin.image.ComfyClient", return_value=client),
            patch(
                "uclone_x.tools.builtin.image.in_process_dependency_problems",
                return_value=problems,
            ),
        ):
            return bootstrap.probe_image_engines()

    installed = probe(())
    lean = probe((DependencyProblem("torch", "torch>=2.2.0", installed=None),))

    assert (installed.in_process_ready, installed.ready, installed.engine) == (
        True,
        True,
        "diffusers-sdxl",
    )
    assert (lean.in_process_ready, lean.ready, lean.engine) == (False, False, "none")
    assert lean.dependency_problems == ("'torch' is not installed (needs torch>=2.2.0)",)
    # Equal once the stubbed field is equalised: nothing else in the report -- the remote
    # address, either probe's verdict, or the checkpoint read off disk -- moves with what
    # happens to be importable.
    assert installed._replace(dependency_problems=lean.dependency_problems) == lean


def test_probe_image_engines_records_a_remote_worker_that_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same probe, answering: the engine is then reported ready and named.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: remote_alive=remote_alive,
    Becomes: remote_alive=False,
    """
    monkeypatch.setenv("UCX_IMAGE_REMOTE_URL", "http://10.0.0.99:8000")
    monkeypatch.delenv("UCX_IMAGE_CHECKPOINT", raising=False)

    client = AsyncMock()
    client.alive.return_value = False

    remote = AsyncMock()
    remote.is_available.return_value = True

    with (
        patch("uclone_x.tools.builtin.image.ComfyClient", return_value=client),
        patch("uclone_x.tools.builtin.image.RemoteCudaImageEngine", return_value=remote),
    ):
        report = bootstrap.probe_image_engines()

    assert report.remote_alive is True
    assert report.engine == "remote-cuda"


def test_probe_image_engines_makes_no_remote_probe_without_a_configured_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No address, no HTTP call -- and `remote_alive` stays false.

    Killed by: src/uclone_x/cli/commands/bootstrap.py ::             if remote_url
    Becomes:             if True
    """
    monkeypatch.delenv("UCX_IMAGE_REMOTE_URL", raising=False)
    monkeypatch.delenv("UCX_MEDIA_REMOTE_URL", raising=False)

    client = AsyncMock()
    client.alive.return_value = False

    remote = AsyncMock()
    remote.is_available.return_value = True

    with (
        patch("uclone_x.tools.builtin.image.ComfyClient", return_value=client),
        patch("uclone_x.tools.builtin.image.RemoteCudaImageEngine", return_value=remote),
    ):
        report = bootstrap.probe_image_engines()

    remote.is_available.assert_not_awaited()
    assert report.remote_url is None
    assert report.remote_alive is False


def test_probe_image_engines_treats_a_raising_remote_probe_as_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A probe that raises is a worker that is not there, not a crashed `ucx media status`.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: remote_alive = False
    Becomes: remote_alive = True
    """
    monkeypatch.setenv("UCX_IMAGE_REMOTE_URL", "http://10.0.0.99:8000")

    client = AsyncMock()
    client.alive.return_value = False

    with (
        patch("uclone_x.tools.builtin.image.ComfyClient", return_value=client),
        patch(
            "uclone_x.tools.builtin.image.RemoteCudaImageEngine",
            side_effect=OSError("refused"),
        ),
    ):
        report = bootstrap.probe_image_engines()

    assert report.remote_alive is False


def test_probe_image_engines_treats_a_failed_probe_as_no_daemon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A probe that raises is a daemon that is not there, not a crashed `ucx start` (P6).

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: comfy_alive = False
    Becomes: comfy_alive = True
    """
    monkeypatch.delenv("UCX_IMAGE_REMOTE_URL", raising=False)

    with patch("uclone_x.tools.builtin.image.ComfyClient", side_effect=OSError("refused")):
        report = bootstrap.probe_image_engines()

    assert report.comfy_alive is False


def test_diffusers_install_command_prefers_uv_where_both_installers_exist() -> None:
    """One order for every in-process install, and it is uv first (#1278).

    Until this, `ucx start`'s image install checked pip first while every other in-process
    install checked uv first. The two agreed by accident on a uv-created environment --
    there is no pip there, so both reached uv -- and disagreed exactly where both exist:
    a pip-seeded venv on a machine with uv on `PATH` repaired itself with a different
    resolver and a different cache depending on which caller asked.

    Killed by: src/uclone_x/core/environment_install.py :: uv_bin = shutil.which("uv")
    Becomes: uv_bin = None
    """
    from uclone_x.tools.builtin.image import IN_PROCESS_REQUIREMENTS

    with (
        patch("shutil.which", _uv_on_path),
        patch("importlib.util.find_spec", return_value=MagicMock()),
    ):
        command = bootstrap.diffusers_install_command()

    assert command == [
        "/opt/uv",
        "pip",
        "install",
        "--python",
        bootstrap.sys.executable,
        *(requirement for _, requirement, _ in IN_PROCESS_REQUIREMENTS),
    ]


def test_diffusers_install_command_installs_torch_and_transformers_too() -> None:
    """Installing `diffusers` alone leaves the engine importable and unable to generate.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: packages = [requirement for _, requirement, _ in IN_PROCESS_REQUIREMENTS]
    Becomes: packages = ["diffusers"]
    """
    with (
        patch("shutil.which", _uv_on_path),
        patch("importlib.util.find_spec", return_value=MagicMock()),
    ):
        command = bootstrap.diffusers_install_command()

    assert command is not None
    assert any(part.startswith("torch") for part in command)
    assert any(part.startswith("transformers") for part in command)


def test_diffusers_install_command_falls_back_to_pip_when_uv_is_absent() -> None:
    """pip is still the second choice, not no choice: a pip-only machine installs."""
    from uclone_x.tools.builtin.image import IN_PROCESS_REQUIREMENTS

    with (
        patch("shutil.which", _nothing_on_path),
        patch("importlib.util.find_spec", return_value=MagicMock()),
    ):
        command = bootstrap.diffusers_install_command()

    assert command == [
        bootstrap.sys.executable,
        "-m",
        "pip",
        "install",
        *(requirement for _, requirement, _ in IN_PROCESS_REQUIREMENTS),
    ]


def test_diffusers_install_command_is_none_without_pip_or_uv() -> None:
    with (
        patch("importlib.util.find_spec", return_value=None),
        patch("shutil.which", _nothing_on_path),
    ):
        assert bootstrap.diffusers_install_command() is None


def test_the_two_in_process_install_paths_never_choose_different_installers() -> None:
    """The `ucx start` path and the shared path must agree in *every* environment (#1278).

    Not a comment asking future editors to keep the two in step: both entry points are
    called under all four availability combinations and compared. Re-inlining an order on
    either side -- which is exactly how they came apart -- makes the both-present row
    disagree, and that row is a failure rather than a coincidence.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: return installer_command(*packages)
    Becomes: return [sys.executable, "-m", "pip", "install", *packages]
    """
    from uclone_x.core.environment_install import installer_command
    from uclone_x.tools.builtin.image import IN_PROCESS_REQUIREMENTS

    packages = [requirement for _, requirement, _ in IN_PROCESS_REQUIREMENTS]
    # Every combination of (uv on `PATH`, pip importable). The divergence was invisible in
    # three of the four: only the last row holds both installers.
    availability = [
        (_nothing_on_path, _no_spec),
        (_uv_on_path, _no_spec),
        (_nothing_on_path, _a_spec),
        (_uv_on_path, _a_spec),
    ]
    seen: list[tuple[list[str] | None, list[str] | None]] = []
    for which, find_spec in availability:
        with patch("shutil.which", which), patch("importlib.util.find_spec", find_spec):
            seen.append((bootstrap.diffusers_install_command(), installer_command(*packages)))

    assert [pair[0] for pair in seen] == [pair[1] for pair in seen]
    # The row that made the divergence observable: both installers present, uv chosen.
    assert seen[-1][0] is not None
    assert seen[-1][0][:3] == ["/opt/uv", "pip", "install"]


def test_only_one_module_in_the_kernel_decides_between_uv_and_pip() -> None:
    """A third installer site is how a fourth would start; the corpus is scanned for one.

    The behavioural guard above compares the two entry points that exist today. This one
    is about the site that does not exist yet: building an installer argv is choosing an
    order, whichever installer the argv names, and there is one place for that. The marker
    is the argv fragment both branches share, so it catches a uv-only third site as well
    as a pip-only one, and it does not fire on prose that merely *mentions* pip.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: return installer_command(*packages)
    Becomes: return [shutil.which("uv") or sys.executable, "pip", "install", *packages]
    """
    from pathlib import Path

    import uclone_x

    package_root = Path(uclone_x.__file__).resolve().parent
    assert package_root.is_relative_to(Path.cwd().resolve()), (
        f"{package_root} is outside this worktree; PYTHONPATH is not set (AGENTS.md §2)."
    )

    argv_fragment = '"pip", "install"'
    sites = sorted(
        str(path.relative_to(package_root))
        for path in package_root.rglob("*.py")
        if argv_fragment in path.read_text(encoding="utf-8")
    )

    assert sites == ["core/environment_install.py"]


def test_install_diffusers_refuses_to_claim_success_on_exit_zero_alone() -> None:
    """Exit 0 is not the question; whether every requirement now imports is (P6).

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: if returncode == 0 and not problems:
    Becomes: if returncode == 0:
    """
    with (
        patch.object(bootstrap, "diffusers_install_command", return_value=["pip", "install", "x"]),
        patch.object(bootstrap.subprocess, "run", return_value=MagicMock(returncode=0)),
        patch("importlib.util.find_spec", return_value=None),
    ):
        assert bootstrap.install_diffusers() is False


def test_install_diffusers_reports_true_when_the_package_becomes_importable() -> None:
    with (
        patch.object(bootstrap, "diffusers_install_command", return_value=["pip", "install", "x"]),
        patch.object(bootstrap.subprocess, "run", return_value=MagicMock(returncode=0)),
        patch("importlib.util.find_spec", return_value=MagicMock()),
    ):
        assert bootstrap.install_diffusers() is True


def test_install_diffusers_still_refuses_an_install_that_landed_below_the_version_floor(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Presence is not the measure this path needs; the floor is (#1278 criterion 2).

    The shared installer reads its outcome back with `still_missing`, which compares no
    version, so a `diffusers 0.30.0` sitting under the `>=0.31.0` floor satisfies it and
    is reported installed. Consolidating the *order* onto that path must not drag its
    weaker re-measurement across with it: this path keeps re-measuring with
    `in_process_dependency_problems`, which reads the floor.

    Killed by: src/uclone_x/tools/builtin/image.py :: ("diffusers", "diffusers>=0.31.0", (0, 31)),
    Becomes: ("diffusers", "diffusers>=0.31.0", (0, 0)),
    """
    import importlib.metadata

    from uclone_x.core.environment_install import still_missing

    # One below the floor, and every other requirement comfortably above it.
    at_floor_minus_one = {"diffusers": "0.30.0", "torch": "2.6.0", "transformers": "4.51.0"}

    def fake_version(name: str) -> str:
        return at_floor_minus_one[name]

    def fake_distribution(name: str) -> Any:
        if name not in at_floor_minus_one:
            raise importlib.metadata.PackageNotFoundError(name)
        return MagicMock()

    with (
        patch.object(bootstrap, "diffusers_install_command", return_value=["uv", "pip"]),
        patch.object(bootstrap.subprocess, "run", return_value=MagicMock(returncode=0)),
        patch("importlib.util.find_spec", _a_spec),
        patch.object(importlib.metadata, "version", fake_version),
        patch.object(importlib.metadata, "distribution", fake_distribution),
    ):
        installed_below_floor = bootstrap.install_diffusers()
        # The presence-only read-back the shared path uses calls the same environment done.
        presence_only = still_missing(tuple(at_floor_minus_one))

    assert installed_below_floor is False
    assert presence_only == (), "the floor is the only thing that can catch this install"
    out = " ".join(capsys.readouterr().out.split())
    assert "0.30.0" in out
    assert "diffusers>=0.31.0" in out


def test_install_diffusers_is_false_when_there_is_no_installer() -> None:
    with patch.object(bootstrap, "diffusers_install_command", return_value=None):
        assert bootstrap.install_diffusers() is False


def test_install_diffusers_points_at_uv_when_there_is_no_installer(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The no-installer refusal names uv, and launches nothing (#971 criterion 2).

    Returning False here is not the behaviour under test -- the branch already did that,
    and `test_install_diffusers_is_false_when_there_is_no_installer` still passes with the
    remedy deleted. What a user needs is the next command to type. A refusal that reports
    only what the machine lacks is the same P6 dead end as the original silence, so the
    wording is pinned rather than the exit path.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: "Install uv (https://docs.astral.sh/uv/), "
    Becomes: ""
    """
    with (
        patch.object(bootstrap, "diffusers_install_command", return_value=None),
        patch.object(bootstrap.subprocess, "run") as run,
    ):
        assert bootstrap.install_diffusers() is False

    out = " ".join(capsys.readouterr().out.split())
    # The only installer that could be launched is the one already known to be absent.
    assert run.call_count == 0
    assert "https://docs.astral.sh/uv/" in out


def test_install_diffusers_points_at_ensurepip_when_there_is_no_installer(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The refusal also names the interpreter that needs pip, not just `ensurepip` (#971).

    `python -m ensurepip` is the wrong instruction on a machine whose `python` is not the
    one running `ucx` -- which is the usual case, since `ucx` runs from a venv. The
    remedy therefore interpolates `sys.executable`, and that is what is pinned here.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: "or seed pip with `{python} -m ensurepip --upgrade`, "
    Becomes: ""
    """
    with (
        patch.object(bootstrap, "diffusers_install_command", return_value=None),
        patch.object(bootstrap.subprocess, "run"),
    ):
        assert bootstrap.install_diffusers() is False

    out = " ".join(capsys.readouterr().out.split())
    assert f"{bootstrap.sys.executable} -m ensurepip --upgrade" in out


def test_install_diffusers_remedy_survives_a_venv_path_containing_square_brackets(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The interpreter goes into rich markup, so an unescaped `[...]` in it is eaten.

    A path like `/Users/me/[email protected]/.venv/bin/python` renders as
    `/Users/me//.venv/bin/python`: rich reads `[email protected]` as a style tag and drops it.
    The user is then handed a remedy naming an interpreter that does not exist, with
    nothing on screen saying anything was removed -- a misdirection that reads exactly
    like a correct instruction, which is worse than the dead end this remedy replaced.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: NO_INSTALLER_REMEDY.format(python=escape(sys.executable))
    Becomes: NO_INSTALLER_REMEDY.format(python=sys.executable)
    """
    bracketed = "/Users/me/[email protected]/.venv/bin/python"

    with (
        patch.object(bootstrap, "diffusers_install_command", return_value=None),
        patch.object(bootstrap.subprocess, "run"),
        patch.object(bootstrap.sys, "executable", bracketed),
    ):
        assert bootstrap.install_diffusers() is False

    out = " ".join(capsys.readouterr().out.split())
    assert f"{bracketed} -m ensurepip --upgrade" in out


def test_ensure_local_image_pipeline_remote_cuda() -> None:
    with patch.object(
        bootstrap,
        "probe_image_engines",
        return_value=_report(remote="http://10.0.0.99:8000", remote_alive=True),
    ):
        ready, engine = bootstrap.ensure_local_image_pipeline(interactive=False)

    assert ready is True
    assert engine == "remote-cuda"


def test_ensure_local_image_pipeline_falls_past_a_remote_worker_that_did_not_answer(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A configured address that does not answer is announced as such, not as an engine.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: if report.remote_ready:
    Becomes: if report.remote_url:
    """
    report = _report(
        remote="http://10.0.0.99:8000",
        remote_alive=False,
        deps_ok=True,
        checkpoint="/models/anillustrious_v4.safetensors",
    )

    with patch.object(bootstrap, "probe_image_engines", return_value=report):
        ready, engine = bootstrap.ensure_local_image_pipeline(interactive=False)

    out = " ".join(capsys.readouterr().out.split())
    assert (ready, engine) == (True, "diffusers-sdxl")
    assert "did not answer" in out


def test_ensure_local_image_pipeline_uses_a_detected_comfyui_without_installing_anything() -> None:
    """A running daemon is adopted; nothing is installed, started or asked about (#1095).

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: return ImageSetup(True, "comfyui-local")
    Becomes: return ImageSetup(False, "none")
    """
    with (
        patch.object(bootstrap, "probe_image_engines", return_value=_report(comfy_alive=True)),
        patch.object(bootstrap, "install_diffusers") as install,
    ):
        ready, engine = bootstrap.ensure_local_image_pipeline(interactive=True)

    assert (ready, engine) == (True, "comfyui-local")
    install.assert_not_called()


def test_ensure_local_image_pipeline_in_process_already_ready() -> None:
    report = _report(deps_ok=True, checkpoint="/models/anillustrious_v4.safetensors")

    with (
        patch.object(bootstrap, "probe_image_engines", return_value=report),
        patch.object(bootstrap, "install_diffusers") as install,
    ):
        ready, engine = bootstrap.ensure_local_image_pipeline(interactive=True)

    assert (ready, engine) == (True, "diffusers-sdxl")
    install.assert_not_called()


def test_ensure_local_image_pipeline_never_installs_packages_before_it_has_weights(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With no checkpoint, the packages are not installed — the weights come first.

    The engine needs both halves, and installing ~1 GB of `torch` beside a machine that
    still has no checkpoint produces exactly as many images as before: none. So a run
    that could not obtain the checkpoint stops there rather than spending the install.

    Killed by: src/uclone_x/cli/commands/bootstrap.py :: return ImageSetup(False, "none", None, outcome.reason or "no-checkpoint")
    Becomes: return ImageSetup(True, "diffusers-sdxl", None, outcome.reason or "no-checkpoint")

    Deleting that return instead — falling through to the dependency install — does not
    kill this test, and was measured not to: the consent gate declines on its own under
    the `stdin_is_interactive` patch below, so `install_diffusers` is never reached either
    way. What this test pins is the verdict, not the absence of the install call.
    """
    with (
        patch.object(bootstrap, "probe_image_engines", return_value=_report()),
        patch.object(bootstrap, "install_diffusers") as install,
        patch.object(bootstrap, "stdin_is_interactive", return_value=False),
        patch.object(bootstrap.Confirm, "ask") as ask,
    ):
        result = bootstrap.setup_local_image(interactive=True)

    assert (result.ready, result.engine, result.reason) == (False, "none", "declined")
    install.assert_not_called()
    ask.assert_not_called()
    assert "media status" in " ".join(capsys.readouterr().out.split())


def test_ensure_local_image_pipeline_offers_the_package_install_when_only_it_is_missing() -> None:
    report = _report(checkpoint="/models/anillustrious_v4.safetensors")

    with (
        patch.object(bootstrap, "probe_image_engines", return_value=report),
        patch.object(bootstrap, "stdin_is_interactive", return_value=True),
        patch.object(bootstrap.Confirm, "ask", return_value=True),
        patch.object(bootstrap, "install_diffusers", return_value=True) as install,
    ):
        ready, engine = bootstrap.ensure_local_image_pipeline(interactive=True)

    assert (ready, engine) == (True, "diffusers-sdxl")
    install.assert_called_once()


def test_ensure_local_image_pipeline_interactive_decline() -> None:
    report = _report(checkpoint="/models/anillustrious_v4.safetensors")

    with (
        patch.object(bootstrap, "probe_image_engines", return_value=report),
        patch.object(bootstrap, "stdin_is_interactive", return_value=True),
        patch.object(bootstrap.Confirm, "ask", return_value=False),
        patch.object(bootstrap, "install_diffusers") as install,
    ):
        ready, engine = bootstrap.ensure_local_image_pipeline(interactive=True)

    assert (ready, engine) == (False, "none")
    install.assert_not_called()


def test_ensure_local_image_pipeline_install_fails() -> None:
    report = _report(checkpoint="/models/anillustrious_v4.safetensors")

    with (
        patch.object(bootstrap, "probe_image_engines", return_value=report),
        patch.object(bootstrap, "stdin_is_interactive", return_value=True),
        patch.object(bootstrap.Confirm, "ask", return_value=True),
        patch.object(bootstrap, "install_diffusers", return_value=False),
    ):
        assert bootstrap.ensure_local_image_pipeline(interactive=True) == (False, "none")


def test_ensure_local_image_pipeline_non_interactive_asks_nothing() -> None:
    report = _report(checkpoint="/models/anillustrious_v4.safetensors")

    with (
        patch.object(bootstrap, "probe_image_engines", return_value=report),
        patch.object(bootstrap.Confirm, "ask") as ask,
    ):
        assert bootstrap.ensure_local_image_pipeline(interactive=False) == (False, "none")
    ask.assert_not_called()


# --- #971: the installer that runs when `ucx start` offers the image engine -------------


class _Completed:
    """The minimal shape of `subprocess.CompletedProcess` this code path reads."""

    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


class _CapturedRun:
    """Records the argv a test's stand-in `subprocess.run` was handed."""

    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], check: bool = False) -> _Completed:
        self.commands.append(list(command))
        return _Completed(self.returncode)


def _no_spec(name: str) -> Any:
    return None


def _a_spec(name: str) -> Any:
    return object()


def _uv_on_path(name: str) -> str | None:
    return "/opt/uv" if name == "uv" else None


def _nothing_on_path(name: str) -> str | None:
    return None


def _nothing_missing(requirements: object) -> tuple[str, ...]:
    """Stands in for the post-install read-back, so a command-shape test stays hermetic."""
    return ()


def test_uv_is_aimed_at_the_running_interpreter_and_not_the_ambient_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare `uv pip install` targets whatever uv discovers, which may not be us.

    Killed by: src/uclone_x/core/environment_install.py :: command = [uv_bin, "pip", "install", "--python", sys.executable, *packages]
    Becomes: command = [uv_bin, "pip", "install", *packages]
    """
    runner = _CapturedRun()
    monkeypatch.setattr("shutil.which", _uv_on_path)
    monkeypatch.setattr("subprocess.run", runner)
    monkeypatch.setattr("uclone_x.core.environment_install.still_missing", _nothing_missing)

    installed, reason = install_into_running_environment("mflux")

    assert installed is True
    assert reason == "installed"
    assert runner.commands == [["/opt/uv", "pip", "install", "--python", sys.executable, "mflux"]]


def test_pip_is_used_only_once_it_has_been_shown_to_be_importable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pip is absent from every uv-created environment, so its presence is checked.

    Killed by: src/uclone_x/core/environment_install.py :: command = [sys.executable, "-m", "pip", "install", *packages]
    Becomes: command = ["pip", "install", *packages]
    """
    runner = _CapturedRun()
    monkeypatch.setattr("shutil.which", _nothing_on_path)
    monkeypatch.setattr("importlib.util.find_spec", _a_spec)
    monkeypatch.setattr("subprocess.run", runner)
    monkeypatch.setattr("uclone_x.core.environment_install.still_missing", _nothing_missing)

    installed, _ = install_into_running_environment("mflux")

    assert installed is True
    assert runner.commands == [[sys.executable, "-m", "pip", "install", "mflux"]]


def test_with_neither_installer_no_subprocess_runs_and_the_remedy_is_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The only subprocess available is the one already known to fail (#971).

    Killed by: src/uclone_x/core/environment_install.py :: elif importlib.util.find_spec("pip") is not None:
    Becomes: elif True:
    """
    runner = _CapturedRun()
    monkeypatch.setattr("shutil.which", _nothing_on_path)
    monkeypatch.setattr("importlib.util.find_spec", _no_spec)
    monkeypatch.setattr("subprocess.run", runner)

    installed, reason = install_into_running_environment("mflux")

    assert installed is False
    assert runner.commands == []
    assert "neither uv nor pip" in reason
    assert "ensurepip" in reason


def test_a_nonzero_exit_is_reported_with_the_command_that_produced_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure the user cannot reproduce is not a report.

    Killed by: src/uclone_x/core/environment_install.py :: return False, f"`{' '.join(command)}` exited with code {result.returncode}."
    Becomes: return False, "install failed."
    """
    runner = _CapturedRun(returncode=2)
    monkeypatch.setattr("shutil.which", _uv_on_path)
    monkeypatch.setattr("subprocess.run", runner)

    installed, reason = install_into_running_environment("mflux")

    assert installed is False
    assert "exited with code 2" in reason
    assert "/opt/uv" in reason
