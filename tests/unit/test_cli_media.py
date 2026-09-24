"""Unit tests for `ucx media` — engine inspection that installs and starts nothing (#1095)."""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer

from uclone_x.cli.commands import media
from uclone_x.cli.commands.bootstrap import ImageEngineReport
from uclone_x.tools.builtin.image import (
    CheckpointResolution,
    CheckpointState,
    LocalDiffusersImageEngine,
)


def _report(
    *,
    remote: str | None = None,
    remote_alive: bool = False,
    comfy_alive: bool = False,
    deps_ok: bool = False,
    checkpoint: str | None = None,
) -> ImageEngineReport:
    return ImageEngineReport(
        remote_url=remote,
        remote_alive=remote_alive,
        comfy_url="http://127.0.0.1:8188",
        comfy_alive=comfy_alive,
        dependency_problems=() if deps_ok else ("'torch' is not installed (needs torch>=2.2.0)",),
        checkpoint=checkpoint,
    )


def _resolution(
    state: CheckpointState = "unconfigured",
    path: str | None = None,
    source: str | None = None,
) -> AbstractContextManager[MagicMock]:
    """Pin the checkpoint state to the test's, so the listing does not read this machine."""
    engine = MagicMock()
    engine.checkpoint_resolution.return_value = CheckpointResolution(state, path, source)
    return patch.object(media, "LocalDiffusersImageEngine", return_value=engine)


def test_human_bytes_switches_unit_at_a_gigabyte() -> None:
    """A size of exactly one gigabyte reads in GB, not as 1024 MB.

    Killed by: src/uclone_x/cli/commands/media.py :: if size >= 1024**3:
    Becomes: if size > 1024**3:
    """
    assert media.human_bytes(1024**3) == "1.00 GB"
    assert media.human_bytes(512 * 1024**2) == "512 MB"


def test_local_checkpoints_lists_the_configured_file_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = tmp_path / "configured.safetensors"
    configured.write_bytes(b"x" * 8)
    monkeypatch.setenv(media.IMAGE_CHECKPOINT_ENV, str(configured))

    with patch.object(media, "DEFAULT_CHECKPOINTS", ()):
        found = media.local_checkpoints()

    assert found == [(str(configured), 8)]


def test_local_checkpoints_omits_a_path_that_is_not_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configured-but-absent checkpoint must not be printed as one that is there (P6).

    Killed by: src/uclone_x/cli/commands/media.py :: if resolved in seen or not path.exists():
    Becomes: if resolved in seen:
    """
    monkeypatch.setenv(media.IMAGE_CHECKPOINT_ENV, str(tmp_path / "absent.safetensors"))

    with patch.object(media, "DEFAULT_CHECKPOINTS", ()):
        assert media.local_checkpoints() == []


def test_local_checkpoints_does_not_list_the_same_file_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "shared.safetensors"
    checkpoint.write_bytes(b"x" * 4)
    monkeypatch.setenv(media.IMAGE_CHECKPOINT_ENV, str(checkpoint))

    with patch.object(media, "DEFAULT_CHECKPOINTS", (str(checkpoint),)):
        assert media.local_checkpoints() == [(str(checkpoint), 4)]


def test_media_status_exits_nonzero_when_no_engine_is_ready(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ "Not ready" must be a failing exit code, so a script cannot read it as success (P6).

    Killed by: src/uclone_x/cli/commands/media.py :: if report.ready:
    Becomes: if True:
    """
    with (
        patch.object(media, "probe_image_engines", return_value=_report()),
        patch.object(media, "local_checkpoints", return_value=[]),
        _resolution("unconfigured"),
    ):
        with pytest.raises(typer.Exit) as exit_info:
            media.media_status()

    assert exit_info.value.exit_code == 1
    out = " ".join(capsys.readouterr().out.split())
    assert "No image engine is ready" in out
    assert "no checkpoint found" in out


def test_media_status_names_the_engine_that_would_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = _report(deps_ok=True, checkpoint="/models/anillustrious_v4.safetensors")

    with (
        patch.object(media, "probe_image_engines", return_value=report),
        _resolution("present", "/models/anillustrious_v4.safetensors"),
        patch.object(
            media,
            "local_checkpoints",
            return_value=[("/models/anillustrious_v4.safetensors", 1024**3)],
        ),
    ):
        media.media_status()

    out = " ".join(capsys.readouterr().out.split())
    assert "'diffusers-sdxl' would run" in out
    assert "1.00 GB" in out


def test_media_status_says_comfyui_is_optional_when_none_is_running(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing daemon is reported as optional, never as something to go and install.

    Killed by: src/uclone_x/cli/commands/media.py :: if not report.comfy_alive:
    Becomes: if False:
    """
    report = _report(deps_ok=True, checkpoint="/models/c.safetensors")

    with (
        patch.object(media, "probe_image_engines", return_value=report),
        _resolution("present", "/models/c.safetensors"),
        patch.object(media, "local_checkpoints", return_value=[("/models/c.safetensors", 10)]),
    ):
        media.media_status()

    out = " ".join(capsys.readouterr().out.split())
    assert "never installs or starts one" in out


def test_media_probe_exits_nonzero_when_no_daemon_answers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = AsyncMock()
    client.alive.return_value = False

    with patch("uclone_x.tools.builtin.comfy_client.ComfyClient", return_value=client):
        with pytest.raises(typer.Exit) as exit_info:
            media.media_probe()

    assert exit_info.value.exit_code == 1
    assert "No ComfyUI answered" in " ".join(capsys.readouterr().out.split())
    client.system_stats.assert_not_awaited()


def test_media_probe_prints_the_stats_of_a_daemon_that_answers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    client = AsyncMock()
    client.alive.return_value = True
    client.system_stats.return_value = {"system": {"comfyui_version": "0.3.60"}}

    with patch("uclone_x.tools.builtin.comfy_client.ComfyClient", return_value=client):
        media.media_probe()

    assert "0.3.60" in " ".join(capsys.readouterr().out.split())
    client.aclose.assert_awaited_once()


def test_media_probe_reports_a_failed_probe_rather_than_crashing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("uclone_x.tools.builtin.comfy_client.ComfyClient", side_effect=OSError("refused")):
        with pytest.raises(typer.Exit) as exit_info:
            media.media_probe()

    assert exit_info.value.exit_code == 1
    assert "Probe failed" in " ".join(capsys.readouterr().out.split())


def test_media_app_exposes_status_and_probe() -> None:
    names = {command.name for command in media.media_app.registered_commands}
    assert {"status", "probe"} <= names


def test_media_is_registered_on_the_cli() -> None:
    from uclone_x.cli.main import app

    assert any(group.name == "media" for group in app.registered_groups)


def test_media_status_prints_a_remote_worker_that_answered_its_probe(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = _report(remote="http://10.0.0.99:8000", remote_alive=True)

    with (
        patch.object(media, "probe_image_engines", return_value=report),
        _resolution("unconfigured"),
        patch.object(media, "local_checkpoints", return_value=[]),
    ):
        media.media_status()

    out = " ".join(capsys.readouterr().out.split())
    assert "http://10.0.0.99:8000 reachable" in out
    assert "'remote-cuda' would run" in out


def test_media_status_reports_a_configured_remote_worker_that_never_answered(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Configured is not running: an unreachable worker is not readiness, and exits 1.

    The command printed "Ready -- 'remote-cuda' would run" and exited 0 for a port with
    nothing listening, while the dispatcher refused the same environment outright
    (reviewer, PR #1096).

    Killed by: src/uclone_x/cli/commands/media.py :: elif report.remote_alive:
    Becomes: elif True:
    """
    report = _report(remote="http://127.0.0.1:59999", remote_alive=False)

    with (
        patch.object(media, "probe_image_engines", return_value=report),
        _resolution("unconfigured"),
        patch.object(media, "local_checkpoints", return_value=[]),
        pytest.raises(typer.Exit) as exit_info,
    ):
        media.media_status()

    out = " ".join(capsys.readouterr().out.split())
    assert exit_info.value.exit_code == 1
    assert "http://127.0.0.1:59999 unreachable" in out
    assert "would run" not in out
    assert "No image engine is ready" in out


def test_media_status_marks_the_checkpoint_that_would_be_loaded(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Two checkpoints on disk, one selected — the marker says which, and only that one.

    Killed by: src/uclone_x/cli/commands/media.py :: marker = "→" if in_process_selected and path == report.checkpoint else " "
    Becomes: marker = "→"
    """
    report = _report(deps_ok=True, checkpoint="/models/b.safetensors")
    listing = [("/models/a.safetensors", 10), ("/models/b.safetensors", 20)]

    with (
        patch.object(media, "probe_image_engines", return_value=report),
        _resolution("present", "/models/b.safetensors"),
        patch.object(media, "local_checkpoints", return_value=listing),
    ):
        media.media_status()

    out = capsys.readouterr().out
    marked = [line for line in out.splitlines() if "→" in line]
    assert len(marked) == 1
    assert "b.safetensors" in marked[0]


def test_local_checkpoints_expands_a_home_relative_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`~/ai_models/...` is a literal directory name until it is expanded.

    Killed by: src/uclone_x/cli/commands/media.py :: path = Path(expand_checkpoint_path(candidate))
    Becomes: path = Path(candidate)
    """
    monkeypatch.delenv(media.IMAGE_CHECKPOINT_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    nested = tmp_path / "ai_models" / "checkpoints"
    nested.mkdir(parents=True)
    (nested / "x.safetensors").write_bytes(b"x" * 2)

    with patch.object(media, "DEFAULT_CHECKPOINTS", ("~/ai_models/checkpoints/x.safetensors",)):
        found = media.local_checkpoints()

    assert found == [(str(nested / "x.safetensors"), 2)]


def test_a_configured_tilde_path_lists_and_resolves_the_same_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The listing and the resolver must not disagree about one configured path (#1123, P6).

    `local_checkpoints()` expanded `~` and `checkpoint_resolution()` did not, so
    `UCX_IMAGE_CHECKPOINT=~/ai_models/checkpoints/foo.safetensors` showed the file in
    `ucx media status` and then reported it `missing` on the next line. "Present" about a
    path nothing will open is the defect, whichever way the tie is broken; this asserts the
    two halves agree, and that the agreed answer is the one the file on disk supports.

    Killed by: src/uclone_x/tools/builtin/image.py :: path = expand_checkpoint_path(configured)
    Becomes: path = configured
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    nested = tmp_path / "ai_models" / "checkpoints"
    nested.mkdir(parents=True)
    real = nested / "foo.safetensors"
    real.write_bytes(b"x" * 8)
    monkeypatch.setenv(media.IMAGE_CHECKPOINT_ENV, "~/ai_models/checkpoints/foo.safetensors")

    listed = media.local_checkpoints()
    resolution = LocalDiffusersImageEngine().checkpoint_resolution()

    assert listed == [(str(real), 8)]
    assert (resolution.state, resolution.path) == ("present", str(real))
    # The `→` marker in `media status` compares the listed string against the resolved one,
    # so agreeing on "present" is not enough — they must agree on the spelling too.
    assert [path for path, _ in listed] == [resolution.path]
    assert LocalDiffusersImageEngine().resolve_checkpoint() == str(real)


def test_a_configured_tilde_path_with_no_file_is_absent_on_both_sides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Agreement has to hold in the negative direction too, or it is half a fix (#1123).

    The `missing` message names the path as typed, because the expanded one appears in no
    file the user can go and edit; it names the expansion as well, because that is where
    the engine actually looked.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "ai_models" / "checkpoints").mkdir(parents=True)
    monkeypatch.setenv(media.IMAGE_CHECKPOINT_ENV, "~/ai_models/checkpoints/typo.safetensors")

    listed = media.local_checkpoints()
    resolution = LocalDiffusersImageEngine().checkpoint_resolution()

    assert listed == []
    assert resolution.state == "missing"
    assert resolution.usable is False
    described = resolution.describe()
    assert "'~/ai_models/checkpoints/typo.safetensors'" in described
    assert str(tmp_path / "ai_models" / "checkpoints" / "typo.safetensors") in described


def test_media_status_names_each_unmet_dependency(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A beginner whose environment lacks torch must be told that, not just "missing".

    Killed by: src/uclone_x/cli/commands/media.py :: for problem in report.dependency_problems:
    Becomes: for problem in ():
    """
    report = _report(checkpoint="/models/b.safetensors")

    with (
        patch.object(media, "probe_image_engines", return_value=report),
        _resolution("present", "/models/b.safetensors"),
        patch.object(media, "local_checkpoints", return_value=[("/models/b.safetensors", 1)]),
    ):
        with pytest.raises(typer.Exit) as exit_info:
            media.media_status()

    assert exit_info.value.exit_code == 1
    out = " ".join(capsys.readouterr().out.split())
    assert "'torch' is not installed" in out
    assert "dependencies missing" in out


def test_media_status_tells_a_mistyped_checkpoint_apart_from_an_unset_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The three checkpoint states must read as three outputs, not two (#1120, P6).

    Configured-but-absent used to print the nothing-configured remedy, so a user who had
    mistyped `UCX_IMAGE_CHECKPOINT` was told to set `UCX_IMAGE_CHECKPOINT`.

    Killed by: src/uclone_x/cli/commands/media.py :: headline = "checkpoint missing" if resolution.state == "missing" else "no checkpoint found"
    Becomes: headline = "no checkpoint found"
    """
    outputs: dict[str, str] = {}
    cases: dict[
        str,
        tuple[ImageEngineReport, AbstractContextManager[MagicMock], list[tuple[str, int]]],
    ] = {
        "unconfigured": (_report(deps_ok=True), _resolution("unconfigured"), []),
        "missing": (
            _report(deps_ok=True),
            _resolution("missing", "/models/typo.safetensors", media.IMAGE_CHECKPOINT_ENV),
            [],
        ),
        "present": (
            _report(deps_ok=True, checkpoint="/models/there.safetensors"),
            _resolution("present", "/models/there.safetensors"),
            [("/models/there.safetensors", 10)],
        ),
    }
    for name, (report, resolution, listing) in cases.items():
        with (
            patch.object(media, "probe_image_engines", return_value=report),
            patch.object(media, "local_checkpoints", return_value=listing),
            resolution,
        ):
            if report.ready:
                media.media_status()
            else:
                with pytest.raises(typer.Exit):
                    media.media_status()
        outputs[name] = " ".join(capsys.readouterr().out.split())

    assert len(set(outputs.values())) == 3
    assert "no checkpoint found" in outputs["unconfigured"]
    assert "typo.safetensors" not in outputs["unconfigured"]
    assert "checkpoint missing" in outputs["missing"]
    assert "typo.safetensors" in outputs["missing"]
    assert "no checkpoint found" not in outputs["missing"]
    assert "there.safetensors" in outputs["present"]
    assert "checkpoint" not in outputs["present"].replace("checkpoint file", "")


def test_media_status_does_not_mark_a_checkpoint_its_engine_cannot_load(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The `→` names the engine that would run; without `torch` that is not this one (#1120).

    The marker matched on the path alone, so a checkpoint on disk was pointed at while the
    in-process engine was reported as missing its dependencies in the line above it — the
    listing promised a load that must fail (P6).

    Killed by: src/uclone_x/cli/commands/media.py :: marker = "→" if in_process_selected and path == report.checkpoint else " "
    Becomes: marker = "→" if path == report.checkpoint else " "
    """
    report = _report(deps_ok=False, checkpoint="/models/b.safetensors")
    assert report.engine == "none"

    with (
        patch.object(media, "probe_image_engines", return_value=report),
        patch.object(media, "local_checkpoints", return_value=[("/models/b.safetensors", 10)]),
        _resolution("present", "/models/b.safetensors"),
    ):
        with pytest.raises(typer.Exit):
            media.media_status()

    out = capsys.readouterr().out
    assert "b.safetensors" in out
    assert "→" not in out


def test_media_status_does_not_mark_the_local_checkpoint_when_comfyui_would_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A warm ComfyUI daemon outranks the in-process engine, so the local file is not selected.

    Killed by: src/uclone_x/cli/commands/media.py :: in_process_selected = report.engine == "diffusers-sdxl"
    Becomes: in_process_selected = True
    """
    report = _report(deps_ok=True, comfy_alive=True, checkpoint="/models/b.safetensors")
    assert report.engine == "comfyui-local"

    with (
        patch.object(media, "probe_image_engines", return_value=report),
        patch.object(media, "local_checkpoints", return_value=[("/models/b.safetensors", 10)]),
        _resolution("present", "/models/b.safetensors"),
    ):
        media.media_status()

    out = capsys.readouterr().out
    assert "b.safetensors" in out
    assert "→" not in out
