"""The agent must be able to repair its own installation, and only its own (#1107).

The tool exists because of a measured failure, not a hypothesis: asked three times to
install a missing image engine, the dashboard agent wrote a `pip install` how-to and
called nothing. So the tests here pin both halves of the fix -- that a named, allowlisted
installer reaches the real installer, and that a name the model invented does not.

The third half arrived from a live install into a throwaway environment: an extra is not
a thing that can be installed, and asking an installer for one that is not declared is
answered with a warning and exit code 0. Those tests are here too.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from uclone_x.core.environment_install import (
    install_into_running_environment,
    requirements_for_extra,
    resolve_installable,
    still_missing,
)
from uclone_x.tools.builtin.install import InstallPackageTool
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import create_default_registry


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(session_id="s", agent_id="a", workspace_root=tmp_path)


class _Recorder:
    """Stands in for the installer, recording what it was asked to install."""

    def __init__(self, result: tuple[bool, str]) -> None:
        self.result = result
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, *packages: str) -> tuple[bool, str]:
        self.calls.append(packages)
        return self.result


class _CapturedRun:
    """Stands in for `subprocess.run`, reporting the exit code it was given."""

    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **_: object) -> _CapturedRun:
        self.commands.append(list(command))
        return self


def _media_requirements(_extra: str) -> tuple[str, ...]:
    return ("pillow>=10.0.0", "diffusers>=0.31.0")


def _no_requirements(_extra: str) -> tuple[str, ...]:
    return ()


def _uv_on_path(_name: str) -> str:
    return "/opt/uv"


def _run(tool: InstallPackageTool, package: str, tmp_path: Path) -> dict[str, Any]:
    result = asyncio.run(tool.execute({"package": package}, _context(tmp_path)))
    assert result.success, result.error
    assert isinstance(result.output, dict)
    return result.output


def test_an_extra_is_installed_as_the_requirements_it_names_not_as_an_extra(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`media` reaches the installer as the distributions it names, never as `uclone-x[media]`.

    Asking uv for `uclone-x[media]` resolves `uclone-x` from an index, and an index copy
    whose metadata predates the extra warns and exits 0: nothing is installed, success is
    reported, and a stale copy of this project lands over the running one. Measured
    2026-09-17 against the published 0.1.2.

    Killed by: src/uclone_x/core/environment_install.py :: return requirements, "extra"
    Becomes: return (f"{PROJECT_DISTRIBUTION}[{extra.lower()}]",), "extra"
    """
    recorder = _Recorder((True, "installed"))
    monkeypatch.setattr(
        "uclone_x.core.environment_install.requirements_for_extra", _media_requirements
    )
    monkeypatch.setattr("uclone_x.tools.builtin.install.install_into_running_environment", recorder)

    output = _run(InstallPackageTool(), "media", tmp_path)

    assert recorder.calls == [("pillow>=10.0.0", "diffusers>=0.31.0")]
    assert output["installed"] is True
    assert output["resolved"] == ["pillow>=10.0.0", "diffusers>=0.31.0"]


def test_an_extra_this_installation_does_not_declare_is_refused_rather_than_attempted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An extra with no requirements behind it is a broken installation, not an install.

    Killed by: src/uclone_x/core/environment_install.py :: if not requirements:
    Becomes: if False:
    """
    recorder = _Recorder((True, "installed"))
    monkeypatch.setattr(
        "uclone_x.core.environment_install.requirements_for_extra", _no_requirements
    )
    monkeypatch.setattr("uclone_x.tools.builtin.install.install_into_running_environment", recorder)

    output = _run(InstallPackageTool(), "media", tmp_path)

    assert recorder.calls == []
    assert output["installed"] is False
    assert "declares no extra named `media`" in output["detail"]


def test_an_installer_that_exits_zero_without_installing_anything_is_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exit code 0 is not the result; the environment afterwards is (#1107).

    uv reports an extra it cannot find exactly as it reports a real install.

    Killed by: src/uclone_x/core/environment_install.py :: missing = still_missing(packages)
    Becomes: missing = ()
    """
    runner = _CapturedRun(returncode=0)
    monkeypatch.setattr("shutil.which", _uv_on_path)
    monkeypatch.setattr("subprocess.run", runner)

    installed, reason = install_into_running_environment("uclone-x-no-such-distribution")

    assert runner.commands != []
    assert installed is False
    assert "still not installed" in reason


def test_a_requirement_is_read_back_by_its_distribution_name_not_its_specifier() -> None:
    """`pillow>=10.0.0` is installed under the name `pillow`, specifier and all.

    Reading the specifier back verbatim would find nothing and report every successful
    install as a failure.

    Killed by: src/uclone_x/core/environment_install.py :: name = distribution_name(requirement)
    Becomes: name = requirement
    """
    assert still_missing(("pytest>=1.0",)) == ()


def test_an_underscored_extra_names_the_hyphenated_one_it_obviously_means(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`code_intel` is `code-intel`; PEP 685 normalises extra names, so this does too.

    Refusing it with "not part of this installation" would be untrue, and would send the
    model looking for a name that does not exist.

    Killed by: src/uclone_x/core/environment_install.py :: extra = extra.replace("_", "-").lower()
    Becomes: extra = extra.lower()
    """
    monkeypatch.setattr(
        "uclone_x.core.environment_install.requirements_for_extra", _media_requirements
    )

    requirements, reason = resolve_installable("code_intel")

    assert requirements == ("pillow>=10.0.0", "diffusers>=0.31.0")
    assert reason == "extra"


def test_a_requirement_whose_name_cannot_be_parsed_is_reported_absent_not_present() -> None:
    """An unparseable requirement is not evidence that anything was installed.

    Skipping it would count it present, which is exactly the false success this function
    exists to prevent.

    Killed by: src/uclone_x/core/environment_install.py :: absent.append(requirement)
    Becomes: pass
    """
    assert still_missing(("",)) == ("",)


def test_a_package_outside_this_project_is_refused_without_running_an_installer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A name the model invented does not become an install on the user's machine.

    The tool's argument is model-authored, so it is reachable by a confabulation or by an
    instruction injected into a page the agent fetched.

    Killed by: src/uclone_x/core/environment_install.py :: if candidate.lower() in INSTALLABLE_PACKAGES:
    Becomes: if True:
    """
    recorder = _Recorder((True, "installed"))
    monkeypatch.setattr("uclone_x.tools.builtin.install.install_into_running_environment", recorder)

    output = _run(InstallPackageTool(), "totally-not-ours", tmp_path)

    assert recorder.calls == []
    assert output["installed"] is False
    assert "media" in output["detail"]


def test_a_failed_install_is_reported_as_a_failure_with_what_the_installer_said(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A refusal to install must not read to the model as a completed install.

    Killed by: src/uclone_x/tools/builtin/install.py :: installed, detail = install_into_running_environment(*requirements)
    Becomes: installed, detail = (True, "installed")
    """
    recorder = _Recorder((False, "`uv pip install torch` exited with code 1."))
    monkeypatch.setattr("uclone_x.tools.builtin.install.install_into_running_environment", recorder)

    output = _run(InstallPackageTool(), "torch", tmp_path)

    assert output["installed"] is False
    assert "exited with code 1" in output["detail"]


def test_a_version_pin_is_refused_rather_than_silently_dropped() -> None:
    """Installing `torch` for a request that said `torch==0.1.0` is a substitution (P6).

    Killed by: src/uclone_x/core/environment_install.py :: if any(op in candidate for op in ("==", ">=", "<=", "~=", "!=", ">", "<")):
    Becomes: if False:
    """
    requirements, reason = resolve_installable("torch==0.1.0")

    assert requirements is None
    assert "pins a version" in reason


def test_extras_of_another_distribution_are_refused() -> None:
    """`requests[socks]` must not be read as "an extra, therefore ours".

    Killed by: src/uclone_x/core/environment_install.py :: if head_name not in {PROJECT_DISTRIBUTION, ".", ""}:
    Becomes: if False:
    """
    requirements, reason = resolve_installable("requests[socks]")

    assert requirements is None
    assert "not this project" in reason


def test_the_default_registry_advertises_the_installer_to_the_model() -> None:
    """A tool the inventory does not carry is a tool the model cannot call.

    This is the half of #1107 that no amount of prompt wording substitutes for: the
    dashboard's agent had `bash_run` advertised and still narrated, and the fix is a
    tool the failure message can name.

    Killed by: src/uclone_x/tools/registry.py :: InstallPackageTool(),
    Becomes:
    """
    registry = create_default_registry(enable_mcp=False)

    names = {tool.name for tool in registry.list_tools()}

    assert "install_package" in names


def _manifest(tree: Path, media: str) -> Path:
    tree.mkdir(parents=True, exist_ok=True)
    (tree / "pyproject.toml").write_text(
        f'[project]\nname = "uclone-x"\n\n[project.optional-dependencies]\nmedia = [{media}]\n',
        encoding="utf-8",
    )
    return tree


def test_an_editable_installs_source_tree_outranks_the_metadata_it_was_installed_with(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The snapshot taken at install time does not decide what the running code needs.

    `.dist-info` metadata is written once, when the project is installed; an editable
    install then goes on changing. #1095 replaced `media`'s `mflux` with `diffusers`,
    `torch` and `transformers`, and every environment installed before that kept
    declaring `mflux>=0.19.0`. That is not a harmless staleness: mflux still resolves on
    PyPI, so `install_package(package='media')` installed it, `still_missing` found the
    name present, the tool answered `installed: true`, and the retry failed exactly as
    before -- the repair loop #1107 exists to end, measured 2026-09-18 (PR #1096).

    Killed by: src/uclone_x/core/environment_install.py :: if declared_in_tree is not None:
    Becomes: if False:
    """
    tree = _manifest(tmp_path / "src-tree", '"diffusers>=0.31.0", "torch>=2.2.0"')
    monkeypatch.setattr("uclone_x.core.environment_install.editable_source_tree", lambda: tree)

    assert requirements_for_extra("media") == ("diffusers>=0.31.0", "torch>=2.2.0")


def test_a_source_tree_without_a_readable_manifest_falls_back_to_the_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The tree is preferred where it can answer, never a way for the question to fail.

    A checkout whose `pyproject.toml` is absent or unparseable must leave the installed
    metadata answering, because the alternative is an exception on the repair path.

    Killed by: src/uclone_x/core/environment_install.py :: except (OSError, tomllib.TOMLDecodeError):
    Becomes: except ():
    """
    empty = tmp_path / "no-manifest"
    empty.mkdir()
    monkeypatch.setattr("uclone_x.core.environment_install.editable_source_tree", lambda: empty)

    from_metadata = requirements_for_extra("media")

    monkeypatch.setattr("uclone_x.core.environment_install.editable_source_tree", lambda: None)
    assert requirements_for_extra("media") == from_metadata


def test_the_engines_the_image_pipeline_replacement_removed_can_no_longer_be_installed() -> None:
    """A refusal message must not name a distribution that repairs nothing.

    `mflux` and `mlx` left the project with #1095. Both still resolve on PyPI, so a model
    reading "Installable packages: mflux, mlx, pillow" would install mflux, be told it
    succeeded, and retry into the same failure -- the loop reproduced through the refusal
    message rather than the prompt (reviewer, PR #1096).

    Killed by: src/uclone_x/core/environment_install.py :: {"diffusers", "torch", "transformers", "pillow", "accelerate"}
    Becomes: {"diffusers", "torch", "transformers", "pillow", "accelerate", "mflux", "mlx"}
    """
    for gone in ("mflux", "mlx"):
        requirements, reason = resolve_installable(gone)
        assert requirements is None, f"{gone} is still installable"
        offered = reason.split("Installable packages:")[-1]
        assert "mflux" not in offered and "mlx" not in offered, reason

    assert resolve_installable("diffusers")[0] == ("diffusers",)
