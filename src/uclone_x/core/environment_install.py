"""Installing a package into the environment `ucx` is itself running in (#971, #1107).

This lives in the kernel rather than in `cli/commands/bootstrap.py`, where it landed
with #971, because it now has two callers on opposite sides of a layer boundary: the
CLI bootstrap flow (a shell) and `install_package` (a kernel tool). `tools` is a kernel
layer and `cli` is a shell, so the tool importing the helper where it sat would be a
kernel-to-shell import, which the layer boundaries refuse outright -- there is no
known-violations escape hatch for that direction.

The allowlist is the other half of the move. #971's helper had exactly one caller and
one literal argument, `"mflux"`, so what it would install was settled at author time.
A tool's argument comes from a model, which means the package name is now attacker- and
hallucination-reachable: a prompt-injected instruction in a fetched page, or simply a
confabulated name, would otherwise become `uv pip install <anything>` on the user's
machine. `resolve_installable` fixes the reachable set to this project's own declared
extras and the engines they name, so the agent can repair *this* installation and
nothing else.

One deviation, named rather than hidden: `docs/core-shell-architecture.md` describes
`core/` as pure Python with no I/O, and `install_into_running_environment` runs a
subprocess. The enforced boundary and the prose disagree here -- the fitness test
classifies all of `tools` as kernel and carries no kernel-to-shell escape hatch, so a
shell placement is not available to the tool. The subprocess is confined to that one
function, and the rest of this module reads metadata off the filesystem -- the same
already-recorded deviation, not purity.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final, cast
from urllib.parse import urlparse
from urllib.request import url2pathname

#: Extras declared in `pyproject.toml` under `[project.optional-dependencies]`, plus the
#: engine distributions those extras name directly. An agent repairing its own
#: installation reaches exactly these; anything else is refused by name.
#:
#: `dev` is deliberately absent: the gate's toolchain is a builder's concern, and a
#: running dashboard has no reason to pull pytest and playwright into itself.
INSTALLABLE_EXTRAS: Final[frozenset[str]] = frozenset(
    {"cli", "http", "code-intel", "telemetry", "llm", "ontology", "media"}
)

#: Individual distributions an extra pulls in, installable on their own because a
#: failure is often specific to one of them -- an in-process image failure names the one
#: package that is missing (`'torch' is not installed`), and a model that reads that name
#: back out of the tool result asks for exactly it.
#:
#: These are the distributions the `media` extra declares, and they must stay that way:
#: #1095 replaced mflux/mlx with diffusers/torch/transformers in the extra but left this
#: set behind, so `install_package(package='torch')` -- the repair the engine's own
#: remedy text leads a model to -- was refused and pointed at `mflux` (reviewer, PR #1096).
INSTALLABLE_PACKAGES: Final[frozenset[str]] = frozenset(
    {"diffusers", "torch", "transformers", "pillow", "accelerate"}
)

PROJECT_DISTRIBUTION: Final[str] = "uclone-x"


def editable_source_tree() -> Path | None:
    """The source tree this project is an *editable* install of, if it is one.

    `None` for an ordinary wheel install, which is the case with nothing to disagree
    about: the metadata was built from the same code that is now running.
    """
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        recorded = distribution(PROJECT_DISTRIBUTION).read_text("direct_url.json")
    except (PackageNotFoundError, OSError):
        return None
    if not recorded:
        return None
    try:
        loaded: object = json.loads(recorded)
    except ValueError:
        return None
    if not isinstance(loaded, dict):
        return None
    record = cast(dict[str, object], loaded)
    dir_info = record.get("dir_info")
    if (
        not isinstance(dir_info, dict)
        or cast(dict[str, object], dir_info).get("editable") is not True
    ):
        return None
    url = record.get("url")
    if not isinstance(url, str) or not url.startswith("file://"):
        return None
    return Path(url2pathname(urlparse(url).path))


def _declared_in_source_tree(tree: Path, extra: str) -> tuple[str, ...] | None:
    """`extra` as `tree/pyproject.toml` declares it, or None if it cannot be read there."""
    import tomllib

    try:
        with (tree / "pyproject.toml").open("rb") as handle:
            document = cast(dict[str, object], tomllib.load(handle))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    project = document.get("project")
    if not isinstance(project, dict):
        return None
    optional = cast(dict[str, object], project).get("optional-dependencies")
    if not isinstance(optional, dict):
        return None
    declared = cast(dict[str, object], optional).get(extra)
    if not isinstance(declared, list):
        return None
    return tuple(str(item) for item in cast(list[object], declared))


def requirements_for_extra(extra: str) -> tuple[str, ...]:
    """The concrete requirements *this* installation declares for `extra`.

    Read from what this installation itself declares rather than assumed, because an
    extra is not a thing that can be installed: `uv pip install uclone-x[media]`
    resolves `uclone-x` from an index, and an index copy whose metadata predates the
    extra answers with a *warning* and exit code 0 (#1107, measured 2026-09-17 against
    the published 0.1.2, which declares no `media`). The install then reports success,
    installs nothing, and drops a stale copy of this project over the running one.
    Installing what the extra *names* avoids both.

    Two places can declare it, and the order matters. `.dist-info` metadata is a snapshot
    taken when the project was installed; for an **editable** install the code has gone on
    changing since, so the two can disagree and the source tree is the one that describes
    what is running. The tree is therefore preferred when there is one, and the metadata is
    the answer everywhere else.

    Reading the metadata first was a silent-wrong-install (#1096, measured 2026-09-18).
    #1095 replaced `media`'s `mflux` with `diffusers`, `torch` and `transformers`, but an
    editable install made before that still declared `mflux>=0.19.0, pillow>=10.0.0`. The
    earlier note here reasoned that a removed distribution would fail to resolve and exit
    nonzero -- untrue for one that still exists on PyPI. `install_package(package='media')`
    installed mflux, `still_missing` found both names present, the tool reported
    `installed: true`, and the retry failed exactly as before.
    """
    wanted = extra.lower()

    tree = editable_source_tree()
    if tree is not None:
        declared_in_tree = _declared_in_source_tree(tree, wanted)
        if declared_in_tree is not None:
            return declared_in_tree

    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import requires as declared_requirements

    try:
        declared = declared_requirements(PROJECT_DISTRIBUTION) or []
    except PackageNotFoundError:
        return ()

    found: list[str] = []
    for requirement in declared:
        base, _, marker = requirement.partition(";")
        if not marker:
            continue
        # The marker is `extra == "media"`, in either quoting.
        if f'extra == "{wanted}"' in marker or f"extra == '{wanted}'" in marker:
            found.append(base.strip())
    return tuple(found)


def distribution_name(requirement: str) -> str:
    """The bare distribution name in a requirement string (`torch>=2.2.0` -> `torch`)."""
    return re.split(r"[<>=!~\[;\s]", requirement, maxsplit=1)[0].strip()


def names_something_installable(package: str) -> tuple[tuple[str, str] | None, str]:
    """What `package` names in this project's allowlist, without reading any metadata.

    The parsing-and-normalisation half of `resolve_installable`, factored out so a second
    caller can ask the policy question alone (#1110). Returns `((kind, name), "")` where
    `kind` is `"extra"` or `"package"` and `name` is the normalised spelling, or
    `(None, reason)` with the refusal `resolve_installable` would have given.

    The two questions are genuinely different, and conflating them made an eval probe
    measure its own machine. `resolve_installable` answers *what should be installed*,
    which only this installation's own declarations can say. This answers *is that a name
    the tool would accept at all*, which is settled by the allowlist above and by nothing
    else. `evals/suites/model_probes.py` asks only the second: a model that replies
    `install_package(package='media')` did the right thing whether or not the machine the
    probe runs on still declares a `media` extra.

    A bare membership test against `INSTALLABLE_EXTRAS | INSTALLABLE_PACKAGES` is not this
    function and is why it exists: it would refuse `uclone-x[media]` and `.[media]`, both
    of which the tool accepts today.
    """
    candidate = package.strip()
    if not candidate:
        return None, "No package was named."

    # Refused by name rather than falling through to the allowlist miss below, which
    # would answer `torch==0.1.0` with "not part of this installation" -- untrue, and it
    # sends a model looking for a different name instead of dropping the pin.
    if any(op in candidate for op in ("==", ">=", "<=", "~=", "!=", ">", "<")):
        return None, (
            f"`{package}` pins a version. Versions are resolved by this project's own "
            f"dependency metadata, so ask for the component without a version specifier."
        )

    extra = candidate
    if candidate.endswith("]") and "[" in candidate:
        head, _, bracketed = candidate.partition("[")
        head_name = head.strip().rstrip("/").replace("_", "-").lower()
        if head_name not in {PROJECT_DISTRIBUTION, ".", ""}:
            return None, (
                f"`{package}` asks for extras of `{head}`, which is not this project. "
                f"Only `{PROJECT_DISTRIBUTION}` extras can be installed this way."
            )
        extra = bracketed[:-1].strip()

    # Normalised the way the head is, and the way PEP 685 requires: `code_intel` names
    # `code-intel`, and answering it with "not part of this installation" is untrue.
    extra = extra.replace("_", "-").lower()
    if extra in INSTALLABLE_EXTRAS:
        return ("extra", extra), ""
    if candidate.lower() in INSTALLABLE_PACKAGES:
        return ("package", candidate.lower()), ""

    return None, (
        f"`{package}` is not part of this installation, so it will not be installed. "
        f"Installable extras: {', '.join(sorted(INSTALLABLE_EXTRAS))}. "
        f"Installable packages: {', '.join(sorted(INSTALLABLE_PACKAGES))}."
    )


def resolve_installable(package: str) -> tuple[tuple[str, ...] | None, str]:
    """Resolve `package` to the requirements to install, or refuse it.

    Accepts a bare extra (`media`), this project with an extra in either spelling
    (`uclone-x[media]`, `.[media]`), or one of the named engine distributions
    (`torch`). Returns `(requirements, reason)`; `requirements` is None when refused and
    `reason` then names what may be installed instead.

    The name is checked by `names_something_installable`, which is metadata-free; what an
    accepted extra resolves to is read from this installation's own declarations, never
    from the allowlist -- see `requirements_for_extra` for the silent-success this avoids.

    Version specifiers are refused rather than stripped. Accepting `torch` while
    silently discarding the `==0.1.0` a caller appended would install something other
    than what was asked for, and report success -- the substitution P6 forbids.
    """
    named, refusal = names_something_installable(package)
    if named is None:
        return None, refusal

    kind, name = named
    if kind == "extra":
        requirements = requirements_for_extra(name)
        if not requirements:
            return None, (
                f"This installation declares no extra named `{name}`, so there is "
                f"nothing to install for it. Its metadata may predate the extra; a person "
                f"has to reinstall this project from its source tree."
            )
        return requirements, "extra"
    return (name,), "package"


def installer_command(*packages: str) -> list[str] | None:
    """The command that installs `packages` into the environment `sys.executable` runs in.

    **This is the only place the installer order is decided** (#1278). Every in-process
    install in this project goes through here, so that two callers cannot reach for
    different resolvers and different caches on the same machine. `None` when neither
    installer exists; the caller says what to do about that, because the sentence differs
    by where the user is standing.

    Two installers, tried in order:

    * `uv pip install --python <sys.executable>`. The `--python` is load-bearing.
      A bare `uv pip install` targets whatever environment uv discovers from the
      cwd and `VIRTUAL_ENV`, which is not necessarily the one `ucx` is running
      in, so the package can land where the running process will never see it.
    * `<sys.executable> -m pip install`, but only after `pip` is confirmed
      importable. It is absent from every uv-created environment, because
      `uv venv` does not seed it.

    uv is first, and the order is only observable in the one environment that holds both
    — a pip-seeded venv on a machine with uv on `PATH`. There, uv is still the right
    choice: `install.sh` builds this project's supported environment with uv against
    `uv.lock`, so uv resolves against the same resolver and cache the environment was
    created from, while pip's presence says only that *something* seeded the venv. uv
    reaches a pip-seeded environment correctly (that is what `--python` is for), whereas
    pip cannot reach a uv-created one at all, so preferring pip buys nothing and costs
    the agreement.
    """
    import importlib.util

    uv_bin = shutil.which("uv")
    if uv_bin:
        command = [uv_bin, "pip", "install", "--python", sys.executable, *packages]
    elif importlib.util.find_spec("pip") is not None:
        command = [sys.executable, "-m", "pip", "install", *packages]
    else:
        return None
    return command


def install_into_running_environment(*packages: str) -> tuple[bool, str]:
    """Install `packages` into the environment `sys.executable` belongs to.

    Two installers, tried in order by `installer_command`, which is where that order and
    its reason live; this function adds the refusal, the subprocess and the read-back.

    When neither is available no subprocess is launched, because the only one
    that could be launched is the one already known to fail. Returns
    (installed, reason); `reason` is a human-readable sentence on failure.

    A zero exit code is not the result. uv reports an extra it cannot find, and an
    already-satisfied requirement, the same way it reports a real install: exit 0. So
    the outcome is read back from the environment afterwards, and a requirement that is
    still absent is a failure however the installer exited (#1107).

    The read-back here is presence-only (`still_missing`). A caller that has a version
    floor to hold — `cli.commands.bootstrap.install_diffusers` and its
    `in_process_dependency_problems` — re-measures against that floor itself afterwards,
    because a distribution below its floor is present and this function will call it
    installed (#1278).

    No allowlist is applied here: this is the mechanism, and `resolve_installable` is
    the policy. The CLI's caller passes a literal it chose itself; the tool's caller
    passes whatever a model produced, and resolves it first.
    """
    if not packages:
        return False, "No package was named."
    named = ", ".join(packages)

    command = installer_command(*packages)
    if command is None:
        return False, (
            f"This environment has neither uv nor pip, so {named} cannot be installed "
            f"automatically. Install uv (https://docs.astral.sh/uv/), or seed pip with "
            f"`{sys.executable} -m ensurepip --upgrade`, and try again."
        )

    try:
        result = subprocess.run(command, check=False)
    except Exception as exc:  # noqa: BLE001 - any OS-level failure is reported, not raised
        return False, f"Running `{' '.join(command)}` failed: {exc}"
    if result.returncode != 0:
        return False, f"`{' '.join(command)}` exited with code {result.returncode}."

    missing = still_missing(packages)
    if missing:
        return False, (
            f"`{' '.join(command)}` exited 0 but {', '.join(missing)} "
            f"{'is' if len(missing) == 1 else 'are'} still not installed, so nothing was "
            f"actually repaired. The installer may have declined the requirement without "
            f"failing."
        )
    return True, "installed"


def still_missing(requirements: Sequence[str]) -> tuple[str, ...]:
    """Which of `requirements` are not installed in this environment, after a refresh.

    Distribution names rather than import names: `pillow` installs a module called
    `PIL`, and the requirement is the only name the caller has.
    """
    import importlib
    from importlib.metadata import PackageNotFoundError, distribution

    importlib.invalidate_caches()
    absent: list[str] = []
    for requirement in requirements:
        name = distribution_name(requirement)
        if not name:
            # Unparseable is not installed. Skipping would count it present, which is the
            # false-success this function exists to prevent (reviewer, PR #1108).
            absent.append(requirement)
            continue
        try:
            distribution(name)
        except PackageNotFoundError:
            absent.append(name)
    return tuple(absent)


__all__ = [
    "INSTALLABLE_EXTRAS",
    "INSTALLABLE_PACKAGES",
    "PROJECT_DISTRIBUTION",
    "distribution_name",
    "editable_source_tree",
    "install_into_running_environment",
    "installer_command",
    "names_something_installable",
    "requirements_for_extra",
    "resolve_installable",
    "still_missing",
]
