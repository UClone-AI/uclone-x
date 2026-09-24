"""Smoke tests for UClone-X package initialization and version."""

import json
import tomllib
from pathlib import Path
from typing import cast

import uclone_x

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_version_is_declared_once_in_effect() -> None:
    """The three version declarations must agree with each other.

    They are three files -- `pyproject.toml`, `src/uclone_x/__init__.py` and
    `frontend/package.json` -- and nothing makes one follow another. A release
    that bumps two of them publishes a wheel whose metadata and `__version__`
    disagree, which no gate would notice.

    This test used to pin the literal `"0.1.0"`, which made it a chore at every
    bump and checked nothing a reader could not read off the file. Pinning
    agreement instead is the assertion with a failure mode behind it.

    Mutation: bump `version` in `pyproject.toml` alone -- this test fails and
    names both values.
    """
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = cast("dict[str, object]", pyproject["project"])
    declared = project["version"]
    assert isinstance(declared, str)

    assert uclone_x.__version__ == declared, (
        f"src/uclone_x/__init__.py declares {uclone_x.__version__!r}, "
        f"pyproject.toml declares {declared!r}"
    )

    # Read unconditionally. Guarding on `is_file()` would make "the file is
    # missing" indistinguishable from "the versions agree", and there is no tree
    # this runs in where it is absent: the tests do not ship in the sdist, and
    # the exported public tree carries `frontend/` and runs this suite.
    package_json = REPO_ROOT / "frontend" / "package.json"
    parsed = cast("dict[str, object]", json.loads(package_json.read_text(encoding="utf-8")))
    ui_version = parsed["version"]
    assert ui_version == declared, (
        f"frontend/package.json declares {ui_version!r}, pyproject.toml declares {declared!r}"
    )


def test_imports() -> None:
    from uclone_x.cli.main import app

    assert app is not None
