"""Embedded developer UI dashboard and server for UClone-X."""

import importlib

from uclone_x.errors import MissingDependencyError

try:
    importlib.import_module("fastapi")
    importlib.import_module("uvicorn")
except ImportError as exc:
    pkg = "fastapi" if "fastapi" in str(exc) else "uvicorn"
    raise MissingDependencyError(
        extra="http",
        package=pkg,
        feature="developer UI dashboard and server",
    ) from exc

from typing import TYPE_CHECKING, Any

from uclone_x.ui.app import create_ui_app

if TYPE_CHECKING:
    from uclone_x.ui.server import start_ui_server

__all__ = ["create_ui_app", "start_ui_server"]


def __getattr__(name: str) -> Any:
    """`start_ui_server` on first use, not at import (#2245).

    The launcher prints to a terminal through `rich`, the `cli` extra's; the app does not,
    so `uclone-x[http]` alone serves it (`uvicorn --factory uclone_x.ui.app:create_ui_app`).
    """
    if name == "start_ui_server":
        from uclone_x.ui.server import start_ui_server

        return start_ui_server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
