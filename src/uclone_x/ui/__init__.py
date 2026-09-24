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

from uclone_x.ui.app import create_ui_app
from uclone_x.ui.server import start_ui_server

__all__ = ["create_ui_app", "start_ui_server"]
