"""CLI package for UClone-X."""

import importlib

from uclone_x.errors import MissingDependencyError

try:
    importlib.import_module("rich")
    importlib.import_module("typer")
except ImportError as exc:
    pkg = "typer" if "typer" in str(exc) else "rich"
    raise MissingDependencyError(
        extra="cli",
        package=pkg,
        feature="CLI shell (ucx)",
    ) from exc
