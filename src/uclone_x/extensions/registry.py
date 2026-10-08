"""Find the installed extensions, and hand the core what they add (#2205).

Two places are searched, and neither is named in code by what it holds:

* **In the tree:** every top-level package of `uclone_x` that has an
  `ucx_extension.py` module. The module's `EXTENSION` is the extension. The file is found
  by name before anything is imported, so a package without one is never loaded.
* **Installed:** every entry point in the group `uclone_x.extensions`. The entry point
  names an `Extension`, or a function of no arguments that returns one.

Why both, and why in this order, is in the design for #2205.

Finding is done once per process, on first use, and never at import: importing the
agent or the tool registry loads no extension. Every failure raises `ExtensionError`
naming the extension and where it came from -- one that does not import, has no
`EXTENSION`, is not an `Extension`, or adds something that conflicts. Nothing is skipped
with a log line, since a registry missing an extension's tools would look like a working
one (P6).
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.metadata
import threading
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from uclone_x.extensions.contract import Extension, ExtensionError, ProtectedRoot

if TYPE_CHECKING:
    from uclone_x.agent.protocols import TurnLifecycleHookProtocol
    from uclone_x.artifacts.kinds import LeasedFolderKind
    from uclone_x.tools.builtin.a2a import CharacterLookup
    from uclone_x.tools.models import ToolContext
    from uclone_x.tools.protocols import ToolProtocol

__all__ = [
    "ENTRY_POINT_GROUP",
    "IN_TREE_MODULE",
    "a2a_character_lookup",
    "discover",
    "extension_lifecycle_hooks",
    "leased_folder_kinds",
    "loaded_extensions",
    "mount_extension_routes",
    "protected_roots",
    "use_extensions",
    "with_extension_tools",
]

#: The entry-point group an installed package declares its extension under.
ENTRY_POINT_GROUP: Final = "uclone_x.extensions"

#: The module a top-level package of `uclone_x` declares its extension in.
IN_TREE_MODULE: Final = "ucx_extension"

#: The package searched for in-tree extensions. Read, not named: this is the core itself.
_ROOT_PACKAGE: Final = __name__.split(".", 1)[0]

_lock = threading.Lock()
_loaded: tuple[Extension, ...] | None = None
_override: tuple[Extension, ...] | None = None


def _resolve(value: object, source: str) -> Extension:
    """`value` as an `Extension`: itself, or what calling it returns."""
    if not isinstance(value, Extension) and callable(value):
        try:
            value = value()
        except ExtensionError:
            raise
        except Exception as exc:
            raise ExtensionError(f"The extension at {source} failed to start: {exc}") from exc
    if not isinstance(value, Extension):
        raise ExtensionError(
            f"{source} is not an extension: it gives a {type(value).__name__}, not an Extension."
        )
    return value if value.source else dataclasses.replace(value, source=source)


def _in_tree() -> list[Extension]:
    """The extensions declared by `ucx_extension.py` modules in the core's own packages."""
    root_module = importlib.import_module(_ROOT_PACKAGE)
    found: list[Extension] = []
    for directory in root_module.__path__:
        for package in sorted(Path(directory).iterdir()):
            if not (package / f"{IN_TREE_MODULE}.py").is_file():
                continue
            module_name = f"{_ROOT_PACKAGE}.{package.name}.{IN_TREE_MODULE}"
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:
                raise ExtensionError(
                    f"The extension module {module_name} could not be imported: {exc}"
                ) from exc
            if not hasattr(module, "EXTENSION"):
                raise ExtensionError(f"The extension module {module_name} has no EXTENSION.")
            found.append(_resolve(module.EXTENSION, module_name))
    return found


def _installed() -> list[Extension]:
    """The extensions installed packages declare under `ENTRY_POINT_GROUP`."""
    found: list[Extension] = []
    for entry in sorted(
        importlib.metadata.entry_points(group=ENTRY_POINT_GROUP), key=lambda e: e.name
    ):
        source = f"entry point {entry.name!r} ({entry.value})"
        try:
            value = entry.load()
        except Exception as exc:
            raise ExtensionError(f"The extension at {source} could not be loaded: {exc}") from exc
        found.append(_resolve(value, source))
    return found


def _checked(extensions: Sequence[Extension]) -> tuple[Extension, ...]:
    """`extensions` in name order, refused if two share a name or a protected folder."""
    by_name: dict[str, Extension] = {}
    roots: dict[str, str] = {}
    for extension in extensions:
        if extension.name in by_name:
            raise ExtensionError(
                f"Two extensions are named {extension.name!r} "
                f"({by_name[extension.name].source or 'given'} and {extension.source or 'given'})."
            )
        by_name[extension.name] = extension
        for root in extension.protected_roots:
            holder = roots.setdefault(root.dirname, extension.name)
            if holder != extension.name:
                raise ExtensionError(
                    f"The extensions {holder!r} and {extension.name!r} both protect the "
                    f"folder {root.dirname!r}."
                )
    return tuple(by_name[name] for name in sorted(by_name))


def discover() -> tuple[Extension, ...]:
    """Every extension, in name order, found now. `loaded_extensions` keeps the answer."""
    return _checked([*_in_tree(), *_installed()])


def loaded_extensions() -> tuple[Extension, ...]:
    """The extensions this process runs with, found on first use."""
    global _loaded
    if _override is not None:
        return _override
    with _lock:
        if _loaded is None:
            _loaded = discover()
        return _loaded


@contextmanager
def use_extensions(extensions: Sequence[Extension]) -> Generator[None]:
    """Run with exactly `extensions`, in place of the ones found (for tests)."""
    global _override
    before = _override
    _override = _checked(extensions)
    try:
        yield
    finally:
        _override = before


# -- what the core asks for -----------------------------------------------------------------


def with_extension_tools(core: Sequence[ToolProtocol]) -> list[ToolProtocol]:
    """`core` with every extension's tools: replacements in place, the rest after it.

    A tool may take a core tool's place only when its extension names that tool in
    `replaces_tools`, and only one extension may replace a tool. Any other repeated name
    is an error: a registry keyed by name would keep one of the two in silence.
    """
    tools = list(core)
    position = {tool.name: index for index, tool in enumerate(tools)}
    core_names = frozenset(position)
    owner: dict[str, str] = {}
    for extension in loaded_extensions():
        for name in extension.replaces_tools:
            if name not in core_names:
                raise ExtensionError(
                    f"The extension {extension.name!r} replaces the tool {name!r}, which the "
                    "core does not have."
                )
        try:
            added = list(extension.tools())
        except ExtensionError:
            raise
        except Exception as exc:
            raise ExtensionError(
                f"The extension {extension.name!r} could not build its tools: {exc}"
            ) from exc
        replaced: set[str] = set()
        for tool in added:
            name = tool.name
            if name in owner:
                raise ExtensionError(
                    f"The extensions {owner[name]!r} and {extension.name!r} both add a tool "
                    f"named {name!r}."
                )
            owner[name] = extension.name
            if name in core_names:
                if name not in extension.replaces_tools:
                    raise ExtensionError(
                        f"The extension {extension.name!r} adds a tool named {name!r}, which "
                        "the core already has. Name it in replaces_tools to replace it."
                    )
                tools[position[name]] = tool
                replaced.add(name)
            else:
                tools.append(tool)
        missing = sorted(set(extension.replaces_tools) - replaced)
        if missing:
            raise ExtensionError(
                f"The extension {extension.name!r} says it replaces {', '.join(missing)} but "
                "adds no tool by that name."
            )
    return tools


def extension_lifecycle_hooks() -> list[TurnLifecycleHookProtocol]:
    """Every extension's turn lifecycle hooks, in extension order."""
    hooks: list[TurnLifecycleHookProtocol] = []
    for extension in loaded_extensions():
        hooks.extend(extension.lifecycle_hooks())
    return hooks


def protected_roots() -> tuple[ProtectedRoot, ...]:
    """Every workspace folder an extension protects from general tools."""
    return tuple(root for extension in loaded_extensions() for root in extension.protected_roots)


def leased_folder_kinds() -> tuple[LeasedFolderKind, ...]:
    """The leased folder kinds the Files screen lists; at most one (`Extension`)."""
    kinds: list[tuple[str, LeasedFolderKind]] = [
        (extension.name, kind)
        for extension in loaded_extensions()
        for kind in extension.leased_folders()
    ]
    if len(kinds) > 1:
        raise ExtensionError(
            "More than one leased folder kind is installed ("
            + ", ".join(f"{name}: {kind.root}" for name, kind in kinds)
            + "), and a conversation keeps one open item. Only one is supported."
        )
    return tuple(kind for _, kind in kinds)


def a2a_character_lookup() -> CharacterLookup | None:
    """The lookup `a2a_call` hands a peer characters with, or `None` when none adds one."""
    lookups = [
        extension.a2a_characters()
        for extension in loaded_extensions()
        if extension.a2a_characters is not None
    ]
    if not lookups:
        return None
    if len(lookups) == 1:
        return lookups[0]

    def every(context: ToolContext, texts: Sequence[str]) -> list[dict[str, Any]]:
        return [found for lookup in lookups for found in lookup(context, texts)]

    return every


def mount_extension_routes(context: object) -> list[str]:
    """Call every extension's `routes` with the head's `context`; name those mounted."""
    mounted: list[str] = []
    for extension in loaded_extensions():
        register: Callable[[Any], None] | None = extension.routes
        if register is None:
            continue
        register(context)
        mounted.append(extension.name)
    return mounted
