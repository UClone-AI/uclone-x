"""Find the installed extensions, and hand the core what they add (#2205).

An extension is an installed package declaring an entry point in the group
`uclone_x.extensions`. The entry point names an `Extension`, or a function of no arguments
that returns one. The core names none of them.

The core's own packages were searched too while the story lived among them (#2205); since
the story became its own example package (#2208) nothing in the core is an extension, and
that search is gone.

Finding is done once per process, on first use, and never at import: importing the
agent or the tool registry loads no extension. Every failure raises `ExtensionError` --
one that does not import, has no `EXTENSION`, is not an `Extension`, or adds something
that conflicts. Nothing is skipped with a log line, since a registry missing an
extension's tools would look like a working one (P6). The error's words name the
extension as it was installed and say what to do; where its entry point points and what
the exception said go to the log, never to the person (#2283).
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import logging
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
    "a2a_character_lookup",
    "discover",
    "extension_persona_dirs",
    "extension_lifecycle_hooks",
    "leased_folder_kinds",
    "loaded_extensions",
    "mount_extension_routes",
    "protected_roots",
    "use_extensions",
    "with_extension_tools",
]

logger = logging.getLogger(__name__)

#: The entry-point group an installed package declares its extension under.
ENTRY_POINT_GROUP: Final = "uclone_x.extensions"

_lock = threading.Lock()
_loaded: tuple[Extension, ...] | None = None
_override: tuple[Extension, ...] | None = None


def _built_for_a_newer_core(exc: Exception) -> bool:
    """Whether `exc` is an extension building `Extension` with a field this core lacks.

    That is an extension written against a newer core: most often a source checkout
    older than the extension the shared environment installed (#2245).
    """
    signature = f"{Extension.__qualname__}.__init__()"
    return isinstance(exc, TypeError) and str(exc).startswith(signature)


def _newer_than_the_core(name: str) -> str:
    """Plain words for `_built_for_a_newer_core`: what happened and what to do."""
    return (
        f"The extension {name!r} was written for a newer version of uclone-x than the one "
        "running here, so it could not start. If you run uclone-x from a source checkout, "
        "that checkout is older than the installed extension: merge origin/main into it. "
        "Otherwise, upgrade uclone-x."
    )


def _logged(error: ExtensionError) -> ExtensionError:
    """`error`, marked as one whose cause this module has logged (`cause_logged`)."""
    error.cause_logged = True
    return error


def _could_not_load(name: str) -> str:
    return (
        f"The extension {name!r} is installed but could not be loaded. Reinstall it, or "
        "uninstall it to run without it."
    )


def _could_not_start(name: str) -> str:
    return (
        f"The extension {name!r} could not start. Reinstall it, or uninstall it to run without it."
    )


def _not_an_extension(name: str) -> str:
    return (
        f"The package installed as the extension {name!r} does not provide a uclone-x "
        "extension. Uninstall it, or install a version made for uclone-x."
    )


def _named_twice(name: str) -> str:
    return (
        f"Two installed extensions are both named {name!r}, so neither can be used. "
        "Uninstall one of them."
    )


def _tools_failed(name: str) -> str:
    return (
        f"The extension {name!r} could not set up its tools. Reinstall it, or uninstall "
        "it to run without it."
    )


def _resolve(value: object, source: str, name: str) -> Extension:
    """`value` as an `Extension`: itself, or what calling it returns.

    `source` says where it was found, for a developer; `name` is the entry point's name,
    what a person installed it as, for the plain-words message (#2256).
    """
    if not isinstance(value, Extension) and callable(value):
        try:
            value = value()
        except ExtensionError as exc:
            # The extension's own words, or the contract's, kept as they are; the cause it
            # chains -- and the file it was raised in -- go to the log here, as for any
            # other failure, since nothing else logs them on the web head (#2294 review).
            logger.error("extension %r at %s failed to start", name, source, exc_info=exc)
            _logged(exc)
            raise
        except Exception as exc:
            logger.error("extension %r at %s failed to start", name, source, exc_info=exc)
            if _built_for_a_newer_core(exc):
                raise _logged(ExtensionError(_newer_than_the_core(name))) from exc
            raise _logged(ExtensionError(_could_not_start(name))) from exc
    if not isinstance(value, Extension):
        logger.error(
            "extension %r at %s gives a %s, not an Extension", name, source, type(value).__name__
        )
        raise _logged(ExtensionError(_not_an_extension(name)))
    return value if value.source else dataclasses.replace(value, source=source)


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
            logger.error("extension %r at %s could not be loaded", entry.name, source, exc_info=exc)
            if _built_for_a_newer_core(exc):
                raise _logged(ExtensionError(_newer_than_the_core(entry.name))) from exc
            raise _logged(ExtensionError(_could_not_load(entry.name))) from exc
        found.append(_resolve(value, source, entry.name))
    return found


def _checked(extensions: Sequence[Extension]) -> tuple[Extension, ...]:
    """`extensions` in name order, refused if two share a name or a protected folder."""
    by_name: dict[str, Extension] = {}
    roots: dict[str, str] = {}
    for extension in extensions:
        if extension.name in by_name:
            logger.error(
                "two extensions are named %r: %s and %s",
                extension.name,
                by_name[extension.name].source or "given",
                extension.source or "given",
            )
            raise _logged(ExtensionError(_named_twice(extension.name)))
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
    return _checked(_installed())


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
        except ExtensionError as error:
            logger.error("extension %r could not build its tools", extension.name, exc_info=error)
            _logged(error)
            raise
        except Exception as exc:
            logger.error("extension %r could not build its tools", extension.name, exc_info=exc)
            raise _logged(ExtensionError(_tools_failed(extension.name))) from exc
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


def extension_persona_dirs() -> tuple[Path, ...]:
    """Every folder of builtin personas an extension ships, in extension order."""
    return tuple(folder for extension in loaded_extensions() for folder in extension.personas)


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
