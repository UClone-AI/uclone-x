"""Declarative persona discovery and registry engine (Principle 0, Principle 4)."""

from __future__ import annotations

import difflib
import logging
from collections.abc import Collection, Sequence
from pathlib import Path

from uclone_x.agent.models import PersonaDefinition
from uclone_x.agent.persona_store import (
    BUILTIN_PERSONAS_DIR,
    DEFAULT_PERSONA_NAME,
    DEFAULT_WORKSPACE_PERSONAS_SUBDIR,
    CompositePersonaStore,
    InMemoryPersonaStore,
    PersonaDraft,
    PersonaLoadError,
    PersonaNotFound,
    PersonaStoreProtocol,
    PersonaWriteConflict,
    PersonaWriteError,
    PersonaWriteRefused,
    YamlFilePersonaStore,
    split_appended_default_prompt,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BUILTIN_PERSONAS_DIR",
    "DEFAULT_PERSONA_NAME",
    "DEFAULT_WORKSPACE_PERSONAS_SUBDIR",
    "CompositePersonaStore",
    "InMemoryPersonaStore",
    "PersonaDraft",
    "PersonaLoadError",
    "PersonaNotFound",
    "PersonaRegistry",
    "PersonaStoreProtocol",
    "PersonaWriteConflict",
    "PersonaWriteError",
    "PersonaWriteRefused",
    "YamlFilePersonaStore",
    "get_default_persona_registry",
    "split_appended_default_prompt",
]


class PersonaRegistry:
    """Catalog and discovery engine for agent persona blueprints (Principle 0).

    Discovers personas from package built-in YAML files and workspace directories,
    enabling zero-code agent extensibility without modifying core code.
    """

    def __init__(
        self,
        workspace_root: Path | None = None,
        extra_dirs: Sequence[Path] | None = None,
        include_defaults: bool = True,
        tool_names: Collection[str] | None = None,
        store: PersonaStoreProtocol | None = None,
    ) -> None:
        self._workspace_root = workspace_root.resolve() if workspace_root else None
        self._extra_dirs = [d.resolve() for d in (extra_dirs or [])]
        #: Whether the package's own `personas/` directory is loaded. The built-in three
        #: are files in it like any other persona, so turning this off means a registry
        #: that knows only what the caller pointed it at.
        self._include_defaults = include_defaults
        #: Tool inventory `allowed_tools` is checked against. `None` disables the check --
        #: see `_validate_tools` for why that is a stated position and not a default.
        self._tool_names: frozenset[str] | None = (
            frozenset(tool_names) if tool_names is not None else None
        )
        self._custom_store = store
        self._store: PersonaStoreProtocol
        self._dynamic_personas: dict[str, PersonaDefinition] = {}
        #: Names the package's own directory defines, whether or not a later file overrides
        #: them -- what lets the head say an edited built-in is an override.
        self._builtin_names: frozenset[str] = frozenset()
        self.reload()

    @property
    def workspace_root(self) -> Path | None:
        """Configured workspace root directory."""
        return self._workspace_root

    @property
    def validates_tools(self) -> bool:
        """Whether this registry was given a tool inventory to check `allowed_tools` against."""
        return self._tool_names is not None

    @property
    def store(self) -> PersonaStoreProtocol:
        """The underlying persona store instance."""
        return self._store

    def reload(self) -> None:
        """Discover and load personas from built-in and workspace directories or configured store."""
        self._dynamic_personas.clear()
        if self._custom_store is not None:
            self._store = self._custom_store
            reload_fn = getattr(self._store, "reload", None)
            if callable(reload_fn):
                reload_fn()
            if BUILTIN_PERSONAS_DIR.is_dir():
                self._builtin_names = frozenset(
                    p.stem
                    for p in sorted(BUILTIN_PERSONAS_DIR.glob("*.yaml"))
                    + sorted(BUILTIN_PERSONAS_DIR.glob("*.yml"))
                )
            else:
                self._builtin_names = frozenset()
            return

        stores: list[PersonaStoreProtocol] = []
        writable_store: YamlFilePersonaStore | None = None

        # 1. Workspace store (.uclone/personas/)
        if self._workspace_root:
            ws_personas_dir = self._workspace_root / DEFAULT_WORKSPACE_PERSONAS_SUBDIR
            writable_store = YamlFilePersonaStore(
                ws_personas_dir,
                tool_names=self._tool_names,
                read_only=False,
            )
            stores.append(writable_store)

        # 2. Extra specified directories
        for ed in self._extra_dirs:
            if ed.is_dir():
                stores.append(
                    YamlFilePersonaStore(
                        ed,
                        tool_names=self._tool_names,
                        read_only=True,
                    )
                )

        # 3. Built-in package YAML personas (src/uclone_x/personas/)
        if self._include_defaults and BUILTIN_PERSONAS_DIR.is_dir():
            builtin_store = YamlFilePersonaStore(
                BUILTIN_PERSONAS_DIR,
                tool_names=self._tool_names,
                read_only=True,
            )
            stores.append(builtin_store)
            self._builtin_names = frozenset(p.name for p in builtin_store.list_personas())
        else:
            self._builtin_names = frozenset()

        self._store = CompositePersonaStore(
            stores=stores,
            writable_store=writable_store,
            overridable_dirs=[BUILTIN_PERSONAS_DIR],
        )

    def _validate_tools(self, persona: PersonaDefinition, *, source: Path | str) -> None:
        """Refuse a persona naming a tool that does not exist."""
        if self._tool_names is None or not persona.allowed_tools:
            return
        unknown = [name for name in persona.allowed_tools if name not in self._tool_names]
        if not unknown:
            return
        known = sorted(self._tool_names)
        details = "; ".join(
            f"{name!r}"
            + (
                f" (did you mean {', '.join(repr(m) for m in matches)}?)"
                if (matches := difflib.get_close_matches(name, known, n=2, cutoff=0.5))
                else ""
            )
            for name in unknown
        )
        raise PersonaLoadError(
            f"{source}: persona {persona.name!r} declares tool(s) that are not registered: "
            f"{details}. Registered: {', '.join(known)}"
        )

    def list_personas(self) -> list[PersonaDefinition]:
        """Return list of all registered persona definitions with default persona ('clone') first."""
        personas_by_name: dict[str, PersonaDefinition] = {}
        for p in self._store.list_personas():
            personas_by_name[p.name] = p
        for p in self._dynamic_personas.values():
            personas_by_name[p.name] = p
        return sorted(
            personas_by_name.values(),
            key=lambda p: (0 if p.name == DEFAULT_PERSONA_NAME else 1, p.name),
        )

    def get_persona(self, name: str) -> PersonaDefinition | None:
        """Get persona definition by unique name."""
        if name in self._dynamic_personas:
            return self._dynamic_personas[name]
        return self._store.get_persona(name)

    def register_persona(self, persona: PersonaDefinition) -> None:
        """Dynamically register or override a persona definition in memory."""
        self._validate_tools(persona, source="<registered in memory>")
        self._dynamic_personas[persona.name] = persona

    def source_of(self, name: str) -> Path | None:
        """The file a persona was loaded from, or `None` for one registered in memory."""
        if name in self._dynamic_personas:
            return None
        source = self._store.source_of(name)
        return source if isinstance(source, Path) else None

    def is_builtin(self, name: str) -> bool:
        """Whether the persona in force comes from the package's own directory."""
        source = self.source_of(name)
        return source is not None and source.parent == BUILTIN_PERSONAS_DIR

    def has_builtin(self, name: str) -> bool:
        """Whether the package ships a persona of this name, whether or not it is in force."""
        return name in self._builtin_names

    def writable_dir(self) -> Path | None:
        """The directory a head's writes go to: `<workspace>/.uclone/personas/`.

        It is the workspace directory `reload` reads, and the one whose files win over the
        package's -- which is what makes a file written there an override of a built-in. A
        registry with no workspace has nowhere to write.
        """
        if self._workspace_root is None:
            return None
        return self._workspace_root / DEFAULT_WORKSPACE_PERSONAS_SUBDIR

    def save_persona(self, draft: PersonaDraft, *, create: bool) -> PersonaDefinition:
        """Write a persona to the workspace directory and put it in force in this registry.

        Create never replaces an existing persona, and an edit never invents one. An edit
        rewrites the file the persona was loaded from. For a built-in, that file is the
        package's, so the edit becomes `<name>.yaml` in the workspace directory instead:
        the loader already lets that file win, and the shipped one stays for the next
        upgrade to replace.
        """
        return self._store.save_persona(draft, create=create)


_default_persona_registry: PersonaRegistry | None = None


def get_default_persona_registry(
    workspace_root: Path | None = None,
    tool_names: Collection[str] | None = None,
) -> PersonaRegistry:
    """Obtain or initialize the default persona registry instance.

    Rebuilt when the workspace changes, and **also** when a caller supplies a tool
    inventory to a cached registry that has none. Without that second condition the
    validation `tool_names` buys is decided by call order rather than by configuration:
    whichever caller reaches the singleton first fixes it for the process, and the
    unvalidated caller is the likelier one to arrive first -- `GET /api/personas` serves
    the dashboard on load, before any chat has created an agent. The check would then be
    configured, passing its own tests, and silently never run in the assembled product.

    Handing an inventory to a registry that already has one does not rebuild it: two
    inventories in one process is a question about which is authoritative, and answering
    it by last-write-wins would be the same order dependence in the other direction.
    """
    global _default_persona_registry
    needs_workspace = (
        workspace_root is not None
        and _default_persona_registry is not None
        and _default_persona_registry.workspace_root != workspace_root
    )
    gains_inventory = (
        tool_names is not None
        and _default_persona_registry is not None
        and not _default_persona_registry.validates_tools
    )
    if _default_persona_registry is None or needs_workspace or gains_inventory:
        _default_persona_registry = PersonaRegistry(
            workspace_root=(
                workspace_root
                if workspace_root is not None or _default_persona_registry is None
                else _default_persona_registry.workspace_root
            ),
            tool_names=tool_names,
        )
    return _default_persona_registry
