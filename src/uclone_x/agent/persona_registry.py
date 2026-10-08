"""Declarative persona discovery and registry engine (Principle 0, Principle 4)."""

from __future__ import annotations

import difflib
import logging
from collections.abc import Collection, Sequence
from pathlib import Path

from uclone_x.agent.clone_store import CloneDirectoryPersonaStore, CloneRecord, ensure_clone_store
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
from uclone_x.core.agent_home import peer_handles
from uclone_x.core.models import AGENT_COMPOSED_TOOLS

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

    Serves the clones under the agents root, one directory each (clone-data-scopes §3.2),
    after importing the workspace's `.uclone/personas/` files and installing the package's
    builtins as clones, enabling zero-code agent extensibility without modifying core code.
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
        #: see `_validate_tools` for why that is a stated position and not a default. The
        #: tools an agent composes itself are added here, once, for every store it builds.
        self._tool_names: frozenset[str] | None = (
            frozenset(tool_names) | AGENT_COMPOSED_TOOLS if tool_names is not None else None
        )
        self._custom_store = store
        self._store: PersonaStoreProtocol
        self._dynamic_personas: dict[str, PersonaDefinition] = {}
        #: Names the package's own directory defines, whether or not a later file overrides
        #: them -- what lets the head say an edited built-in is an override.
        self._builtin_names: frozenset[str] = frozenset()
        self._clone_store: CloneDirectoryPersonaStore | None = None
        self._package: YamlFilePersonaStore | None = None
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

        # Before anything reads a clone: migrate, import this workspace's personas, and
        # install the package's (clone-data-scopes §3.8). A no-op once done. The package
        # directory is read here, at call time, so a test that swaps it is honoured.
        package_dir: Path = BUILTIN_PERSONAS_DIR
        ensure_clone_store(
            self._workspace_root,
            builtin_dir=package_dir if package_dir.is_dir() else None,
            install=self._include_defaults,
        )

        # 1. The clones: one directory each under the agents root, the only writable store.
        clone_store = CloneDirectoryPersonaStore(tool_names=self._tool_names)
        self._clone_store = clone_store
        stores: list[PersonaStoreProtocol] = [clone_store]

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

        # 3. The package's own definitions: no longer served directly, since each is
        # installed as a clone, but kept to say whether a clone is still its builtin.
        if self._include_defaults and package_dir.is_dir():
            package = YamlFilePersonaStore(package_dir, read_only=True)
            self._package = package
            self._builtin_names = frozenset(p.name for p in package.list_personas())
        else:
            self._package = None
            self._builtin_names = frozenset()

        self._store = CompositePersonaStore(
            stores=stores,
            writable_store=clone_store,
            overridable_dirs=[],
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

    def handle_for(self, ref: str) -> str:
        """`ref` as the name a persona goes by: a clone id is read back to its handle.

        Seats and chats are keyed by clone id (clone-data-scopes §4 step 3) while a
        persona is named by its clone's handle; every lookup here takes either.
        """
        if self._clone_store is not None:
            return self._clone_store.handle_for(ref)
        return ref

    def id_of(self, name: str) -> str | None:
        """The clone id behind the persona `name` (a handle or an id), or None for one without."""
        record = self.clone_record(name)
        return record.agent_id if record is not None else None

    def get_persona(self, name: str) -> PersonaDefinition | None:
        """Get a persona definition by its name, or by its clone's id."""
        if name in self._dynamic_personas:
            return self._dynamic_personas[name]
        return self._store.get_persona(self.handle_for(name))

    def register_persona(self, persona: PersonaDefinition) -> None:
        """Dynamically register or override a persona definition in memory."""
        self._validate_tools(persona, source="<registered in memory>")
        self._dynamic_personas[persona.name] = persona

    def source_of(self, name: str) -> Path | None:
        """The file a persona was loaded from, or `None` for one registered in memory."""
        if name in self._dynamic_personas:
            return None
        source = self._store.source_of(self.handle_for(name))
        return source if isinstance(source, Path) else None

    def is_builtin(self, name: str) -> bool:
        """Whether the clone `name` is still the package's builtin, unchanged.

        True when it was installed from the package persona of the same name and its
        definition still equals that one, peers compared by handle. An edited builtin is
        a clone of its own, which `has_builtin` still reports as overriding the package's.
        """
        name = self.handle_for(name)
        record = self.clone_record(name)
        package = self._package
        if record is None or package is None or record.template != name:
            return False
        shipped = package.get_persona(name)
        held = self.get_persona(name)
        if shipped is None or held is None or name in self._dynamic_personas:
            return False
        root = self._clone_store.root if self._clone_store is not None else None
        return held.model_copy(
            update={"a2a_peers": peer_handles(held.a2a_peers, root)}
        ) == shipped.model_copy(update={"a2a_peers": peer_handles(shipped.a2a_peers, root)})

    def has_builtin(self, name: str) -> bool:
        """Whether the package ships a persona of this name, whether or not it is in force."""
        return name in self._builtin_names

    def package_dir(self) -> Path | None:
        """The package's own personas directory, when this registry reads it."""
        return self._package.directory if self._package is not None else None

    def clone_record(self, name: str) -> CloneRecord | None:
        """The clone directory behind the persona `name`, or None for one without."""
        if name in self._dynamic_personas or self._clone_store is None:
            return None
        return self._clone_store.record_of(name)

    def display_name_of(self, name: str) -> dict[str, str]:
        """The clone's display name per locale; empty when it has none."""
        record = self.clone_record(name)
        return dict(record.display_name) if record is not None else {}

    def writable_dir(self) -> Path | None:
        """The agents root a head's writes go to: one directory per clone beneath it.

        None for a registry over a caller's own store, which has no clone directories.
        """
        if self._clone_store is None:
            return None
        return self._clone_store.root

    def save_persona(self, draft: PersonaDraft, *, create: bool) -> PersonaDefinition:
        """Write a persona as a clone and put it in force in this registry.

        Create never replaces an existing persona, and an edit never invents one. A create
        makes the clone's directory in one step; an edit rewrites its `clone.yaml`, a
        builtin's included, whose package file stays for the next upgrade to replace.
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
    unvalidated caller is the likelier one to arrive first -- `GET /api/clones` serves
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
