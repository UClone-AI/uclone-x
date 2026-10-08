"""Extension registration: the core finds what a domain adds without naming it (#2205).

Three things are pinned here. The core reads every kind of contribution from the
registry -- tools, hooks, protected folders, the leased folder kind, routes -- so a fake
extension added in a test reaches each place the story's used to be named. A broken or
conflicting extension raises `ExtensionError` rather than being left out. And the story,
registered as the first extension, gives the core exactly what it had before: the same
tools, the story's `character_sheet` in the core's place, its hooks, `stories/` refused to
general tools, and the story library on the Files screen.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from uclone_x.agent.clone_builder import with_app_lifecycle_hooks
from uclone_x.agent.composition import HostDependencies
from uclone_x.agent.models import ToolExecutionRecord
from uclone_x.agent.session import SessionStore
from uclone_x.artifacts.kinds import FolderItem, FolderItemUnreadableError
from uclone_x.artifacts.library import ArtifactLibrary
from uclone_x.browser.turn import BrowserTurnHook
from uclone_x.engine.event_bus import EventBus
from uclone_x.extensions import (
    ENTRY_POINT_GROUP,
    Extension,
    ExtensionError,
    ProtectedRoot,
    discover,
    leased_folder_kinds,
    loaded_extensions,
    mount_extension_routes,
    use_extensions,
    with_extension_tools,
)
from uclone_x.extensions import registry as extension_registry
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.room.service import RoomService
from uclone_x.room.store import RoomStore
from uclone_x.sandbox import story_jail
from uclone_x.story import StoryLifecycleHook
from uclone_x.story.character import StoryCharacterSheetTool
from uclone_x.story.scene_turn import SceneTurnHook
from uclone_x.story.ucx_extension import StoryFolderKind
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.builtin.character import CharacterSheetTool
from uclone_x.tools.builtin.filesystem import FileWriteTool
from uclone_x.tools.models import NoIsolation, ToolContext
from uclone_x.tools.registry import LocalTool, ToolRegistry, create_default_registry

#: The story tools the default registry held before #2205, in its order then.
STORY_TOOLS = (
    "story_library",
    "muse_spark",
    "story_outline",
    "story_codex",
    "story_manuscript",
    "story_context",
    "story_audit",
    "story_start",
)

_REFUSAL = "{path} is the vault's, so it was not written."


@pytest.fixture(autouse=True)
def found_afresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test finds the extensions itself, so a patched entry point is not cached away."""
    monkeypatch.setattr(extension_registry, "_loaded", None)


def _tool(name: str) -> LocalTool:
    return LocalTool(name=name, description=f"The {name} tool.")


def _host(tmp_path: Path) -> HostDependencies:
    return HostDependencies(
        bus=EventBus(),
        llm=MockLLMConnector(default_response="Done."),
        tools=ToolRegistry(),
        tracer=TelemetryTracer(),
        store=SessionStore(tmp_path / "sessions"),
    )


def _ctx(workspace: Path) -> ToolContext:
    return ToolContext(
        agent_id="a", session_id="s", workspace_root=workspace, isolation=NoIsolation()
    )


class _VaultHook:
    """A lifecycle hook a fake extension adds; it changes nothing."""

    def after_tool_step(
        self, records: Sequence[ToolExecutionRecord], context: ToolContext
    ) -> ToolContext:
        return context


class _VaultKind:
    """A leased folder kind in which every folder is an item nobody holds."""

    root = "vault"

    def is_item(self, folder: Path) -> bool:
        return True

    def load(self, base: Path, item_id: str) -> FolderItem:
        return FolderItem(title=item_id.upper(), holder=None)

    def open(self, workspace: Path, item_id: str, room_id: str) -> tuple[FolderItem, bool]:
        return FolderItem(title=item_id, holder=None), True

    def take_over(self, workspace: Path, item_id: str, room_id: str) -> FolderItem:
        return FolderItem(title=item_id, holder=room_id)

    def release(self, workspace: Path, item_id: str, holder: str) -> bool:
        return False


def _vault(**overrides: Any) -> Extension:
    fields: dict[str, Any] = {
        "name": "vault",
        "tools": lambda: (_tool("vault_open"),),
        "lifecycle_hooks": lambda: (_VaultHook(),),
        "protected_roots": (ProtectedRoot(dirname="vault", refusal=_REFUSAL),),
        "leased_folders": lambda: (_VaultKind(),),
    }
    fields.update(overrides)
    return Extension(**fields)


# -- discovery -----------------------------------------------------------------------------


def test_the_story_is_found_in_the_tree_without_being_named() -> None:
    """`story/ucx_extension.py` is found by its module name alone."""
    found = discover()

    assert [e.name for e in found] == ["story"]
    assert found[0].source == "uclone_x.story.ucx_extension"


class _EntryPoint:
    def __init__(self, name: str, value: object) -> None:
        self.name = name
        self.value = f"fake:{name}"
        self._loaded = value

    def load(self) -> object:
        if isinstance(self._loaded, Exception):
            raise self._loaded
        return self._loaded


def _installed(monkeypatch: pytest.MonkeyPatch, *entries: _EntryPoint) -> None:
    def entry_points(*, group: str) -> list[_EntryPoint]:
        assert group == ENTRY_POINT_GROUP
        return list(entries)

    monkeypatch.setattr(importlib.metadata, "entry_points", entry_points)


def test_an_installed_extension_is_found_through_its_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An entry point naming an `Extension`, or a function returning one, both count.

    Killed by: src/uclone_x/extensions/registry.py :: return _checked([*_in_tree(), *_installed()])
    Becomes: return _checked([*_in_tree()])
    """
    _installed(
        monkeypatch,
        _EntryPoint("vault", _vault()),
        _EntryPoint("atlas", lambda: Extension(name="atlas")),
    )

    found = discover()

    assert [e.name for e in found] == ["atlas", "story", "vault"]
    assert found[2].source == "entry point 'vault' (fake:vault)"


def test_an_extension_that_does_not_load_is_an_error_not_a_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken extension stops discovery, naming where it came from (P6).

    Killed by: src/uclone_x/extensions/registry.py :: raise ExtensionError(f"The extension at {source} could not be loaded: {exc}") from exc
    Becomes: continue
    """
    _installed(monkeypatch, _EntryPoint("vault", ModuleNotFoundError("No module named 'vault'")))

    with pytest.raises(ExtensionError, match="'vault'.*could not be loaded"):
        discover()


def test_an_entry_point_that_is_not_an_extension_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: src/uclone_x/extensions/registry.py :: if not isinstance(value, Extension):
    Becomes: if False:
    """
    _installed(monkeypatch, _EntryPoint("vault", {"name": "vault"}))

    with pytest.raises(ExtensionError, match="not an extension: it gives a dict"):
        discover()


def test_two_extensions_with_one_name_are_an_error() -> None:
    """Killed by: src/uclone_x/extensions/registry.py :: if extension.name in by_name:
    Becomes: if False:
    """
    with pytest.raises(ExtensionError, match="Two extensions are named 'vault'"):
        with use_extensions([_vault(), _vault(tools=lambda: ())]):
            pass


def test_two_extensions_cannot_protect_one_folder() -> None:
    """Killed by: src/uclone_x/extensions/registry.py :: if holder != extension.name:
    Becomes: if False:
    """
    with pytest.raises(ExtensionError, match="both protect the folder 'vault'"):
        with use_extensions([_vault(), _vault(name="vault2")]):
            pass


def test_finding_is_done_once() -> None:
    """The extensions are found on first use and kept; finding imports packages."""
    assert loaded_extensions() is loaded_extensions()


# -- tools --------------------------------------------------------------------------------


def test_an_extension_tool_comes_after_the_core_tools() -> None:
    core = [_tool("file_read"), _tool("character_sheet")]
    with use_extensions([_vault()]):
        assert [t.name for t in with_extension_tools(core)] == [
            "file_read",
            "character_sheet",
            "vault_open",
        ]


def test_a_tool_with_a_core_name_is_refused_unless_declared() -> None:
    """A registry keyed by name would keep one of the two in silence.

    Killed by: src/uclone_x/extensions/registry.py :: if name not in extension.replaces_tools:
    Becomes: if False:
    """
    with use_extensions([_vault(tools=lambda: (_tool("character_sheet"),))]):
        with pytest.raises(ExtensionError, match="the core already has"):
            with_extension_tools([_tool("character_sheet")])


def test_a_declared_replacement_takes_the_core_tools_place() -> None:
    """Killed by: src/uclone_x/extensions/registry.py :: tools[position[name]] = tool
    Becomes: tools.append(tool)
    """
    mine = _tool("character_sheet")
    with use_extensions([_vault(tools=lambda: (mine,), replaces_tools=("character_sheet",))]):
        tools = with_extension_tools([_tool("file_read"), _tool("character_sheet")])

    assert [t.name for t in tools] == ["file_read", "character_sheet"]
    assert tools[1] is mine


def test_a_replacement_of_nothing_is_an_error() -> None:
    """Killed by: src/uclone_x/extensions/registry.py :: if name not in core_names:
    Becomes: if False:
    """
    with use_extensions([_vault(tools=lambda: (), replaces_tools=("nope",))]):
        with pytest.raises(ExtensionError, match="replaces the tool 'nope'"):
            with_extension_tools([_tool("file_read")])


def test_a_declared_replacement_with_no_tool_is_an_error() -> None:
    """Killed by: src/uclone_x/extensions/registry.py :: if missing:
    Becomes: if False:
    """
    with use_extensions([_vault(tools=lambda: (), replaces_tools=("character_sheet",))]):
        with pytest.raises(ExtensionError, match="adds no tool by that name"):
            with_extension_tools([_tool("character_sheet")])


def test_two_extensions_adding_one_tool_name_are_an_error() -> None:
    """Killed by: src/uclone_x/extensions/registry.py :: if name in owner:
    Becomes: if False:
    """
    two = [_vault(), Extension(name="atlas", tools=lambda: (_tool("vault_open"),))]
    with use_extensions(two):
        with pytest.raises(ExtensionError, match="both add a tool named 'vault_open'"):
            with_extension_tools([])


def test_an_extension_whose_tools_fail_to_build_is_an_error() -> None:
    def broken() -> Sequence[LocalTool]:
        raise ImportError("cannot import name 'VaultTool'")

    with use_extensions([_vault(tools=broken)]):
        with pytest.raises(ExtensionError, match="'vault' could not build its tools"):
            with_extension_tools([])


# -- hooks, folders, routes ---------------------------------------------------------------


def test_extension_hooks_compose_ahead_of_the_cores_once(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/clone_builder.py :: for hook in extension_lifecycle_hooks():
    Becomes: for hook in ():
    """
    with use_extensions([_vault()]):
        once = with_app_lifecycle_hooks(_host(tmp_path))
        twice = with_app_lifecycle_hooks(once)

    assert [type(h) for h in twice.lifecycle_hooks] == [_VaultHook, BrowserTurnHook]


async def test_a_protected_folder_is_refused_to_general_tools(tmp_path: Path) -> None:
    """The write guard reads the folders from the extensions, with each one's sentence.

    Killed by: src/uclone_x/tools/base.py :: for protected in protected_roots():
    Becomes: for protected in ():
    """
    with use_extensions([_vault()]):
        result = await FileWriteTool().execute(
            {"path": "vault/key.txt", "content": "x"}, _ctx(tmp_path)
        )
        allowed = await FileWriteTool().execute(
            {"path": "stories/note.txt", "content": "x"}, _ctx(tmp_path)
        )

    assert not result.success
    assert result.error is not None and "'vault/key.txt' is the vault's" in result.error
    assert not (tmp_path / "vault" / "key.txt").exists()
    # Without the story extension, `stories/` is an ordinary folder.
    assert allowed.success, allowed.error


def test_the_jail_covers_every_protected_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/sandbox/story_jail.py :: for dirname in dirnames:
    Becomes: for dirname in dirnames[:1]:
    """
    fake = tmp_path / "sandbox-exec"
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setattr(story_jail, "PLATFORM", "darwin")
    monkeypatch.setattr(story_jail, "SANDBOX_EXEC", fake)
    two = [_vault(), _vault(name="atlas", protected_roots=(ProtectedRoot("atlas", _REFUSAL),))]

    with use_extensions(two):
        argv = story_jail.story_library_jail(tmp_path)
    with use_extensions([]):
        none = story_jail.story_library_jail(tmp_path)

    root = tmp_path.resolve()
    assert f"LIBRARY_0={root / 'atlas'}" in argv
    assert f"LIBRARY_1={root / 'vault'}" in argv
    assert none == []


def test_more_than_one_leased_folder_kind_is_an_error() -> None:
    """A conversation keeps one open item, so a second kind could not be opened into one.

    Killed by: src/uclone_x/extensions/registry.py :: if len(kinds) > 1:
    Becomes: if False:
    """
    with use_extensions(
        [_vault(), Extension(name="atlas", leased_folders=lambda: (_VaultKind(),))]
    ):
        with pytest.raises(ExtensionError, match="More than one leased folder kind"):
            leased_folder_kinds()


def test_the_files_screen_lists_an_extensions_folder_kind(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/artifacts/library.py :: kind.root: kind for kind in (leased_folder_kinds() if kinds is None else kinds)
    Becomes: kind.root: kind for kind in ()
    """
    (tmp_path / "vault" / "ledger").mkdir(parents=True)
    (tmp_path / "vault" / "ledger" / "page.md").write_text("one", encoding="utf-8")
    rooms = RoomService(RoomStore(tmp_path / "rooms"))

    with use_extensions([_vault()]):
        survey = ArtifactLibrary(tmp_path, rooms, turn_in_flight=lambda _: False).survey()

    (entry,) = [e for e in survey.entries if e.path == "vault/ledger"]
    assert entry.kind == "story"
    assert entry.name == "LEDGER"


def test_each_extensions_routes_are_mounted_with_the_heads_context() -> None:
    """Killed by: src/uclone_x/extensions/registry.py :: register(context)
    Becomes: pass
    """
    seen: list[object] = []
    context = object()
    with use_extensions([_vault(routes=seen.append), Extension(name="atlas")]):
        mounted = mount_extension_routes(context)

    assert seen == [context]
    assert mounted == ["vault"]


# -- the story, as the first extension ----------------------------------------------------


def test_the_story_gives_the_default_registry_what_it_had() -> None:
    """Every story tool, and the story's `character_sheet` where the core's stood.

    Killed by: src/uclone_x/story/ucx_extension.py :: replaces_tools=("character_sheet",),
    Becomes: replaces_tools=(),
    """
    registry = create_default_registry(enable_mcp=False)
    names = [tool.name for tool in registry.list_tools()]

    assert names[-len(STORY_TOOLS) :] == list(STORY_TOOLS)
    assert names.count("character_sheet") == 1
    assert names.index("character_sheet") < names.index("install_package")
    assert type(registry.get("character_sheet")) is StoryCharacterSheetTool


def test_without_the_story_the_core_runs_alone(tmp_path: Path) -> None:
    """The core has no story tools of its own, and its `character_sheet` is the core's."""
    with use_extensions([]):
        registry = create_default_registry(enable_mcp=False)
        hooks = with_app_lifecycle_hooks(_host(tmp_path)).lifecycle_hooks

    assert not {tool.name for tool in registry.list_tools()} & set(STORY_TOOLS)
    assert type(registry.get("character_sheet")) is CharacterSheetTool
    assert [type(h) for h in hooks] == [BrowserTurnHook]


def test_the_story_hooks_lead_the_cores(tmp_path: Path) -> None:
    hooks = with_app_lifecycle_hooks(_host(tmp_path)).lifecycle_hooks

    assert [type(h) for h in hooks] == [StoryLifecycleHook, SceneTurnHook, BrowserTurnHook]


async def test_the_story_library_is_still_refused_to_general_tools(tmp_path: Path) -> None:
    """The sentence the story's protected folder carries is the one the guard had (#1583)."""
    result = await FileWriteTool().execute(
        {"path": "stories/x/codex.yaml", "content": "x"}, _ctx(tmp_path)
    )

    assert result.error is not None
    assert (
        "'stories/x/codex.yaml' is in the story library, so it was not written. Stories are "
        "changed with the story tools (story_manuscript, story_outline, story_codex), which "
        "check which conversation is writing the story and ask a person before a codex "
        "change. Reading the file is still allowed."
    ) in result.error


def test_the_story_library_is_the_files_screens_folder_kind() -> None:
    (kind,) = leased_folder_kinds()

    assert isinstance(kind, StoryFolderKind)
    assert kind.root == "stories"


def test_an_unreadable_story_says_why(tmp_path: Path) -> None:
    """A story whose `story.yaml` will not load is reported, never taken for missing."""
    folder = tmp_path / "stories" / "salt-road"
    folder.mkdir(parents=True)
    (folder / "story.yaml").write_text("title: [unclosed\n", encoding="utf-8")
    kind = StoryFolderKind()

    assert kind.is_item(folder)
    with pytest.raises(FolderItemUnreadableError):
        kind.load(tmp_path, "salt-road")
