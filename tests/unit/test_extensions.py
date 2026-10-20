"""Extension registration: the core finds what a domain adds without naming it (#2205).

Three things are pinned here. The core reads every kind of contribution from the
registry -- tools, hooks, protected folders, the leased folder kind, routes -- so a fake
extension added in a test reaches each place the story's used to be named. A broken or
conflicting extension raises `ExtensionError` rather than being left out. And the story,
the first extension and its own example package (#2208), gives the core exactly what it
had before: the same
tools, the story's `character_sheet` in the core's place, its hooks, `stories/` refused to
general tools, and the story library on the Files screen.
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
import shutil
import subprocess
import tomllib
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from uclone_x.agent.clone_builder import with_app_lifecycle_hooks
from uclone_x.agent.clone_store import install_builtin_personas
from uclone_x.agent.composition import HostDependencies
from uclone_x.agent.models import ToolExecutionRecord
from uclone_x.agent.persona_avatar import PersonaAvatarStore
from uclone_x.agent.persona_registry import (
    BUILTIN_PERSONAS_DIR,
    PersonaLoadError,
    PersonaRegistry,
)
from uclone_x.agent.session import SessionStore
from uclone_x.artifacts.kinds import FolderItem, FolderItemUnreadableError
from uclone_x.artifacts.library import ArtifactLibrary
from uclone_x.browser.turn import BrowserTurnHook
from uclone_x.core.agent_home import AGENTS_DIR_ENV_VAR
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
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.builtin.character import CharacterSheetTool
from uclone_x.tools.builtin.filesystem import FileWriteTool
from uclone_x.tools.models import NoIsolation, ToolContext
from uclone_x.tools.registry import LocalTool, ToolRegistry, create_default_registry
from uclone_x_story import StoryLifecycleHook
from uclone_x_story.character import StoryCharacterSheetTool
from uclone_x_story.scene_turn import SceneTurnHook
from uclone_x_story.ucx_extension import EXTENSION as STORY_EXTENSION
from uclone_x_story.ucx_extension import PERSONAS_DIR, StoryFolderKind

#: The story example's own distribution metadata (#2208).
REPO_ROOT = Path(__file__).resolve().parents[2]
STORY_PYPROJECT = REPO_ROOT / "examples" / "story" / "pyproject.toml"

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


def _declared_entry_point() -> importlib.metadata.EntryPoint:
    """The one entry point the story example's `pyproject.toml` declares."""
    declared = tomllib.loads(STORY_PYPROJECT.read_text(encoding="utf-8"))
    (group,) = declared["project"]["entry-points"]
    ((name, value),) = declared["project"]["entry-points"][group].items()
    return importlib.metadata.EntryPoint(name=name, value=value, group=group)


def test_the_story_example_is_found_through_its_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The example declares `story` in the core's group, naming its `EXTENSION` (#2208).

    The entry point is read from the example's `pyproject.toml` and loaded as an installed
    one would be, so a wrong group, module or attribute there fails here, not only in a
    built wheel.
    """
    declared = _declared_entry_point()

    def entry_points(*, group: str) -> list[importlib.metadata.EntryPoint]:
        return [declared] if group == declared.group else []

    monkeypatch.setattr(importlib.metadata, "entry_points", entry_points)

    found = discover()

    assert declared.group == ENTRY_POINT_GROUP
    assert [e.name for e in found] == ["story"]
    assert found[0] == STORY_EXTENSION
    assert found[0].source == f"entry point 'story' ({declared.value})"


def test_nothing_installed_finds_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The core's own packages hold no extension since the story left them (#2208)."""
    _installed(monkeypatch)

    assert discover() == ()


def test_an_installed_extension_is_found_through_its_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An entry point naming an `Extension`, or a function returning one, both count.

    Killed by: src/uclone_x/extensions/registry.py :: return _checked(_installed())
    Becomes: return _checked([])
    """
    _installed(
        monkeypatch,
        _EntryPoint("vault", _vault()),
        _EntryPoint("atlas", lambda: Extension(name="atlas")),
    )

    found = discover()

    assert [e.name for e in found] == ["atlas", "vault"]
    assert found[1].source == "entry point 'vault' (fake:vault)"


_REGISTRY_LOGGER = "uclone_x.extensions.registry"


def _plain(words: str, *internals: str) -> None:
    """`words` name the extension as a person installed it, and none of `internals` (#2283).

    Where the entry point points and what the exception said go to the log instead.
    """
    assert "'vault'" in words
    for internal in ("entry point", "fake:", "Error", "Traceback", *internals):
        assert internal not in words, f"{internal!r} reached a person: {words!r}"


def test_an_extension_that_does_not_load_is_an_error_not_a_gap(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A broken extension stops discovery, in plain words; the log says where and why (P6).

    Killed by: src/uclone_x/extensions/registry.py :: raise _logged(ExtensionError(_could_not_load(entry.name))) from exc
    Becomes: continue
    """
    problem = ModuleNotFoundError("No module named 'vault_impl'")
    _installed(monkeypatch, _EntryPoint("vault", problem))

    with caplog.at_level("ERROR", logger=_REGISTRY_LOGGER):
        with pytest.raises(ExtensionError) as caught:
            discover()

    _plain(str(caught.value), "No module named", "vault_impl")
    assert "could not be loaded" in str(caught.value)
    assert "fake:vault" in caplog.text and "No module named 'vault_impl'" in caplog.text


def test_an_extension_whose_factory_fails_says_so_in_plain_words(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A factory that raises: the extension's name for a person, the cause for the log.

    Killed by: src/uclone_x/extensions/registry.py :: ExtensionError(_could_not_start(name))
    Becomes: ExtensionError(f"The extension at {source} failed to start: {exc}")
    """

    def factory() -> Extension:
        raise RuntimeError("no vault index at /srv/vault/index.db")

    _installed(monkeypatch, _EntryPoint("vault", factory))

    with caplog.at_level("ERROR", logger=_REGISTRY_LOGGER):
        with pytest.raises(ExtensionError) as caught:
            discover()

    _plain(str(caught.value), "/srv/vault", "index")
    assert "could not start" in str(caught.value)
    assert "fake:vault" in caplog.text and "/srv/vault/index.db" in caplog.text


class _NewerEntryPoint(_EntryPoint):
    """An extension written against a newer core: it builds `Extension` with a field
    this core's contract does not have, as the story did in a worktree based before #2243."""

    def load(self) -> object:
        unknown_to_this_core: dict[str, Any] = {"quests": ()}
        return Extension(name=self.name, **unknown_to_this_core)


def test_an_extension_newer_than_the_core_says_so_in_plain_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A checkout older than the installed extension is told to update itself (#2245).

    The words say what happened and what to do, and carry no Python internals.

    Killed by: src/uclone_x/extensions/registry.py :: return isinstance(exc, TypeError) and
    Becomes: return isinstance(exc, KeyError) and
    """
    _installed(monkeypatch, _NewerEntryPoint("vault", None))

    with pytest.raises(ExtensionError) as caught:
        discover()

    words = str(caught.value)
    assert "'vault'" in words
    assert "newer version of uclone-x" in words
    assert "merge origin/main" in words
    for internal in ("__init__", "keyword argument", "quests", "TypeError", "fake:"):
        assert internal not in words


def test_a_newer_extension_behind_a_factory_is_named_as_a_person_knows_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The factory path words a newer extension by its entry name too (#2256).

    It used to pass where the entry point points, so the sentence read "The extension
    \"entry point 'vault' (fake:vault)\" was written for a newer version...".

    Killed by: src/uclone_x/extensions/registry.py :: ExtensionError(_newer_than_the_core(name))
    Becomes: ExtensionError(_newer_than_the_core(source))
    """

    def factory() -> Extension:
        unknown_to_this_core: dict[str, Any] = {"quests": ()}
        return Extension(name="vault", **unknown_to_this_core)

    _installed(monkeypatch, _EntryPoint("vault", factory))

    with pytest.raises(ExtensionError) as caught:
        discover()

    words = str(caught.value)
    assert words.startswith("The extension 'vault' was written for a newer version")
    for internal in ("entry point", "fake:", "__init__", "keyword argument", "quests"):
        assert internal not in words


def test_ucx_puts_this_checkouts_examples_on_the_path(tmp_path: Path) -> None:
    """`./ucx` runs this checkout's example code, not the editable install's (#2256).

    The shared venv installs `examples/story` editable from the primary checkout, so a
    worktree's dev app and evals loaded the primary's story. The real `ucx` is copied
    beside a stub interpreter that prints the `PYTHONPATH` it was given.

    Killed by: ucx :: EXAMPLES_PATH="$EXAMPLES_PATH$example_src:"
    Becomes: EXAMPLES_PATH="$EXAMPLES_PATH"
    """
    sandbox = tmp_path / "repo"
    (sandbox / ".venv" / "bin").mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "ucx", sandbox / "ucx")
    (sandbox / "examples" / "story" / "src").mkdir(parents=True)
    (sandbox / "examples" / "notes").mkdir(parents=True)  # no src: not put on the path
    stub = sandbox / ".venv" / "bin" / "python"
    stub.write_text('#!/bin/sh\nprintf "%s" "$PYTHONPATH"\n', encoding="utf-8")
    stub.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}

    result = subprocess.run(
        ["bash", str(sandbox / "ucx"), "--help"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )

    entries = [Path(entry) for entry in result.stdout.split(os.pathsep) if entry]
    resolved = sandbox.resolve()
    assert resolved / "examples" / "story" / "src" in entries
    assert resolved / "src" in entries
    assert not any("notes" in entry.parts for entry in entries)


def _causes_logged(caplog: pytest.LogCaptureFixture, cause: str) -> list[logging.LogRecord]:
    """The records whose traceback carries `cause`."""
    return [
        record
        for record in caplog.records
        if record.exc_info is not None
        and cause in "".join(traceback.format_exception(*record.exc_info))
    ]


def test_an_extensions_own_chained_error_is_logged_once_where_it_failed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Its own `ExtensionError`, chained to the cause, is logged by the registry (#2294 review).

    The registry re-raised it unlogged, so the web head, which has no launcher, kept no
    record of the cause. It is logged once, and the error says so for the launcher.

    Killed by: src/uclone_x/extensions/registry.py :: _logged(exc)
    Becomes: pass
    """

    def factory() -> Extension:
        try:
            raise OSError("no vault index at /srv/vault/index.db")
        except OSError as exc:
            raise ExtensionError("The vault index is missing.") from exc

    _installed(monkeypatch, _EntryPoint("vault", factory))

    with caplog.at_level("ERROR", logger=_REGISTRY_LOGGER):
        with pytest.raises(ExtensionError) as caught:
            discover()

    assert str(caught.value) == "The vault index is missing."
    assert caught.value.cause_logged
    assert len(_causes_logged(caplog, "/srv/vault/index.db")) == 1
    assert "fake:vault" in caplog.text


def test_an_extensions_own_error_from_its_tools_is_logged_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Killed by: src/uclone_x/extensions/registry.py :: _logged(error)
    Becomes: pass
    """

    def broken() -> Sequence[LocalTool]:
        try:
            raise OSError("no vault index at /srv/vault/index.db")
        except OSError as exc:
            raise ExtensionError("The vault index is missing.") from exc

    with use_extensions([_vault(tools=broken)]):
        with caplog.at_level("ERROR", logger=_REGISTRY_LOGGER):
            with pytest.raises(ExtensionError) as caught:
                with_extension_tools([])

    assert caught.value.cause_logged
    assert len(_causes_logged(caplog, "/srv/vault/index.db")) == 1


def test_an_entry_point_that_is_not_an_extension_is_an_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Killed by: src/uclone_x/extensions/registry.py :: if not isinstance(value, Extension):
    Becomes: if False:

    Killed by: src/uclone_x/extensions/registry.py :: ExtensionError(_not_an_extension(name))
    Becomes: ExtensionError(f"{source} is not an extension: it gives a {type(value).__name__}.")
    """
    _installed(monkeypatch, _EntryPoint("vault", {"name": "vault"}))

    with caplog.at_level("ERROR", logger=_REGISTRY_LOGGER):
        with pytest.raises(ExtensionError) as caught:
            discover()

    _plain(str(caught.value), "dict")
    assert "does not provide a uclone-x extension" in str(caught.value)
    assert "fake:vault" in caplog.text and "dict" in caplog.text


def test_two_extensions_with_one_name_are_an_error() -> None:
    """Killed by: src/uclone_x/extensions/registry.py :: if extension.name in by_name:
    Becomes: if False:
    """
    with pytest.raises(ExtensionError, match="Two installed extensions are both named 'vault'"):
        with use_extensions([_vault(), _vault(tools=lambda: ())]):
            pass


def test_two_installed_extensions_with_one_name_name_no_entry_point(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Both found through entry points: the log says which two, the words only the name.

    Killed by: src/uclone_x/extensions/registry.py :: ExtensionError(_named_twice(extension.name))
    Becomes: ExtensionError(f"Two extensions are named {extension.name!r} ({extension.source}).")
    """
    _installed(
        monkeypatch,
        _EntryPoint("vault_a", Extension(name="vault")),
        _EntryPoint("vault_b", Extension(name="vault")),
    )

    with caplog.at_level("ERROR", logger=_REGISTRY_LOGGER):
        with pytest.raises(ExtensionError) as caught:
            discover()

    _plain(str(caught.value), "vault_a", "vault_b")
    assert "fake:vault_a" in caplog.text and "fake:vault_b" in caplog.text


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


def test_an_extension_whose_tools_fail_to_build_is_an_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Killed by: src/uclone_x/extensions/registry.py :: ExtensionError(_tools_failed(extension.name))
    Becomes: ExtensionError(f"The extension {extension.name!r} could not build its tools: {exc}")
    """

    def broken() -> Sequence[LocalTool]:
        raise ImportError("cannot import name 'VaultTool'")

    with use_extensions([_vault(tools=broken)]):
        with caplog.at_level("ERROR", logger=_REGISTRY_LOGGER):
            with pytest.raises(ExtensionError) as caught:
                with_extension_tools([])

    _plain(str(caught.value), "VaultTool", "cannot import")
    assert "could not set up its tools" in str(caught.value)
    assert "cannot import name 'VaultTool'" in caplog.text


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

    Killed by: examples/story/src/uclone_x_story/ucx_extension.py :: replaces_tools=("character_sheet",),
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


# -- personas an extension ships (#2208) ----------------------------------------------------


def _persona_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PersonaRegistry:
    """A registry checking tools against the default registry, as the web head's does."""
    monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(tmp_path / "agents"))
    names = [tool.name for tool in create_default_registry(enable_mcp=False).list_tools()]
    return PersonaRegistry(workspace_root=tmp_path / "workspace", tool_names=names)


def test_without_the_story_every_core_persona_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core alone lists its clones; no builtin names a story tool it lacks (#2208).

    Measured before the Writer moved: the core's `writer.yaml` named the story tools, and a
    registry checking tools refused it with `PersonaLoadError`, so no clone listed at all.
    The defect is a file, not a line: copying `writer.yaml` back into
    `src/uclone_x/personas/` fails this test with that error (checked by hand, #2208).
    """
    with use_extensions([]):
        registry = _persona_registry(tmp_path, monkeypatch)
        names = [persona.name for persona in registry.list_personas()]

    assert "writer" not in names
    assert {"clone", "artist", "scout"} <= set(names)


def test_without_the_story_the_core_clone_is_still_its_builtin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Core alone, `clone` is unchanged from what shipped, so it reads as builtin (#2256).

    Its `writer` peer is pending, not gone: the comparison with the shipped peers counts
    it where it stands, rather than calling the clone edited for a peer it never lost.

    Killed by: src/uclone_x/agent/persona_registry.py :: held_peers = peers_with_pending(record, peer_handles(held.a2a_peers, root))
    Becomes: held_peers = peer_handles(held.a2a_peers, root)
    """
    with use_extensions([]):
        registry = _persona_registry(tmp_path, monkeypatch)

        assert registry.is_builtin("clone")
        assert registry.get_persona("writer") is None


def test_the_story_ships_the_writer_as_a_builtin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Writer is installed as a clone from the example, a builtin with its picture.

    Killed by: examples/story/src/uclone_x_story/ucx_extension.py :: personas=(PERSONAS_DIR,),
    Becomes: personas=(),
    """
    registry = _persona_registry(tmp_path, monkeypatch)
    writer = registry.get_persona("writer")
    picture = PersonaAvatarStore(registry).find("writer")

    assert writer is not None and "story_start" in writer.allowed_tools
    assert registry.is_builtin("writer") and registry.has_builtin("writer")
    assert picture is not None and picture.path == PERSONAS_DIR / "writer.png"


def test_a_builtin_shipped_twice_is_refused(tmp_path: Path) -> None:
    """An extension shipping a core builtin's name is an error, not a coin toss.

    Killed by: src/uclone_x/agent/clone_store.py :: if handle in found:
    Becomes: if False:
    """
    shipped = tmp_path / "vault-personas"
    shipped.mkdir()
    (shipped / "clone.yaml").write_text(
        "name: clone\nrole: Mine\nsystem_prompt: Mine.\n", encoding="utf-8"
    )

    with pytest.raises(PersonaLoadError, match="'clone' is already shipped"):
        install_builtin_personas(
            BUILTIN_PERSONAS_DIR, extension_dirs=(shipped,), root=tmp_path / "agents"
        )
