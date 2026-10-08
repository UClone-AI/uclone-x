"""Clones as directories: install, migration, import, handles, the lock and the write guard.

clone-data-scopes §3 (Rev 5). Every test works under the per-test agents root the
conftest sets through `UCLONE_AGENTS_DIR`, or under `tmp_path`; none touches `~/.uclone`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

import uclone_x
from uclone_x.agent.clone_store import (
    INSTALLED_FILE_NAME,
    CloneDirectoryPersonaStore,
    ensure_clone_store,
    import_workspace_personas,
)
from uclone_x.agent.persona_registry import PersonaRegistry
from uclone_x.agent.persona_store import (
    BUILTIN_PERSONAS_DIR,
    PersonaDraft,
    PersonaWriteConflict,
    PersonaWriteRefused,
)
from uclone_x.core.agent_home import (
    AgentHome,
    AgentHomeError,
    AgentHomeFault,
    CloneNotFoundError,
    DuplicateHandleError,
    clone_handles,
    clone_root_lock,
    create_clone,
    default_agents_root,
    is_agent_id,
    list_agent_homes,
    peer_handles,
    resolve_handle,
)
from uclone_x.errors import PlainRefusalError
from uclone_x.memory.store import default_cross_session_memory, read_saved_facts
from uclone_x.tools.builtin.filesystem import FileWriteTool

BUILTINS = ("artist", "clone", "guardian", "pioneer", "scout", "writer")


def _clone_file(root: Path, handle: str) -> dict[str, object]:
    (agent_id,) = clone_handles(root)[handle]
    loaded = yaml.safe_load((root / agent_id / "clone.yaml").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded  # pyright: ignore[reportUnknownVariableType]


def _legacy_home(root: Path, name: str, *, memory: str = '{"facts": []}') -> str:
    """An `agents/<name>/` directory as it was before 2026-09-27, with an id and memory."""
    home = AgentHome.for_legacy_name(name, root)
    agent_id = home.agent_id()
    home.memory_path.write_text(memory, encoding="utf-8")
    return agent_id


def _workspace_persona(workspace: Path, name: str, **fields: object) -> Path:
    folder = workspace / ".uclone" / "personas"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.yaml"
    data: dict[str, object] = {"name": name, "role": "Helper", "system_prompt": f"I am {name}."}
    data.update(fields)
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _log_text(root: Path) -> str:
    return "".join(p.read_text(encoding="utf-8") for p in sorted(root.glob(".migration-*.log")))


# --- fresh install and the second start -------------------------------------------------


def test_a_fresh_install_has_the_six_builtins_as_clones(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/clone_store.py :: if handle in installed:
    Becomes: if handle not in installed:
    """
    root = default_agents_root()
    ensure_clone_store(tmp_path, builtin_dir=BUILTIN_PERSONAS_DIR)

    claims = clone_handles(root)
    assert sorted(claims) == list(BUILTINS)
    assert all(len(ids) == 1 and is_agent_id(ids[0]) for ids in claims.values())
    installed = (root / INSTALLED_FILE_NAME).read_text(encoding="utf-8").split()
    assert sorted(installed) == list(BUILTINS)
    clone = _clone_file(root, "clone")
    assert clone["template"] == "clone"
    assert clone["display_name"] == {"en": "Clone", "ko": "클론"}
    # The shipped picture stays in the package; resetting a chosen one falls back to it.
    (artist_id,) = claims["artist"]
    assert not list((root / artist_id).glob("avatar*"))


def test_a_second_start_changes_nothing(tmp_path: Path) -> None:
    root = default_agents_root()
    _legacy_home(root, "scout")
    _workspace_persona(tmp_path, "notes")
    ensure_clone_store(tmp_path, builtin_dir=BUILTIN_PERSONAS_DIR)
    before = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
    log_before = _log_text(root)

    report = ensure_clone_store(tmp_path, builtin_dir=BUILTIN_PERSONAS_DIR)

    assert report.lines == []
    assert sorted(str(p.relative_to(root)) for p in root.rglob("*")) == before
    assert _log_text(root) == log_before


def test_a_deleted_builtin_does_not_come_back(tmp_path: Path) -> None:
    """Recording, not presence, stops a reinstall (§3.4)."""
    root = default_agents_root()
    ensure_clone_store(tmp_path, builtin_dir=BUILTIN_PERSONAS_DIR)
    (scout_id,) = clone_handles(root)["scout"]
    for leftover in (root / scout_id).iterdir():
        leftover.unlink()
    (root / scout_id).rmdir()

    ensure_clone_store(tmp_path, builtin_dir=BUILTIN_PERSONAS_DIR)

    assert "scout" not in clone_handles(root)


# --- migration --------------------------------------------------------------------------


def test_legacy_homes_workspace_personas_and_a_conflict_migrate_with_a_report(
    tmp_path: Path,
) -> None:
    """Step 1 renames old homes to their ids; step 2 imports; a conflict is kept and reported.

    Killed by: src/uclone_x/agent/clone_store.py :: if held != offered:
    Becomes: if held == offered:
    """
    root = default_agents_root()
    workspace = tmp_path / "ws"
    scout_id = _legacy_home(root, "scout", memory='{"facts": ["s"]}')
    mine_id = _legacy_home(root, "mine")
    _workspace_persona(workspace, "mine", role="Mine")
    picture = b"\x89PNG\r\n\x1a\n" + b"0" * 16
    (workspace / ".uclone" / "personas" / "mine.png").write_bytes(picture)
    _workspace_persona(workspace, "fresh", display_name={"ko": "새것"})
    create_clone("notes", "handle: notes\nrole: Kept\nsystem_prompt: kept\n", root=root)
    _workspace_persona(workspace, "notes", role="Different")

    ensure_clone_store(workspace, builtin_dir=BUILTIN_PERSONAS_DIR)

    # Step 1: renamed, memory kept, persona written in, nothing deleted.
    assert not (root / "scout").exists() and not (root / "mine").exists()
    assert (root / scout_id / "memory.json").read_text(encoding="utf-8") == '{"facts": ["s"]}'
    scout = _clone_file(root, "scout")
    assert scout["handle"] == "scout" and scout["template"] == "scout"
    mine = _clone_file(root, "mine")
    assert mine["role"] == "Mine" and "template" not in mine
    assert (root / mine_id / "avatar.png").read_bytes() == picture
    assert (workspace / ".uclone" / "personas" / "mine.yaml").is_file()
    # Step 2: imported, and the conflicting file kept out.
    assert _clone_file(root, "fresh")["display_name"] == {"ko": "새것"}
    assert _clone_file(root, "notes")["role"] == "Kept"
    # Install: the migrated builtin is only recorded, so six builtins and no duplicates.
    claims = clone_handles(root)
    assert all(len(claims[name]) == 1 for name in BUILTINS)
    log = _log_text(root)
    assert f"agents/scout -> agents/{scout_id}" in log
    assert f"agents/mine -> agents/{mine_id}" in log
    assert "kept clone 'notes'" in log
    assert "builtin 'scout': a clone of that handle exists; recorded only." in log

    # The conflict is reported once: it is recorded in the clone's directory.
    again = ensure_clone_store(workspace, builtin_dir=BUILTIN_PERSONAS_DIR)
    assert again.lines == []


def test_a_migrated_workspace_persona_keeps_the_name_a_person_gave_it(tmp_path: Path) -> None:
    """A clone named under #1947 that has an old home is migrated with its name.

    Its home is migrated from the workspace file, which is not read again afterwards, so a
    name left out of `clone.yaml` here is gone (#1949 review D1).

    Killed by: src/uclone_x/agent/clone_store.py :: display_name = {} if source is None else display_name_from_mapping(mapping, source)
    Becomes: display_name = {}
    """
    root = default_agents_root()
    workspace = tmp_path / "ws"
    _legacy_home(root, "sleepy")
    _workspace_persona(workspace, "sleepy", display_name={"ko": "잠 꾸러기"})

    ensure_clone_store(workspace, builtin_dir=BUILTIN_PERSONAS_DIR)

    assert _clone_file(root, "sleepy")["display_name"] == {"ko": "잠 꾸러기"}
    assert "kept clone 'sleepy'" not in _log_text(root)


def test_a_workspace_file_the_clone_already_matches_is_not_reported_as_different(
    tmp_path: Path,
) -> None:
    """The comparison reads the clone's name as the store does, so an unchanged file is quiet.

    The file is offered again with its import record gone, as a copied or restored agents
    root would have it: nothing differs, so nothing is reported (#1949 review D2).

    Killed by: src/uclone_x/agent/clone_store.py :: if "display_name" in fields:
    Becomes: if False:
    """
    root = default_agents_root()
    workspace = tmp_path / "ws"
    _workspace_persona(workspace, "sleepy", display_name={"ko": "잠 꾸러기"})
    ensure_clone_store(workspace, builtin_dir=BUILTIN_PERSONAS_DIR)
    (agent_id,) = clone_handles(root)["sleepy"]
    (root / agent_id / "imports.yaml").unlink()

    report = import_workspace_personas(workspace)

    assert not any("kept clone 'sleepy'" in line for line in report.lines), report.lines


def test_a_move_interrupted_between_writing_and_renaming_is_finished(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/clone_store.py :: if (entry / CLONE_FILE_NAME).is_file():
    Becomes: if False:
    """
    root = default_agents_root()
    agent_id = _legacy_home(root, "writer")
    (root / "writer" / "clone.yaml").write_text(
        "handle: writer\nrole: Mine now\nsystem_prompt: edited\n", encoding="utf-8"
    )

    ensure_clone_store(tmp_path, builtin_dir=BUILTIN_PERSONAS_DIR)

    assert not (root / "writer").exists()
    assert clone_handles(root)["writer"] == (agent_id,)
    assert _clone_file(root, "writer")["role"] == "Mine now"


def test_the_cli_default_home_is_kept_as_a_clone_with_the_clone_builtins_fields(
    tmp_path: Path,
) -> None:
    """A persona-less home (the CLI's old `default`) keeps its memory under its own handle."""
    root = default_agents_root()
    agent_id = _legacy_home(root, "default", memory='{"facts": ["d"]}')

    ensure_clone_store(tmp_path, builtin_dir=BUILTIN_PERSONAS_DIR)

    default = _clone_file(root, "default")
    shipped = yaml.safe_load((BUILTIN_PERSONAS_DIR / "clone.yaml").read_text(encoding="utf-8"))
    assert default["system_prompt"] == shipped["system_prompt"]
    assert "template" not in default
    assert (root / agent_id / "memory.json").read_text(encoding="utf-8") == '{"facts": ["d"]}'
    # Not merged into the builtin `clone`, which is installed beside it.
    assert clone_handles(root)["clone"] != (agent_id,)
    assert "no persona of its own" in _log_text(root)


def test_the_cli_runs_the_builtin_clone_by_default() -> None:
    """A CLI run with no name reaches `clone`, which a fresh install has (Rev 4 item 3)."""
    import inspect

    from uclone_x.cli.commands.a2a import a2a_serve, start_a2a_server
    from uclone_x.cli.commands.loop import (
        _run_loop_agent,  # pyright: ignore[reportPrivateUsage]
        run_loop_cmd,
    )
    from uclone_x.cli.commands.run import run_agent_repl, run_agent_repl_async
    from uclone_x.cli.main import run as cli_run

    for runner, parameter in (
        (run_agent_repl, "agent_name"),
        (run_agent_repl_async, "agent_name"),
        (_run_loop_agent, "agent_name"),
        (start_a2a_server, "agent_id"),
    ):
        assert inspect.signature(runner).parameters[parameter].default == "clone"
    # The typer commands carry the default inside their `Argument` / `Option`.
    for command, parameter in (
        (cli_run, "agent_name"),
        (run_loop_cmd, "agent_name"),
        (a2a_serve, "agent_id"),
    ):
        default = inspect.signature(command).parameters[parameter].default
        assert getattr(default, "default", default) == "clone"
    ensure_clone_store(None, builtin_dir=BUILTIN_PERSONAS_DIR)
    storage = default_cross_session_memory("clone").storage_path
    assert storage is not None and is_agent_id(storage.parent.name)


# --- handles ----------------------------------------------------------------------------


def test_a_handle_resolves_to_its_clone_and_an_unknown_one_creates_nothing() -> None:
    """Killed by: src/uclone_x/core/agent_home.py :: agent_id = resolve_handle(handle, base)
    Becomes: agent_id = handle
    """
    root = default_agents_root()
    home = create_clone("scribe", "handle: scribe\nrole: r\nsystem_prompt: p\n", root=root)

    assert resolve_handle("scribe") == home.path.name
    assert AgentHome.for_handle("scribe").path == home.path
    with pytest.raises(CloneNotFoundError):
        default_cross_session_memory("nobody")
    with pytest.raises(CloneNotFoundError):
        read_saved_facts("nobody")
    assert not (root / "nobody").exists()
    assert sorted(p.name for p in root.iterdir() if not p.name.startswith(".")) == [home.path.name]


def test_a_duplicate_handle_is_refused_on_write_and_reported_on_read() -> None:
    root = default_agents_root()
    create_clone("twin", "handle: twin\nrole: r\nsystem_prompt: p\n", root=root)
    with pytest.raises(DuplicateHandleError):
        create_clone("twin", "handle: twin\nrole: r\nsystem_prompt: p\n", root=root)

    # Two directories claiming one handle, as a copy by hand would leave them.
    other = create_clone("other", "handle: other\nrole: r\nsystem_prompt: p\n", root=root)
    (other.path / "clone.yaml").write_text(
        "handle: twin\nrole: r\nsystem_prompt: p\n", encoding="utf-8"
    )

    faults = {entry.fault for entry in list_agent_homes(root).homes}
    assert AgentHomeFault.DUPLICATE_HANDLE in faults
    with pytest.raises(DuplicateHandleError):
        resolve_handle("twin")
    assert CloneDirectoryPersonaStore(root).get_persona("twin") is None


def test_a_korean_display_name_round_trips_through_a_save(tmp_path: Path) -> None:
    registry = PersonaRegistry(workspace_root=tmp_path, include_defaults=False)
    draft = PersonaDraft(
        name="helper",
        role="Helper",
        system_prompt="Help.",
        display_name={"en": "Helper", "ko": "도우미"},
    )

    registry.save_persona(draft, create=True)

    raw = next(default_agents_root().glob("agt_*/clone.yaml")).read_text(encoding="utf-8")
    assert "도우미" in raw
    assert PersonaRegistry(workspace_root=tmp_path).display_name_of("helper") == {
        "en": "Helper",
        "ko": "도우미",
    }


def test_a_spaced_korean_name_is_a_display_name_and_never_a_handle(tmp_path: Path) -> None:
    """`잠 꾸러기` is what a person calls the clone; the handle stays ASCII (§3.3).

    The display name keeps its space and Hangul through `clone.yaml` and a fresh read, and
    the same text offered as a handle is refused before any directory is made.
    """
    wanted = {"ko": "잠 꾸러기"}
    registry = PersonaRegistry(workspace_root=tmp_path, include_defaults=False)
    registry.save_persona(
        PersonaDraft(name="sleepy", role="Sleeper", system_prompt="Yawn.", display_name=wanted),
        create=True,
    )

    assert _clone_file(default_agents_root(), "sleepy")["display_name"] == wanted
    assert PersonaRegistry(workspace_root=tmp_path).display_name_of("sleepy") == wanted
    before = sorted(p.name for p in default_agents_root().iterdir())
    with pytest.raises(AgentHomeError):
        create_clone("잠 꾸러기", "handle: 잠 꾸러기\n", root=default_agents_root())
    with pytest.raises((PersonaWriteRefused, ValueError)):
        registry.save_persona(
            PersonaDraft(name="잠 꾸러기", role="Sleeper", system_prompt="Yawn."), create=True
        )
    assert sorted(p.name for p in default_agents_root().iterdir()) == before


# --- peers ------------------------------------------------------------------------------


def test_peers_are_stored_by_id_and_read_back_to_handles(tmp_path: Path) -> None:
    """Step 2b on import, and on every save (Rev 5 a).

    The persona carries its peers as clone ids -- the clone resource shows them so too
    (clone-data-scopes §4) -- and a handle is read back from each id where one is wanted.

    Killed by: src/uclone_x/agent/clone_store.py :: fields["a2a_peers"] = self._peer_ids(draft.a2a_peers)
    Becomes: fields["a2a_peers"] = list(draft.a2a_peers)
    """
    root = default_agents_root()
    _workspace_persona(tmp_path, "lead", a2a_peers=["writer", "ghost"])
    registry = PersonaRegistry(workspace_root=tmp_path)
    (writer_id,) = clone_handles(root)["writer"]
    (scout_id,) = clone_handles(root)["scout"]

    assert _clone_file(root, "lead")["a2a_peers"] == [writer_id]
    assert "peer 'ghost' names no clone and was dropped" in _log_text(root)
    lead = registry.get_persona("lead")
    assert lead is not None and lead.a2a_peers == (writer_id,)
    assert peer_handles(lead.a2a_peers, registry.writable_dir()) == ("writer",)

    registry.save_persona(
        PersonaDraft(name="lead", role="Lead", system_prompt="Lead.", a2a_peers=["scout"]),
        create=False,
    )
    assert _clone_file(root, "lead")["a2a_peers"] == [scout_id]
    with pytest.raises(PersonaWriteRefused):
        registry.save_persona(
            PersonaDraft(name="lead", role="Lead", system_prompt="L.", a2a_peers=["ghost"]),
            create=False,
        )


def test_a_builtin_is_builtin_until_it_is_edited(tmp_path: Path) -> None:
    registry = PersonaRegistry(workspace_root=tmp_path)
    assert registry.is_builtin("writer") and registry.has_builtin("writer")

    registry.save_persona(
        PersonaDraft(name="writer", role="Mine", system_prompt="Edited."), create=False
    )

    assert not registry.is_builtin("writer") and registry.has_builtin("writer")
    with pytest.raises(PersonaWriteConflict):
        registry.save_persona(PersonaDraft(name="writer", role="r", system_prompt="p"), create=True)


def test_import_is_callable_for_another_workspace(tmp_path: Path) -> None:
    _workspace_persona(tmp_path, "late")
    report = import_workspace_personas(tmp_path)
    assert "late" in clone_handles(default_agents_root())
    assert any("late" in line for line in report.lines)


def test_a_workspace_file_that_is_not_a_persona_is_reported_not_skipped(tmp_path: Path) -> None:
    """A mistyped key in the launch directory's file is named in the report, with why.

    Before 2026-09-27 the registry refused such a file at load (`test_persona_registry`);
    the import leaves it out instead, and a clone that quietly never appears is P6's
    silent fallback unless the report says which file and what is wrong with it.

    Killed by: src/uclone_x/agent/clone_store.py :: report.add(f"{path} was not imported: {exc}")
    Becomes: pass
    """
    folder = tmp_path / ".uclone" / "personas"
    folder.mkdir(parents=True)
    (folder / "maybe.yaml").write_text(
        'name: maybe\nrole: Maybe\nsystem_prompt: Hello.\nappend_default_prompt: "no"\n',
        encoding="utf-8",
    )

    report = import_workspace_personas(tmp_path)

    assert "maybe" not in clone_handles(default_agents_root())
    (line,) = [line for line in report.lines if "maybe.yaml" in line]
    assert "append_default_prompt" in line


# --- the lock ---------------------------------------------------------------------------


def test_a_second_process_waits_for_the_migration_and_then_finds_nothing(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/clone_store.py :: with clone_root_lock(root) as base:  # the whole migration, one process at a time
    Becomes: for base in (root or default_agents_root(),):
    """
    root = default_agents_root()
    _legacy_home(root, "scout")
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from uclone_x.agent.clone_store import ensure_clone_store\n"
        "from uclone_x.agent.persona_store import BUILTIN_PERSONAS_DIR\n"
        "report = ensure_clone_store(Path(sys.argv[1]), builtin_dir=BUILTIN_PERSONAS_DIR)\n"
        "print(len(report.lines))\n"
    )
    env = dict(os.environ, PYTHONPATH=str(Path(uclone_x.__file__).resolve().parents[1]))
    with clone_root_lock(root):
        second = subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(1.5)
        assert second.poll() is None, second.communicate()
        assert (root / "scout").is_dir()
        first = ensure_clone_store(tmp_path, builtin_dir=BUILTIN_PERSONAS_DIR)
    out, err = second.communicate(timeout=60)

    assert second.returncode == 0, err
    assert first.lines
    assert out.strip() == "0"
    assert all(len(ids) == 1 for ids in clone_handles(root).values())


def _spelled_through_dot_dot(tmp_path: Path) -> Path:
    return tmp_path / "elsewhere" / ".." / "agents"


def _spelled_through_a_symlinked_parent(tmp_path: Path) -> Path:
    # macOS's `/tmp` and `/var` are symlinks into `/private`, so a first start there is this.
    (tmp_path / "real").mkdir()
    (tmp_path / "linked").symlink_to(tmp_path / "real", target_is_directory=True)
    return tmp_path / "linked" / "agents"


@pytest.mark.parametrize("spell", [_spelled_through_dot_dot, _spelled_through_a_symlinked_parent])
def test_a_first_start_on_a_root_not_yet_made_and_not_canonical_does_not_wait_on_itself(
    tmp_path: Path, spell: Callable[[Path], Path]
) -> None:
    """The lock is reentrant under any spelling of a root that does not exist yet.

    Keyed before the root existed, the outer hold and the nested one (the install inside
    the migration) had different keys, and `flock` in one process waited on itself forever
    (#1949 review D3). A thread and a timeout, so a regression fails instead of hanging.

    The mutation keys the outer hold one way and a nested one the other, which is the
    defect; on a root spelled canonically the two keys agree and nothing waits, hence the
    two non-canonical spellings.

    Killed by: src/uclone_x/core/agent_home.py :: key = str(base.resolve())
    Becomes: key = str(base.resolve()) if getattr(_held_locks, "depths", None) else str(base.absolute())
    """
    root = spell(tmp_path)
    assert not root.exists()
    done = threading.Event()
    failures: list[BaseException] = []

    def first_start() -> None:
        try:
            with clone_root_lock(root), clone_root_lock(root):
                pass
            ensure_clone_store(
                None, builtin_dir=BUILTIN_PERSONAS_DIR, root=spell(tmp_path / "second")
            )
            done.set()
        except BaseException as exc:  # reported on the test's thread below
            failures.append(exc)

    (tmp_path / "second").mkdir()
    worker = threading.Thread(target=first_start, daemon=True)
    worker.start()
    worker.join(timeout=20)

    assert not failures, failures
    assert done.is_set(), "the first start waited on its own lock"


# --- the write guard --------------------------------------------------------------------


def _write_path(target: str, workspace: Path) -> Path:
    return FileWriteTool().resolve_write_path(target, workspace)


def test_a_general_tool_cannot_write_into_the_agents_root(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/tools/base.py :: if in_app_state_dir(resolved):
    Becomes: if False:
    """
    root = default_agents_root()
    root.mkdir(parents=True, exist_ok=True)
    workspace = root.parent  # a workspace that holds the agents root
    with pytest.raises(PlainRefusalError):
        _write_path(str(root / "agt_x" / "clone.yaml"), workspace)
    assert _write_path("elsewhere.txt", workspace) == (workspace / "elsewhere.txt").resolve()


def test_launching_from_home_keeps_uclone_out_of_reach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    (home / ".uclone" / "agents").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))

    assert _write_path("notes.txt", home) == (home / "notes.txt").resolve()
    with pytest.raises(PlainRefusalError):
        _write_path(".uclone/agents/agt_x/memory.json", home)


# --- plain copy -------------------------------------------------------------------------


def test_what_a_person_reads_names_no_path_class_or_traceback(tmp_path: Path) -> None:
    root = default_agents_root()
    create_clone("taken", "handle: taken\nrole: r\nsystem_prompt: p\n", root=root)
    registry = PersonaRegistry(workspace_root=tmp_path, include_defaults=False)
    messages: list[str] = []
    for attempt in (
        lambda: resolve_handle("missing"),
        lambda: create_clone("taken", "handle: taken\nrole: r\nsystem_prompt: p\n", root=root),
        lambda: default_cross_session_memory("missing"),
        lambda: registry.save_persona(
            PersonaDraft(name="taken", role="r", system_prompt="p", a2a_peers=["missing"]),
            create=False,
        ),
    ):
        with pytest.raises(Exception) as caught:
            attempt()
        messages.append(str(caught.value))
    try:
        _write_path(str(root / "x.txt"), root.parent)
    except PlainRefusalError as exc:
        messages.append(str(exc).replace(str(root / "x.txt"), "<the path asked for>"))

    for message in messages:
        assert str(tmp_path) not in message and "/Users/" not in message, message
        assert "Error" not in message and "Traceback" not in message, message
        assert "agt_" not in message, message
