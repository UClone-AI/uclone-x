"""Unit tests for PersonaStore backends and protocols (Issue #538 / #892)."""

from __future__ import annotations

from pathlib import Path

import pytest

from uclone_x.agent.models import (
    PersonaDefinition,
)
from uclone_x.agent.persona_registry import PersonaRegistry
from uclone_x.agent.persona_store import (
    CompositePersonaStore,
    InMemoryPersonaStore,
    PersonaDraft,
    PersonaLoadError,
    PersonaNotFound,
    PersonaStoreProtocol,
    PersonaWriteConflict,
    PersonaWriteRefused,
    YamlFilePersonaStore,
)
from uclone_x.agent.prompts import compose_system_prompt


def _make_draft(
    name: str = "custom",
    role: str = "Specialist",
    system_prompt: str = "You specialize.",
    **kwargs: object,
) -> PersonaDraft:
    return PersonaDraft(
        name=name,
        role=role,
        system_prompt=system_prompt,
        **kwargs,  # type: ignore[arg-type]
    )


def test_persona_store_protocol_conformance(tmp_path: Path) -> None:
    yaml_store = YamlFilePersonaStore(tmp_path)
    mem_store = InMemoryPersonaStore()
    comp_store = CompositePersonaStore([mem_store])

    assert isinstance(yaml_store, PersonaStoreProtocol)
    assert isinstance(mem_store, PersonaStoreProtocol)
    assert isinstance(comp_store, PersonaStoreProtocol)


# --- YamlFilePersonaStore Tests ----------------------------------------------------------


def test_yaml_file_store_discovers_yaml_and_yml(tmp_path: Path) -> None:
    (tmp_path / "clone.yaml").write_text(
        "name: clone\nrole: Default Clone\nsystem_prompt: Clone prompt.\n",
        encoding="utf-8",
    )
    (tmp_path / "helper.yml").write_text(
        "name: helper\nrole: Assistant\nsystem_prompt: Helper prompt.\n",
        encoding="utf-8",
    )

    store = YamlFilePersonaStore(tmp_path)

    assert store.has_persona("clone")
    assert store.has_persona("helper")
    assert not store.has_persona("unknown")

    clone = store.get_persona("clone")
    assert clone is not None
    assert clone.role == "Default Clone"

    helper = store.get_persona("helper")
    assert helper is not None
    assert helper.role == "Assistant"

    source = store.source_of("clone")
    assert source == (tmp_path / "clone.yaml").resolve()

    # Default persona 'clone' comes first
    personas = list(store.list_personas())
    assert len(personas) == 2
    assert personas[0].name == "clone"
    assert personas[1].name == "helper"


def test_yaml_file_store_saves_and_updates_in_place(tmp_path: Path) -> None:
    store = YamlFilePersonaStore(tmp_path)

    draft = _make_draft(
        name="tester",
        role="Tester",
        system_prompt="Test thoroughly.",
        allowed_tools=["test_tool"],
        model_name="gpt-4o",
        model_tier="fast",
        temperature=0.5,
    )
    created = store.save_persona(draft, create=True)

    assert created.name == "tester"
    assert created.role == "Tester"
    assert (tmp_path / "tester.yaml").exists()

    # Update in-place
    update_draft = _make_draft(
        name="tester",
        role="Lead Tester",
        system_prompt="Test everything.",
    )
    updated = store.save_persona(update_draft, create=False)

    assert updated.role == "Lead Tester"
    reloaded = YamlFilePersonaStore(tmp_path).get_persona("tester")
    assert reloaded is not None
    assert reloaded.role == "Lead Tester"


def test_yaml_file_store_atomic_write_cleans_up_temp_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = YamlFilePersonaStore(tmp_path, tool_names=["valid_tool"])

    # Draft with invalid tool when store validates tools
    draft = _make_draft(name="invalid", allowed_tools=["unknown_tool"])

    with pytest.raises(PersonaWriteRefused):
        store.save_persona(draft, create=True)

    # No target file and no .tmp residue left
    assert not (tmp_path / "invalid.yaml").exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_yaml_file_store_read_only_refuses_writes(tmp_path: Path) -> None:
    store = YamlFilePersonaStore(tmp_path, read_only=True)
    draft = _make_draft(name="forbidden")

    with pytest.raises(PersonaWriteConflict, match="read-only"):
        store.save_persona(draft, create=True)


def test_yaml_file_store_error_handling(tmp_path: Path) -> None:
    store = YamlFilePersonaStore(tmp_path)

    # Unusable name
    with pytest.raises(PersonaWriteRefused, match="cannot be a persona name"):
        store.save_persona(_make_draft(name="../bad"), create=True)

    # Conflict on create when already exists
    store.save_persona(_make_draft(name="existing"), create=True)
    with pytest.raises(PersonaWriteConflict, match="already exists"):
        store.save_persona(_make_draft(name="existing"), create=True)

    # Not found on update when doesn't exist
    with pytest.raises(PersonaNotFound, match="no persona named 'missing' is loaded"):
        store.save_persona(_make_draft(name="missing"), create=False)


def test_yaml_file_store_corrupt_files_raise_load_error(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text("invalid: [unclosed", encoding="utf-8")

    with pytest.raises(PersonaLoadError, match="bad.yaml"):
        YamlFilePersonaStore(tmp_path)


def test_yaml_file_store_append_default_prompt(tmp_path: Path) -> None:
    (tmp_path / "appended.yaml").write_text(
        "name: appended\nrole: Appended\nsystem_prompt: Base prompt.\nappend_default_prompt: true\n",
        encoding="utf-8",
    )

    store = YamlFilePersonaStore(tmp_path)
    p = store.get_persona("appended")
    assert p is not None
    assert p.system_prompt == f"Base prompt.\n\n{compose_system_prompt(writes_permitted=False)}"


# --- InMemoryPersonaStore Tests ----------------------------------------------------------


def test_in_memory_store_crud() -> None:
    p1 = PersonaDefinition(
        name="clone",
        role="Clone",
        system_prompt="Clone prompt.",
    )
    store = InMemoryPersonaStore([p1])

    assert store.has_persona("clone")
    assert store.get_persona("clone") == p1
    assert store.source_of("clone") == "<memory:clone>"
    assert store.source_of("missing") is None

    # Register in memory
    p2 = PersonaDefinition(
        name="scripter",
        role="Scripter",
        system_prompt="Script prompt.",
    )
    store.register_persona(p2)
    assert store.has_persona("scripter")

    # List personas has 'clone' first
    listed = store.list_personas()
    assert [p.name for p in listed] == ["clone", "scripter"]

    # Save persona create
    draft = _make_draft(name="analyst", role="Data Analyst")
    created = store.save_persona(draft, create=True)
    assert created.role == "Data Analyst"
    assert store.has_persona("analyst")

    # Save persona update
    update_draft = _make_draft(name="analyst", role="Senior Analyst")
    updated = store.save_persona(update_draft, create=False)
    assert updated.role == "Senior Analyst"

    # Save persona conflict
    with pytest.raises(PersonaWriteConflict):
        store.save_persona(draft, create=True)

    # Save persona not found
    with pytest.raises(PersonaNotFound):
        store.save_persona(_make_draft(name="ghost"), create=False)


def test_in_memory_store_read_only() -> None:
    store = InMemoryPersonaStore(read_only=True)
    with pytest.raises(PersonaWriteConflict, match="read-only"):
        store.save_persona(_make_draft(name="test"), create=True)


# --- CompositePersonaStore Tests ---------------------------------------------------------


def test_composite_store_precedence_and_overrides(tmp_path: Path) -> None:
    ws_dir = tmp_path / "workspace"
    builtin_dir = tmp_path / "builtin"
    ws_dir.mkdir()
    builtin_dir.mkdir()

    # Builtin defines 'writer' and 'artist'
    (builtin_dir / "writer.yaml").write_text(
        "name: writer\nrole: Shipped Writer\nsystem_prompt: Shipped.\n",
        encoding="utf-8",
    )
    (builtin_dir / "artist.yaml").write_text(
        "name: artist\nrole: Shipped Artist\nsystem_prompt: Shipped artist.\n",
        encoding="utf-8",
    )

    # Workspace defines custom 'writer' (override)
    (ws_dir / "writer.yaml").write_text(
        "name: writer\nrole: Custom Workspace Writer\nsystem_prompt: Custom.\n",
        encoding="utf-8",
    )

    ws_store = YamlFilePersonaStore(ws_dir, read_only=False)
    builtin_store = YamlFilePersonaStore(builtin_dir, read_only=True)

    composite = CompositePersonaStore(
        stores=[ws_store, builtin_store],
        writable_store=ws_store,
        overridable_dirs=[builtin_dir],
    )

    # Precedence: Workspace wins for 'writer'
    writer = composite.get_persona("writer")
    assert writer is not None
    assert writer.role == "Custom Workspace Writer"
    assert composite.source_of("writer") == (ws_dir / "writer.yaml").resolve()

    # Fallback: Builtin resolved for 'artist'
    artist = composite.get_persona("artist")
    assert artist is not None
    assert artist.role == "Shipped Artist"
    assert composite.source_of("artist") == (builtin_dir / "artist.yaml").resolve()

    # Listing de-duplicates
    names = [p.name for p in composite.list_personas()]
    assert names == ["artist", "writer"]

    # Editing a builtin persona writes an override to workspace_store
    edit_artist = _make_draft(
        name="artist",
        role="Overridden Artist",
        system_prompt="My custom artist prompt.",
    )
    saved_artist = composite.save_persona(edit_artist, create=False)
    assert saved_artist.role == "Overridden Artist"
    assert (ws_dir / "artist.yaml").exists()

    # Now composite resolves 'artist' from workspace store
    artist_res = composite.get_persona("artist")
    assert artist_res is not None
    assert artist_res.role == "Overridden Artist"
    assert composite.source_of("artist") == (ws_dir / "artist.yaml").resolve()


def test_composite_store_refuses_create_if_exists_in_any_store(tmp_path: Path) -> None:
    ws_dir = tmp_path / "workspace"
    builtin_dir = tmp_path / "builtin"
    ws_dir.mkdir()
    builtin_dir.mkdir()

    (builtin_dir / "clone.yaml").write_text(
        "name: clone\nrole: Shipped Clone\nsystem_prompt: Shipped.\n",
        encoding="utf-8",
    )

    ws_store = YamlFilePersonaStore(ws_dir, read_only=False)
    builtin_store = YamlFilePersonaStore(builtin_dir, read_only=True)

    composite = CompositePersonaStore(
        stores=[ws_store, builtin_store],
        writable_store=ws_store,
    )

    with pytest.raises(PersonaWriteConflict, match="already exists"):
        composite.save_persona(_make_draft(name="clone"), create=True)


def test_composite_store_refuses_update_for_unknown_persona(tmp_path: Path) -> None:
    ws_dir = tmp_path / "workspace"
    ws_dir.mkdir()
    ws_store = YamlFilePersonaStore(ws_dir, read_only=False)

    composite = CompositePersonaStore(stores=[ws_store], writable_store=ws_store)

    with pytest.raises(PersonaNotFound, match="no persona named 'ghost' is loaded"):
        composite.save_persona(_make_draft(name="ghost"), create=False)


def test_composite_store_refuses_write_without_writable_store(tmp_path: Path) -> None:
    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()
    ro_store = YamlFilePersonaStore(readonly_dir, read_only=True)

    composite = CompositePersonaStore(stores=[ro_store], writable_store=None)

    with pytest.raises(PersonaWriteConflict, match="no workspace directory"):
        composite.save_persona(_make_draft(name="new_persona"), create=True)


def test_composite_store_refuses_editing_external_non_overridable_dir(tmp_path: Path) -> None:
    ws_dir = tmp_path / "workspace"
    external_dir = tmp_path / "external"
    ws_dir.mkdir()
    external_dir.mkdir()

    (external_dir / "vendor.yaml").write_text(
        "name: vendor\nrole: Vendor Agent\nsystem_prompt: Vendor.\n",
        encoding="utf-8",
    )

    ws_store = YamlFilePersonaStore(ws_dir, read_only=False)
    ext_store = YamlFilePersonaStore(external_dir, read_only=True)

    composite = CompositePersonaStore(
        stores=[ws_store, ext_store],
        writable_store=ws_store,
        overridable_dirs=[],  # external_dir is not overridable
    )

    with pytest.raises(PersonaWriteConflict, match="Edit that file directly"):
        composite.save_persona(_make_draft(name="vendor", role="New Vendor"), create=False)


# --- PersonaRegistry with Store Injection ------------------------------------------------


def test_persona_registry_with_in_memory_store() -> None:
    custom_p = PersonaDefinition(
        name="custom_bot",
        role="Custom Bot",
        system_prompt="Be helpful.",
    )
    mem_store = InMemoryPersonaStore([custom_p])
    registry = PersonaRegistry(store=mem_store)

    assert registry.get_persona("custom_bot") == custom_p
    assert registry.store is mem_store

    # Dynamic registration overlays in memory
    dyn_p = PersonaDefinition(
        name="dyn_bot",
        role="Dynamic Bot",
        system_prompt="Be dynamic.",
    )
    registry.register_persona(dyn_p)
    assert registry.get_persona("dyn_bot") == dyn_p
    assert registry.source_of("dyn_bot") is None
