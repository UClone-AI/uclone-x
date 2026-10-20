"""A clone's "Learn from conversations" switch: its default, its file and its wire (#1840).

Owner ruling 2026-10-10: one switch per clone, on by default. The room side (off stops the
extraction, and only it) is in `test_knowledge_extractor.py`. Every test here works under the
per-test agents root the conftest sets through `UCLONE_AGENTS_DIR`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from fastapi.testclient import TestClient

from uclone_x.agent.persona_registry import PersonaRegistry
from uclone_x.agent.persona_store import (
    InMemoryPersonaStore,
    PersonaDraft,
    YamlFilePersonaStore,
)
from uclone_x.core.agent_home import clone_handles, default_agents_root
from uclone_x.core.models import PersonaDefinition
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.ui.app import create_ui_app


def _draft(**fields: Any) -> PersonaDraft:
    return PersonaDraft(name="helper", role="Helper", system_prompt="Help.", **fields)


def _file(handle: str = "helper") -> dict[str, Any]:
    root = default_agents_root()
    (agent_id,) = clone_handles(root)[handle]
    loaded = yaml.safe_load((root / agent_id / "clone.yaml").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded  # pyright: ignore[reportUnknownVariableType]


def _loaded(tmp_path: Path, handle: str = "helper") -> PersonaDefinition:
    persona = PersonaRegistry(workspace_root=tmp_path, include_defaults=False).get_persona(handle)
    assert persona is not None
    return persona


def test_a_clone_learns_unless_it_is_told_not_to() -> None:
    """On by default, and written out only when off, so every clone saved before the switch
    existed reads and serialises exactly as it did.

    Killed by: src/uclone_x/core/models.py :: if data.get("learn_from_conversations") is True:
    Becomes: if False:
    """
    on = PersonaDefinition(name="helper", role="Helper", system_prompt="Help.")
    off = on.model_copy(update={"learn_from_conversations": False})

    assert on.learn_from_conversations is True
    assert "learn_from_conversations" not in on.model_dump()
    assert off.model_dump()["learn_from_conversations"] is False


def test_a_clone_file_without_the_switch_loads_as_learning(tmp_path: Path) -> None:
    PersonaRegistry(workspace_root=tmp_path, include_defaults=False).save_persona(
        _draft(), create=True
    )

    assert "learn_from_conversations" not in _file()
    assert _loaded(tmp_path).learn_from_conversations is True


def test_switching_off_is_written_to_the_file_and_switching_on_removes_it(
    tmp_path: Path,
) -> None:
    """Killed by: src/uclone_x/agent/clone_store.py :: fields["learn_from_conversations"] = False
    Becomes: pass
    """
    registry = PersonaRegistry(workspace_root=tmp_path, include_defaults=False)
    registry.save_persona(_draft(), create=True)

    registry.save_persona(_draft(learn_from_conversations=False), create=False)
    assert _file()["learn_from_conversations"] is False
    assert _loaded(tmp_path).learn_from_conversations is False

    registry.save_persona(_draft(learn_from_conversations=True), create=False)
    assert "learn_from_conversations" not in _file()
    assert _loaded(tmp_path).learn_from_conversations is True


def test_an_edit_that_leaves_the_switch_out_keeps_it(tmp_path: Path) -> None:
    """An editor or script that predates the switch does not turn learning back on.

    Killed by: src/uclone_x/agent/clone_store.py :: if "learn_from_conversations" in draft.model_fields_set or current is None
    Becomes: if True
    """
    registry = PersonaRegistry(workspace_root=tmp_path, include_defaults=False)
    registry.save_persona(_draft(learn_from_conversations=False), create=True)

    registry.save_persona(_draft(description="Edited."), create=False)

    assert _file()["learn_from_conversations"] is False
    assert _loaded(tmp_path).learn_from_conversations is False


def test_the_clone_list_carries_the_switch_and_an_edit_saves_it(tmp_path: Path) -> None:
    """`/api/clones` reports the switch, and the editor's PUT of the listed entry changes it.

    Killed by: src/uclone_x/ui/app.py :: "learn_from_conversations": persona.learn_from_conversations,
    Becomes: "learn_from_conversations": True,
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    app = create_ui_app(
        static_dir=workspace / "static",
        storage_dir=workspace / "sessions",
        llm=MockLLMConnector(),
        workspace_dir=workspace,
    )
    with TestClient(app) as client:
        res = client.post(
            "/api/clones", json={"name": "helper", "role": "Helper", "system_prompt": "Help."}
        )
        assert res.status_code in (200, 201), res.text

        def listed() -> dict[str, Any]:
            return {p["name"]: p for p in client.get("/api/clones").json()["clones"]}["helper"]

        entry = listed()
        assert entry["learn_from_conversations"] is True

        edit = {
            k: v
            for k, v in entry.items()
            if k in ("name", "role", "description", "system_prompt", "allowed_tools")
        }
        res = client.put("/api/clones/helper", json={**edit, "learn_from_conversations": False})
        assert res.status_code == 200, res.text

        assert listed()["learn_from_conversations"] is False


def test_a_workspace_persona_file_carries_the_switch_only_when_off(tmp_path: Path) -> None:
    """The YAML persona store writes through `persona_file_contents` itself (#2228).

    Killed by: src/uclone_x/agent/persona_store.py :: if not draft.learn_from_conversations:
    Becomes: if False:
    """
    store = YamlFilePersonaStore(tmp_path)
    store.save_persona(_draft(learn_from_conversations=False), create=True)

    written = yaml.safe_load((tmp_path / "helper.yaml").read_text(encoding="utf-8"))
    assert written["learn_from_conversations"] is False
    reread = YamlFilePersonaStore(tmp_path).get_persona("helper")
    assert reread is not None and reread.learn_from_conversations is False

    store.save_persona(_draft(learn_from_conversations=True), create=False)
    written = yaml.safe_load((tmp_path / "helper.yaml").read_text(encoding="utf-8"))
    assert "learn_from_conversations" not in written


def test_an_in_memory_persona_saved_off_stays_off() -> None:
    """A persona defined in-process (no file) keeps the switch it was saved with (#2228).

    Killed by: src/uclone_x/agent/persona_store.py :: learn_from_conversations=draft.learn_from_conversations,
    Becomes: learn_from_conversations=True,
    """
    store = InMemoryPersonaStore()

    saved = store.save_persona(_draft(learn_from_conversations=False), create=True)

    assert saved.learn_from_conversations is False
    stored = store.get_persona("helper")
    assert stored is not None and stored.learn_from_conversations is False
