"""Follow-ups to the unreadable clone of #2220, from #2237's review (#2241).

* The start-up peer pass rewrote an unreadable clone's file when its peers were still
  handles. It now skips a file it cannot parse; its peers are stored by id once the file
  can be read. It runs before the tool registry exists, so a clone unreadable only for an
  unknown tool is still translated.
* The reason a clone could not be loaded reached the screen in English only. The listing
  now sends a `reason_code` the head words in the reader's language; the English
  sentence stays as the fallback.
* Setting a picture on an unreadable clone said "There is no clone named ...", which is
  untrue: the clone is there and cannot be loaded. It now says that, with the reason.
* Only the loader's own refusals make a clone unreadable. Any other exception while a
  persona is parsed is a defect, and it fails loudly instead of being listed as one bad
  clone.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from uclone_x.agent import clone_store
from uclone_x.agent.clone_store import ensure_clone_store
from uclone_x.agent.persona_avatar import AvatarRefused, PersonaAvatarStore
from uclone_x.agent.persona_registry import PersonaRegistry
from uclone_x.core.agent_home import AGENTS_DIR_ENV_VAR, create_clone, default_agents_root
from uclone_x.llm import MockLLMConnector
from uclone_x.ui.app import create_ui_app

GOOD = "handle: good\nrole: Helper\nsystem_prompt: I help.\n"
BARE = (
    "handle: bad\nrole: Helper\nsystem_prompt: I help.\nllm_config:\n  model_name: gemini-3-pro\n"
)
ODD = "handle: odd\nrole: R\nsystem_prompt: S\nappend_default_prompt: maybe\n"

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


# --- the peer pass leaves an unreadable clone's file alone ------------------------------


def test_the_peer_pass_does_not_rewrite_a_clone_it_cannot_read(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/clone_store.py :: if _load_problem(fields, handle, folder) is not None:
    Becomes: if False:
    """
    root = default_agents_root()
    good = create_clone("good", GOOD, root=root)
    bad = create_clone("bad", BARE + "a2a_peers:\n  - good\n", root=root)
    kept = create_clone("kept", GOOD.replace("good", "kept") + "a2a_peers:\n  - good\n", root=root)
    before = (bad.path / "clone.yaml").read_bytes()

    ensure_clone_store(tmp_path, builtin_dir=None, install=False)

    assert (bad.path / "clone.yaml").read_bytes() == before
    # A clone that can be read is still translated: the skip is for the unreadable one.
    assert good.path.name in (kept.path / "clone.yaml").read_text(encoding="utf-8")


def test_a_clone_unreadable_only_for_its_tools_still_has_its_peers_stored_by_id(
    tmp_path: Path,
) -> None:
    """The pass cannot see tools, so this is the one unreadable clone it does rewrite.

    Pinned so the scope stays what the docstring and §3.8 say: the peers become ids and
    every other key is kept as written.

    Killed by: src/uclone_x/agent/clone_store.py :: persona_from_mapping(_clone_persona_mapping(fields, handle), source)
    Becomes: raise PersonaLoadError(handle)
    """
    root = default_agents_root()
    good = create_clone("good", GOOD, root=root)
    tooled = create_clone(
        "tooled",
        GOOD.replace("good", "tooled")
        + "allowed_tools:\n  - no_such_tool_xyz\na2a_peers:\n  - good\n",
        root=root,
    )

    ensure_clone_store(tmp_path, builtin_dir=None, install=False)

    written = yaml.safe_load((tooled.path / "clone.yaml").read_text(encoding="utf-8"))
    assert written["a2a_peers"] == [good.path.name]
    assert written["allowed_tools"] == ["no_such_tool_xyz"]
    registry = PersonaRegistry(
        workspace_root=tmp_path, include_defaults=False, tool_names=["file_read"]
    )
    assert registry.unreadable_reason("tooled") is not None


# --- only the loader's own refusals make a clone unreadable -----------------------------


def _parse_raises(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
    def broken(fields: dict[str, Any], file_path: Path) -> Any:
        raise exc

    monkeypatch.setattr(clone_store, "persona_from_mapping", broken)


def test_a_defect_in_parsing_is_not_passed_off_as_one_unreadable_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/agent/clone_store.py :: except ValidationError as exc:
    Becomes: except Exception as exc:
    """
    create_clone("good", GOOD, root=default_agents_root())
    _parse_raises(monkeypatch, RuntimeError("a defect"))

    with pytest.raises(RuntimeError, match="a defect"):
        PersonaRegistry(workspace_root=tmp_path, include_defaults=False)


def test_a_value_error_from_a_defect_is_not_caught_as_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/agent/clone_store.py :: except (PersonaLoadError, ValidationError) as exc:
    Becomes: except (PersonaLoadError, ValidationError, ValueError) as exc:
    """
    create_clone("good", GOOD, root=default_agents_root())
    _parse_raises(monkeypatch, ValueError("a defect"))

    with pytest.raises(ValueError, match="a defect"):
        PersonaRegistry(workspace_root=tmp_path, include_defaults=False)


# --- the listing carries a code the head can word ---------------------------------------


@pytest.mark.usefixtures("builtin_personas_absent")
def test_each_unreadable_row_carries_a_reason_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed by: src/uclone_x/ui/clones.py :: reason_code=None if persona is None else persona.reason_code,
    Becomes: reason_code=None,
    """
    agents_root = tmp_path / "agents"
    monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(agents_root))
    create_clone("good", GOOD, root=agents_root)
    create_clone("bad", BARE, root=agents_root)
    create_clone("odd", ODD, root=agents_root)
    client = TestClient(
        create_ui_app(
            static_dir=tmp_path / "static",
            storage_dir=tmp_path / "sessions",
            llm=MockLLMConnector(),
        )
    )

    rows = {row["handle"]: row for row in client.get("/api/clones").json()["clones"]}

    assert rows["bad"]["reason_code"] == "bare_model"
    assert rows["odd"]["reason_code"] == "unreadable_file"
    assert "reason_code" not in rows["good"]
    # The English sentence stays, for a head that does not know the code.
    assert rows["bad"]["reason"]


# --- the picture refusal says what is true ----------------------------------------------


def test_a_picture_for_an_unreadable_clone_is_refused_with_its_reason(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/persona_avatar.py :: if unreadable is not None:
    Becomes: if False:
    """
    root = default_agents_root()
    create_clone("good", GOOD, root=root)
    bad = create_clone("bad", BARE, root=root)
    store = PersonaAvatarStore(PersonaRegistry(workspace_root=tmp_path, include_defaults=False))

    with pytest.raises(AvatarRefused) as caught:
        store.set("bad", PNG)

    assert caught.value.reason_code == "clone_unreadable"
    message = str(caught.value)
    assert "no clone" not in message.lower()
    assert "connection" in message
    assert not any(bad.path.glob("avatar*"))


# --- an unreadable clone keeps the name a person gave it, when that much can be read ------


@pytest.mark.usefixtures("builtin_personas_absent")
def test_an_unreadable_row_carries_its_display_name_when_it_can_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare model leaves the display name readable, so the row is not shown by handle (#2275).

    Killed by: src/uclone_x/ui/clones.py :: row["display_name"] = dict(found.display_name)
    Becomes: row["display_name"] = {}
    """
    agents_root = tmp_path / "agents"
    monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(agents_root))
    create_clone("bad", BARE + "display_name:\n  ko: 잠 꾸러기\n", root=agents_root)
    create_clone(
        "worse", "handle: worse\nrole: R\nsystem_prompt: S\ndisplay_name: 7\n", root=agents_root
    )
    client = TestClient(
        create_ui_app(
            static_dir=tmp_path / "static",
            storage_dir=tmp_path / "sessions",
            llm=MockLLMConnector(),
        )
    )

    rows = {row["handle"]: row for row in client.get("/api/clones").json()["clones"]}

    assert rows["bad"]["status"] == "unreadable"
    assert rows["bad"]["reason_code"] == "bare_model"
    assert rows["bad"]["display_name"] == {"ko": "잠 꾸러기"}
    # A display name that is itself what could not be read: the row falls back to the handle.
    assert rows["worse"]["status"] == "unreadable"
    assert rows["worse"]["display_name"] == {}


# --- a damaged home still says which clone it is, when the Core knows ---------------------


@pytest.mark.usefixtures("builtin_personas_absent")
def test_a_damaged_home_row_carries_the_id_its_directory_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two clones claiming one handle are each listed by their own id (#2260).

    With `id: null` a head keyed the row by its handle, so a room seating the clone by its
    id was not matched, and was reopened although nobody in it could answer.

    Killed by: src/uclone_x/ui/clones.py :: id=known_id,
    Becomes: id=None if persona is None else known_id,
    """
    agents_root = tmp_path / "agents"
    monkeypatch.setenv(AGENTS_DIR_ENV_VAR, str(agents_root))
    ids = ["agt_" + "a" * 32, "agt_" + "b" * 32]
    for agent_id in ids:
        folder = agents_root / agent_id
        folder.mkdir(parents=True)
        (folder / "id").write_text(f"{agent_id}\n", encoding="utf-8")
        (folder / "clone.yaml").write_text(GOOD.replace("good", "twin"), encoding="utf-8")
    client = TestClient(
        create_ui_app(
            static_dir=tmp_path / "static",
            storage_dir=tmp_path / "sessions",
            llm=MockLLMConnector(),
        )
    )

    rows = [row for row in client.get("/api/clones").json()["clones"] if row["handle"] == "twin"]

    assert [row["status"] for row in rows] == ["unreadable", "unreadable"]
    assert sorted(row["id"] for row in rows) == ids
