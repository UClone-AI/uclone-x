# pyright: reportPrivateUsage=false
"""The persona write path: create, edit and persist a persona from the head (#892).

The owner decided scope B on 2026-09-19 and ruled the RFC's security deferral lower priority
(). What these
tests keep anyway is correctness, not a threat model: a persona's name becomes a filename, so
a write must land in the one directory the loader reads and nowhere else, and a payload that
cannot become a valid persona file is refused rather than trimmed into one.

Every test runs against its own workspace under `tmp_path`, passed to `create_ui_app`
explicitly: the default workspace is the process's cwd, which here is the checkout.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from tests.support.app_clone import app_clone
from uclone_x.agent import persona_registry
from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import BASE_PERSONA_TOOLS
from uclone_x.agent.persona_registry import PersonaRegistry
from uclone_x.agent.prompts import compose_system_prompt
from uclone_x.core.agent_home import CLONE_FILE_NAME, AgentHome, default_agents_root
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import LLMRequest, MessageRole, ModelResponse
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import ToolRegistry
from uclone_x.ui.app import AgentSessionManager, create_ui_app

PERSONAS_SUBDIR = Path(".uclone") / "personas"


class _EchoParams(BaseModel):
    text: str = Field(default="", description="Text to echo")


class _MapTool(BaseTool[_EchoParams]):
    name = "map_area"
    writes_files = False  # read-only, so the persona flags (off by default) leave it (#1167)
    description = "Maps an area."

    def run(self, params: _EchoParams, context: ToolContext) -> str:
        return "mapped"


class _DigTool(BaseTool[_EchoParams]):
    name = "dig_site"
    writes_files = False  # read-only, so the persona flags (off by default) leave it (#1167)
    description = "Digs at a site."

    def run(self, params: _EchoParams, context: ToolContext) -> str:
        return "dug"


class _RecordingConnector(MockLLMConnector):
    """Answers normally, and keeps every request it was sent."""

    def __init__(self) -> None:
        super().__init__(default_response="ok")
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        return await super().generate(request)


def _system_sent(request: LLMRequest) -> str:
    systems = [m.content or "" for m in request.messages if m.role is MessageRole.SYSTEM]
    assert len(systems) == 1, f"expected exactly one system message, got {len(systems)}"
    return systems[0]


def _own_tools_sent(request: LLMRequest) -> list[str]:
    """The tools a request offered, less the base set every persona is given (#1402).

    The chat head gives each agent its own memory, so the memory tools are offered to
    every persona; what these tests pin is the persona's *own* list following an edit.
    """
    names = [t.name for t in request.tools]
    assert "record_memory_fact" in names
    return [name for name in names if name not in BASE_PERSONA_TOOLS]


def _tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_MapTool())
    registry.register(_DigTool())
    return registry


#: A model ref on the `mock` connection `_with_box` saves (model-gateway §3.1).
_BOX_MODEL = "box/hermes3:8b"


def _with_box(workspace: Path) -> None:
    """Save a `box` connection, so a persona may name a model on it (§3.7.1)."""
    sessions = workspace / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    (sessions / "settings.json").write_text(
        '{"connections": [{"id": "box", "kind": "mock"}]}', encoding="utf-8"
    )


def _client(workspace: Path, llm: MockLLMConnector | None = None) -> TestClient:
    app = create_ui_app(
        static_dir=workspace / "static",
        storage_dir=workspace / "sessions",
        llm=llm or MockLLMConnector(default_response="ok"),
        tools=_tools(),
        workspace_dir=workspace,
    )
    return TestClient(app)


async def _take_turn(agent: BaseAgent) -> None:
    """One turn with the persona, the way a room seat drives its clone."""
    result = await agent.execute_turn("go")
    assert result.error is None, result.error


def _manager(client: TestClient) -> AgentSessionManager:
    return cast(AgentSessionManager, cast(Any, client.app).state.session_manager)


def _saved(client: TestClient, name: str) -> Any:
    """Persona `name` as the registry now reads it, after the head's writes."""
    manager = _manager(client)
    return persona_registry.get_default_persona_registry(
        manager.workspace_dir, tool_names=[tool.name for tool in manager.tools.list_tools()]
    ).get_persona(name)


def _turn(client: TestClient, agent: BaseAgent) -> None:
    asyncio.run(_take_turn(agent))


def _draft(name: str = "surveyor", **overrides: Any) -> dict[str, Any]:
    draft: dict[str, Any] = {
        "name": name,
        "role": "Site Surveyor",
        "description": "Maps a site before anyone digs.",
        "system_prompt": "You survey.\nReport  \n what you map.",
        "allowed_tools": ["map_area"],
        # A model ref on a saved connection (model-gateway §3.4); `_with_box` saves `box`.
        "model_name": None,
        "model_tier": "fast",
        "temperature": 0.3,
        "max_tokens": 512,
        "enable_write_tools": False,
        "enable_subagent_tools": True,
    }
    draft.update(overrides)
    return draft


def _listed(client: TestClient) -> dict[str, dict[str, Any]]:
    res = client.get("/api/clones")
    assert res.status_code == 200
    return {p["name"]: p for p in res.json()["clones"]}


def _clone_file(name: str) -> Path:
    """The `clone.yaml` a persona called `name` is stored in (clone-data-scopes §3.2)."""
    return AgentHome.for_handle(name).clone_path


def _files_under(root: Path) -> set[Path]:
    return {p.relative_to(root) for p in root.rglob("*") if p.is_file() or p.is_symlink()}


@pytest.fixture
def workspace(tmp_path: Path, builtin_personas_absent: None) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root


# --- create ------------------------------------------------------------------------------


def test_create_writes_one_yaml_file_in_the_directory_the_loader_reads(workspace: Path) -> None:
    """A created persona is a clone directory -- `id` and `clone.yaml` -- and is listed.

    Since 2026-09-27 the clone root, not `<workspace>/.uclone/personas/`, is what
    `PersonaRegistry.reload` reads (clone-data-scopes §3.2), so it is the one a write must
    land in for the next process to see it, and the workspace is left without a file.

    Killed by: src/uclone_x/core/agent_home.py :: os.rename(staged, target)
    Becomes: pass
    """
    _with_box(workspace)
    client = _client(workspace)

    res = client.post("/api/clones", json=_draft(model_name=_BOX_MODEL))

    assert res.status_code == 201, res.text
    clone_dir = _clone_file("surveyor").parent
    assert clone_dir.parent == default_agents_root()
    assert _files_under(clone_dir) == {Path("id"), Path(CLONE_FILE_NAME)}
    assert not (workspace / PERSONAS_SUBDIR).exists()
    listed = _listed(client)["surveyor"]
    assert listed["role"] == "Site Surveyor"
    assert listed["system_prompt"] == "You survey.\nReport  \n what you map."
    assert listed["allowed_tools"] == ["map_area"]
    assert listed["model_name"] == _BOX_MODEL
    assert listed["llm_config"] == {
        "model_name": _BOX_MODEL,
        "fast_model": None,
        "image_model": None,
    }
    assert listed["model_tier"] == "fast"
    assert listed["temperature"] == 0.3
    assert listed["max_tokens"] == 512
    assert listed["enable_write_tools"] is False
    assert listed["enable_subagent_tools"] is True
    assert listed["builtin"] is False
    assert listed["overrides_builtin"] is False


def test_a_written_persona_round_trips_through_a_fresh_loader(workspace: Path) -> None:
    """What was saved is what a new process loads, byte for byte in the prompt.

    Multi-line text with trailing spaces and non-ASCII text are the two things a hand-rolled
    YAML emitter gets wrong; the file is written by `yaml.safe_dump` and read back by the
    loader itself before it replaces anything.

    Killed by: src/uclone_x/agent/persona_store.py :: "allowed_tools": list(draft.allowed_tools),
    Becomes: "allowed_tools": [],

    The tier is the case that could not load before this write path existed: the model
    config is strict and YAML has no enum, so `model_tier: fast` was refused by the loader.

    Killed by: src/uclone_x/agent/persona_store.py :: llm_dict["model_tier"] = ModelTier(tier)
    Becomes: pass
    """
    _with_box(workspace)
    client = _client(workspace)
    prompt = "당신은 측량사입니다.\n  indented line  \n\ntrailing   "
    created = client.post("/api/clones", json=_draft(system_prompt=prompt, model_name=_BOX_MODEL))
    assert created.status_code == 201

    fresh = PersonaRegistry(workspace_root=workspace, include_defaults=False)
    loaded = fresh.get_persona("surveyor")

    assert loaded is not None
    assert loaded.system_prompt == prompt
    assert loaded.allowed_tools == ("map_area",)
    assert loaded.llm_config.model_name == _BOX_MODEL
    assert loaded.llm_config.model_tier == "fast"
    assert loaded.llm_config.max_tokens == 512
    assert loaded.enable_subagent_tools is True


def test_create_refuses_a_name_that_is_already_defined(workspace: Path) -> None:
    """Create never overwrites: an existing name is a 409 and its file is left as it was.

    Killed by: src/uclone_x/agent/clone_store.py :: if create and (record is not None or clone_handles(self.root).get(draft.name)):
    Becomes: if False:
    """
    client = _client(workspace)
    assert client.post("/api/clones", json=_draft()).status_code == 201
    before = _clone_file("surveyor").read_bytes()

    res = client.post("/api/clones", json=_draft(role="Impostor"))

    assert res.status_code == 409
    assert "surveyor" in res.json()["detail"]
    assert _clone_file("surveyor").read_bytes() == before


# --- update ------------------------------------------------------------------------------


def test_update_rewrites_the_file_the_persona_was_loaded_from(workspace: Path) -> None:
    """An edit replaces the clone's own file, even for a persona imported from another name.

    A workspace file not named after its persona is imported at start into the clone of
    that name (clone-data-scopes §3.8 step 2). The edit rewrites that clone's `clone.yaml`
    in place: a second clone would be refused as a duplicate handle, and the import source
    is never written back to (the migration deletes and changes nothing it read).

    Killed by: src/uclone_x/agent/clone_store.py :: replace_clone_file(record.agent_id, text, root=self.root)
    Becomes: create_clone(draft.name, text, root=self.root)
    """
    personas_dir = workspace / PERSONAS_SUBDIR
    personas_dir.mkdir(parents=True)
    legacy = personas_dir / "zz_legacy.yml"
    legacy.write_text("name: surveyor\nrole: Old\nsystem_prompt: old prompt\n", encoding="utf-8")
    legacy_bytes = legacy.read_bytes()
    client = _client(workspace)
    clone_file = _clone_file("surveyor")

    res = client.put("/api/clones/surveyor", json=_draft(role="New"))

    assert res.status_code == 200, res.text
    assert _clone_file("surveyor") == clone_file
    assert yaml.safe_load(clone_file.read_text(encoding="utf-8"))["role"] == "New"
    assert legacy.read_bytes() == legacy_bytes
    assert _listed(client)["surveyor"]["role"] == "New"


def test_update_of_an_unknown_persona_is_404(workspace: Path) -> None:
    """Update is not create: an unknown name is a 404 and nothing is written.

    The refusal is made more than once on the way down (the clone store and each persona
    store check), so no one guard pins it; the route's mapping of it to a 404 does.

    Killed by: src/uclone_x/ui/app.py :: except PersonaNotFound as exc:
    Becomes: except PersonaWriteRefused as exc:
    """
    client = _client(workspace)

    res = client.put("/api/clones/nobody", json=_draft(name="nobody"))

    assert res.status_code == 404
    assert not (workspace / PERSONAS_SUBDIR).exists()


def test_update_refuses_a_body_naming_a_different_persona(workspace: Path) -> None:
    """Renaming is not an edit; a body whose name differs from the path is refused.

    Killed by: src/uclone_x/ui/app.py :: if addressed is not None and draft.name != addressed:
    Becomes: if False:
    """
    client = _client(workspace)
    assert client.post("/api/clones", json=_draft()).status_code == 201

    res = client.put("/api/clones/surveyor", json=_draft(name="digger"))

    assert res.status_code == 422
    assert set(_listed(client)) == {"surveyor"}


def test_an_edit_addressed_by_the_clone_id_is_the_edit_addressed_by_its_handle(
    workspace: Path,
) -> None:
    """`PUT /api/clones/{ref}` takes the clone's id as well as its handle (§3.7).

    Seats and chats address a clone by id, so a head holding only the id edits the same
    clone, in the same file, with the same answer; and a body naming another handle is
    still refused, since the address resolves to the clone's handle before the check.
    """
    client = _client(workspace)
    assert client.post("/api/clones", json=_draft()).status_code == 201
    clone_id = _listed(client)["surveyor"]["id"]
    assert isinstance(clone_id, str) and clone_id.startswith("agt_")
    clone_file = _clone_file("surveyor")

    by_id = client.put(f"/api/clones/{clone_id}", json=_draft(role="By id"))

    assert by_id.status_code == 200, by_id.text
    persona = by_id.json()["persona"]
    assert (persona["id"], persona["handle"], persona["role"]) == (clone_id, "surveyor", "By id")
    assert _clone_file("surveyor") == clone_file
    assert yaml.safe_load(clone_file.read_text(encoding="utf-8"))["role"] == "By id"

    by_handle = client.put("/api/clones/surveyor", json=_draft(role="By handle"))

    assert by_handle.status_code == 200, by_handle.text
    assert by_handle.json()["persona"]["id"] == clone_id
    assert client.get(f"/api/clones/{clone_id}").json() == client.get("/api/clones/surveyor").json()
    assert client.get(f"/api/clones/{clone_id}").json()["persona"]["role"] == "By handle"

    renamed = client.put(f"/api/clones/{clone_id}", json=_draft(name="digger"))

    assert renamed.status_code == 422, renamed.text
    assert set(_listed(client)) == {"surveyor"}


def test_the_clone_listing_names_each_clone_by_id_and_handle_and_its_peers_by_id(
    workspace: Path,
) -> None:
    """Each `GET /api/clones` row carries `id`, `handle` and `display_name` (§3.7).

    `a2a_peers` are clone ids, as stored: a draft may name a peer by handle, and the
    listing gives it back as that peer's id, which a head shows by the peer's own row.
    """
    client = _client(workspace)
    assert client.post("/api/clones", json=_draft(name="digger")).status_code == 201
    created = client.post(
        "/api/clones",
        json=_draft(display_name={"en": "Surveyor"}, a2a_peers=["digger"]),
    )
    assert created.status_code == 201, created.text

    listed = _listed(client)

    digger, surveyor = listed["digger"], listed["surveyor"]
    for row, handle in ((digger, "digger"), (surveyor, "surveyor")):
        assert isinstance(row["id"], str) and row["id"].startswith("agt_"), row
        assert row["handle"] == row["name"] == handle
        assert row["id"] == AgentHome.for_handle(handle).path.name
    assert digger["id"] != surveyor["id"]
    assert surveyor["display_name"] == {"en": "Surveyor"}
    assert digger["display_name"] == {}
    assert surveyor["a2a_peers"] == [digger["id"]]
    assert created.json()["persona"]["a2a_peers"] == [digger["id"]]


def test_editing_a_builtin_writes_an_override_and_leaves_the_shipped_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A built-in persona is edited in its installed clone, never in the package.

    The shipped file is what an upgrade replaces; editing it would lose the change on the
    next install and, in a read-only site-packages, fail outright. Install copied it into a
    clone directory (clone-data-scopes §3.5), so the edit is that clone's `clone.yaml`.
    A built-in whose prompt appends the default prompt round-trips as its own text plus the
    flag, not as a frozen copy of the composed default.

    Killed by: src/uclone_x/ui/app.py :: own_prompt, appended = split_appended_default_prompt(persona.system_prompt)
    Becomes: own_prompt, appended = persona.system_prompt, False
    """
    shipped = tmp_path / "shipped"
    shipped.mkdir()
    shipped_file = shipped / "guide.yaml"
    shipped_file.write_text(
        "name: guide\nrole: Guide\nappend_default_prompt: true\nsystem_prompt: You guide.\n",
        encoding="utf-8",
    )
    shipped_bytes = shipped_file.read_bytes()
    monkeypatch.setattr(persona_registry, "BUILTIN_PERSONAS_DIR", shipped)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    client = _client(workspace)

    before = _listed(client)["guide"]
    assert before["builtin"] is True
    assert before["system_prompt"] == "You guide."
    assert before["append_default_prompt"] is True

    edit = {k: v for k, v in before.items() if k in _draft()}
    edit.update(role="Senior Guide", append_default_prompt=True)
    res = client.put("/api/clones/guide", json=edit)

    assert res.status_code == 200, res.text
    assert shipped_file.read_bytes() == shipped_bytes
    written = yaml.safe_load(_clone_file("guide").read_text(encoding="utf-8"))
    assert written["system_prompt"] == "You guide."
    assert written["append_default_prompt"] is True
    after = _listed(client)["guide"]
    assert after["role"] == "Senior Guide"
    assert after["builtin"] is False
    assert after["overrides_builtin"] is True
    fresh = PersonaRegistry(workspace_root=workspace).get_persona("guide")
    assert fresh is not None
    assert fresh.system_prompt == f"You guide.\n\n{compose_system_prompt(writes_permitted=False)}"


# --- refusals ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_name",
    [
        "../escaped",
        "..",
        "a/b",
        "a\\b",
        "/abs",
        "Upper",
        "",
        "-dash",
        "x" * 65,
        "sp ace",
        "dot.ted",
    ],
)
def test_a_name_that_cannot_be_a_filename_is_refused_and_nothing_is_written(
    tmp_path: Path, builtin_personas_absent: None, bad_name: str
) -> None:
    """A persona name becomes a filename and an agent id, so the loader's own rule applies.

    The refusal comes from the same `refuse_an_unusable_username` the loader applies to every
    persona it reads, so a name the head accepts is one the next restart can load. The
    message is that rule's, which is what separates this refusal from the directory check
    behind it (that one would also refuse `../escaped`, with different words).

    Killed by: src/uclone_x/agent/clone_store.py :: refuse_an_unusable_username(draft.name)
    Becomes: pass
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    client = _client(workspace)
    before = _files_under(tmp_path)

    res = client.post("/api/clones", json=_draft(name=bad_name))

    assert res.status_code == 422
    assert "cannot be a persona name" in str(res.json()["detail"])
    assert _files_under(tmp_path) - before <= {Path("ws/sessions")}  # nothing new but storage
    assert not any(p.suffix in {".yaml", ".yml"} for p in _files_under(tmp_path) - before)


def test_a_write_that_resolves_outside_the_personas_directory_is_refused(
    tmp_path: Path, builtin_personas_absent: None
) -> None:
    """A clone file that is a symlink out of its directory is replaced, not written through.

    A name cannot reach outside the root (the name rule above), but the file an edit
    rewrites can be a link. Writing through it would change a file the user never pointed
    the head at, so the new text replaces the link in one step and the target is untouched.

    Killed by: src/uclone_x/core/agent_home.py :: os.replace(staged, home.clone_path)
    Becomes: home.clone_path.write_text(clone_yaml_text, encoding="utf-8")
    """
    outside = tmp_path / "elsewhere.yaml"
    outside.write_text("handle: linked\nrole: Linked\nsystem_prompt: linked\n", encoding="utf-8")
    outside_bytes = outside.read_bytes()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    client = _client(workspace)
    assert client.post("/api/clones", json=_draft(name="linked")).status_code == 201
    clone_file = _clone_file("linked")
    clone_file.unlink()
    clone_file.symlink_to(outside)

    res = client.put("/api/clones/linked", json=_draft(name="linked", role="Changed"))

    assert res.status_code == 200, res.text
    assert outside.read_bytes() == outside_bytes
    assert not clone_file.is_symlink()
    assert _listed(client)["linked"]["role"] == "Changed"


def test_an_unregistered_tool_is_refused_and_the_previous_definition_stands(
    workspace: Path,
) -> None:
    """A tool name the runtime does not have is a 422, and the file and catalogue are unchanged.

    The candidate text is parsed by the loader before anything is written, and the file is
    then replaced in one step, so a refused edit leaves no half-written file behind either.

    Killed by: src/uclone_x/agent/clone_store.py :: self._parse(checked, draft.name, label)
    Becomes: pass
    """
    client = _client(workspace)
    assert client.post("/api/clones", json=_draft()).status_code == 201
    clone_dir = _clone_file("surveyor").parent
    before = _clone_file("surveyor").read_bytes()

    res = client.put("/api/clones/surveyor", json=_draft(allowed_tools=["map_aera"]))

    assert res.status_code == 422
    assert "map_aera" in res.json()["detail"]
    assert "map_area" in res.json()["detail"]  # the loader's did-you-mean
    assert _clone_file("surveyor").read_bytes() == before
    assert _files_under(clone_dir) == {Path("id"), Path(CLONE_FILE_NAME)}
    assert _listed(client)["surveyor"]["allowed_tools"] == ["map_area"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"unexpected_field": 1},
        {"temperature": "warm"},
        {"temperature": 5.0},
        {"max_tokens": 0},
        {"model_tier": "turbo"},
        {"role": ""},
        {"system_prompt": ""},
        {"enable_write_tools": "yes"},
        {"allowed_tools": "map_area"},
    ],
)
def test_an_invalid_payload_is_refused_not_trimmed(
    workspace: Path, overrides: dict[str, Any]
) -> None:
    """Every field the file cannot hold is a 422; none is dropped or coerced into a value.

    Killed by: src/uclone_x/agent/persona_store.py :: model_config = ConfigDict(extra="forbid", strict=True)
    Becomes: model_config = ConfigDict(extra="ignore")
    """
    client = _client(workspace)

    res = client.post("/api/clones", json=_draft(**overrides))

    assert res.status_code == 422, res.text
    assert not (workspace / PERSONAS_SUBDIR).exists()


# --- a running agent ---------------------------------------------------------------------


def test_a_running_agent_takes_the_edited_prompt_and_tools_on_its_next_turn(
    workspace: Path,
) -> None:
    """An agent given the edited persona uses the edit on its next turn, tools included.

    The prompt alone would follow the registry by itself -- the agent re-resolves it on every
    read -- so asserting the prompt could not see the defect this pins. The tools are the
    half that is stored: the session manager used to copy the persona's list in as the
    *operator's* list, which the agent's scope rule lets win over any persona, and wrap the
    registry in a proxy fixed at creation. Either one kept the old tools in force behind the
    new prompt.

    The agent is built as the app builds a clone and handed the edit by `define_persona`,
    the call that puts an edited persona in force on a running agent (#1893 item 4: the
    chat-only agent cache that used to make that call is no longer written).

    Killed by: src/uclone_x/agent/bootstrap.py :: allowed_tools=(),  # the persona's list, resolved by the agent
    Becomes: allowed_tools=persona.granted_tools,
    """
    llm = _RecordingConnector()
    client = _client(workspace, llm)
    assert client.post("/api/clones", json=_draft()).status_code == 201
    agent = app_clone(_manager(client), "surveyor", "sess_survey")

    _turn(client, agent)
    assert _system_sent(llm.requests[-1]) == "You survey.\nReport  \n what you map."
    assert _own_tools_sent(llm.requests[-1]) == ["map_area"]

    res = client.put(
        "/api/clones/surveyor",
        json=_draft(system_prompt="You dig now.", allowed_tools=["dig_site"]),
    )
    assert res.status_code == 200, res.text
    agent.define_persona(_saved(client, "surveyor"))

    _turn(client, agent)
    assert _system_sent(llm.requests[-1]) == "You dig now."
    assert _own_tools_sent(llm.requests[-1]) == ["dig_site"]


def test_a_sub_agent_of_a_tool_scoped_persona_gets_only_the_parent_s_tools_edits_included(
    workspace: Path,
) -> None:
    """A delegating persona's child is held to the parent's tool list as it stands at spawn.

    Before #892 the persona agent's registry was wrapped in a scope proxy, and a child
    inherited the proxy. Resolving the persona's list inside the agent (so an edit reaches
    it) removed the proxy, and the child was then offered every registered tool. The child
    now takes the parent's resolved list as its own `allowed_tools`, so it is neither
    offered nor allowed to run anything outside it -- and an edit made after the parent was
    created governs the next child too.

    Killed by: src/uclone_x/agent/base.py :: allowed_tools=child_allowed,
    Becomes: allowed_tools=(),
    """
    llm = _RecordingConnector()
    app = create_ui_app(
        static_dir=workspace / "static",
        storage_dir=workspace / "sessions",
        llm=llm,
        tools=_tools(),
        workspace_dir=workspace,
    )
    manager = cast(AgentSessionManager, app.state.session_manager)
    with TestClient(app) as client:
        portal = client.portal
        assert portal is not None
        assert client.post("/api/clones", json=_draft()).status_code == 201
        parent = app_clone(manager, "surveyor", "sess_sub")
        portal.call(_take_turn, parent)

        def offered_to_a_child(parent: BaseAgent) -> list[str]:
            child = portal.call(parent.spawn_subagent, "helper", "help")
            portal.call(parent.delegate_task, child, "go")
            return [t.name for t in llm.requests[-1].tools if t.name not in BASE_PERSONA_TOOLS]

        assert offered_to_a_child(parent) == ["map_area"]

        edit = client.put("/api/clones/surveyor", json=_draft(allowed_tools=["dig_site"]))
        assert edit.status_code == 200, edit.text
        parent.define_persona(_saved(client, "surveyor"))
        assert offered_to_a_child(parent) == ["dig_site"]

        child = portal.call(parent.spawn_subagent, "helper", "help")
        with pytest.raises(PermissionError):
            portal.call(child.execute_tool_call, "map_area", dict[str, Any]())
