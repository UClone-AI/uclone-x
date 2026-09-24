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

from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from uclone_x.agent import persona_registry
from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import BASE_PERSONA_TOOLS
from uclone_x.agent.persona_registry import PersonaRegistry
from uclone_x.agent.prompts import compose_system_prompt
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


def _client(workspace: Path, llm: MockLLMConnector | None = None) -> TestClient:
    app = create_ui_app(
        static_dir=workspace / "static",
        storage_dir=workspace / "sessions",
        llm=llm or MockLLMConnector(default_response="ok"),
        tools=_tools(),
        workspace_dir=workspace,
    )
    return TestClient(app)


def _draft(name: str = "surveyor", **overrides: Any) -> dict[str, Any]:
    draft: dict[str, Any] = {
        "name": name,
        "role": "Site Surveyor",
        "description": "Maps a site before anyone digs.",
        "system_prompt": "You survey.\nReport  \n what you map.",
        "allowed_tools": ["map_area"],
        "model_name": "hermes3:8b",
        "model_tier": "fast",
        "temperature": 0.3,
        "max_tokens": 512,
        "enable_write_tools": False,
        "enable_subagent_tools": True,
    }
    draft.update(overrides)
    return draft


def _listed(client: TestClient) -> dict[str, dict[str, Any]]:
    res = client.get("/api/personas")
    assert res.status_code == 200
    return {p["name"]: p for p in res.json()["personas"]}


def _files_under(root: Path) -> set[Path]:
    return {p.relative_to(root) for p in root.rglob("*") if p.is_file() or p.is_symlink()}


@pytest.fixture
def workspace(tmp_path: Path, builtin_personas_absent: None) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    return root


# --- create ------------------------------------------------------------------------------


def test_create_writes_one_yaml_file_in_the_directory_the_loader_reads(workspace: Path) -> None:
    """A created persona is a file in `<workspace>/.uclone/personas/`, and the catalogue lists it.

    That directory, not `~/.uclone/personas/`, is the one `PersonaRegistry.reload` reads
    (`DEFAULT_WORKSPACE_PERSONAS_SUBDIR` under the workspace root), so it is the one a write
    must land in for the next process to see it.

    Killed by: src/uclone_x/agent/persona_store.py :: os.replace(temp_path, target)
    Becomes: os.remove(temp_path)
    """
    client = _client(workspace)

    res = client.post("/api/personas", json=_draft())

    assert res.status_code == 201, res.text
    assert _files_under(workspace / PERSONAS_SUBDIR) == {Path("surveyor.yaml")}
    listed = _listed(client)["surveyor"]
    assert listed["role"] == "Site Surveyor"
    assert listed["system_prompt"] == "You survey.\nReport  \n what you map."
    assert listed["allowed_tools"] == ["map_area"]
    assert listed["model_name"] == "hermes3:8b"
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
    client = _client(workspace)
    prompt = "당신은 측량사입니다.\n  indented line  \n\ntrailing   "
    assert client.post("/api/personas", json=_draft(system_prompt=prompt)).status_code == 201

    fresh = PersonaRegistry(workspace_root=workspace, include_defaults=False)
    loaded = fresh.get_persona("surveyor")

    assert loaded is not None
    assert loaded.system_prompt == prompt
    assert loaded.allowed_tools == ("map_area",)
    assert loaded.llm_config.model_name == "hermes3:8b"
    assert loaded.llm_config.model_tier == "fast"
    assert loaded.llm_config.max_tokens == 512
    assert loaded.enable_subagent_tools is True


def test_create_refuses_a_name_that_is_already_defined(workspace: Path) -> None:
    """Create never overwrites: an existing name is a 409 and its file is left as it was.

    Killed by: src/uclone_x/agent/persona_store.py :: if create and draft.name in self._personas:
    Becomes: if False:
    """
    client = _client(workspace)
    assert client.post("/api/personas", json=_draft()).status_code == 201
    before = (workspace / PERSONAS_SUBDIR / "surveyor.yaml").read_bytes()

    res = client.post("/api/personas", json=_draft(role="Impostor"))

    assert res.status_code == 409
    assert "surveyor" in res.json()["detail"]
    assert (workspace / PERSONAS_SUBDIR / "surveyor.yaml").read_bytes() == before


# --- update ------------------------------------------------------------------------------


def test_update_rewrites_the_file_the_persona_was_loaded_from(workspace: Path) -> None:
    """An edit replaces the persona's own file, even one not named after it.

    A second file would be shadowed or would shadow depending on sort order, so an edit
    written beside the original can silently disappear on the next restart.

    Killed by: src/uclone_x/agent/persona_store.py :: in_place = source is not None and source.parent == directory
    Becomes: in_place = False
    """
    personas_dir = workspace / PERSONAS_SUBDIR
    personas_dir.mkdir(parents=True)
    (personas_dir / "zz_legacy.yml").write_text(
        "name: surveyor\nrole: Old\nsystem_prompt: old prompt\n", encoding="utf-8"
    )
    client = _client(workspace)

    res = client.put("/api/personas/surveyor", json=_draft(role="New"))

    assert res.status_code == 200, res.text
    assert _files_under(personas_dir) == {Path("zz_legacy.yml")}
    assert yaml.safe_load((personas_dir / "zz_legacy.yml").read_text())["role"] == "New"
    assert _listed(client)["surveyor"]["role"] == "New"


def test_update_of_an_unknown_persona_is_404(workspace: Path) -> None:
    """Update is not create: an unknown name is a 404 and nothing is written.

    Killed by: src/uclone_x/agent/persona_store.py :: if not create and not self.has_persona(draft.name):
    Becomes: if False:
    """
    client = _client(workspace)

    res = client.put("/api/personas/nobody", json=_draft(name="nobody"))

    assert res.status_code == 404
    assert not (workspace / PERSONAS_SUBDIR).exists()


def test_update_refuses_a_body_naming_a_different_persona(workspace: Path) -> None:
    """Renaming is not an edit; a body whose name differs from the path is refused.

    Killed by: src/uclone_x/ui/app.py :: if name is not None and draft.name != name:
    Becomes: if False:
    """
    client = _client(workspace)
    assert client.post("/api/personas", json=_draft()).status_code == 201

    res = client.put("/api/personas/surveyor", json=_draft(name="digger"))

    assert res.status_code == 422
    assert set(_listed(client)) == {"surveyor"}


def test_editing_a_builtin_writes_an_override_and_leaves_the_shipped_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A built-in persona is edited by an override file in the workspace, never in the package.

    The shipped file is what an upgrade replaces; editing it would lose the change on the
    next install and, in a read-only site-packages, fail outright. The loader already lets
    a workspace file win over a built-in of the same name, so the override is that file.
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
    res = client.put("/api/personas/guide", json=edit)

    assert res.status_code == 200, res.text
    assert shipped_file.read_bytes() == shipped_bytes
    override = workspace / PERSONAS_SUBDIR / "guide.yaml"
    written = yaml.safe_load(override.read_text(encoding="utf-8"))
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

    Killed by: src/uclone_x/agent/persona_store.py :: _refuse_an_unwritable_name(draft.name)
    Becomes: pass
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    client = _client(workspace)
    before = _files_under(tmp_path)

    res = client.post("/api/personas", json=_draft(name=bad_name))

    assert res.status_code == 422
    assert "cannot be a persona name" in str(res.json()["detail"])
    assert _files_under(tmp_path) - before <= {Path("ws/sessions")}  # nothing new but storage
    assert not any(p.suffix in {".yaml", ".yml"} for p in _files_under(tmp_path) - before)


def test_a_write_that_resolves_outside_the_personas_directory_is_refused(
    tmp_path: Path, builtin_personas_absent: None
) -> None:
    """A persona file that is a symlink out of the directory is not written through.

    A name cannot reach outside the directory (the name rule above), but the file an edit
    rewrites is the one the persona was *loaded* from, and that can be a link. Writing
    through it would change a file the user never pointed the head at.

    Killed by: src/uclone_x/agent/persona_store.py :: if resolved_target.parent != directory.resolve():
    Becomes: if False:
    """
    outside = tmp_path / "elsewhere.yaml"
    outside.write_text("name: linked\nrole: Linked\nsystem_prompt: linked\n", encoding="utf-8")
    outside_bytes = outside.read_bytes()
    workspace = tmp_path / "ws"
    personas_dir = workspace / PERSONAS_SUBDIR
    personas_dir.mkdir(parents=True)
    (personas_dir / "linked.yaml").symlink_to(outside)
    client = _client(workspace)

    res = client.put("/api/personas/linked", json=_draft(name="linked", role="Changed"))

    assert res.status_code == 422
    assert "outside" in res.json()["detail"]
    assert outside.read_bytes() == outside_bytes
    assert _listed(client)["linked"]["role"] == "Linked"


def test_an_unregistered_tool_is_refused_and_the_previous_definition_stands(
    workspace: Path,
) -> None:
    """A tool name the runtime does not have is a 422, and the file and catalogue are unchanged.

    The candidate file is written to a temporary name, loaded by the loader, and only then
    moved over the real one, so a refused edit leaves no half-written file behind either.

    Killed by: src/uclone_x/agent/persona_store.py :: persona = self._parse_file(temp_path, label=target)
    Becomes: persona = self._parse_file(temp_path, label=target) if False else PersonaDefinition(name=draft.name, role=draft.role, system_prompt=draft.system_prompt)
    """
    client = _client(workspace)
    assert client.post("/api/personas", json=_draft()).status_code == 201
    before = (workspace / PERSONAS_SUBDIR / "surveyor.yaml").read_bytes()

    res = client.put("/api/personas/surveyor", json=_draft(allowed_tools=["map_aera"]))

    assert res.status_code == 422
    assert "map_aera" in res.json()["detail"]
    assert "map_area" in res.json()["detail"]  # the loader's did-you-mean
    assert (workspace / PERSONAS_SUBDIR / "surveyor.yaml").read_bytes() == before
    assert _files_under(workspace / PERSONAS_SUBDIR) == {Path("surveyor.yaml")}
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

    res = client.post("/api/personas", json=_draft(**overrides))

    assert res.status_code == 422, res.text
    assert not (workspace / PERSONAS_SUBDIR).exists()


# --- a running agent ---------------------------------------------------------------------


def test_a_running_agent_takes_the_edited_prompt_and_tools_on_its_next_turn(
    workspace: Path,
) -> None:
    """An open conversation with the persona uses the edit on its next turn, tools included.

    The prompt alone would follow the registry by itself -- the agent re-resolves it on every
    read -- so asserting the prompt could not see the defect this pins. The tools are the
    half that is stored: the session manager used to copy the persona's list in as the
    *operator's* list, which the agent's scope rule lets win over any persona, and wrap the
    registry in a proxy fixed at creation. Either one kept the old tools in force behind the
    new prompt.

    Killed by: src/uclone_x/agent/bootstrap.py :: allowed_tools=(),  # the persona's list, resolved by the agent
    Becomes: allowed_tools=persona.granted_tools,
    """
    llm = _RecordingConnector()
    client = _client(workspace, llm)
    assert client.post("/api/personas", json=_draft()).status_code == 201
    turn = {"message": "go", "agent_id": "surveyor", "session_id": "sess_survey"}

    assert client.post("/api/turn", json=turn).status_code == 200
    assert _system_sent(llm.requests[-1]) == "You survey.\nReport  \n what you map."
    assert _own_tools_sent(llm.requests[-1]) == ["map_area"]

    res = client.put(
        "/api/personas/surveyor",
        json=_draft(system_prompt="You dig now.", allowed_tools=["dig_site"]),
    )
    assert res.status_code == 200, res.text
    assert res.json()["live_agents_updated"] == 1

    assert client.post("/api/turn", json=turn).status_code == 200
    assert _system_sent(llm.requests[-1]) == "You dig now."
    assert _own_tools_sent(llm.requests[-1]) == ["dig_site"]


def test_an_edit_reaches_every_live_agent_seated_as_that_persona_and_no_other(
    workspace: Path,
) -> None:
    """Two conversations with the persona both take the edit; another persona's does not.

    Killed by: src/uclone_x/ui/app.py :: if agent.persona != persona.name or id(agent) in seen:
    Becomes: if id(agent) in seen:
    """
    llm = _RecordingConnector()
    client = _client(workspace, llm)
    assert client.post("/api/personas", json=_draft()).status_code == 201
    assert client.post("/api/personas", json=_draft(name="digger")).status_code == 201
    for agent_id, session_id in (("surveyor", "s1"), ("surveyor", "s2"), ("digger", "s3")):
        chat = {"message": "go", "agent_id": agent_id, "session_id": session_id}
        assert client.post("/api/turn", json=chat).status_code == 200

    res = client.put(
        "/api/personas/surveyor", json=_draft(system_prompt="Edited.", allowed_tools=["dig_site"])
    )

    assert res.json()["live_agents_updated"] == 2
    assert (
        client.post(
            "/api/turn", json={"message": "go", "agent_id": "digger", "session_id": "s3"}
        ).status_code
        == 200
    )
    assert _system_sent(llm.requests[-1]) == "You survey.\nReport  \n what you map."
    assert _own_tools_sent(llm.requests[-1]) == ["map_area"]


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
        assert client.post("/api/personas", json=_draft()).status_code == 201
        turn = {"message": "go", "agent_id": "surveyor", "session_id": "sess_sub"}
        assert client.post("/api/turn", json=turn).status_code == 200
        parent = manager.get_agent("surveyor", "sess_sub")
        assert parent is not None

        def offered_to_a_child(parent: BaseAgent) -> list[str]:
            child = portal.call(parent.spawn_subagent, "helper", "help")
            portal.call(parent.delegate_task, child, "go")
            return [t.name for t in llm.requests[-1].tools]

        assert offered_to_a_child(parent) == ["map_area"]

        edit = client.put("/api/personas/surveyor", json=_draft(allowed_tools=["dig_site"]))
        assert edit.status_code == 200, edit.text
        assert offered_to_a_child(parent) == ["dig_site"]

        child = portal.call(parent.spawn_subagent, "helper", "help")
        with pytest.raises(PermissionError):
            portal.call(child.execute_tool_call, "map_area", dict[str, Any]())
