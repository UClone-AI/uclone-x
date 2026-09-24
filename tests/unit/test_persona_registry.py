"""Tests for PersonaRegistry (Principle 0, Principle 4).

Verifies declarative YAML persona discovery, workspace override hierarchy,
and BaseAgent persona resolution.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from uclone_x.agent import persona_registry as persona_registry_module
from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import (
    BASE_PERSONA_TOOLS,
    AgentConfig,
    AgentContext,
    PersonaDefinition,
)
from uclone_x.agent.persona_registry import (
    PersonaLoadError,
    PersonaRegistry,
    get_default_persona_registry,
)
from uclone_x.agent.prompts import compose_system_prompt
from uclone_x.tools import ToolRegistry


def test_builtin_personas_discovered() -> None:
    registry = PersonaRegistry()
    ordered = registry.list_personas()
    personas = {p.name: p for p in ordered}

    assert set(personas.keys()) == {"clone", "writer", "artist", "guardian", "pioneer", "scout"}

    # Default persona clone is ordered first
    assert ordered[0].name == "clone"
    clone = registry.get_persona("clone")
    assert clone is not None
    assert "Personal AI Clone" in clone.role
    assert clone.enable_write_tools is True
    assert clone.enable_subagent_tools is True

    # Verify writer specifics
    writer = registry.get_persona("writer")
    assert writer is not None
    assert "Storyteller" in writer.role
    assert "file_write" in writer.allowed_tools
    assert "file_read" in writer.allowed_tools
    assert "file_search" in writer.allowed_tools
    assert writer.enable_write_tools is True
    assert writer.enable_subagent_tools is False

    # Verify artist specifics
    artist = registry.get_persona("artist")
    assert artist is not None
    assert "Visual Artist" in artist.role
    assert "generate_image" in artist.allowed_tools
    assert "file_read" in artist.allowed_tools
    assert artist.enable_write_tools is True

    # Verify guardian specifics
    guardian = registry.get_persona("guardian")
    assert guardian is not None
    assert "Risk Analyst" in guardian.role
    assert "file_read" in guardian.allowed_tools
    assert "web_search" in guardian.allowed_tools
    assert "web_fetch" in guardian.allowed_tools
    assert guardian.enable_write_tools is False

    # Verify pioneer specifics
    pioneer = registry.get_persona("pioneer")
    assert pioneer is not None
    assert "Visionary Innovator" in pioneer.role
    assert "file_write" in pioneer.allowed_tools
    assert pioneer.enable_write_tools is True

    # Verify scout specifics
    scout = registry.get_persona("scout")
    assert scout is not None
    assert "Research & Search Specialist" in scout.role
    assert "web_search" in scout.allowed_tools


def test_workspace_override_persona(tmp_path: Path) -> None:
    ws_personas = tmp_path / ".uclone" / "personas"
    ws_personas.mkdir(parents=True)

    # Override writer with custom instructions
    custom_yaml = ws_personas / "writer.yaml"
    custom_yaml.write_text(
        """
name: writer
role: Custom Workspace Writer
description: Overridden writer persona.
system_prompt: Custom prompt for workspace writer.
allowed_tools:
  - write_to_file
  - custom_tool
llm:
  model_name: custom-model
  temperature: 0.9
enable_write_tools: true
""",
        encoding="utf-8",
    )

    registry = PersonaRegistry(workspace_root=tmp_path)
    writer = registry.get_persona("writer")
    assert writer is not None
    assert writer.role == "Custom Workspace Writer"
    assert writer.system_prompt == "Custom prompt for workspace writer."
    assert "custom_tool" in writer.allowed_tools
    assert writer.llm_config.model_name == "custom-model"


def test_workspace_new_custom_persona(tmp_path: Path) -> None:
    ws_personas = tmp_path / ".uclone" / "personas"
    ws_personas.mkdir(parents=True)

    custom_yaml = ws_personas / "poet.yaml"
    custom_yaml.write_text(
        """
name: poet
role: Lyric Poet
description: Generates lyrical poetry.
system_prompt: You are a lyrical poet.
allowed_tools:
  - write_to_file
""",
        encoding="utf-8",
    )

    registry = PersonaRegistry(workspace_root=tmp_path)
    poet = registry.get_persona("poet")
    assert poet is not None
    assert poet.name == "poet"
    assert poet.role == "Lyric Poet"
    assert poet.allowed_tools == ("write_to_file",)


def test_dynamic_persona_registration() -> None:
    registry = PersonaRegistry()
    dynamic_p = PersonaDefinition(
        name="dynamic_scripter",
        role="Automation Scripter",
        description="Writes scripts",
        system_prompt="You write scripts.",
        allowed_tools=("file_write", "bash_run"),
    )
    registry.register_persona(dynamic_p)

    fetched = registry.get_persona("dynamic_scripter")
    assert fetched is not None
    assert fetched.name == "dynamic_scripter"
    assert fetched.allowed_tools == ("file_write", "bash_run")


def test_unloadable_persona_file_raises_naming_the_file(tmp_path: Path) -> None:
    """A corrupt persona file is an error, not a persona that quietly does not exist.

    Amends `test_invalid_yaml_file_gracefully_skipped`, which pinned the opposite. The
    skip is P6's silent fallback: a user who mistypes a key in their own file gets a
    runtime missing one agent and a log line nobody reads, which looks exactly like
    having written no file at all.

    Killed by: src/uclone_x/agent/persona_store.py :: raise PersonaLoadError(f"{file_path}: {exc}") from exc
    Becomes: logger.warning("skipped %s: %s", file_path, exc)
    """
    ws_personas = tmp_path / ".uclone" / "personas"
    ws_personas.mkdir(parents=True)
    (ws_personas / "corrupt.yaml").write_text("invalid: [yaml: unclosed", encoding="utf-8")

    with pytest.raises(PersonaLoadError) as excinfo:
        PersonaRegistry(workspace_root=tmp_path)

    assert "corrupt.yaml" in str(excinfo.value)


def test_append_default_prompt_must_be_a_boolean(tmp_path: Path) -> None:
    """A non-boolean in that key is refused rather than read for its truthiness.

    `"no"` and `"false"` are both true as strings. Coercing here would hand the author the
    opposite of what their file says, with nothing reported -- the P6 substitution.

    Killed by: src/uclone_x/agent/persona_store.py :: if not isinstance(append_default, bool):
    Becomes: if False:
    """
    ws_personas = tmp_path / ".uclone" / "personas"
    ws_personas.mkdir(parents=True)
    (ws_personas / "maybe.yaml").write_text(
        'name: maybe\nrole: Maybe\nsystem_prompt: Hello.\nappend_default_prompt: "no"\n',
        encoding="utf-8",
    )

    with pytest.raises(PersonaLoadError) as excinfo:
        PersonaRegistry(workspace_root=tmp_path)

    assert "append_default_prompt" in str(excinfo.value)
    assert "maybe.yaml" in str(excinfo.value)


def test_a_file_that_does_not_ask_for_the_default_prompt_keeps_its_own(tmp_path: Path) -> None:
    """Appending is opt-in, so an operator's prompt is served exactly as written.

    The shipped workspace personas carry prompts that stand alone; silently appending the
    composed default to them would change every one of them.

    Killed by: src/uclone_x/agent/persona_store.py :: append_default: object = data.pop("append_default_prompt", False)
    Becomes: append_default: object = data.pop("append_default_prompt", True)
    """
    ws_personas = tmp_path / ".uclone" / "personas"
    ws_personas.mkdir(parents=True)
    (ws_personas / "standalone.yaml").write_text(
        "name: standalone\nrole: Standalone\nsystem_prompt: Answer briefly.\n",
        encoding="utf-8",
    )
    (ws_personas / "grounded.yaml").write_text(
        "name: grounded\nrole: Grounded\nsystem_prompt: Answer briefly."
        "\nappend_default_prompt: true\n",
        encoding="utf-8",
    )

    registry = PersonaRegistry(workspace_root=tmp_path)

    standalone = registry.get_persona("standalone")
    assert standalone is not None
    assert standalone.system_prompt == "Answer briefly."

    grounded = registry.get_persona("grounded")
    assert grounded is not None
    assert (
        grounded.system_prompt
        == f"Answer briefly.\n\n{compose_system_prompt(writes_permitted=False)}"
    )


def test_persona_naming_an_unregistered_tool_is_refused(tmp_path: Path) -> None:
    """`allowed_tools` is checked against the inventory, and the error is actionable.

    `list_tools(filter_names=...)` filters by set membership, so an unresolvable name is
    invisible: the agent advertises fewer tools than its file declares and nothing says
    so. This is the only place that still knows which file the name came from.

    Killed by: src/uclone_x/agent/persona_store.py :: self._validate_tools(persona, source=source)
    Becomes:
    """
    ws_personas = tmp_path / ".uclone" / "personas"
    ws_personas.mkdir(parents=True)
    (ws_personas / "typo.yaml").write_text(
        "name: typo\nrole: R\nsystem_prompt: S\nallowed_tools:\n  - file_wrote\n",
        encoding="utf-8",
    )

    # The real inventory, so the shipped personas load cleanly and the only file that can
    # fail is the one this test wrote.
    inventory = [tool.name for tool in ToolRegistry.with_builtins(enable_mcp=False).list_tools()]

    with pytest.raises(PersonaLoadError) as excinfo:
        PersonaRegistry(workspace_root=tmp_path, tool_names=inventory)

    message = str(excinfo.value)
    assert "typo.yaml" in message
    assert "file_wrote" in message
    # The near-match is what makes the error worth reading rather than merely correct.
    assert "file_write" in message


def test_no_tool_inventory_means_no_tool_check(tmp_path: Path) -> None:
    """Without an inventory the check is skipped, not guessed at.

    A registry built with no `tool_names` cannot distinguish a typo from a tool an MCP
    server registers later, so it refuses to rule either way.
    """
    ws_personas = tmp_path / ".uclone" / "personas"
    ws_personas.mkdir(parents=True)
    (ws_personas / "later.yaml").write_text(
        "name: later\nrole: R\nsystem_prompt: S\nallowed_tools:\n  - mcp_tool_added_later\n",
        encoding="utf-8",
    )

    registry = PersonaRegistry(workspace_root=tmp_path)

    assert registry.get_persona("later") is not None


def test_every_shipped_persona_resolves_against_the_default_registry() -> None:
    """The declaration a shipped persona makes is one the runtime can actually honour.

    `scout` shipped naming `grep_search` and `find_by_name`, which are registered by
    nothing, and `novelist`/`story_writer` shipped naming `write_to_file` while telling
    the model in their own prompts to save chapters with it. All three were advertised a
    strictly smaller toolset than they declared, silently.

    Killed by: src/uclone_x/personas/scout.yaml :: - directory_list
    Becomes: - grep_search
    """
    registered = {tool.name for tool in ToolRegistry.with_builtins(enable_mcp=False).list_tools()}
    registry = PersonaRegistry()

    unresolved = {
        persona.name: [name for name in persona.allowed_tools if name not in registered]
        for persona in registry.list_personas()
    }

    assert {name: missing for name, missing in unresolved.items() if missing} == {}


def test_a_persona_defined_on_one_agent_is_not_visible_to_another() -> None:
    """`define_persona` registers on the agent, not on the class.

    `_persona_store` was a class attribute, so `self._persona_store[name] = persona`
    mutated the single dict on the class body: every persona any agent defined leaked to
    every other agent in the process, sub-agents in isolated contexts included, and
    nothing cleared it between sessions.

    Killed by: src/uclone_x/agent/base.py :: self._persona_store: dict[str, PersonaDefinition] = {p.name: p for p in personas}
    Becomes:
    """
    first = BaseAgent(config=AgentConfig(agent_id="first", name="First"))
    second = BaseAgent(config=AgentConfig(agent_id="second", name="Second"))
    private = PersonaDefinition(name="private_helper", role="Helper", system_prompt="You help.")

    first.define_persona(private)

    assert first.get_persona("private_helper") is private
    assert second.get_persona("private_helper") is None


def test_a_registered_persona_named_scout_overrides_the_shipped_one() -> None:
    """No constant is consulted after the store and the registry.

    `get_persona` returned `SCOUT_PERSONA`/`CRITIC_PERSONA` from module constants ahead of
    the store, so a persona registered under either name was neither honoured nor refused.

    The store is consulted *first*; that ordering is what the declaration below mutates.
    Returning the constant from the fallback branch is not what this pins -- the store hit
    short-circuits before any fallback runs, which is the point.

    Killed by: src/uclone_x/agent/base.py :: if name in self._persona_store:
    Becomes: if False:
    """
    agent = BaseAgent(config=AgentConfig(agent_id="host", name="Host"))
    replacement = PersonaDefinition(
        name="scout", role="Replaced Scout", system_prompt="You are the replacement."
    )

    agent.define_persona(replacement)
    resolved = agent.get_persona("scout")

    assert resolved is not None
    assert resolved is replacement
    assert resolved.role == "Replaced Scout"


def test_workspace_persona_tools_reach_the_agent_that_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `.uclone/personas/` definition contributes its tools at construction.

    Persona resolution ran in `__init__` *before* `self._context` was assigned, behind a
    `hasattr(self, "_context")` guard that was therefore always false. The workspace root
    was never found, so only shipped personas were searched and a user's own persona --
    the entire point of the declarative path -- could not contribute anything.

    Killed by: src/uclone_x/agent/base.py :: self._workspace_root_hint: Path | None = self._context.workspace_root
    Becomes: self._workspace_root_hint: Path | None = None
    """
    ws_personas = tmp_path / ".uclone" / "personas"
    ws_personas.mkdir(parents=True)
    (ws_personas / "surveyor.yaml").write_text(
        "name: surveyor\nrole: Surveyor\nsystem_prompt: You survey.\n"
        "allowed_tools:\n  - file_read\n  - directory_list\n",
        encoding="utf-8",
    )
    # The module-level default registry is a cached singleton keyed on workspace root.
    # Priming it with this tmp_path here would make the test pass whatever the agent
    # passes -- including `None` -- so it is cleared instead, leaving the agent's own
    # hint as the only thing that can reach `.uclone/personas/`.
    monkeypatch.setattr(persona_registry_module, "_default_persona_registry", None)

    agent = BaseAgent(
        config=AgentConfig(agent_id="surveyor", name="Surveyor", persona="surveyor"),
        context=AgentContext(
            session_id="sess_surveyor", agent_id="surveyor", workspace_root=tmp_path
        ),
    )

    # Its own list, then the base set every persona is given (#1402).
    assert agent.config.allowed_tools[:2] == ("file_read", "directory_list")
    assert set(agent.config.allowed_tools) == {"file_read", "directory_list", *BASE_PERSONA_TOOLS}


def test_base_agent_resolves_persona_and_inherits_tools() -> None:
    config = AgentConfig(
        agent_id="test-writer",
        name="Test Writer",
        persona="writer",
    )
    agent = BaseAgent(config=config)

    persona = agent.get_persona("writer")
    assert persona is not None
    assert persona.name == "writer"

    # Effective system prompt should come from writer persona
    prompt = agent.effective_system_prompt
    assert "Writer" in prompt or "storyteller" in prompt

    # Allowed tools should be inherited from writer persona
    assert "file_write" in agent.config.allowed_tools
    assert "file_read" in agent.config.allowed_tools
    assert "file_search" in agent.config.allowed_tools


def test_get_default_persona_registry_singleton(tmp_path: Path) -> None:
    reg1 = get_default_persona_registry()
    reg2 = get_default_persona_registry()
    assert reg1 is reg2

    reg3 = get_default_persona_registry(workspace_root=tmp_path)
    assert reg3.workspace_root == tmp_path.resolve()
    assert reg3 is not reg1


def test_a_cached_registry_without_an_inventory_is_rebuilt_when_one_arrives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validation is decided by configuration, not by which caller reaches the singleton first.

    `GET /api/personas` serves the dashboard on load and so usually runs before any chat
    has created an agent. If the first caller's registry -- built with no inventory -- were
    kept, the tool check would be configured, would pass its own tests, and would never
    run in the assembled product.

    Killed by: src/uclone_x/agent/persona_registry.py :: or _default_persona_registry is None
    Becomes: and False
    """
    monkeypatch.setattr(persona_registry_module, "_default_persona_registry", None)
    inventory = [tool.name for tool in ToolRegistry.with_builtins(enable_mcp=False).list_tools()]

    unvalidated = get_default_persona_registry(workspace_root=tmp_path)
    assert unvalidated.validates_tools is False

    validated = get_default_persona_registry(workspace_root=tmp_path, tool_names=inventory)

    assert validated.validates_tools is True
    assert validated is not unvalidated
    # And an inventory arriving a second time does not churn the registry again.
    assert get_default_persona_registry(tool_names=inventory) is validated


def test_a_persona_name_no_directory_can_carry_is_refused_at_its_file(tmp_path: Path) -> None:
    """A persona's name is the `agent_id` the dashboard sends, so it is a directory name.

    `WorkspaceSidebar` posted `p.name` as `agent_id` (so did `PlaygroundTab`, until #1208
    retired it), which reaches
    `default_cross_session_memory` and becomes `<agents root>/<name>/`. A persona named
    `Code Reviewer` therefore loaded, appeared in the sidebar and the model dropdown, and
    failed on its first message -- with a refusal naming a rule this loader never applied
    and no file at all. This is the only place that still knows which YAML to change.

    Killed by: src/uclone_x/agent/persona_store.py :: refuse_an_unusable_username(persona_name)
    Becomes: pass
    """
    ws_personas = tmp_path / ".uclone" / "personas"
    ws_personas.mkdir(parents=True)
    (ws_personas / "reviewer.yaml").write_text(
        "name: Code Reviewer\nrole: R\nsystem_prompt: S\n", encoding="utf-8"
    )

    with pytest.raises(PersonaLoadError) as excinfo:
        PersonaRegistry(workspace_root=tmp_path)

    message = str(excinfo.value)
    assert "reviewer.yaml" in message, "the error must name the file that has to change"
    assert "Code Reviewer" in message
    assert "agent id" in message, "the reader has to learn why a persona name is restricted"
