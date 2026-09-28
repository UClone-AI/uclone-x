import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "src"))


import pytest

from tests.support.app_clone import app_clone
from uclone_x.agent.models import PersonaDefinition
from uclone_x.agent.persona_registry import PersonaRegistry


def test_default_personas_ship_as_package_files():
    """The default built-ins are discovered from `personas/`, not defined in code.

    Killed by: src/uclone_x/agent/persona_registry.py :: if self._include_defaults and BUILTIN_PERSONAS_DIR.is_dir():
    Becomes: if False:
    """
    registry = PersonaRegistry()

    clone = registry.get_persona("clone")
    assert isinstance(clone, PersonaDefinition)
    assert clone.role == "Personal AI Clone & Collaborator"
    assert clone.enable_write_tools
    assert clone.enable_subagent_tools

    scout = registry.get_persona("scout")
    assert isinstance(scout, PersonaDefinition)
    assert scout.role == "Research & Search Specialist"
    assert not scout.enable_write_tools

    guardian = registry.get_persona("guardian")
    assert isinstance(guardian, PersonaDefinition)
    assert guardian.role == "Risk Analyst & Strategic Guardian"
    assert not guardian.enable_write_tools

    pioneer = registry.get_persona("pioneer")
    assert isinstance(pioneer, PersonaDefinition)
    assert pioneer.role == "Visionary Innovator & Growth Strategist"
    assert pioneer.enable_write_tools


def test_no_hardcoded_agent_name_stands_in_for_a_missing_one():
    """No route may fill a missing `agent_id` with a name it made up.

    This test used to require the opposite -- it asserted that
    `str(req.get("agent_id", "champion"))` was present -- which pinned the very
    substitution P6 forbids: on an install with no `champion`, a request that named
    nobody was answered by, and attributed to, an agent that does not exist there.

    A grep over the source rather than a behavioural check, because the failure it
    guards against is a *re-introduction*: someone adding a sixth route with a
    convenient default. The behaviour itself is covered by the 400 and 422 tests in
    `test_ui_agent_id_required.py`.
    """
    app_py_path = pathlib.Path(__file__).parent.parent.parent / "src" / "uclone_x" / "ui" / "app.py"
    content = app_py_path.read_text()

    for invented in ('"agent_id", "champion"', '"agent_id", "scout"', '"agent_id", "critic"'):
        assert invented not in content, f"a route still substitutes {invented} for a missing name"
    assert 'agent_id: str = "champion"' not in content
    assert 'role = str(req.get("role", "assistant"))' in content
    assert '"agent-orchestrator"' not in content


@pytest.mark.asyncio
async def test_topology_integration(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An agent the user asked for directly is nobody's child.

    Killed by: src/uclone_x/agent/clone_builder.py :: context=AgentContext(
    Becomes: context=AgentContext(parent_agent_id="champion" if clone_id != "champion" else None,
    """
    # This test builds an agent but is not about provider resolution, so it names the
    # connector it wants.
    from uclone_x.llm import MockLLMConnector
    from uclone_x.ui.app import AgentSessionManager

    session_mgr = AgentSessionManager(storage_dir=tmp_path / "storage", llm=MockLLMConnector())

    # Two agents the UI can build. Neither is anybody's parent: the retired chat helper
    # used to set `parent_agent_id="champion" if agent_id != "champion"`, so every agent
    # but one was recorded as having been spawned by `champion` -- an edge the dashboard's
    # agent graph then drew between two agents that had never spoken to each other.
    clone_agent = app_clone(session_mgr, "clone")
    assert clone_agent.agent_id == "clone"
    assert clone_agent.context.parent_agent_id is None

    guardian_agent = app_clone(session_mgr, "guardian")
    assert guardian_agent.agent_id == "guardian"
    assert guardian_agent.context.parent_agent_id is None
