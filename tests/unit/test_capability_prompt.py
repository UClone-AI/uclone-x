"""The default prompt follows the tools the agent holds, and one shell is advertised (#1424).

Pins four things:
1. A fragment of tool guidance reaches only an agent that holds its tool, and a writing
   tool's guidance only an agent allowed to write.
2. Every tool name a prompt mentions is a registered tool -- the check that would have
   caught `write_to_file`, which named nothing the registry serves.
3. The shell is advertised under one name; the other stays registered as an alias.
4. The composition is byte-stable: the same persona composes to the same bytes whatever
   order its tools are listed in, on every load, and in every process.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.composition import HostDependencies
from uclone_x.agent.models import AgentConfig, PersonaDefinition
from uclone_x.agent.persona_registry import PersonaRegistry
from uclone_x.agent.persona_store import default_prompt_for, split_appended_default_prompt
from uclone_x.agent.prompts import (
    ARTIFACT_REPORTING,
    CAPABILITY_FRAGMENTS,
    ENVIRONMENT_REPAIR,
    HERMES_STEERABILITY_POLICY,
    IMAGE_GENERATION,
    compose_system_prompt,
)
from uclone_x.agent.session import SessionStore
from uclone_x.engine.event_bus import EventBus
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import LLMRequest, ModelResponse
from uclone_x.room.models import Participant, ParticipantKind
from uclone_x.room.resolver import RoomAgentResolver
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.base import drop_shadowed_aliases, tool_writes_files
from uclone_x.tools.registry import ToolRegistry, create_default_registry
from uclone_x.ui.app import AgentSessionManager

#: A lowercase word with an underscore in it: how every tool name is spelled, and how
#: almost nothing else in a prose prompt is.
_SNAKE_CASE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")

#: Snake-case words a prompt may use that are not tool names. Each is a field of a tool's
#: result the prompt tells the model to read, not something it can call.
_NOT_TOOL_NAMES = frozenset({"relative_url"})


class _RecordingConnector(MockLLMConnector):
    def __init__(self) -> None:
        super().__init__(default_response="ok")
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        return await super().generate(request)


def _builtin(name: str) -> PersonaDefinition:
    persona = PersonaRegistry().get_persona(name)
    assert persona is not None, name
    return persona


def _registered_tool_names() -> set[str]:
    return {tool.name for tool in create_default_registry(enable_mcp=False).list_tools()}


# --- guidance follows the tools ----------------------------------------------------------


def test_a_persona_without_image_or_package_tools_is_told_nothing_about_them() -> None:
    """Scout holds four tools and neither of these, so neither fragment reaches it.

    Artist is the control: it holds `generate_image` and not `install_package`, so the
    check can see a fragment arrive as well as stay away.

    Killed by: src/uclone_x/agent/prompts.py :: if (tools is None or fragment.tool in tools)
    Becomes: if (True)
    """
    scout = _builtin("scout").system_prompt
    for absent in (IMAGE_GENERATION, ENVIRONMENT_REPAIR, "generate_image", "install_package"):
        assert absent not in scout

    artist = _builtin("artist").system_prompt
    assert IMAGE_GENERATION in artist
    assert ENVIRONMENT_REPAIR not in artist
    assert "install_package" not in artist


def test_a_persona_refused_writes_is_not_told_to_write_a_file() -> None:
    """Guardian lists `file_write` but its `enable_write_tools` is off, so it cannot call it.

    Pioneer lists the same tools with writes on, and is told.

    Killed by: src/uclone_x/agent/prompts.py :: and (writes_permitted or not fragment.writes_files)
    Becomes: and True
    """
    guardian = _builtin("guardian")
    assert "file_write" in guardian.allowed_tools
    assert guardian.enable_write_tools is False
    assert ARTIFACT_REPORTING not in guardian.system_prompt

    pioneer = _builtin("pioneer")
    assert set(pioneer.allowed_tools) == set(guardian.allowed_tools)
    assert ARTIFACT_REPORTING in pioneer.system_prompt


def test_an_unrestricted_persona_is_told_about_every_tool() -> None:
    """An empty `allowed_tools` is no restriction, so it holds every fragment's tool."""
    clone = _builtin("clone")
    assert clone.allowed_tools == ()
    for fragment in CAPABILITY_FRAGMENTS:
        assert fragment.text in clone.system_prompt


# --- every named tool exists -------------------------------------------------------------


def test_every_tool_name_a_prompt_mentions_is_registered() -> None:
    """A prompt naming an unregistered tool routes the model to nothing.

    Scans every fragment, both steerability framings, the unrestricted default and every
    built-in persona's composed prompt.

    Killed by: src/uclone_x/agent/prompts.py :: via file_write into the workspace
    Becomes: via write_to_file into the workspace
    """
    registered = _registered_tool_names()
    texts = {
        "default": compose_system_prompt(),
        "hermes": HERMES_STEERABILITY_POLICY,
        **{f"fragment:{f.tool}": f.text for f in CAPABILITY_FRAGMENTS},
        **{p.name: p.system_prompt for p in PersonaRegistry().list_personas()},
    }
    unknown = {
        source: sorted(set(_SNAKE_CASE.findall(text)) - registered - _NOT_TOOL_NAMES)
        for source, text in texts.items()
    }
    assert {source: names for source, names in unknown.items() if names} == {}


def test_each_fragment_is_keyed_by_a_registered_tool_it_names() -> None:
    """The key is a registered tool, the text names it, and the write flag is the tool's own.

    Killed by: src/uclone_x/agent/prompts.py :: CapabilityFragment(tool="install_package", text=ENVIRONMENT_REPAIR, writes_files=True)
    Becomes: CapabilityFragment(tool="install_package", text=ENVIRONMENT_REPAIR, writes_files=False)
    """
    registry = create_default_registry(enable_mcp=False)
    for fragment in CAPABILITY_FRAGMENTS:
        tool = registry.get(fragment.tool)
        assert tool is not None, fragment.tool
        assert fragment.tool in fragment.text
        assert fragment.writes_files is tool_writes_files(tool), fragment.tool


# --- one shell ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_default_registry_advertises_one_shell() -> None:
    """`bash_run` and `run_command` are one tool; a request carries it once, as `bash_run`.

    Killed by: src/uclone_x/agent/base.py :: for t in drop_shadowed_aliases(self._tools.list_tools(filter_names=allowed)):
    Becomes: for t in self._tools.list_tools(filter_names=allowed):
    """
    llm = _RecordingConnector()
    registry = create_default_registry(enable_mcp=False)
    agent = BaseAgent(config=AgentConfig(agent_id="a", name="A"), tools=registry, llm=llm)

    await agent.execute_turn("go")

    advertised = [tool.name for tool in llm.requests[-1].tools]
    assert [name for name in advertised if name in {"bash_run", "run_command"}] == ["bash_run"]
    # Still registered: a stored call or a persona file naming it resolves.
    assert registry.get("run_command") is not None


def test_run_command_is_registered_as_an_alias_of_bash_run() -> None:
    """Killed by: src/uclone_x/tools/registry.py :: alias_of="bash_run"
    Becomes: alias_of=None
    """
    registry = create_default_registry(enable_mcp=False)
    names = [tool.name for tool in drop_shadowed_aliases(registry.list_tools())]
    assert "bash_run" in names
    assert "run_command" not in names


def test_an_alias_is_kept_when_its_canonical_tool_is_not_offered() -> None:
    """A persona that lists only `run_command` keeps a shell.

    Killed by: src/uclone_x/tools/base.py :: if getattr(tool, "alias_of", None) not in names
    Becomes: if getattr(tool, "alias_of", None) is None
    """
    registry = create_default_registry(enable_mcp=False)
    only_alias = registry.list_tools(filter_names=("run_command", "file_read"))
    assert [tool.name for tool in drop_shadowed_aliases(only_alias)] == [
        tool.name for tool in only_alias
    ]
    assert "run_command" in [tool.name for tool in only_alias]


# --- byte stability ----------------------------------------------------------------------


def test_the_order_tools_are_listed_in_never_reaches_the_prompt() -> None:
    """Killed by: src/uclone_x/agent/prompts.py :: for fragment in CAPABILITY_FRAGMENTS
    Becomes: for fragment in sorted(CAPABILITY_FRAGMENTS, key=lambda f: [*tools].index(f.tool) if tools is not None and f.tool in tools else 0)
    """
    tools = ["web_search", "file_write", "install_package", "generate_image"]
    composed = compose_system_prompt(tools=tools)
    assert compose_system_prompt(tools=list(reversed(tools))) == composed
    assert compose_system_prompt(tools=frozenset(tools)) == composed
    assert compose_system_prompt(tools=["generate_image", *tools]) == composed


def test_a_persona_composes_to_the_same_bytes_on_every_load() -> None:
    """Two independent loads -- what a restart does -- agree, and agree with the derivation."""
    first = {p.name: p.system_prompt for p in PersonaRegistry().list_personas()}
    second = {p.name: p.system_prompt for p in PersonaRegistry().list_personas()}
    assert first == second
    scout = _builtin("scout")
    assert scout.system_prompt.endswith(f"\n\n{default_prompt_for(scout)}")


_PRINT_PROMPTS = (
    "from uclone_x.agent.persona_registry import PersonaRegistry;"
    "import sys;"
    "[sys.stdout.write(p.name + '\\0' + p.system_prompt + '\\0') "
    "for p in sorted(PersonaRegistry().list_personas(), key=lambda p: p.name)]"
)


def test_a_persona_composes_to_the_same_bytes_in_every_process(tmp_path: Path) -> None:
    """String hashing is salted per process, so an iteration over a set could differ.

    Three processes with different hash seeds must print identical prompts. The kill below
    is probabilistic: three fragments ordered by a salted hash agree across three seeds
    about one time in 36.

    Killed by: src/uclone_x/agent/prompts.py :: for fragment in CAPABILITY_FRAGMENTS
    Becomes: for fragment in sorted(CAPABILITY_FRAGMENTS, key=lambda f: hash(f.tool))
    """
    outputs: list[bytes] = []
    for seed in ("1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-c", _PRINT_PROMPTS],
            capture_output=True,
            check=True,
            cwd=tmp_path,
            env=env,
        )
        outputs.append(result.stdout)
    assert outputs[0]
    assert outputs[0] == outputs[1] == outputs[2]


def test_the_editor_recovers_a_restricted_persona_s_own_words() -> None:
    """The editor strips the appended default whichever tools it was composed for.

    Killed by: src/uclone_x/agent/persona_store.py :: for composed in composed_default_prompts():
    Becomes: for composed in (compose_system_prompt(),):
    """
    for name in ("scout", "writer", "artist", "guardian"):
        persona = _builtin(name)
        own, appended = split_appended_default_prompt(persona.system_prompt)
        assert appended is True, name
        assert f"{own}\n\n{default_prompt_for(persona)}" == persona.system_prompt


# --- both seat paths use the derivation --------------------------------------------------


@pytest.mark.asyncio
async def test_a_room_seat_gets_its_persona_s_derived_prompt(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/persona_store.py :: return _with_default_prompt(persona) if append_default else persona
    Becomes: return persona.model_copy(update={"system_prompt": f"{persona.system_prompt}\\n\\n{compose_system_prompt()}"}) if append_default else persona
    """
    host = HostDependencies(
        bus=EventBus(),
        llm=MockLLMConnector(),
        tools=ToolRegistry(),
        tracer=TelemetryTracer(),
        store=SessionStore(tmp_path / "sessions"),
    )
    resolver = RoomAgentResolver(host, persona_registry=PersonaRegistry(workspace_root=tmp_path))
    participant = Participant(
        id="scout",
        kind=ParticipantKind.AGENT,
        display_name="Scout",
        persona="scout",
        session_id="sess_room__r1__scout",
    )

    agent = cast(BaseAgent, await resolver.resolve(participant))

    prompt = agent.effective_system_prompt
    assert prompt.endswith(default_prompt_for(_builtin("scout")))
    assert IMAGE_GENERATION not in prompt


@pytest.mark.asyncio
async def test_a_one_to_one_session_gets_its_persona_s_derived_prompt(tmp_path: Path) -> None:
    """Killed by: src/uclone_x/agent/persona_store.py :: return _with_default_prompt(persona) if append_default else persona
    Becomes: return persona.model_copy(update={"system_prompt": f"{persona.system_prompt}\\n\\n{compose_system_prompt()}"}) if append_default else persona
    """
    manager = AgentSessionManager(
        storage_dir=tmp_path / "sessions", llm=MockLLMConnector(), workspace_dir=tmp_path
    )

    agent = await manager.get_or_create_agent("scout")

    prompt = agent.effective_system_prompt
    assert prompt.endswith(default_prompt_for(_builtin("scout")))
    assert IMAGE_GENERATION not in prompt
