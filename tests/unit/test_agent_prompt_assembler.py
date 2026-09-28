"""`agent/prompt_assembler.py` on its own: identity, turn context and the request record (#1736).

The code moved out of `agent/base.py` unchanged, and the behavioural tests that drive it
through `BaseAgent` stay where they were (`test_agent_base.py`, `test_context_snapshot.py`,
`test_read_roots.py`, ...). This file pins the module directly, and pins the one property
the move itself introduced: the assembler holds no copy of agent state, so a value swapped
after construction -- as `evals/harness_ladder/runner.py` swaps `agent._tools` -- is the
value it reads.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import (
    AgentConfig,
    AgentContext,
    PersonaDefinition,
    PlanState,
    PlanStep,
)
from uclone_x.agent.prompt_assembler import (
    TURN_CONTEXT_HEADER,
    AnchorWriter,
    PromptAssembler,
    PromptScope,
    compose_identity_prompt,
    persisted_anchor_provenance,
    restored_anchor_provenance,
    turn_context_block,
)
from uclone_x.agent.session import ContextSnapshot
from uclone_x.llm.models import ChatMessage, LLMRequest, MessageRole
from uclone_x.tools.builtin.filesystem import FileReadTool
from uclone_x.tools.protocols import ToolRegistryProtocol
from uclone_x.tools.registry import ToolRegistry


@dataclass
class _Session:
    stored_bodies: set[str] = field(default_factory=set[str])
    pending_bodies: dict[str, str] = field(default_factory=dict[str, str])
    context_snapshots: list[ContextSnapshot] = field(default_factory=list[ContextSnapshot])
    last_conversation: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    last_request: int | None = None


class _FileToolRegistry:
    """A registry holding only `file_read`, which is all the workspace section asks about."""

    def get(self, name: str) -> object | None:
        return object() if name == "file_read" else None


@dataclass
class _State:
    """Mutable agent-shaped state the scope reads through, so a test can swap it."""

    config: AgentConfig
    tools: ToolRegistryProtocol | None = None
    workspace: Path | None = None
    plan: PlanState | None = None
    history: list[ChatMessage] = field(default_factory=list[ChatMessage])
    session: _Session = field(default_factory=_Session)
    effective_prompt: str = "EFFECTIVE"


def _assembler(state: _State) -> PromptAssembler:
    return PromptAssembler(
        PromptScope(
            ontology=lambda: None,
            skills=lambda: None,
            config=lambda: state.config,
            tools=lambda: state.tools,
            memory=lambda: None,
            workspace_root=lambda: state.workspace,
            current_plan=lambda: state.plan,
            history=lambda: state.history,
            active_session=lambda: state.session,
            turn_counter=lambda: 1,
            anchor_is_stale=lambda: False,
            system_prompt_base=lambda: "BASE",
            effective_system_prompt=lambda: state.effective_prompt,
        )
    )


def _config(**overrides: Any) -> AgentConfig:
    return AgentConfig(agent_id="assembler", name="Assembler", **overrides)


def test_seat_framing_puts_the_persona_instructions_under_a_labelled_header() -> None:
    """Framing plus a persona with instructions: framing first, then the labelled persona.

    Killed by: src/uclone_x/agent/prompt_assembler.py :: if persona is not None and persona.system_prompt:
    Becomes: if False:
    """
    persona = PersonaDefinition(name="scout", role="Scout", system_prompt="You scout.")

    prompt = compose_identity_prompt(
        config_prompt="CONFIG", persona=persona, seat_framing="You sit in room R."
    )

    assert prompt == "You sit in room R.\n\n[Persona Instructions: Scout]\nYou scout."


def test_the_turn_context_block_opens_with_its_header_and_is_empty_without_sections() -> None:
    r"""The block says what it is, and is absent when there is nothing to say.

    Killed by: src/uclone_x/agent/prompt_assembler.py :: return "\n\n".join((TURN_CONTEXT_HEADER, *sections))
    Becomes: return "\n\n".join(sections)
    """
    assert turn_context_block([]) == ""
    assert turn_context_block(["[Plan] a", "[Memory] b"]) == (
        f"{TURN_CONTEXT_HEADER}\n\n[Plan] a\n\n[Memory] b"
    )


def test_an_unrecorded_stamp_round_trips_as_unrecorded_never_as_no_persona() -> None:
    """A record with no stamp stays a gap; it is not read as "composed under no persona"."""
    assert persisted_anchor_provenance(AnchorWriter.UNRECORDED) is None
    assert restored_anchor_provenance(None) is AnchorWriter.UNRECORDED


def test_the_assembler_reads_a_registry_swapped_in_after_construction(tmp_path: Path) -> None:
    """The scope is read on every call, so a later `_tools` swap is what the section sees.

    Built with no registry, the workspace section is withheld: nothing in the allowed list
    is a file tool that exists. Swapping a registry that holds `file_read` in afterwards
    must bring the section back without rebuilding the assembler.

    Killed by: src/uclone_x/agent/prompt_assembler.py :: return self._scope.tools()
    Becomes: return None
    """
    state = _State(config=_config(allowed_tools=("file_read",)), workspace=tmp_path)
    assembler = _assembler(state)
    assert assembler.get_workspace_prompt_section() is None

    state.tools = cast(ToolRegistryProtocol, _FileToolRegistry())

    section = assembler.get_workspace_prompt_section()
    assert section is not None
    assert str(tmp_path) in section


def test_with_no_anchor_the_turn_sends_the_effective_prompt_and_the_plan_rides_at_the_tail() -> (
    None
):
    """No `SYSTEM` turn in history: identity is the effective prompt; the plan is turn context.

    Killed by: src/uclone_x/agent/prompt_assembler.py :: turn_context = turn_context_block(turn_sections)
    Becomes: turn_context = ""
    """
    plan = PlanState(title="Ship it", steps=(PlanStep(index=1, description="write"),))
    state = _State(
        config=_config(),
        plan=plan,
        history=[ChatMessage(role=MessageRole.USER, content="go")],
    )

    layers = _assembler(state).prepare_turn_layers(extra_sections=("[Extra] x",))

    assert layers.identity == "EFFECTIVE"
    assert layers.slow_context == ""
    assert layers.conversation == (ChatMessage(role=MessageRole.USER, content="go"),)
    assert layers.turn_context.startswith(TURN_CONTEXT_HEADER)
    assert "[Extra] x" in layers.turn_context
    assert "### Active Execution Plan: Ship it" in layers.turn_context


def test_request_context_fields_chain_each_request_to_the_one_before() -> None:
    """The second request names the first as its base and records only what it added.

    Killed by: src/uclone_x/agent/prompt_assembler.py :: session.last_request = request_number
    Becomes: pass
    """
    state = _State(config=_config(), history=[ChatMessage(role=MessageRole.USER, content="a")])
    assembler = _assembler(state)

    def fields() -> dict[str, Any]:
        layers = assembler.prepare_turn_layers()
        messages: Sequence[ChatMessage] = layers.conversation
        return assembler.request_context_fields(1, LLMRequest(messages=tuple(messages)), layers)

    first = fields()
    state.history.append(ChatMessage(role=MessageRole.ASSISTANT, content="b"))
    second = fields()

    assert (first["request"], first["base_request"]) == (1, None)
    assert (second["request"], second["base_request"]) == (2, 1)
    assert second["kept_message_count"] == 1
    assert [m["content"] for m in second["appended_messages"]] == ["b"]
    assert len(state.session.context_snapshots) == 1


def test_an_agent_whose_tools_are_swapped_after_construction_is_assembled_from_the_new_ones(
    tmp_path: Path,
) -> None:
    """The agent hands the assembler a live read of `_tools`, not the registry it was built with.

    `evals/harness_ladder/runner.py` assigns `agent._tools` after construction. A scope
    that captured the registry at construction would keep describing the old one.

    Killed by: src/uclone_x/agent/base.py :: tools=lambda: self._tools,
    Becomes: tools=lambda _t=self._tools: _t,
    """
    agent = BaseAgent(
        config=_config(allowed_tools=("file_read",)),
        tools=ToolRegistry(),
        context=AgentContext(session_id="s", agent_id="assembler", workspace_root=tmp_path),
    )
    assert agent._get_workspace_prompt_section() is None  # pyright: ignore[reportPrivateUsage]

    agent._tools = ToolRegistry(tools=[FileReadTool()])  # pyright: ignore[reportPrivateUsage]

    section = agent._get_workspace_prompt_section()  # pyright: ignore[reportPrivateUsage]
    assert section is not None and section.startswith("[Workspace]")
